"""
Pose Re-ID Skeleton-based Camera Processor

Este script procesa video en tiempo real desde la cámara del dispositivo para detectar poses humanas,
asignar IDs únicos a personas usando ReID (embeddings visuales) o descriptores de esqueleto como fallback,
y opcionalmente IoU para seguimiento a corto plazo.

Flujo principal:
1. Detección de poses con YOLOv8.
2. Extracción de features (ReID o esqueleto).
3. Asignación de IDs: IoU -> Similitud -> Reaparición.
4. Mostrar video anotado en tiempo real.

Uso: python pose_reid_skeleton_camera.py --use-reid
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from typing import Optional

# Configuración para evitar conflictos con MKL/OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constantes por defecto para umbrales y configuraciones
DEFAULT_SIM_THRESHOLD = 0.75
DEFAULT_IOU_THRESHOLD = 0.95
DEFAULT_MAX_AGE = 1
DEFAULT_FEATURE_HISTORY = 10
DEFAULT_CONF = 0.40
DEFAULT_SCALE = 1.0  # Cambiado a 1.0 para ventana más grande en cámara
DEFAULT_REID_THRESHOLD = 0.8
DEFAULT_REAPPEAR_THRESHOLD = 0.6

class PersonDatabase:
    """
    Clase para manejar la base de datos de personas en el sistema de Re-ID.
    """
    def __init__(self):
        self.db = {}
        self.next_id = 1

    def add_person(self, features, bbox, frame_index, feature_history):
        """
        Agrega una nueva persona a la base de datos.

        Args:
            features: Vector de características.
            bbox: Caja delimitadora (x1,y1,x2,y2).
            frame_index: Índice del frame actual.
            feature_history: Longitud del historial de features.

        Returns:
            int: ID asignado a la nueva persona.
        """
        pid = self.next_id
        self.next_id += 1
        hist = deque([features.copy()], maxlen=feature_history) if features is not None else deque(maxlen=feature_history)
        self.db[pid] = {
            'feat': features.copy() if features is not None else None,
            'hist': hist,
            'bbox': bbox,
            'last_seen': frame_index,
            'misses': 0
        }
        return pid

    def update_person(self, pid, features, bbox, frame_index):
        """
        Actualiza la información de una persona existente.

        Args:
            pid: ID de la persona.
            features: Nuevas características.
            bbox: Nueva caja delimitadora.
            frame_index: Índice del frame actual.
        """
        if pid not in self.db:
            return
        if features is not None:
            self.db[pid]['hist'].append(features.copy())
            avg = np.mean(np.stack(self.db[pid]['hist'], axis=0), axis=0)
            n = np.linalg.norm(avg)
            self.db[pid]['feat'] = (avg / n) if n > 1e-6 else avg
        self.db[pid]['bbox'] = bbox
        self.db[pid]['last_seen'] = frame_index
        self.db[pid]['misses'] = 0

    def get_recent_ids(self, frame_index, max_age):
        """
        Obtiene IDs de personas vistas recientemente.

        Args:
            frame_index: Índice del frame actual.
            max_age: Máximo frames sin ver.

        Returns:
            list: Lista de IDs recientes con bbox válido.
        """
        return [pid for pid in self.db.keys() if (frame_index - self.db[pid]['last_seen']) <= max_age and self.db[pid]['bbox'] is not None]

    def increment_misses(self, frame_index):
        """
        Incrementa el contador de misses para personas no vistas.

        Args:
            frame_index: Índice del frame actual.
        """
        for pid in list(self.db.keys()):
            if (frame_index - self.db[pid]['last_seen']) >= 0:
                self.db[pid]['misses'] = self.db[pid].get('misses', 0) + 1

    def clear(self):
        """Limpia la base de datos."""
        self.db.clear()
        self.next_id = 1

    def __len__(self):
        return len(self.db)

    def keys(self):
        return self.db.keys()

    def __getitem__(self, key):
        return self.db[key]

    def __contains__(self, key):
        return key in self.db


def parse_args():
    """
    Parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Argumentos parseados.
    """
    parser = argparse.ArgumentParser(description='Pose Re-ID skeleton basado en cámara')
    parser.add_argument('--model', type=str, default=r'C:\Users\Carlo\Desktop\Camera-Processor-ros2\src\camera_processor\models\yolov8n-pose.pt', help='Modelo YOLOv8 pose (.pt)')
    parser.add_argument('--device', type=str, default=None, help='Dispositivo torch: cuda, cuda:0, cpu')
    parser.add_argument('--sim-threshold', type=float, default=DEFAULT_SIM_THRESHOLD, help='Umbral de similitud (0-1)')
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD, help='Umbral IoU para emparejar con historial reciente')
    parser.add_argument('--max-age', type=int, default=DEFAULT_MAX_AGE, help='Cuadros max sin ver una persona para usar IoU (memoria corta)')
    parser.add_argument('--feature-history', type=int, default=DEFAULT_FEATURE_HISTORY, help='Longitud del historial para promediar el descriptor por persona')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Confianza minima deteccion YOLO')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, help='Factor de escala para reducir tamaño (por defecto 0.5 = mitad)')
    # Opciones ReID por imagen (OSNet via torchreid)
    parser.add_argument('--use-reid', action='store_true', help='Usar un modelo ReID (OSNet) para embeddings por persona (mejor precisión).')
    parser.add_argument('--reid-model', type=str, default='osnet_x0_25', help='Nombre del modelo ReID en torchreid (ej: osnet_x0_25, osnet_x0_5, osnet_x1_0).')
    parser.add_argument('--reid-size', type=int, default=256, help='Tamaño de resize cuadrado para el crop de persona (ej: 256).')
    parser.add_argument('--reid-threshold', type=float, default=DEFAULT_REID_THRESHOLD, help='Umbral de similitud coseno para embeddings ReID (0-1).')
    parser.add_argument('--reappear-threshold', type=float, default=DEFAULT_REAPPEAR_THRESHOLD, help='Umbral de similitud para reapariciones de IDs viejos (0-1).')
    parser.add_argument('--no-iou', action='store_true', help='Desactivar asignación por IoU')
    return parser.parse_args()


def extract_skeleton_features(keypoints: np.ndarray):
    """
    Extrae un descriptor de características basado en keypoints del esqueleto.

    Normaliza por escala y orientación para robustez. Incluye longitudes de huesos,
    ángulos articulares y otras métricas.

    Args:
        keypoints (np.ndarray): Array de keypoints (17, 3) con x, y, conf.

    Returns:
        np.ndarray or None: Vector de características normalizado, o None si falla.
    """
    conf_thr = 0.5
    if keypoints is None or keypoints.shape[0] < 17:
        return None

    pts = keypoints[:, :2].astype(np.float32)
    vis = keypoints[:, 2] > conf_thr

    # Selecciona raíz (pelvis) y escala (ancho hombros/ caderas / torso)
    def have(i):
        return bool(vis[i])

    def dist(a, b):
        if have(a) and have(b):
            v = pts[a] - pts[b]
            return float(np.hypot(v[0], v[1]))
        return None

    # Raíz: centro de caderas si disponible, si no centro de hombros, si no nariz
    root = None
    if have(11) and have(12):
        root = (pts[11] + pts[12]) / 2.0
    elif have(5) and have(6):
        root = (pts[5] + pts[6]) / 2.0
    elif have(0):
        root = pts[0]
    else:
        return None

    # Escala preferente: ancho de hombros, luego caderas, luego altura torso
    scale_candidates = []
    d_sh = dist(5, 6)
    d_hip = dist(11, 12)
    if d_sh: scale_candidates.append(d_sh)
    if d_hip: scale_candidates.append(d_hip)
    if have(5) and have(11):
        scale_candidates.append(dist(5, 11))
    if have(6) and have(12):
        scale_candidates.append(dist(6, 12))
    scale = None
    if len(scale_candidates) > 0:
        scale = float(np.median([s for s in scale_candidates if s is not None and s > 1e-3]))
    if not scale or scale <= 1e-3:
        # Fallback: max distancia válida
        dists = [dist(i, j) for i in range(17) for j in range(i + 1, 17)]
        dists = [d for d in dists if d is not None]
        if len(dists) == 0:
            return None
        scale = max(dists)
    if scale <= 1e-3:
        return None

    # Coordenadas normalizadas centradas en raíz
    norm = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        if vis[i]:
            norm[i] = (pts[i] - root) / scale
        else:
            norm[i] = 0.0

    # Longitudes óseas normalizadas
    bones = [
        (5, 7), (7, 9),   # brazo izq
        (6, 8), (8, 10),  # brazo der
        (11, 13), (13, 15),  # pierna izq
        (12, 14), (14, 16),  # pierna der
        (5, 6), (11, 12),    # hombros, caderas
        (5, 11), (6, 12)     # torso
    ]
    bone_lengths = []
    for a, b in bones:
        d = dist(a, b)
        bone_lengths.append((d / scale) if d else 0.0)

    # Ángulos en articulaciones principales (codo, rodilla, hombro, cadera)
    def angle_at(a, b, c):
        # ángulo en b formado por a-b-c (usa coords normalizadas)
        va = norm[a] - norm[b]
        vc = norm[c] - norm[b]
        na = np.linalg.norm(va)
        nc = np.linalg.norm(vc)
        if na < 1e-6 or nc < 1e-6:
            return 0.0, 1.0  # cos=1, sin=0 (neutral)
        va /= na; vc /= nc
        cos_t = float(np.clip(np.dot(va, vc), -1.0, 1.0))
        # Para continuidad añade sin con signo usando producto cruzado 2D
        sin_t = float(np.clip(va[0] * vc[1] - va[1] * vc[0], -1.0, 1.0))
        return cos_t, sin_t

    angle_triplets = [
        (5, 7, 9), (6, 8, 10),    # codos
        (11, 13, 15), (12, 14, 16),  # rodillas
        (11, 5, 7), (12, 6, 8),    # hombros con torso
        (5, 11, 13), (6, 12, 14)   # caderas con piernas
    ]
    angle_feats = []
    for a, b, c in angle_triplets:
        cos_t, sin_t = angle_at(a, b, c)
        angle_feats += [cos_t, sin_t]

    # Orientación del tronco (vector hombros) en cos/sin
    orient_cos, orient_sin = 1.0, 0.0
    if have(5) and have(6):
        v = norm[6] - norm[5]
        ang = np.arctan2(v[1], v[0])
        orient_cos = float(np.cos(ang))
        orient_sin = float(np.sin(ang))

    # Visibilidad: proporción de puntos visibles
    vis_ratio = float(np.mean(vis.astype(np.float32)))

    feat = np.array(bone_lengths + angle_feats + [orient_cos, orient_sin, vis_ratio], dtype=np.float32)
    # Normaliza vector final para estabilizar la similitud coseno
    n = np.linalg.norm(feat)
    if n > 1e-6:
        feat = feat / n
    return feat


def find_or_create_person_id(skeleton_features: np.ndarray, db: dict, next_id_ref: dict, similarity_threshold: float):
    # Ya no se usa en el flujo principal; se mantiene por compatibilidad si alguien lo llama.
    if skeleton_features is None:
        return None
    if not db:
        pid = next_id_ref['value']
        next_id_ref['value'] += 1
        db[pid] = {
            'feat': skeleton_features.copy(),
            'hist': deque([skeleton_features.copy()], maxlen=10),
            'bbox': None,
            'last_seen': 0,
            'misses': 0
        }
        return pid


def iou_xyxy(a, b):
    """
    Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras.

    Args:
        a, b: Tuplas (x1, y1, x2, y2) de las cajas.

    Returns:
        float: Valor IoU entre 0 y 1.
    """
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def assign_ids_greedy(det_features: list, det_boxes: list, person_db: PersonDatabase,
                      similarity_threshold: float, iou_threshold: float, frame_index: int,
                      max_age: int, feature_history: int, reappear_threshold: float = 0.6, counters: dict = None, use_iou: bool = True):
    """
    Asigna IDs a detecciones usando un algoritmo greedy con etapas:
    1. IoU (opcional): Matching por proximidad espacial.
    2. Similitud: Matching por embeddings (ReID o esqueleto).
    3. Reaparición: Reasignación de IDs viejos con umbral bajo.

    Args:
        det_features: Lista de vectores de características por detección.
        det_boxes: Lista de cajas (x1,y1,x2,y2) por detección.
        person_db: Instancia de PersonDatabase.
        similarity_threshold: Umbral para matching por similitud.
        iou_threshold: Umbral para IoU.
        frame_index: Índice del frame actual.
        max_age: Máx frames para considerar IoU.
        feature_history: Longitud del historial de features.
        reappear_threshold: Umbral para reapariciones.
        counters: Diccionario para contar asignaciones.
        use_iou: Si usar IoU o no.

    Returns:
        list: Lista de IDs asignados por detección.
    """
    if counters is None:
        counters = {}
    counters.setdefault('iou_assignments', 0)
    counters.setdefault('sim_assignments', 0)
    counters.setdefault('reappear_assignments', 0)
    assigned = [None] * len(det_features)
    existing_ids = list(person_db.keys())

    used_dets = set()
    used_ids = set()

    if use_iou:
        # 1) Intento por IoU con historiales recientes
        recent_ids = person_db.get_recent_ids(frame_index, max_age)
        iou_pairs = []  # (sim, det_i, pid)
        for i, box in enumerate(det_boxes):
            if box is None:
                continue
            for pid in recent_ids:
                iou = iou_xyxy(box, person_db[pid]['bbox'])
                if iou >= iou_threshold:
                    iou_pairs.append((iou, i, pid))
        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        for iou, det_i, pid in iou_pairs:
            if det_i in used_dets or pid in used_ids:
                continue
            assigned[det_i] = pid
            used_dets.add(det_i)
            used_ids.add(pid)
            counters['iou_assignments'] += 1

    # 2) Emparejamiento por descriptor (coseno) para los que quedaron
    sim_pairs = []  # (sim, det_i, pid)
    for i, feat in enumerate(det_features):
        if i in used_dets or feat is None:
            continue
        for pid in existing_ids:
            if pid in used_ids:
                continue
            ref = person_db[pid]['feat']
            sim = float(cosine_similarity(feat.reshape(1, -1), ref.reshape(1, -1))[0][0])
            if sim >= similarity_threshold:
                sim_pairs.append((sim, i, pid))
    sim_pairs.sort(key=lambda x: x[0], reverse=True)
    for sim, det_i, pid in sim_pairs:
        if det_i in used_dets or pid in used_ids:
            continue
        assigned[det_i] = pid
        used_dets.add(det_i)
        used_ids.add(pid)
        counters['sim_assignments'] += 1

    # 2.5) Etapa de reaparición: emparejar con IDs viejos usando similitud alta
    reappear_pairs = []
    for i, feat in enumerate(det_features):
        if assigned[i] is not None or feat is None:
            continue
        for pid in existing_ids:
            if pid in used_ids:
                continue
            ref = person_db[pid]['feat']
            sim = float(cosine_similarity(feat.reshape(1, -1), ref.reshape(1, -1))[0][0])
            if sim >= reappear_threshold:
                reappear_pairs.append((sim, i, pid))
    reappear_pairs.sort(key=lambda x: x[0], reverse=True)
    for sim, det_i, pid in reappear_pairs:
        if assigned[det_i] is not None or pid in used_ids:
            continue
        assigned[det_i] = pid
        used_ids.add(pid)
        counters['reappear_assignments'] += 1

    # 3) Actualiza tracks emparejados
    for i, pid in enumerate(assigned):
        if pid is None:
            continue
        feat = det_features[i]
        box = det_boxes[i]
        person_db.update_person(pid, feat, box, frame_index)

    # 4) Crea nuevos IDs para detecciones no emparejadas con descriptor válido
    for i, feat in enumerate(det_features):
        if assigned[i] is None and feat is not None:
            pid = person_db.add_person(feat, det_boxes[i], frame_index, feature_history)
            assigned[i] = pid

    # 5) Incrementa misses para no emparejados (opcional limpieza futura)
    person_db.increment_misses(frame_index)

    return assigned


def setup_reid_model(model_name: str, device: str):
    try:
        import torchreid
    except Exception:
        return None
    model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=True)
    model.eval().to(device)
    return model


def compute_reid_embedding(img_bgr: np.ndarray, reid_model, device: str, size: int) -> Optional[np.ndarray]:
    try:
        import torch
        from torchvision import transforms
    except Exception:
        return None
    if img_bgr is None or img_bgr.size == 0 or reid_model is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize cuadrado con pad
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    img_res = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    img_sq = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        tensor = tfm(img_sq).unsqueeze(0).to(device)
        feat = reid_model(tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        vec = feat.squeeze().detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 1e-6:
            vec = vec / n
        return vec
    return None


def initialize_models(args):
    """
    Inicializa los modelos YOLO y ReID.

    Args:
        args: Argumentos parseados.

    Returns:
        tuple: (model, reid_model, device)
    """
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    model = YOLO(args.model).to(device)
    # ReID opcional
    reid_model = None
    if args.use_reid:
        try:
            reid_model = setup_reid_model(args.reid_model, device)
        except Exception as e:
            reid_model = None
            print(f'Error cargando ReID: {e}')
        if reid_model is None:
            print('Aviso: torchreid no está disponible. Usaré descriptor por esqueleto (fallback).')
            args.use_reid = False
        else:
            print(f'ReID activado: {args.reid_model}')

    return model, reid_model, device


def setup_camera(args):
    """
    Configura la captura de video desde la cámara.

    Args:
        args: Argumentos parseados.

    Returns:
        tuple: (cap, out_width, out_height)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError('No se pudo abrir la cámara')

    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.scale != 1.0 and args.scale > 0:
        out_width = max(1, int(out_width * args.scale))
        out_height = max(1, int(out_height * args.scale))

    return cap, out_width, out_height


def process_frame(frame, model, reid_model, args, person_db: PersonDatabase, frame_index, counters, out_width, out_height):
    """
    Procesa un frame individual: detección, extracción de features, asignación de IDs, anotación.

    Args:
        frame: Frame de video.
        model: Modelo YOLO.
        reid_model: Modelo ReID (opcional).
        args: Argumentos parseados.
        person_database: Base de datos de personas.
        frame_index: Índice del frame actual.
        counters: Diccionario de contadores.
        out_width, out_height: Dimensiones de salida.

    Returns:
        np.ndarray: Frame anotado.
    """
    results = model.predict(frame, device=model.device, verbose=False, conf=args.conf)
    if not results:
        return cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

    annotated_frame = results[0].plot()
    orig_h, orig_w = annotated_frame.shape[:2]
    scale_x = out_width / float(orig_w) if orig_w > 0 else 1.0
    scale_y = out_height / float(orig_h) if orig_h > 0 else 1.0
    if (annotated_frame.shape[1] != out_width) or (annotated_frame.shape[0] != out_height):
        annotated_frame = cv2.resize(annotated_frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

    if results[0].boxes is not None and results[0].keypoints is not None:
        boxes = results[0].boxes
        kpts_all = results[0].keypoints
        n = len(boxes)
        det_features = []
        det_xy = []
        det_boxes = []
        for i in range(n):
            conf = float(boxes.conf[i]) if boxes.conf is not None else 1.0
            if conf < args.conf:
                det_features.append(None)
                det_xy.append((0, 0))
                det_boxes.append(None)
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            kpts_array = kpts_all.data[i].cpu().numpy()
            features = None
            if args.use_reid:
                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(frame.shape[1], x2); y2c = min(frame.shape[0], y2)
                crop = frame[y1c:y2c, x1c:x2c]
                features = compute_reid_embedding(crop, reid_model, model.device, args.reid_size)
                if features is not None:
                    counters['reid_detections'] += 1
            if features is None:
                features = extract_skeleton_features(kpts_array)
                if features is not None:
                    counters['skeleton_detections'] += 1
            det_features.append(features)
            det_xy.append((x1, y1))
            det_boxes.append((x1, y1, x2, y2))

        assigned_ids = assign_ids_greedy(
            det_features=det_features,
            det_boxes=det_boxes,
            person_db=person_db,
            similarity_threshold=(args.reid_threshold if args.use_reid else args.sim_threshold),
            iou_threshold=args.iou_threshold,
            frame_index=frame_index,
            max_age=args.max_age,
            feature_history=args.feature_history,
            reappear_threshold=args.reappear_threshold,
            counters=counters,
            use_iou=not args.no_iou,
        )

        for i in range(n):
            pid = assigned_ids[i]
            if pid is None:
                continue
            x1, y1 = det_xy[i]
            label = f'ID: {pid}'
            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            y_text = max(0, sy1 - int(15 * scale_y))
            cv2.putText(annotated_frame, label, (sx1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(annotated_frame, label, (sx1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return annotated_frame


def display_statistics(person_db: PersonDatabase, counters, fps):
    """
    Muestra estadísticas finales del procesamiento.

    Args:
        person_database: Base de datos de personas.
        counters: Diccionario de contadores.
        fps: FPS promedio.
    """
    print('\n=== ESTADISTICAS FINALES ===')
    print(f'FPS promedio: {fps:.2f}')
    print(f'Personas unicas detectadas: {len(person_db)}')
    print(f'IDs asignados: {list(person_db.keys())}')
    print(f'ReID embeddings usados: {counters["reid_detections"]} | Fallback esqueleto: {counters["skeleton_detections"]} | Asignaciones por IoU: {counters["iou_assignments"]} | ReID or Skele assigments: {counters["sim_assignments"] + counters["reappear_assignments"]}')


def main():
    """
    Función principal: Carga argumentos, inicializa modelos, procesa video en tiempo real desde la cámara,
    asigna IDs, muestra video anotado y estadísticas finales.
    """
    args = parse_args()
    if not os.path.exists(args.model):
        print(f'No existe el modelo: {args.model}')
        return 1

    model, reid_model, device = initialize_models(args)
    cap, out_width, out_height = setup_camera(args)

    person_db = PersonDatabase()
    similarity_threshold = args.sim_threshold

    # Determinar umbral activo basado en si se usa ReID
    umbral_activo = args.reid_threshold if args.use_reid else similarity_threshold
    tipo_umbral = "(reid)" if args.use_reid else "(skele)"

    fps_counter = 0
    start_time = time.time()
    fps = 0.0
    frame_index = 0

    # Contadores para diagnostico
    counters = {
        'reid_detections': 0,
        'skeleton_detections': 0,
        'iou_assignments': 0,
        'sim_assignments': 0,
        'reappear_assignments': 0
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No se pudo capturar frame de la cámara')
                break
            frame_index += 1

            annotated_frame = process_frame(frame, model, reid_model, args, person_db, frame_index, counters, out_width, out_height)

            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()

            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Personas unicas: {len(person_db)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f'Umbral similitud {tipo_umbral}: {umbral_activo:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f'Frame: {frame_index}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow('Skeleton-based Re-ID (Camera)', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                if args.use_reid:
                    args.reid_threshold = min(0.99, args.reid_threshold + 0.02)
                    umbral_activo = args.reid_threshold
                else:
                    similarity_threshold = min(0.99, similarity_threshold + 0.02)
                    umbral_activo = similarity_threshold
                print(f'Umbral {tipo_umbral}: {umbral_activo:.2f}')
            elif key == ord('-'):
                if args.use_reid:
                    args.reid_threshold = max(0.50, args.reid_threshold - 0.02)
                    umbral_activo = args.reid_threshold
                else:
                    similarity_threshold = max(0.50, similarity_threshold - 0.02)
                    umbral_activo = similarity_threshold
                print(f'Umbral {tipo_umbral}: {umbral_activo:.2f}')
            elif key == ord('r'):
                person_db.clear()
                print('Base de datos reseteada')

        display_statistics(person_db, counters, fps)
    except KeyboardInterrupt:
        print('\nInterrupcion manual')
    except Exception as e:
        print(f'Error: {e}')
        import traceback; traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    sys.exit(main())