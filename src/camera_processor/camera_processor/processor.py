import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
from typing import Optional
import scipy.optimize
from sklearn.metrics.pairwise import cosine_similarity

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

# Configuração para evitar conflitos com MKL/OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
Processa um único frame de vídeo a preto e branco.

Args:
    frame (np.ndarray): Imagem OpenCV BGR

Returns:
    processed_frame (np.ndarray): Frame processado
"""
def process_frame_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


"""
Processa um único frame de vídeo a preto e branco.

Args:
    frame (np.ndarray): Imagem OpenCV BGR

Returns:
    red_highlighted (np.ndarray): Frame processado (apenas regiões vermelhas visíveis)
    detected (boolean): True se algum pixel vermelho foi detetado
"""
def color_process_frame(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color ranges
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([5, 255, 255])

    lower_red2 = np.array([175, 150, 70])
    upper_red2 = np.array([180, 255, 255])


    # Threshold for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Binary output for visualization
    red_highlighted = cv2.bitwise_and(frame, frame, mask=mask)

    # Determine if any red was detected
    detected = np.any(mask > 0)

    return red_highlighted, detected

def person_process_frame_old(frame):
    pkg_share = get_package_share_directory('camera')
    model_dir = os.path.join(pkg_share, 'model')
    model_path = os.path.join(model_dir, 'yolov8n-pose.pt')

    model = YOLO(model_path)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    people_count = len(results[0].boxes)
    people_detected = people_count > 0

    return annotated_frame, people_detected, people_count


# --- Auxiliares de ReID baseados em pose com CNN e IoU ---
try:
    _pkg_share_processor = get_package_share_directory('camera_processor')
except PackageNotFoundError:
    _pkg_share_processor = None

_local_model_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-pose.pt')
)

def _resolve_model_path() -> str:
    candidates = []
    if _pkg_share_processor:
        candidates.append(os.path.join(_pkg_share_processor, 'models', 'yolov8n-pose.pt'))
    candidates.append(_local_model_path)

    try:
        camera_share = get_package_share_directory('camera')
    except PackageNotFoundError:
        camera_share = None

    if camera_share:
        candidates.append(os.path.join(camera_share, 'model', 'yolov8n-pose.pt'))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return _local_model_path


_POSE_MODEL_PATH = _resolve_model_path()
if not os.path.exists(_POSE_MODEL_PATH):
    raise FileNotFoundError(f"YOLO pose model not found at {_POSE_MODEL_PATH}")

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_POSE_MODEL = YOLO(_POSE_MODEL_PATH)
if _DEVICE == 'cuda':
    _POSE_MODEL.to(_DEVICE)

# Configuraciones para ReID e IoU
_USE_REID = True  # Cambia a False para usar solo skeleton
_REID_MODEL_NAME = 'osnet_x0_25'
_REID_SIZE = 256
_SIMILARITY_THRESHOLD = 0.75
_IOU_THRESHOLD = 0.95
_MAX_AGE = 1
_FEATURE_HISTORY = 10
_KEYPOINT_CONF_THRESHOLD = 0.4
DEFAULT_CONF = 0.40
_REAPPEAR_THRESHOLD = 0.6
_REID_THRESHOLD = 0.7

_PERSON_DATABASE = {}
_NEXT_PERSON_ID = 1
_FRAME_INDEX = 0  # Contador global de frames para ROS2

# Modelo ReID
_REID_MODEL = None
if _USE_REID:
    try:
        import torchreid
        _REID_MODEL = torchreid.models.build_model(name=_REID_MODEL_NAME, num_classes=1000, pretrained=True)
        _REID_MODEL.eval().to(_DEVICE)
    except Exception as e:
        print(f"ReID model not available: {e}. Falling back to skeleton features.")
        _USE_REID = False


class PersonDatabase:
    """
    Classe para gerir a base de dados de pessoas no sistema de Re-ID.
    """
    def __init__(self):
        self.db = {}
        self.next_id = 1

    def add_person(self, features, bbox, frame_index, feature_history):
        """
        Adiciona uma nova pessoa à base de dados.

        Args:
            features: Vetor de características.
            bbox: Caixa delimitadora (x1,y1,x2,y2).
            frame_index: Índice do frame atual.
            feature_history: Comprimento do histórico de features.

        Returns:
            int: ID atribuído à nova pessoa.
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
        Atualiza a informação de uma pessoa existente.

        Args:
            pid: ID da pessoa.
            features: Novas características.
            bbox: Nova caixa delimitadora.
            frame_index: Índice do frame atual.
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
        Obtém IDs de pessoas vistas recentemente.

        Args:
            frame_index: Índice do frame atual.
            max_age: Máximo frames sem ver.

        Returns:
            list: Lista de IDs recentes com bbox válido.
        """
        return [pid for pid in self.db.keys() if (frame_index - self.db[pid]['last_seen']) <= max_age and self.db[pid]['bbox'] is not None]

    def increment_misses(self, frame_index):
        """
        Incrementa o contador de misses para pessoas não vistas.

        Args:
            frame_index: Índice do frame atual.
        """
        for pid in list(self.db.keys()):
            if (frame_index - self.db[pid]['last_seen']) >= 0:
                self.db[pid]['misses'] += 1

    def clear(self):
        """Limpa a base de dados."""
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


_PERSON_DB = PersonDatabase()


def get_similarity_threshold() -> float:
    return _SIMILARITY_THRESHOLD


def set_similarity_threshold(value: float) -> float:
    global _SIMILARITY_THRESHOLD
    _SIMILARITY_THRESHOLD = max(0.5, min(0.99, float(value)))
    return _SIMILARITY_THRESHOLD


def adjust_similarity_threshold(delta: float) -> float:
    return set_similarity_threshold(_SIMILARITY_THRESHOLD + delta)


def reset_person_database() -> None:
    global _PERSON_DB
    _PERSON_DB.clear()


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def extract_skeleton_features(keypoints: np.ndarray):
    """
    Extrai um descritor de características baseado nos keypoints do esqueleto.

    Normaliza por escala e orientação para robustez. Inclui comprimentos de ossos,
    ângulos articulares e outras métricas.

    Args:
        keypoints (np.ndarray): Array de keypoints (17, 3) com x, y, conf.

    Returns:
        np.ndarray or None: Vetor de características normalizado, ou None se falhar.
    """
    conf_thr = _KEYPOINT_CONF_THRESHOLD
    if keypoints is None or keypoints.shape[0] < 17:
        return None

    pts = keypoints[:, :2].astype(np.float32)
    vis = keypoints[:, 2] > conf_thr

    # Seleciona raiz (pelve) e escala (largura ombros/ ancas / torso)
    def have(i):
        return bool(vis[i])

    def dist(a, b):
        if have(a) and have(b):
            return float(np.sqrt((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2))
        return None

    # Raiz: centro das ancas se disponível, senão centro dos ombros, senão nariz
    root = None
    if have(11) and have(12):
        root = (pts[11] + pts[12]) / 2.0
    elif have(5) and have(6):
        root = (pts[5] + pts[6]) / 2.0
    elif have(0):
        root = pts[0]
    else:
        return None

    # Escala preferencial: largura dos ombros, depois ancas, depois altura do torso
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
        # Fallback: max distância válida
        dists = [dist(i, j) for i in range(17) for j in range(i + 1, 17)]
        dists = [d for d in dists if d is not None]
        if len(dists) == 0:
            return None
        scale = max(dists)
    if scale <= 1e-3:
        return None

    # Coordenadas normalizadas centradas na raiz
    norm = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        if vis[i]:
            norm[i] = (pts[i] - root) / scale
        else:
            norm[i] = np.array([0.0, 0.0])

    # Comprimentos ósseos normalizados
    bones = [
        (5, 7), (7, 9),   # braço esq
        (6, 8), (8, 10),  # braço dir
        (11, 13), (13, 15),  # perna esq
        (12, 14), (14, 16),  # perna dir
        (5, 6), (11, 12),    # ombros, ancas
        (5, 11), (6, 12)     # torso
    ]
    bone_lengths = []
    for a, b in bones:
        d = dist(a, b)
        bone_lengths.append((d / scale) if d else 0.0)

    # Ângulos nas articulações principais (cotovelo, joelho, ombro, anca)
    def angle_at(a, b, c):
        # ângulo em b formado por a-b-c (usa coords normalizadas)
        va = norm[a] - norm[b]
        vc = norm[c] - norm[b]
        na = np.linalg.norm(va)
        nc = np.linalg.norm(vc)
        if na < 1e-6 or nc < 1e-6:
            return 0.0, 0.0
        va /= na; vc /= nc
        cos_t = float(np.clip(np.dot(va, vc), -1.0, 1.0))
        # Para continuidade adiciona sin com sinal usando produto cruzado 2D
        sin_t = float(np.clip(va[0] * vc[1] - va[1] * vc[0], -1.0, 1.0))
        return cos_t, sin_t

    angle_triplets = [
        (5, 7, 9), (6, 8, 10),    # cotovelos
        (11, 13, 15), (12, 14, 16),  # joelhos
        (11, 5, 7), (12, 6, 8),    # ombros com torso
        (5, 11, 13), (6, 12, 14)   # ancas com pernas
    ]
    angle_feats = []
    for a, b, c in angle_triplets:
        cos_t, sin_t = angle_at(a, b, c)
        angle_feats += [cos_t, sin_t]

    # Orientação do tronco (vetor ombros) em cos/sin
    orient_cos, orient_sin = 1.0, 0.0
    if have(5) and have(6):
        v = norm[6] - norm[5]
        ang = np.arctan2(v[1], v[0])
        orient_cos = float(np.cos(ang))
        orient_sin = float(np.sin(ang))

    # Visibilidade: proporção de pontos visíveis
    vis_ratio = float(np.mean(vis.astype(np.float32)))

    feat = np.array(bone_lengths + angle_feats + [orient_cos, orient_sin, vis_ratio], dtype=np.float32)
    # Normaliza vetor final para estabilizar a similaridade coseno
    n = np.linalg.norm(feat)
    if n > 1e-6:
        feat = feat / n
    return feat


def compute_reid_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Computa embedding ReID usando o modelo CNN.

    Args:
        img_bgr: Imagem BGR do crop da pessoa.

    Returns:
        np.ndarray or None: Vetor de embedding normalizado.
    """
    if img_bgr is None or img_bgr.size == 0 or _REID_MODEL is None:
        return None
    try:
        from torchvision import transforms
    except ImportError:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize cuadrado con pad
    h, w = img_rgb.shape[:2]
    scale = _REID_SIZE / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    img_res = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (_REID_SIZE - nh) // 2
    bottom = _REID_SIZE - nh - top
    left = (_REID_SIZE - nw) // 2
    right = _REID_SIZE - nw - left
    img_sq = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        tensor = tfm(img_sq).unsqueeze(0).to(_DEVICE)
        feat = _REID_MODEL(tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[0] if len(feat) > 0 else feat
        vec = feat.squeeze().detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 1e-6:
            vec = vec / n
        return vec
    return None


def iou_xyxy(a, b):
    """
    Calcula a Intersecção sobre União (IoU) entre duas caixas delimitadoras.

    Args:
        a, b: Tuplas (x1, y1, x2, y2) das caixas.

    Returns:
        float: Valor IoU entre 0 e 1.
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
    Atribui IDs a deteções usando um algoritmo greedy com etapas:
    1. IoU (opcional): Matching por proximidade espacial.
    2. Similaridade: Matching por embeddings (ReID ou esqueleto).
    3. Reaparência: Reatribuição de IDs velhos com umbral baixo.

    Args:
        det_features: Lista de vetores de características por deteção.
        det_boxes: Lista de caixas (x1,y1,x2,y2) por deteção.
        person_db: Instância de PersonDatabase.
        similarity_threshold: Umbral para matching por similaridade.
        iou_threshold: Umbral para IoU.
        frame_index: Índice do frame atual.
        max_age: Máx frames para considerar IoU.
        feature_history: Comprimento do histórico de features.
        reappear_threshold: Umbral para reaparências.
        counters: Dicionário para contar atribuições.
        use_iou: Se usar IoU ou não.

    Returns:
        list: Lista de IDs atribuídos por deteção.
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
        # 1) Tentativa por IoU com históricos recentes
        recent_ids = person_db.get_recent_ids(frame_index, max_age)
        iou_pairs = []  # (sim, det_i, pid)
        for i, box in enumerate(det_boxes):
            for pid in recent_ids:
                if pid in used_ids:
                    continue
                prev_box = person_db[pid]['bbox']
                if prev_box is not None:
                    iou_val = iou_xyxy(box, prev_box)
                    if iou_val >= iou_threshold:
                        iou_pairs.append((iou_val, i, pid))
        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        for iou, det_i, pid in iou_pairs:
            if det_i in used_dets or pid in used_ids:
                continue
            assigned[det_i] = pid
            used_dets.add(det_i)
            used_ids.add(pid)
            counters['iou_assignments'] += 1

    # 2) Emparelhamento por descritor (coseno) para os que restaram usando atribuição ótima
    unmatched_dets = [i for i in range(len(det_features)) if i not in used_dets and det_features[i] is not None]
    available_ids = [pid for pid in existing_ids if pid not in used_ids]
    if unmatched_dets and available_ids:
        # Criar matriz de custo
        cost_matrix = np.full((len(unmatched_dets), len(available_ids)), np.inf)
        det_to_idx = {det: idx for idx, det in enumerate(unmatched_dets)}
        id_to_idx = {pid: idx for idx, pid in enumerate(available_ids)}
        for i, feat in enumerate(det_features):
            if i not in unmatched_dets or feat is None:
                continue
            for pid in available_ids:
                ref = person_db[pid]['feat']
                sim = float(cosine_similarity(feat.reshape(1, -1), ref.reshape(1, -1))[0][0])
                if sim >= similarity_threshold:
                    cost_matrix[det_to_idx[i], id_to_idx[pid]] = -sim  # negativo para minimização
        # Resolver atribuição ótima
        if np.any(np.isfinite(cost_matrix)):
            try:
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] != np.inf:
                        det_i = unmatched_dets[r]
                        pid = available_ids[c]
                        assigned[det_i] = pid
                        used_dets.add(det_i)
                        used_ids.add(pid)
                        counters['sim_assignments'] += 1
            except ValueError:
                # Se a matriz for infactível (ex. filas sem matches), saltar atribuição
                pass

    # 2.5) Etapa de reaparência: emparelhar com IDs velhos usando similaridade alta
    reappear_pairs = []
    for i, feat in enumerate(det_features):
        if assigned[i] is not None or feat is None:
            continue
        for pid in existing_ids:
            if pid in used_ids:
                continue
            stored_feat = person_db[pid]['feat']
            if stored_feat is not None:
                sim = float(cosine_similarity(feat.reshape(1, -1), stored_feat.reshape(1, -1))[0][0])
                if sim >= reappear_threshold:
                    reappear_pairs.append((sim, i, pid))
    reappear_pairs.sort(key=lambda x: x[0], reverse=True)
    for sim, det_i, pid in reappear_pairs:
        if assigned[det_i] is not None or pid in used_ids:
            continue
        assigned[det_i] = pid
        used_ids.add(pid)
        counters['reappear_assignments'] += 1

    # 3) Atualiza tracks emparelhados
    for i, pid in enumerate(assigned):
        if pid is None:
            continue
        feat = det_features[i]
        box = det_boxes[i]
        person_db.update_person(pid, feat, box, frame_index)

    # 4) Cria novos IDs para deteções não emparelhadas com descritor válido
    for i, feat in enumerate(det_features):
        if assigned[i] is None and feat is not None:
            box = det_boxes[i]
            assigned[i] = person_db.add_person(feat, box, frame_index, feature_history)

    # 5) Incrementa misses para não emparelhados (opcional limpeza futura)
    person_db.increment_misses(frame_index)

    return assigned


def person_process_frame(frame):
    global _FRAME_INDEX
    _FRAME_INDEX += 1
    results = _POSE_MODEL(frame, verbose=False, device=_DEVICE, conf=DEFAULT_CONF)
    annotated_frame = results[0].plot()

    persistent_ids = set()
    raw_people_count = 0
    if results[0].boxes is not None and results[0].keypoints is not None:
        boxes = results[0].boxes
        keypoints_tensor = results[0].keypoints.data
        raw_people_count = len(boxes)
        det_features = []
        det_boxes = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kpts_array = keypoints_tensor[idx].cpu().numpy()
            # Extrair features: ReID se disponível, senão skeleton
            if _USE_REID:
                # Crop da pessoa para ReID
                crop = frame[y1:y2, x1:x2]
                feat = compute_reid_embedding(crop)
            else:
                feat = extract_skeleton_features(kpts_array)
            det_features.append(feat)
            det_boxes.append((x1, y1, x2, y2))

        # Atribuir IDs usando greedy
        similarity_threshold = _REID_THRESHOLD if _USE_REID else _SIMILARITY_THRESHOLD
        assigned_ids = assign_ids_greedy(
            det_features=det_features,
            det_boxes=det_boxes,
            person_db=_PERSON_DB,
            similarity_threshold=similarity_threshold,
            iou_threshold=_IOU_THRESHOLD,
            frame_index=_FRAME_INDEX,
            max_age=_MAX_AGE,
            feature_history=_FEATURE_HISTORY,
            reappear_threshold=_REAPPEAR_THRESHOLD,
            use_iou=True
        )

        for idx, pid in enumerate(assigned_ids):
            if pid is not None:
                persistent_ids.add(pid)
            display_id = pid if pid is not None else idx
            x1, y1, _, _ = det_boxes[idx]
            label = f'ID: {display_id}'
            cv2.putText(annotated_frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            cv2.putText(annotated_frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    fps_label = f'Device: {_DEVICE}'
    cv2.putText(annotated_frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Unique IDs: {len(_PERSON_DB)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    threshold_type = 'ReID' if _USE_REID else 'Skeleton'
    threshold_label = _REID_THRESHOLD if _USE_REID else _SIMILARITY_THRESHOLD
    cv2.putText(annotated_frame, f'{threshold_type} Threshold: {threshold_label:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    people_count = max(raw_people_count, len(persistent_ids))
    people_detected = people_count > 0

    return annotated_frame, people_detected, people_count