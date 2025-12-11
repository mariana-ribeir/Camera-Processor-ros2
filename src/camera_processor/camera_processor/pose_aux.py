import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def pose_process_frame(frame):
    # Variáveis para calcular FPS
    fps_counter = 0
    start_time = time.time()
    fps = 0

    pkg_share = get_package_share_directory('camera')
    model_dir = os.path.join(pkg_share, 'models')
    model_path = os.path.join(model_dir, 'yolov8n-pose.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f"A usar dispositivo: {device}")

    #model = YOLO(model_path).to(device)
    model = YOLO(model_path).to(device)

    # Usar tracking em vez de deteção simples
    results = model.track(frame, persist=True, device=device, verbose=False)
        
    # Obter frame anotado
    annotated_frame = results[0].plot()
        
    # Classificar poses para cada deteção
    detected_poses = []
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
            pose = classify_pose(kpts)
            detected_poses.append(pose)
            x1, y1, x2, y2 = map(int, box)
                
            # Mostrar a classificação no frame
            label = f'{pose}'
            cv2.putText(annotated_frame, label, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            print(f"Pessoa {i+1}: {pose}")
        
    # Calcular FPS
    fps_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Atualizar a cada segundo
        fps = fps_counter / elapsed_time
        fps_counter = 0
        start_time = time.time()
        
    # Mostrar FPS no frame
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_frame, detected_poses
        


def classify_pose(keypoints):
    """
    Classifica a pose geral baseada em keypoints do YOLOv8-pose.
    Keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                left_knee, right_knee, left_ankle, right_ankle]
    """
    conf_threshold = 0.5
    if keypoints.shape[0] < 17:
        return "Desconhecida"
    
    pts = keypoints[:, :2]
    vis = keypoints[:, 2] > conf_threshold
    
    # Verificar keypoints chave
    has_nose = vis[0]
    has_shoulders = vis[5] and vis[6]
    has_hips = vis[11] and vis[12]
    has_knees = vis[13] and vis[14]
    has_ankles = vis[15] and vis[16]
    
    if not (has_shoulders and has_hips):
        return "Desconhecida"
    
    # Calcular posições relativas
    shoulder_center = (pts[5] + pts[6]) / 2
    hip_center = (pts[11] + pts[12]) / 2
    knee_center = (pts[13] + pts[14]) / 2 if has_knees else hip_center
    
    # Altura relativa (dos ombros às ancas)
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    
    # Ângulo das pernas (aprox. se joelhos estiverem dobrados)
    leg_angle = "estendida"
    if has_knees and has_ankles:
        knee_to_hip = pts[13] - pts[11] if vis[11] else np.array([0, 0])
        knee_to_ankle = pts[15] - pts[13] if vis[13] and vis[15] else np.array([0, 0])
        if np.linalg.norm(knee_to_hip) > 0 and np.linalg.norm(knee_to_ankle) > 0:
            cos_angle = np.dot(knee_to_hip, knee_to_ankle) / (np.linalg.norm(knee_to_hip) * np.linalg.norm(knee_to_ankle))
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            if angle < 120:  # Ângulo agudo indica dobrada
                leg_angle = "dobrada"
    
    # Orientação geral
    if has_nose:
        head_to_torso = pts[0] - shoulder_center
        orientation = "vertical" if abs(head_to_torso[1]) > abs(head_to_torso[0]) else "horizontal"
    else:
        orientation = "desconhecida"
    
    # Classificação heurística melhorada
    # Classificação heurística
    if orientation == "horizontal":
        return "Deitada"
    elif leg_angle == "dobrada":
        return "Sentada"
    elif leg_angle == "estendida":
        return "De pé"
    else:
        return "Desconhecida"



       