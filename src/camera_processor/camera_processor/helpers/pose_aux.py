import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def pose_process_frame_model(frame):
    pkg_share = get_package_share_directory('camera_processor')
    model_dir = os.path.join(pkg_share, 'models')
    model_path = os.path.join(model_dir, 'best.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = YOLO(model_path).to(device)

    # Use simple detection instead of tracking (no lap required)
    results = model(frame)  # <-- change from track() to just call the model

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Extract boxes if you want to print them, optional
    detected_poses = []
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
            pose = "unknown"  # temporarily skip classify_pose
            detected_poses.append(pose)
            x1, y1, x2, y2 = map(int, box)
            label = f'{pose}'
            cv2.putText(annotated_frame, label, (x1, y1 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return annotated_frame, detected_poses



def pose_process_frame_keypoints(frame):
    # Variáveis para calcular FPS
    fps_counter = 0
    start_time = time.time()
    fps = 0

    pkg_share = get_package_share_directory('camera_processor')
    model_dir = os.path.join(pkg_share, 'models')
    model_path = os.path.join(model_dir, 'best.pt')

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
            cv2.putText(annotated_frame, label, (x1, y1 - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
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
        return "unknown"
    
    pts = keypoints[:, :2]
    vis = keypoints[:, 2] > conf_threshold
    
    # Verificar keypoints chave
    has_nose = vis[0]
    has_shoulders = vis[5] and vis[6]
    has_hips = vis[11] and vis[12]
    has_knees = vis[13] and vis[14]
    has_ankles = vis[15] and vis[16]
    
    if not (has_shoulders and has_hips):
        return "unknown"
    
    # Calcular posições relativas
    shoulder_center = (pts[5] + pts[6]) / 2
    hip_center = (pts[11] + pts[12]) / 2
    knee_center = (pts[13] + pts[14]) / 2 if has_knees else hip_center
    
    # Altura relativa (dos ombros às ancas)
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    
    # Ângulo das pernas - ángulo grande indica doblada (sentada), pequeño indica extendida (de pie)
    leg_angle = "estendida"
    angle_l = 180
    angle_r = 180
    if has_knees and has_ankles:
        # Pierna izquierda
        knee_to_hip_l = pts[13] - pts[11] if vis[11] and vis[13] else np.array([0, 0])
        knee_to_ankle_l = pts[15] - pts[13] if vis[13] and vis[15] else np.array([0, 0])
        angle_l = 180
        if np.linalg.norm(knee_to_hip_l) > 0 and np.linalg.norm(knee_to_ankle_l) > 0:
            cos_angle_l = np.dot(knee_to_hip_l, knee_to_ankle_l) / (np.linalg.norm(knee_to_hip_l) * np.linalg.norm(knee_to_ankle_l))
            angle_l = np.arccos(np.clip(cos_angle_l, -1, 1)) * 180 / np.pi
        
        # Pierna derecha
        knee_to_hip_r = pts[14] - pts[12] if vis[12] and vis[14] else np.array([0, 0])
        knee_to_ankle_r = pts[16] - pts[14] if vis[14] and vis[16] else np.array([0, 0])
        angle_r = 180
        if np.linalg.norm(knee_to_hip_r) > 0 and np.linalg.norm(knee_to_ankle_r) > 0:
            cos_angle_r = np.dot(knee_to_hip_r, knee_to_ankle_r) / (np.linalg.norm(knee_to_hip_r) * np.linalg.norm(knee_to_ankle_r))
            angle_r = np.arccos(np.clip(cos_angle_r, -1, 1)) * 180 / np.pi
        
        # Si alguna pierna tiene ángulo > 90, considerar doblada (sentada)
        if angle_l > 90 or angle_r > 90:
            leg_angle = "dobrada"
    
    # Orientação geral baseada no torso
    torso_vector = shoulder_center - hip_center
    orientation = "vertical" if abs(torso_vector[1]) > abs(torso_vector[0]) else "horizontal"
    
    # Classificação heurística melhorada
    if orientation == "horizontal":
        pose = "lying"
    elif leg_angle == "dobrada":
        pose = "sitting"
    elif leg_angle == "estendida":
        pose = "standing"
    else:
        pose = "unknown"
    
    return pose



       