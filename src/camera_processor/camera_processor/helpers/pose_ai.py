import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def pose_process_frame_model(frame, model, logger):

    # Use simple detection instead of tracking (no lap required)
    results = model(frame, verbose=False)  

    logger.info(f"Results length: {len(results)}")
    #logger.info(f"Results: {results}")

    # Get annotated frame
    annotated_frame = results[0].plot()

    detected_poses = []
    
    # Process results
    for r in results:
        # Get labels from boxes
        if r.boxes:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = r.names[class_id]
                detected_poses.append(label)
        
        # Create the visual frame with boxes drawn on it
        annotated_frame = r.plot()

    return annotated_frame, detected_poses
