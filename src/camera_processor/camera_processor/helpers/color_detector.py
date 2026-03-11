import os
import cv2
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

"""
Processes a single black-and-white video frame.

Args:
    frame (np.ndarray): OpenCV BGR image

Returns:
    processed_frame (np.ndarray): Processed frame
"""
def process_frame_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


"""
Detects color regions in a video frame using a threshold in HSV.
In this case the color is red.

Args:
    frame (np.ndarray): Image frame in BGR format (OpenCV).

Returns:
    color_highlighted (np.ndarray): Resulting frame with only pixels belonging to the color regions.
    detected (bool): Indicates whether at least one pixel within the defined color range was detected.
"""
def color_process_frame(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([5, 255, 255])

    lower_red2 = np.array([175, 150, 70])
    upper_red2 = np.array([180, 255, 255])

    # Threshold for the color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Binary output for visualization
    color_highlighted = cv2.bitwise_and(frame, frame, mask=mask)

    # Determine if the color red was detected
    detected = np.any(mask > 0)

    return color_highlighted, detected
