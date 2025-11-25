import cv2
import numpy as np

"""
Processes a single video frame in black and white.

Args:
    frame (np.ndarray): OpenCV BGR image

Returns:
    processed_frame (np.ndarray): Processed frame
"""
def process_frame_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


"""
Processes a single video frame in black and white.

Args:
    frame (np.ndarray): OpenCV BGR image

Returns:
    red_highlighted (np.ndarray): Processed frame (only red regions visible)
    detected (boolean): True if any red pixels were detected
"""
def process_frame(frame):
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