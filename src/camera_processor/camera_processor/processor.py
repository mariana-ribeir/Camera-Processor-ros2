import cv2

"""
Processes a single video frame.

Args:
    frame (np.ndarray): OpenCV BGR image

Returns:
    processed_frame (np.ndarray): Processed frame
"""
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray
