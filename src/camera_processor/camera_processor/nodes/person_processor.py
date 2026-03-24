import rclpy
import os
import torch
import threading
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from ultralytics import YOLO
from camera_interfaces.msg import PersonDetection, PersonDetectionArray

from camera_processor.helpers.person_detector import (
    person_process_frame,
)

"""
ROS2 Node for real-time human detection and tracking.

This node acts as the primary detector for the system. It processes raw camera 
images using a YOLOv8-pose model to identify people, generates tracking IDs, 
and provides bounding box coordinates for use by downstream pose-estimation nodes.

Subscriptions:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Publishers:
    /person/detections (camera_interfaces/PersonDetectionArray): Array containing tracking IDs and bounding boxes (x, y, w, h).
    /person/detected (std_msgs/Bool): True if at least one person is in frame.
    /person/processed_image (sensor_msgs/Image): Annotated image with bounding boxes and IDs (if show_gui is enabled).

Parameters:
    show_gui (bool): If True, publishes the annotated visualization frame.

Attributes:
    model (YOLO): YOLOv8-pose model used as a person detector.
    bridge (CvBridge): Utility for ROS-to-OpenCV image conversion.
    last_frame (ndarray): The most recent frame stored for the processing timer.
    detections_pub (rclpy.Publisher): Publisher for the structured detection data.
"""

class PersonProcessorNode(Node):
    def __init__(self):
        super().__init__('person_processor')  
        self.get_logger().info("Node 'person_processor' started!")

        self.bridge = CvBridge()
        self.last_frame = None
        self.processing = False

        #publish the processed image so we can see it remotely
        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value

        # path setup for model
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n-pose.pt')

        # load the model
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        #publishers for detection and count
        self.detected_pub = self.create_publisher(Bool, 'person/detected', 10)
        #publish detections (bbox + id)
        self.detections_pub = self.create_publisher(
            PersonDetectionArray,
            'person/detections',
            10
        )

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'person/processed_image', 1)

        #subscribe the image topic  
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1)
               
        self.create_timer(0.1, self.process)
        

    def image_callback(self, msg):
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process(self):
        if self.processing or self.last_frame is None:
            return

        # mark as processing immediately
        self.processing = True

        # copy latest frame
        frame = self.last_frame.copy()

        # run heavy YOLO in a separate thread to avoid blocking subscriber
        threading.Thread(target=self._process_frame, args=(frame,), daemon=True).start()

    def _process_frame(self, frame):
        with torch.inference_mode():
            processed_frame, people_detected, count, detections = person_process_frame(frame, self.model)

        # publish detections
        msg = PersonDetectionArray()
        for det in detections:
            pid, x, y, w, h, confidence = det
            det_msg = PersonDetection()
            det_msg.id = pid
            det_msg.x = x
            det_msg.y = y
            det_msg.width = w
            det_msg.height = h
            msg.detections.append(det_msg)

        self.detections_pub.publish(msg)
        self.detected_pub.publish(Bool(data=bool(people_detected)))

        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)

        self.processing = False

def main(args=None):
    rclpy.init(args=args)

    node = PersonProcessorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

if __name__ == '__main__':
    main()
