import rclpy
import os
import torch
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
    get_similarity_threshold,
)

"""
ROS2 Node for real-time human detection and counting.

Subscribes to raw camera images, processes them using computer vision
to detect people, and publishes the detection status and count.
Also allows runtime adjustment of the similarity threshold via keyboard input.

Subscribes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Publishers:
    person/detected (std_msgs/Bool): Indicates whether at least one person is detected.
    person/count (std_msgs/Int32): Number of detected persons in the current frame.
    person/processed_image (sensor_msgs/Image): Annotated image with detections (only if 'show_gui' is enabled).

Parameters:
    show_gui (bool): If True, publishes the processed image with detections.

Attributes:
    subscription (rclpy.Subscription): Subscriber to '/camera/image_raw'.
    detected_pub (rclpy.Publisher): Publisher for the 'person/detected' topic.
    image_pub (rclpy.Publisher): Publisher for processed images (if enabled).
    bridge (CvBridge): Converter between ROS Image messages and OpenCV images.
    model (YOLO): YOLOv8 pose model used for person detection.
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
        if self.processing:
            return

        if self.last_frame is None:
            return

        self.processing = True

        frame = self.last_frame.copy()

        with torch.inference_mode():
            processed_frame, people_detected, count, detections = person_process_frame(
                frame,
                self.model
            )

        msg = PersonDetectionArray()

        for det in detections:
            det_msg = PersonDetection()
            pid, x, y, w, h, confidence = det

            det_msg.id = pid
            det_msg.x = x
            det_msg.y = y
            det_msg.width = w
            det_msg.height = h
            msg.detections.append(det_msg)

        self.detections_pub.publish(msg)

        #publish the processed image for the Web Server 
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
