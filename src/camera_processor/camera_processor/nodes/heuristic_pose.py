import os

import torch
from ultralytics import YOLO

from ament_index_python import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from camera_interfaces.msg import PersonDetection, PersonDetectionArray, PoseDetectionArray, PoseDetection
from rclpy.executors import MultiThreadedExecutor
from camera_processor.helpers.pose_detector import pose_process_frame_keypoints

"""
ROS2 Node for real-time human detection and counting.

This node subscribes to raw camera images, processes each frame using a YOLOv8
model to detect people, and publishes both the detection status and the number
of detected persons. Optionally, it can also publish an annotated image for
visualization.

Subscriptions:
    /camera/image_raw (sensor_msgs/Image): Raw camera video stream.

Publishers:
    person/detected (std_msgs/Bool): Indicates whether at least one person is detected.
    person/count (std_msgs/Int32): Number of detected persons in the current frame.
    person/processed_image (sensor_msgs/Image): Annotated image with detections (only if 'show_gui' is enabled).

Parameters:
    show_gui (bool): If True, publishes the processed image with detections.

Attributes:
    subscription (rclpy.Subscription): Subscriber to '/camera/image_raw'.
    detected_pub (rclpy.Publisher): Publisher for the 'person/detected' topic.
    count_pub (rclpy.Publisher): Publisher for the 'person/count' topic.
    image_pub (rclpy.Publisher): Publisher for processed images (if enabled).
    bridge (CvBridge): Converter between ROS Image messages and OpenCV images.
    model (YOLO): YOLOv8 pose model used for person detection.
"""
class HeuristicPoseNode(Node):
    def __init__(self):
        super().__init__('heuristic_pose')  # ROS node name
        self.get_logger().info("Node 'heuristic_pose' started!")


        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value

        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n-pose.pt')

        # load the model
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loading YOLO model from {model_path}...")

        #ROS
        self.bridge = CvBridge()

        # State (thread-safe-ish)
        self.last_frame = None
        self.latest_detections = None
        self.processing = False

        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1)

        self.create_subscription(
            PersonDetectionArray,
            '/person/detections',
            self.detections_callback,
            1
        )

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'pose_heuristic/processed_image', 1)

        self.pose_pub = self.create_publisher(
            PoseDetectionArray,
            '/pose/heuristic/detected',
            1
        )
        
        self.create_timer(0.05, self.process)

    def image_callback(self, msg):
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def detections_callback(self, msg):
        self.latest_detections = msg.detections

    def process(self):
        if self.processing:
            return

        if self.last_frame is None or self.latest_detections is None:
            return

        self.processing = True

        frame = self.last_frame.copy()

        pose_array = PoseDetectionArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()

        with torch.inference_mode():
            processed_frame, detected_poses = pose_process_frame_keypoints(
                frame,
                self.model
            )

        for det, pose in zip(self.latest_detections, detected_poses):
            pose_msg = PoseDetection()
            pose_msg.id = det.id
            pose_msg.pose = pose
            pose_array.pose_detections.append(pose_msg)

        self.pose_pub.publish(pose_array)

        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)

        self.processing = False


def main(args=None):
    rclpy.init(args=args)

    node = HeuristicPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

if __name__ == '__main__':
    main()
