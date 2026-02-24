import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# Importing both processing helpers
from camera_processor.helpers.pose_ai import pose_process_frame_model
from camera_processor.helpers.pose_heuristic import pose_process_frame_keypoints

class PoseProcessorNode(Node):
    def __init__(self):
        super().__init__('pose_processor')
        self.get_logger().info("Node 'pose_processor' (AI + Heuristic) started!")

       # State variables to store the latest from each node
        self.latest_ai_pose = None
        self.latest_heuristic_pose = None

        # Subscribers
        self.ai_sub = self.create_subscription(
            String, '/pose/ia/detected', self.ai_callback, 10)
        
        self.heuristic_sub = self.create_subscription(
            String, '/pose/heuristic/detected', self.heuristic_callback, 10)
        
        # Publisher
        self.detected_pub = self.create_publisher(String, 'pose/detected', 10)
        
        # Timer to check and compare at 10Hz (0.1s)
        self.create_timer(0.1, self.compare_and_publish)

    def ai_callback(self, msg):
        # Logic to extract the pose list from the string "Poses: standing, sitting"
        self.latest_ai_pose = msg.data
        
    def heuristic_callback(self, msg):
        self.latest_heuristic_pose = msg.data

    def compare_and_publish(self):
        # If we don't even have AI data, we can't do much.
        if self.latest_ai_pose is None:
            self.get_logger().debug("Waiting for AI data...")
            return 

        final_msg = String()
        
        # Case 1: We have both - Compare them
        if self.latest_heuristic_pose is not None:
            if self.latest_ai_pose == self.latest_heuristic_pose:
                certainty = 100
                status = "Agreed"
            else:
                certainty = 50
                status = "Disputed"
            
            final_msg.data = f"Certainty: {certainty}% | Mode: {status} | AI: {self.latest_ai_pose}"

        # Case 2: Heuristic is missing - Trust AI but note the lack of verification
        else:
            certainty = 75 
            status = "AI Only (Heuristic Offline)"
            final_msg.data = f"Certainty: {certainty}% | Mode: {status} | AI: {self.latest_ai_pose}"

        self.detected_pub.publish(final_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()