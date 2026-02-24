import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

from camera_processor.helpers.pose_ai import pose_process_frame_model

"""
ROS2 Node for real-time human pose detection.

Subscribes to raw camera images, processes them using computer vision
to detect people, and the pose of them.

Subscribes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Attributes:
    subscription (rclpy.Subscription): Subscriber to the '/camera/image_raw' topic.
    bridge (CvBridge): Converter between OpenCV images and ROS2 Image messages.
"""
class IaPoseNode(Node):
    def __init__(self):
        super().__init__('ia_pose')  # ROS node name
        self.get_logger().info("Node 'ia_pose' started!")
        
        # path setup
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'best.pt')

        # load the model
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)

        #create an boolean topic to see if some person is present in frame or not 
        self.detected_pub = self.create_publisher(String, 'pose/ia/detected', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.namedWindow("Real Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real Frame", 800, 600)
        cv2.imshow("Real Frame", frame)

        # process the current frame in computer vision script
        processed_frame, detected_poses= pose_process_frame_model(frame, self.model, self.get_logger())

        # 1. Publish the detection message (e.g., the combined pose string)
        if detected_poses:
            # Combine all detected poses into a single string for publishing
            pose_string = ", ".join(detected_poses)
            
            det_msg = String()
            det_msg.data = f"Detected {len(detected_poses)} people. Poses: {pose_string}"
            self.detected_pub.publish(det_msg)
            
            # Use ROS logging for status updates
            self.get_logger().info(f"Published detection: {det_msg.data}")
        else:
            # Publish a message if no person is detected
            det_msg = String()
            det_msg.data = "No person detected."
            self.detected_pub.publish(det_msg)
            self.get_logger().debug("No person detected.") # Use debug for frequent non-critical updates


        # Display the frames using the annotated_frame result
        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Frame", 800, 600)
        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(1) # Important for cv2.imshow to work


def main(args=None):
    rclpy.init(args=args)
    node = IaPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
