import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from camera_processor.helpers.pose_ai import pose_process_frame_model

"""
ROS2 node for real-time human pose detection using an AI model.

This node subscribes to raw camera images, processes each frame using a
YOLO-based pose detection model, and publishes the detected poses along
with an optional annotated image.

Subscriptions:
    /camera/image_raw (sensor_msgs/Image): Raw camera image stream.

Publishers:
    pose/ia/detected (std_msgs/String): Text description of detected people and their poses.
    pose_ia/processed_image (sensor_msgs/Image): Annotated image with pose detections (published only if 'show_gui' is enabled).

Parameters:
    show_gui (bool): If True, publishes the processed image for visualization.

Attributes:
    subscription (rclpy.Subscription): Subscriber to the camera image topic.
    detected_pub (rclpy.Publisher): Publisher for the pose detection results.
    image_pub (rclpy.Publisher): Publisher for the processed image when visualization is enabled.
    bridge (CvBridge): Utility for converting between ROS Image messages and OpenCV images.
    model (YOLO): Loaded YOLO pose detection model used for inference.
"""
class IaPoseNode(Node):
    def __init__(self):
        super().__init__('ai_pose')  # ROS node name
        self.get_logger().info("Node 'ai_pose' started!")

        #publish the processed image so we can see it remotely
        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value
        
        # path setup for model
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'best.pt')

        # load the model
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'pose_ia/processed_image', 1)

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

        #publish the processed image for the Web Server ---
        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IaPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
