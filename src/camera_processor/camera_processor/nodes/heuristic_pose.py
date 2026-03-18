import os

from ultralytics import YOLO

from ament_index_python import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

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

        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value

        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n-pose.pt')

        # load the model
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loading YOLO model from {model_path}...")

        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'pose_heuristic/processed_image', 1)

        #create an boolean topic to see if some person is present in frame or not 
        self.detected_pub = self.create_publisher(String, 'pose/heuristic/detected', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process the current frame in computer vision script
        processed_frame, detected_poses  = pose_process_frame_keypoints(frame, self.model)

        # 1. Publish the detection message (e.g., the combined pose string)
        if detected_poses:
            # Combine all detected poses into a single string for publishing
            pose_string = ", ".join(detected_poses)
            
            det_msg = String()
            det_msg.data = f"Detected {len(detected_poses)} people. Poses: {pose_string}"
            self.detected_pub.publish(det_msg)
            
        else:
            # Publish a message if no person is detected
            det_msg = String()
            det_msg.data = "No person detected."
            self.detected_pub.publish(det_msg)
            self.get_logger().debug("No person detected.") # Use debug for frequent non-critical updates


        #publish the processed image for the Web Server
        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()