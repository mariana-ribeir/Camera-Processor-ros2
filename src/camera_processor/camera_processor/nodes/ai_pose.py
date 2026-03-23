import torch

import rclpy
import os
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs import msg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from camera_processor.helpers.pose_detector import pose_process_frame_model
from camera_interfaces.msg import PersonDetection, PersonDetectionArray, PoseDetectionArray, PoseDetection

"""
ROS2 node for real-time human pose detection using an AI model.

This node subscribes to raw camera images, processes each frame using a
YOLO-based pose detection model, and publishes the detected poses along
with an optional annotated image.

Subscriptions:
    /camera/image_raw (sensor_msgs/Image): Raw camera image stream.
    /person/detections (camera_interfaces/PersonDetectionArray): Detected person bounding boxes.

Publishers:
    pose/ia/detected (std_msgs/String): Text description of detected people and their poses.
    /pose_ia/processed_image (sensor_msgs/Image): Annotated image with pose detections (published only if 'show_gui' is enabled).

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

        # Params
        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value
        
        # Model
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'best.pt')

        # load the model
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        # State (thread-safe-ish)
        self.last_frame = None
        self.latest_detections = None
        self.processing = False

        #ROS
        self.bridge = CvBridge()
        self.create_timer(0.05, self.process)

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'pose_ia/processed_image', 1)

        #subscribe the camera topic 
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1
        )

        #subscribe the bbox topic 
        self.create_subscription(
            PersonDetectionArray,
            '/person/detections',
            self.detections_callback,
            1
        )

        self.pose_pub = self.create_publisher(
            PoseDetectionArray,
            '/pose/ia/detected',
            1
        )
    
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

        frame = self.last_frame
        vis_frame = self.last_frame.copy()

        pose_array = PoseDetectionArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()

        with torch.inference_mode(): 
            for det in self.latest_detections:

                x, y, w, h = det.x, det.y, det.width, det.height

                crop = frame[y:y+h, x:x+w]

                if crop.size == 0:
                    continue

                processed_crop, poses = pose_process_frame_model(
                    crop,
                    self.model,
                    self.get_logger()
                )

                # 👇 criar mensagens estruturadas
                for pose in poses:
                    pose_msg = PoseDetection()
                    pose_msg.id = det.id
                    pose_msg.pose = pose
                    pose_array.pose_detections.append(pose_msg)

                if self.show_gui:
                    vis_frame[y:y+h, x:x+w] = processed_crop

        # Publish poses
        self.pose_pub.publish(pose_array)

        # Publish imagem
        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)

        self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = IaPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()

if __name__ == '__main__':
    main()