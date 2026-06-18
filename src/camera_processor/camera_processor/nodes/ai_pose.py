import torch
import rclpy
import os
import threading
from ultralytics import YOLO
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from camera_processor.helpers.pose_detector import pose_process_frame_model
from camera_interfaces.msg import PersonDetectionArray, PoseDetectionArray, PoseDetection

"""
ROS2 node for real-time human pose detection using a crop-based AI approach.

This node subscribes to raw camera images and person bounding boxes. It crops
the image around each detected person and runs a YOLO pose model on the 
individual crops to achieve higher accuracy or specific pose analysis.

Subscriptions:
    /camera/image_raw (sensor_msgs/Image): Raw camera image stream.
    /person/detections (camera_interfaces/PersonDetectionArray): Bounding boxes used for cropping.

Publishers:
    /pose/ia/detected (camera_interfaces/PoseDetectionArray): Array of poses linked to person IDs.
    /pose/ia/processed_image (sensor_msgs/Image): Annotated full frame with pose crops re-inserted.

Parameters:
    show_gui (bool): If True, reconstructs the full frame with pose overlays and publishes it.

Attributes:
    model (YOLO): The YOLO pose model loaded from 'best.pt'.
    bridge (CvBridge): Utility for ROS-OpenCV image conversion.
    last_frame (ndarray): Storage for the most recent camera frame.
    last_detections (list): Storage for the most recent bounding boxes.
"""

class IaPoseNode(Node):
    def __init__(self):
        super().__init__('ai_pose')  # ROS node name
        self.get_logger().info("Node 'ai_pose' started!")

        # params
        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value
        
        # load the model
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'best.onnx')
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        # state and thread
        self.last_frame = None
        self.last_detections = None
        self.processing = False

        #ROS
        self.bridge = CvBridge()
        self.create_timer(0.05, self.process)   # Check for new data every 50ms

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'pose/ia/processed_image', 1)

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

        #publish the pose detections 
        self.pose_pub = self.create_publisher(
            PoseDetectionArray,
            '/pose/ia/detected',
            1
        )
    
    def image_callback(self, msg):
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def detections_callback(self, msg):
        self.last_detections = msg.detections


    def process(self):
        if self.processing or self.last_frame is None or self.last_detections is None:
            return

        self.processing = True
        frame = self.last_frame.copy()
        detections = list(self.last_detections)  # make a copy to avoid race conditions

        threading.Thread(target=self._process_frame, args=(frame, detections), daemon=True).start()

    def _process_frame(self, frame, detections):
        height, width = frame.shape[:2]
        pose_array = PoseDetectionArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()

        processed_frame = frame.copy()  # Start with a copy of the original frame

        with torch.inference_mode(): 
            for det in detections:
                x1 = max(0, int(det.x))
                y1 = max(0, int(det.y))
                x2 = min(width, int(det.x + det.width))
                y2 = min(height, int(det.y + det.height))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                annotated_crop, poses = pose_process_frame_model(crop, self.model, self.get_logger())
                
                # Insert the annotated crop back into the processed frame
                processed_frame[y1:y2, x1:x2] = annotated_crop
                
                # Take only the first detected pose for this person to avoid duplicates
                if poses:
                    pose = poses[0]
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