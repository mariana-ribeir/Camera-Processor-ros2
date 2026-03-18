import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from camera_interfaces.msg import PersonDetection, PersonDetectionArray

from camera_processor.helpers.person_detector import (
    person_process_frame,
    adjust_similarity_threshold,
    reset_person_database,
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
    count_pub (rclpy.Publisher): Publisher for the 'person/count' topic.
    image_pub (rclpy.Publisher): Publisher for processed images (if enabled).
    bridge (CvBridge): Converter between ROS Image messages and OpenCV images.
    model (YOLO): YOLOv8 pose model used for person detection.
"""
class PersonProcessor(Node):
    def __init__(self):
        super().__init__('person_processor')  
        self.get_logger().info("Node 'person_processor' started!")
        self.get_logger().info(f"Similarity threshold start value: {get_similarity_threshold():.2f}")

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
        self.count_pub = self.create_publisher(Int32, 'person/count', 10)
        
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
            self.listener_callback,
            1)
       
        self.bridge = CvBridge()


    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process the current frame in computer vision script
        processed_frame, people_detected, people_count, detections = person_process_frame(frame, self.model)

        #publish detection message
        det_msg = Bool()
        det_msg.data = people_detected
        self.detected_pub.publish(det_msg)

        # publish count message
        count_msg = Int32()
        count_msg.data = people_count  # set the Python int into the ROS message
        self.count_pub.publish(count_msg)

        # publish bounding boxes
        detections_msg = PersonDetectionArray()
        detections_msg.header = msg.header

        for i, det in enumerate(detections):

            person_id, x, y, w, h, conf = det

            msg = PersonDetection()
            msg.id = person_id
            msg.x = int(x)
            msg.y = int(y)
            msg.width = int(w)
            msg.height = int(h)
            msg.confidence = float(conf)

            detections_msg.detections.append(msg)

        self.detections_pub.publish(detections_msg)

        #publish the processed image for the Web Server 
        if self.show_gui:
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PersonProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
