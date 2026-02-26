import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

from camera_processor.processor import (
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

Publishes:
    person/detected (std_msgs/Bool): Boolean flag indicating if some person was detected.
    person/count (std_msgs/Int32): Integer number indicating the number of persons was detected.

Attributes:
    subscription (rclpy.Subscription): Subscriber to the '/camera/image_raw' topic.
    detected_pub (rclpy.Publisher): Publisher for the 'person/detected' (Bool) topic.
    count_pub (rclpy.Publisher): Publisher for the 'person/count' (Int32) topic.
    bridge (CvBridge): Converter between OpenCV images and ROS2 Image messages.
"""
class PersonProcessor(Node):
    def __init__(self):
        super().__init__('person_processor')  # ROS node name
        self.get_logger().info("Node 'person_processor' started!")
        self.get_logger().info(f"Similarity threshold start value: {get_similarity_threshold():.2f}")

        # path setup for model
        pkg_share = get_package_share_directory('camera_processor')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n-pose.pt')

        # load the model
        self.get_logger().info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'person/processed_image', 1)
        
        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        #create an boolean topic to see if some person is present in frame or not 
        self.detected_pub = self.create_publisher(Bool, 'person/detected', 10)
        #create an iny topic to count how many person are present in frame
        self.count_pub = self.create_publisher(Int32, 'person/count', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process the current frame in computer vision script
        processed_frame, people_detected, people_count  = person_process_frame(frame, self.model)

        #publish detection message
        det_msg = Bool()
        det_msg.data = people_detected
        self.detected_pub.publish(det_msg)

        # Publish count message
        count_msg = Int32()
        count_msg.data = people_count  # set the Python int into the ROS message
        self.count_pub.publish(count_msg)

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
