import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from camera_processor.processor import (
    person_process_frame,
    adjust_similarity_threshold,
    reset_person_database,
    get_similarity_threshold,
)

"""
ROS2 Node that simulates a camera using a video file.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
    timer (rclpy.Timer): Timer to periodically publish frames
"""
class PersonProcessor(Node):
    def __init__(self):
        super().__init__('person_processor')  # ROS node name
        self.get_logger().info("Node 'person_processor' started!")
        self.get_logger().info(f"Similarity threshold start value: {get_similarity_threshold():.2f}")
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

        cv2.namedWindow("Real Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real Frame", 800, 600)
        cv2.imshow("Real Frame", frame)

        # process the current frame in computer vision script
        processed_frame, people_detected, people_count  = person_process_frame(frame)

        #publish detection message
        det_msg = Bool()
        det_msg.data = people_detected
        self.detected_pub.publish(det_msg)

        # Publish count message
        count_msg = Int32()
        count_msg.data = people_count  # set the Python int into the ROS message
        self.count_pub.publish(count_msg)

        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Frame", 800, 600)
        cv2.imshow("Processed Frame", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('+'), ord('=')):
            new_threshold = adjust_similarity_threshold(0.02)
            self.get_logger().info(f"Similarity threshold increased to {new_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            new_threshold = adjust_similarity_threshold(-0.02)
            self.get_logger().info(f"Similarity threshold decreased to {new_threshold:.2f}")
        elif key in (ord('r'), ord('R')):
            reset_person_database()
            self.get_logger().info("Person database reset")


def main(args=None):
    rclpy.init(args=args)
    node = PersonProcessor()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
