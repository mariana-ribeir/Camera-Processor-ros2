import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from camera_processor.processor import color_process_frame

"""
ROS2 Node that simulates a camera using a video file.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
    timer (rclpy.Timer): Timer to periodically publish frames
"""
class ColorProcessor(Node):
    def __init__(self):
        super().__init__('color_processor')  # ROS node name
        self.get_logger().info("Node 'color_processor' started!")
        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        #create an boolean topic to see if red is present in frame or not 
        self.red_pub = self.create_publisher(Bool, 'color/red_detected', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.namedWindow("Real Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real Frame", 800, 600)
        cv2.imshow("Real Frame", frame)

        # process the current frame in computer vision script
        processed, red_detected = color_process_frame(frame)

        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Frame", 800, 600)
        cv2.imshow("Processed Frame", processed)
        cv2.waitKey(1)

        #publish detection message
        det_msg = Bool()
        det_msg.data = bool(red_detected)
        self.red_pub.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ColorProcessor()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
