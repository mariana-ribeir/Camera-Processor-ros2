import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from camera_processor.helpers.color_detector import color_process_frame

"""
ROS 2 node responsible for receiving image streams, delegating color 
detection processing (e.g., red color presence) to an external backend 
script, and publishing the detection results and the processed image.

Subscribes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Publishes:
    color/red_detected (std_msgs/Bool): Boolean flag indicating if the target color (red) was detected.
    color/processed_image (sensor_msgs/Image): The frame after the computer vision processing, including annotations.

Attributes:
    bridge (CvBridge): Utility for converting between ROS Image messages and OpenCV images (numpy arrays).
    subscription (rclpy.Subscription): Subscription object for the input image topic.
    red_pub (rclpy.Publisher): Publisher for the boolean detection flag.
    image_pub (rclpy.Publisher): Publisher for processed images when visualization is enabled.
"""
class ColorProcessor(Node):
    def __init__(self):
        super().__init__('color_processor')  # ROS node name
        self.get_logger().info("Node 'color_processor' started!")

        #publish the processed image so we can see it remotely
        self.declare_parameter('show_gui', False)
        self.show_gui = self.get_parameter('show_gui').value

        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        #create an boolean topic to see if red is present in frame or not 
        self.red_pub = self.create_publisher(Bool, 'color/red_detected', 10)
        self.bridge = CvBridge()

        if self.show_gui:
            self.image_pub = self.create_publisher(Image, 'color/processed_image', 1)


    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process the current frame in computer vision script
        processed, red_detected = color_process_frame(frame)

        # publish the processed image itself
        if self.show_gui:   
            img_msg = self.bridge.cv2_to_imgmsg(processed, encoding="bgr8")
            self.image_pub.publish(img_msg)

        #publish detection message
        det_msg = Bool()
        det_msg.data = bool(red_detected)
        self.red_pub.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ColorProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
