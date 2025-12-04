import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory

"""
ROS2 Node that simulates a camera using a video file.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    timer (rclpy.Timer): Timer to periodically publish frames
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
"""
class CameraSimulator(Node):
    def __init__(self, video_path):
        super().__init__('camera_simulator')
        self.get_logger().info("Node 'camera_simulator' started!")
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS
        self.bridge = CvBridge()

        # Open video (0 = webcam, ou "video.mp4")
        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Error open the video(simulation of camera): {video_path}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            #self.get_logger().info("End of video or error capturing the frame of camera.")
            # Start over the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Publish camera image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        #elf.get_logger().info("Publicando frame...")

def main(args=None):
    print("Init camera simulation...")
    #Start the rcply mesages
    rclpy.init(args=args)

    #Define the name of the folder of videos data
    pkg_share = get_package_share_directory('camera')
    data_dir = os.path.join(pkg_share, 'data')
    video=os.path.join(data_dir, 'Video Project.mp4')

    #Create the node
    node = CameraSimulator(video)
    #Run the node
    rclpy.spin(node)
    #Clean the node after the executation
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
