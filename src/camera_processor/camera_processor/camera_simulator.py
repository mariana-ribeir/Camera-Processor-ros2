import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

import cv2
from camera_processor.processor import process_frame

"""
ROS2 Node that simulates a camera using a video file.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
    timer (rclpy.Timer): Timer to periodically publish frames
"""
class CameraSimulator(Node):
    def __init__(self, video_path):
        super().__init__('camera_simulator')  # ROS node name
        #publish an image topic 
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        #camera
        #self.cap = cv2.VideoCapture(0)
        #video
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Error open the video(simulation of camera): {video_path}")
        self.bridge = CvBridge()  

        self.timer = self.create_timer(1/30, self.timer_callback)  # 30 fps
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # Reinicia o vídeo quando terminar
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
         # process the current frame in computer vision script
        processed = process_frame(frame)

        # show process frame
        cv2.imshow("Frame Processado", processed)
        cv2.waitKey(1)

        # publish message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    video=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../walk_people.mp4')
    node = CameraSimulator(video)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
