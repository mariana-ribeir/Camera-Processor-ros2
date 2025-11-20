import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from camera_processor.processor import process_frame


#Define the name of the package
pkg_share = get_package_share_directory('camera_processor')
#Define the name of the folder of videos data
data_dir = os.path.join(pkg_share, 'data')
video=os.path.join(data_dir, 'orange.mp4')

# listar todos os arquivos mp4
#videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
# exemplo 2: escolher aleatório
# video_path = os.path.join(video_dir, random.choice(videos))
#self.get_logger().info(f"Using video: {video_path}")

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
        #create an image topic 
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        #create an boolean topic to see if red is present in frame or not 
        self.red_pub = self.create_publisher(Bool, '/camera/red_detected', 10)

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
        processed, red_detected = process_frame(frame)

        # show process frame
        cv2.namedWindow("Frame Processado", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame Processado", 800, 600)
        cv2.imshow("Frame Processado", processed)
        cv2.waitKey(1)

        #publish camera image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

        #publish detection message
        det_msg = Bool()
        det_msg.data = bool(red_detected)
        self.red_pub.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSimulator(video)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
