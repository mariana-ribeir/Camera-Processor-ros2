import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory

"""
ROS2 Node that simulates a camera using a video file.

Publishes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    timer (rclpy.Timer): Timer to periodically publish frames
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
"""
class CameraSimulator(Node):
    def __init__(self, video_path, rotate_video=False):
        super().__init__('camera_simulator')
        self.get_logger().info("Node 'camera_simulator' started!")
        
        # Declarar parámetros para que el Launch file pueda controlarlos
        self.declare_parameter('video_path', video_path)
        self.declare_parameter('rotate_video', rotate_video)
        
        # Obtener los valores (prioriza lo que diga el launch file)
        self.video_path = self.get_parameter('video_path').value
        self.rotate_video = self.get_parameter('rotate_video').value

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 1)
        # Seteado a 0.25 (4 FPS) para mayor fluidez. Si el Jetson laguea, subir a 0.3 o 0.4
        self.timer = self.create_timer(0.25, self.timer_callback)
        self.bridge = CvBridge()

        # Open video (0 = webcam, ou "video.mp4")
        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f"Error open the video(simulation of camera): {self.video_path}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Optimization: Resize frame to 640x480 to reduce load on AI nodes (INTER_AREA is better for downscaling)
        #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # Rotate the frame 90 degrees clockwise if enabled
        if self.rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Publish camera image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher_.publish(msg)
        #elf.get_logger().info("Publicando frame...")

def main(args=None):
    print("Init camera simulation...")
    #Start the rcply mesages
    rclpy.init(args=args)

    #Define the name of the folder of videos data
    pkg_share = get_package_share_directory('camera')
    data_dir = os.path.join(pkg_share, 'data')
    video=os.path.join(data_dir, 'OBRA1.mp4')

    # Flag to rotate the video 90 degrees clockwise
    rotate_video = False

    #Create the node
    node = CameraSimulator(video, rotate_video=rotate_video)
    #Run the node
    rclpy.spin(node)
    #Clean the node after the executation
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
