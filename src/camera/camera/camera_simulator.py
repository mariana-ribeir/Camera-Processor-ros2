import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory

"""
ROS 2 node that publishes frames from a video file or a live camera.

Publishes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Attributes:
    publisher_ (rclpy.Publisher): Publisher for /camera/image_raw
    timer (rclpy.Timer): Timer to periodically publish frames
    cap (cv2.VideoCapture): OpenCV video capture object
    bridge (CvBridge): Converter from OpenCV images to ROS2 Image messages
"""


def resolve_capture_source(camera_source, video_path, camera_device):
    source = str(camera_source).strip().lower()
    if source == 'video':
        return source, video_path
    if source == 'camera':
        return source, camera_device
    raise ValueError(
        'CAMERA_SOURCE debe ser "video" o "camera"; '
        f'se recibio: {camera_source!r}'
    )


class CameraSimulator(Node):
    def __init__(self, video_path, rotate_video=False):
        super().__init__('camera_simulator')
        self.get_logger().info("Node 'camera_simulator' started!")
        
        self.declare_parameter('video_path', video_path)
        self.declare_parameter('rotate_video', rotate_video)
        self.declare_parameter(
            'camera_source', os.getenv('CAMERA_SOURCE', 'video')
        )
        self.declare_parameter(
            'camera_device', os.getenv('CAMERA_DEVICE', '/dev/video0')
        )
        
        self.video_path = self.get_parameter('video_path').value
        self.rotate_video = self.get_parameter('rotate_video').value
        requested_source = self.get_parameter('camera_source').value
        self.camera_device = self.get_parameter('camera_device').value
        self.camera_source, capture_source = resolve_capture_source(
            requested_source, self.video_path, self.camera_device
        )

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 1)
        # Publica a un maximo de 4 FPS.
        self.timer = self.create_timer(0.25, self.timer_callback)
        self.bridge = CvBridge()

        if self.camera_source == 'camera':
            self.cap = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
            self.get_logger().info(
                f'Abriendo camara en {capture_source}'
            )
        else:
            self.cap = cv2.VideoCapture(capture_source)
            self.get_logger().info(
                f'Abriendo video en {capture_source}'
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f'No se pudo abrir {self.camera_source}: {capture_source}'
            )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.camera_source == 'video':
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
    node = None
    try:
        node = CameraSimulator(video, rotate_video=rotate_video)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.cap.release()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
