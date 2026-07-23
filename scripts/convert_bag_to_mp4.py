import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class BagToMp4(Node):
    def __init__(self):
        super().__init__('bag_to_mp4')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/rgb/color/rect/image/compressed',
            self.image_callback,
            10)
        self.out = None

    def image_callback(self, msg):
        try:
            # Convert received CompressedImage to cv2/numpy array
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if self.out is None:
                height, width, _ = frame.shape
                # Use mp4v codec for mp4 output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter('zed_video.mp4', fourcc, 30.0, (width, height))
                self.get_logger().info(f"Started writing to zed_video.mp4 at {width}x{height}")
                
            self.out.write(frame)
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def destroy_node(self):
        if self.out is not None:
            self.out.release()
            self.get_logger().info("Video completely saved to zed_video.mp4")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BagToMp4()
    try:
        print("Waiting for images on /zed/zed_node/rgb/color/rect/image/compressed...")
        print("--> PLAY YOUR BAG FILE IN ANOTHER TERMINAL NOW <--")
        print("Press Ctrl+C here when the bag finishes to save the mp4.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
