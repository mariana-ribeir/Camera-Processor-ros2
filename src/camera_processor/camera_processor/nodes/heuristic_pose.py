import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from camera_processor.helpers.pose_heuristic import pose_process_frame_keypoints

"""
ROS2 Node for real-time human pose detection.

Subscribes to raw camera images, processes them using computer vision
to detect people, and the pose of them.

Subscribes:
    /camera/image_raw (sensor_msgs/Image): The raw video stream input.

Attributes:
    subscription (rclpy.Subscription): Subscriber to the '/camera/image_raw' topic.
    bridge (CvBridge): Converter between OpenCV images and ROS2 Image messages.
"""
class HeuristicPoseNode(Node):
    def __init__(self):
        super().__init__('heuristic_pose')  # ROS node name
        self.get_logger().info("Node 'heuristic_pose' started!")

        # parameter for GUI toggle
        self.declare_parameter('show_gui', True)
        self.show_gui = self.get_parameter('show_gui').value

        if self.show_gui:
            cv2.namedWindow("Real Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Real Frame", 800, 600)
            cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Processed Frame", 800, 600)

        #subscribe the image topic 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        
        #create an boolean topic to see if some person is present in frame or not 
        self.detected_pub = self.create_publisher(String, 'pose/heuristic/detected', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process the current frame in computer vision script
        processed_frame, detected_poses  = pose_process_frame_keypoints(frame)

        if self.show_gui:
            cv2.imshow("Real Frame", frame)
            cv2.imshow("Processed Frame", processed_frame)
            cv2.waitKey(1)

        # 1. Publish the detection message (e.g., the combined pose string)
        if detected_poses:
            # Combine all detected poses into a single string for publishing
            pose_string = ", ".join(detected_poses)
            
            det_msg = String()
            det_msg.data = f"Detected {len(detected_poses)} people. Poses: {pose_string}"
            self.detected_pub.publish(det_msg)
            
            # Use ROS logging for status updates
            self.get_logger().info(f"Published detection: {det_msg.data}")
        else:
            # Publish a message if no person is detected
            det_msg = String()
            det_msg.data = "No person detected."
            self.detected_pub.publish(det_msg)
            self.get_logger().debug("No person detected.") # Use debug for frequent non-critical updates


        # Display the frames using the annotated_frame result
        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Frame", 800, 600)
        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(1) # Important for cv2.imshow to work


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    if node.show_gui:
        cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()