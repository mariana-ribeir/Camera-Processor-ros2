import rclpy
from rclpy.node import Node
from std_msgs.msg import String

"""
ROS2 node responsible for combining pose detection results from two sources:
an AI-based model and a heuristic-based algorithm.

The node subscribes to pose detection messages from both systems, compares
their outputs, and publishes a final pose detection result with an estimated
certainty level.

Subscriptions:
    /pose/ia/detected (std_msgs/String): Pose detection results produced by the AI-based pose detection node.
    /pose/heuristic/detected (std_msgs/String): Pose detection results produced by the heuristic pose detection node.

Publishers:
    pose/detected (std_msgs/String): Final fused pose detection result including certainty level and source agreement.

Attributes:
    latest_ai_pose (str | None): Most recent pose detection result received from the AI node.
    latest_heuristic_pose (str | None):  Most recent pose detection result received from the heuristic node.
    ai_sub (rclpy.Subscription): Subscriber for the AI pose detection topic.
    heuristic_sub (rclpy.Subscription): Subscriber for the heuristic pose detection topic.
    detected_pub (rclpy.Publisher): Publisher for the final fused pose detection result.
"""
class PoseProcessorNode(Node):
    def __init__(self):
        super().__init__('pose_processor')
        self.get_logger().info("Node 'pose_processor' (AI + Heuristic) started!")

       # State variables to store the latest from each node
        self.latest_ai_pose = None
        self.latest_heuristic_pose = None

        # Subscribers
        self.ai_sub = self.create_subscription(
            String, '/pose/ia/detected', self.ai_callback, 10)
        
        self.heuristic_sub = self.create_subscription(
            String, '/pose/heuristic/detected', self.heuristic_callback, 10)
        
        # Publisher
        self.detected_pub = self.create_publisher(String, 'pose/detected', 10)
        
        # Timer to check and compare at 10Hz (0.1s)
        self.create_timer(0.1, self.compare_and_publish)

    
    def ai_callback(self, msg):
        """
        Callback executed when a new AI pose detection message is received.

        Args:
            msg (std_msgs.msg.String): Message containing AI pose detection results.
        """
        # Logic to extract the pose list from the string "Poses: standing, sitting"
        self.latest_ai_pose = msg.data
    

    def heuristic_callback(self, msg):
        """
        Callback executed when a new heuristic pose detection message is received.

        Args:
            msg (std_msgs.msg.String): Message containing heuristic pose detection results.
        """
        self.latest_heuristic_pose = msg.data

    def compare_and_publish(self):
        """
        Periodic timer callback that compares the latest AI and heuristic pose
        detections and publishes a fused result with a certainty estimate.
        """
        # If we don't even have AI data, we can't do much.
        if self.latest_ai_pose is None:
            self.get_logger().debug("Waiting for AI data...")
            return 

        final_msg = String()
        
        # Case 1: We have both - Compare them
        if self.latest_heuristic_pose is not None:
            if self.latest_ai_pose == self.latest_heuristic_pose:
                certainty = 100
                status = "Agreed"
            else:
                certainty = 50
                status = "Disputed"
            
            final_msg.data = f"Certainty: {certainty}% | Mode: {status} | AI: {self.latest_ai_pose}"

        # Case 2: Heuristic is missing - Trust AI but note the lack of verification
        else:
            certainty = 75 
            status = "AI Only (Heuristic Offline)"
            final_msg.data = f"Certainty: {certainty}% | Mode: {status} | AI: {self.latest_ai_pose}"

        self.detected_pub.publish(final_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()