import rclpy
from rclpy.node import Node
from collections import deque
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from camera_interfaces.msg import PoseDetection, PoseDetectionArray

"""
ROS2 Node for Pose Fusion and Temporal Filtering.

This node acts as a central processor that fuses pose detections from two 
different sources (AI-based and Heuristic-based). It synchronizes messages by 
timestamp and applies a sliding window consensus algorithm to filter out 
jitter and transient false positives.

Subscriptions:
    /pose/ia/detected (camera_interfaces/PoseDetectionArray): High-accuracy AI pose data.
    /pose/heuristic/detected (camera_interfaces/PoseDetectionArray): Heuristic-calculated pose data.

Publishers:
    /pose/detected (camera_interfaces/PoseDetectionArray): The final, fused, and stabilized pose data.

Attributes:
    pose_history (dict): Dictionary mapping person IDs to a deque of recent poses.
    window_size (int): Number of historical frames to consider (default: 10).
    consensus_ratio (float): Required agreement percentage to confirm a pose change (0.8).
"""
class PoseProcessorNode(Node):
    def __init__(self):
        super().__init__('pose_processor')
        self.get_logger().info("Node 'pose_processor' (AI + Heuristic) started!")

        self.bridge = CvBridge()

        # Sliding window parameters
        self.window_size = 10
        self.consensus_ratio = 0.8
        self.pose_history = {} 

        # Latest raw messages
        self.latest_ai_msg = None
        self.latest_heuristic_msg = None
        self.processing = False

        # Subscribers
        self.ai_sub = self.create_subscription(
            PoseDetectionArray, '/pose/ia/detected', self.ai_callback, 10)
        self.heuristic_sub = self.create_subscription(
            PoseDetectionArray, '/pose/heuristic/detected', self.heuristic_callback, 10)

        # Publisher
        self.detected_pub = self.create_publisher(PoseDetectionArray, 'pose/detected', 10)

        # Timer to process messages periodically
        self.create_timer(0.05, self.process_poses)

    def ai_callback(self, msg: PoseDetectionArray):
        """Stores the latest message from the AI pose node."""
        self.latest_ai_msg = msg

    def heuristic_callback(self, msg: PoseDetectionArray):
        """Stores the latest message from the Heuristic pose node."""
        self.latest_heuristic_msg = msg

    def fuse_pose(self, ai_pose, heuristic_pose):
        """
        Combines results from both sources. 
        Currently prioritizes AI if they disagree.
        """
        if ai_pose == heuristic_pose:
            return ai_pose
        return ai_pose  

    def process_poses(self):
        if self.processing:
            return

        if self.latest_ai_msg is None or self.latest_heuristic_msg is None:
            return

        #  timestamp synchronization 
        ai_time = self.latest_ai_msg.header.stamp
        h_time = self.latest_heuristic_msg.header.stamp

        ai_sec = ai_time.sec + ai_time.nanosec * 1e-9
        h_sec = h_time.sec + h_time.nanosec * 1e-9

        time_diff = abs(ai_sec - h_sec)

        #if frames are more than 100ms apart, they are likely not the same frame
        if time_diff > 0.1: 
            self.get_logger().debug(f"Skipping unsynced frames: {time_diff:.3f}s")
            return

        self.processing = True

        # prepare data 
        ai_detections = self.latest_ai_msg.pose_detections
        heuristic_detections = self.latest_heuristic_msg.pose_detections

        heuristic_dict = {p.id: p.pose for p in heuristic_detections}
        final_poses = []

        # process each detected person
        for p in ai_detections:
            pid = p.id
            ai_pose = p.pose
            h_pose = heuristic_dict.get(pid, None)

            #initial fusion
            fused_pose = self.fuse_pose(ai_pose, h_pose)

            # sliding window
            if pid not in self.pose_history:
                self.pose_history[pid] = deque(maxlen=self.window_size)

            self.pose_history[pid].append(fused_pose)

            # count frequency of each pose in history
            counts = {
                pose: self.pose_history[pid].count(pose)
                for pose in set(self.pose_history[pid])
            }

            most_common_pose = max(counts, key=counts.get)
            consensus_ratio_actual = counts[most_common_pose] / len(self.pose_history[pid])

            # if AI and Heuristic match immediately, trust it.
            # otherwise, only use the 'most common' if it hits the consensus threshold.
            if h_pose is not None and ai_pose == h_pose:
                final_pose = fused_pose
            elif consensus_ratio_actual >= self.consensus_ratio:
                final_pose = most_common_pose
            else:
                final_pose = fused_pose  

            pose_msg = PoseDetection()
            pose_msg.id = pid
            pose_msg.pose = final_pose

            final_poses.append(pose_msg)

        # remove IDs that are no longer being detected
        active_ids = {p.id for p in ai_detections}

        for pid in list(self.pose_history.keys()):
            if pid not in active_ids:
                del self.pose_history[pid]

        # publish the stabilized result
        msg_out = PoseDetectionArray()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = "camera"
        msg_out.pose_detections.extend(final_poses)

        self.detected_pub.publish(msg_out)
        self.get_logger().debug(f"Published {len(final_poses)} poses")
        self.processing = False

def main(args=None):
    rclpy.init(args=args)
    node = PoseProcessorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()

if __name__ == '__main__':
    main()