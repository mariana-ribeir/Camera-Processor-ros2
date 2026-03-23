import rclpy
from rclpy.node import Node
from collections import deque
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from camera_interfaces.msg import PoseDetection, PoseDetectionArray

class PoseProcessorNode(Node):
    def __init__(self):
        super().__init__('pose_processor')
        self.get_logger().info("Node 'pose_processor' (AI + Heuristic) started!")

        self.bridge = CvBridge()

        # Sliding window parameters
        self.window_size = 10
        self.consensus_ratio = 0.8
        self.pose_history = {}  # {id: deque([...], maxlen=window_size)}

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
        self.latest_ai_msg = msg

    def heuristic_callback(self, msg: PoseDetectionArray):
        self.latest_heuristic_msg = msg

    def fuse_pose(self, ai_pose, heuristic_pose):
        """
        Simple fusion logic:
        - If AI and Heuristic agree → return that
        - Else → return AI (could refine with confidence later)
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

        if time_diff > 0.1: 
            self.get_logger().debug(f"Skipping unsynced frames: {time_diff:.3f}s")
            return

        self.processing = True

        # prepare data 
        ai_detections = self.latest_ai_msg.pose_detections
        heuristic_detections = self.latest_heuristic_msg.pose_detections

        heuristic_dict = {p.id: p.pose for p in heuristic_detections}
        final_poses = []

        # process person
        for p in ai_detections:
            pid = p.id
            ai_pose = p.pose
            h_pose = heuristic_dict.get(pid, None)

            fused_pose = self.fuse_pose(ai_pose, h_pose)

            # sliding window
            if pid not in self.pose_history:
                self.pose_history[pid] = deque(maxlen=self.window_size)

            self.pose_history[pid].append(fused_pose)

            counts = {
                pose: self.pose_history[pid].count(pose)
                for pose in set(self.pose_history[pid])
            }

            most_common_pose = max(counts, key=counts.get)
            consensus_ratio_actual = counts[most_common_pose] / len(self.pose_history[pid])

            # fuse logic ---
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

        # clean old IDs 
        active_ids = {p.id for p in ai_detections}

        for pid in list(self.pose_history.keys()):
            if pid not in active_ids:
                del self.pose_history[pid]

        # publish result
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