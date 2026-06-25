from launch import LaunchDescription
from launch_ros.actions import Node

"""
Launch file for the Full Human Pose Detection and Fusion Pipeline, without GUI.

This script orchestrates the following data flow:
1. Camera Simulator: Provides the raw video stream.
2. Person Processor: Detects and tracks people (Bounding Boxes).
3. AI & Heuristic Nodes: Calculate poses based on detections.
4. Pose Processor: Fuses and stabilizes pose data.

Note: 'show_gui' is disabled for processing nodes to allow remote visualization.
"""
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera',
            executable='camera_simulator',
            parameters=[{'show_gui': False}],
            name='camera_simulator'
        ),
        
        Node(
            package='camera_processor',
            executable='person_processor',
            name='person_processor',
            parameters=[{'show_gui': False}],
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='ai_pose',
            parameters=[{'show_gui': False}],
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='heuristic_pose',
            parameters=[{'show_gui': False}],
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='pose_processor',
            output='screen'
        ),

        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            name='ros_tcp_endpoint',
            emulate_tty=True,
            parameters=[{
                'ROS_IP': '0.0.0.0',
                'ROS_TCP_PORT': 10000
            }],
            output='screen'
        ),
    ])