from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_processor',
            executable='empatica_gmm_monitor',
            name='empatica_gmm_monitor',
            output='screen',
        ),
    ])