from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

from camera_processor.helpers.polar_rosbag import polar_rosbag_command


def generate_launch_description():
    actions = [
        Node(
            package='camera_processor',
            executable='empatica_gmm_monitor',
            name='empatica_gmm_monitor',
            output='screen',
        ),
        Node(
            package='camera_processor',
            executable='polarh10_monitor',
            name='polarh10_monitor',
            output='screen',
        ),
    ]

    bag_command = polar_rosbag_command()
    if bag_command is not None:
        actions.insert(0, ExecuteProcess(cmd=bag_command, output='screen'))

    return LaunchDescription(actions)