from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera',
            executable='camera_simulator',
            name='camera_simulator'
        ),

        Node(
            package='camera_processor',
            executable='color_processor',
            name='color_processor',
            output='screen'
        ),
        
        Node(
            package='camera_processor',
            executable='person_processor',
            name='person_processor',
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='ia_pose',
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='heuristic_pose',
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='pose_processor',
            output='screen'
        ),

        # Node Web Video Server (The Eyes)
        # allows you to see the 'processed_image's in a browser
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            parameters=[{'port': 8080}],
            output='screen'
        )
    ])