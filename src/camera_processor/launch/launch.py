from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera',
            executable='camera_simulator',
            parameters=[{'show_gui': True}],
            name='camera_simulator'
        ),

        Node(
            package='camera_processor',
            executable='color_processor',
            name='color_processor',
            parameters=[{'show_gui': True}],
            output='screen'
        ),
        
        Node(
            package='camera_processor',
            executable='person_processor',
            name='person_processor',
            parameters=[{'show_gui': True}],
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='ia_pose',
            parameters=[{'show_gui': True}],
            output='screen'
        ),

        Node(
            package='camera_processor',
            executable='heuristic_pose',
            parameters=[{'show_gui': True}],
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