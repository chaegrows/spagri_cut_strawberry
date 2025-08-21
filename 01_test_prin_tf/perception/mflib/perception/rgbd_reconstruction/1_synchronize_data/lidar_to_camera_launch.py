from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.03', '0.00', '-0.05', '-1.5707', '0', '-1.5707', 'lidar', 'femto_bolt'],
            output='screen'
        )
    ])
