import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace


def generate_launch_description():

    fairino_bringup =IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('fairino5_v6_moveit2_config'), 'launch'), '/demo.launch.py']),
      )

    moveit_driver_node = Node(
        package='mf_robot_driver',
        executable='moveit_driver_node.py',
        name='robot_driver_node',
        output='screen',
        parameters=[
            os.path.join(get_package_share_directory('mf_robot_driver'), 'config', 'fairino5.yaml')
        ]
    )

    fairino_server = Node(
        package='fairino_hardware',
        executable='ros2_cmd_server',
        name='ros2_cmd_server',
        output='screen'
    )

    return LaunchDescription([
      GroupAction(
            actions=[
                fairino_bringup,
                fairino_server,
                moveit_driver_node
            ]
        )
    ])
