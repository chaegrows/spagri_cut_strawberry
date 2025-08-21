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

    rb_bringup = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('rb5_850e_moveit_config'), 'launch'), '/moveit.launch.py']),
      )

    moveit_params = os.path.join(get_package_share_directory('mf_robot_driver'), 'config', 'rb5.yaml')
    print("--------------------------------\n")
    print("moveit_params: ", moveit_params)
    print("--------------------------------\n")

    moveit_driver_node = Node(
        package='mf_robot_driver',
        executable='moveit_driver_node.py',
        name='robot_driver_node',
        output='screen',
        parameters=[
            moveit_params
        ]
    )

    return LaunchDescription([
        GroupAction(
            actions=[
                rb_bringup,
                moveit_driver_node
            ])
    ])
