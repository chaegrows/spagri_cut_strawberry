#-*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    # 리프트 드라이버 노드
    lift_driver = Node(
        package='mf_serial',
        executable='lift_main',
        name='lift_driver',
        output='screen'
    )

    # 매니퓰레이터 드라이버 노드
    manipulator_driver = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_robot_driver'), 'launch'), '/rb5_driver.launch.py']),
    )

    endeffector = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/endeffector_driver.launch.py']),
    )

    # 수동 제어 서버 노드
    control_server = Node(
        package='mf_bringup',
        executable='control_server.py',
        name='control_server_manual',
        output='screen'
    )

    # 조이스틱 노드
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    return LaunchDescription([
        GroupAction(
            actions=[
                lift_driver,
                endeffector,
                control_server,
                joy_node,
                manipulator_driver,
            ])
    ])
