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

def generate_launch_description():

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('mf_robot_driver'), 'launch'), '/ur_control.launch.py']),
        launch_arguments={'ur_type': 'ur5e',
                          'robot_ip': '192.168.50.102'}.items()
        )

    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('mf_robot_driver'), 'launch'), '/ur_moveit.launch.py']),
        launch_arguments={'ur_type': 'ur5e',
                          'robot_ip': '192.168.50.102'}.items()
        )

    return LaunchDescription([
        GroupAction(
            actions=[
                ur_control,
                ur_moveit,
            ])
    ])
