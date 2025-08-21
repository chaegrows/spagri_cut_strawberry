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
    device_health_check_node = Node(
        package='health_manager',
        executable='device_health_check_node.py',
        name='device_health_check_node',
        prefix='chrt -f 89',
        output='screen',
        parameters=[
            {'unit_device': "['MANIPULATOR', 'CAM_HAND', 'EEF']"}, # , 
            {'cpu_timeout': 10},
            {'cpu_threshold': 80},
            {'heartbeat_timeout': 3.0}
        ]
    )

    return LaunchDescription([
        GroupAction(
            actions=[
                device_health_check_node,
            ])
    ])
