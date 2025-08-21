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

    eef_driver_node = Node(
        package='mf_serial',
        executable='mf_eef_driver',
        name='mf_eef_driver_node',
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
    )

    return LaunchDescription([
        eef_driver_node,
    ])
