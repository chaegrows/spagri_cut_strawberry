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

    mobile_driver = Node(
        package='mf_serial',
        executable='md_main',
        name='mobile_driver',
        output='screen'
    )


    return LaunchDescription([
        mobile_driver,
    ])
