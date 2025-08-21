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
    workspace_check_node = Node(
      package='health_manager',
      executable = 'workspace_check_node.py',
      name='workspace_check_node',
      output = 'screen'
    )

    return LaunchDescription([
        GroupAction(
            actions=[
                workspace_check_node,
            ])
    ])
