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
  job_manager_node = Node(
      package='job_manager',
      executable='job_manager_node.py',
      output='screen'
  )

  return LaunchDescription([
    GroupAction(
      actions=[
        job_manager_node,
      ])
  ])
