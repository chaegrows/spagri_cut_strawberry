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

    job_manager = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('job_manager'), 'launch'), '/job_manager.launch.py']),
    )
    statistics_collector = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('statistics_collector'), 'launch'), '/statistics_collector.launch.py']),
    )

    rosbag_recorder = Node(
      package='rosbag_recorder_server',
      executable='rosbag_recorder.py',
      output='screen'
      )


    return LaunchDescription([
        GroupAction(
            actions=[

                job_manager,
                statistics_collector,

                # rosbag_recorder,

            ])
    ])
