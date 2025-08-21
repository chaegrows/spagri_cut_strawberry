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

    camera_hand = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/realsense.launch.py']),
      launch_arguments={
            'camera_name': 'camera_hand',
            'output': 'screen'
            }.items()
    )

    # camera_base_front = IncludeLaunchDescription(
    # PythonLaunchDescriptionSource([os.path.join(
    #   get_package_share_directory('mf_bringup'), 'launch'), '/realsense.launch.py']),
    #   launch_arguments={
    #         'camera_name': 'camera_base_front',
    #         'output': 'screen'
    #         }.items()
    # )


    return LaunchDescription([
        GroupAction(
            actions=[
                PushRosNamespace('/camera'),
                camera_hand,
                # camera_base_front
            ])
    ])
