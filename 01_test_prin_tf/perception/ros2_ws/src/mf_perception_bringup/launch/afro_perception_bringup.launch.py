from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
import rclpy
from launch.actions import TimerAction


def generate_launch_description():
 
  yolo = Node(
        package='mf_mot_object_tracker',
        executable='yolo.py',
        output='screen',
        emulate_tty=True,
        prefix='nice -n 5', 
        arguments=['strawberry_harvesting_pollination']
  )



  mot = Node(
        package='mf_mot_object_tracker',
        # executable='mot_by_bbox_server.py',
        # executable='mot_by_bbox.py',
        executable='mot_by_bbox_pre_a.py',
        output='screen',
        emulate_tty=True,
        arguments=['strawberry_harvesting_pollination']
    )


  return LaunchDescription([
    yolo,
    mot,

  ])
