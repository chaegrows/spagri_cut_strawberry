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

    endeffector = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/endeffector_driver.launch.py']),
    )

    sensors = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/bringup_camera.launch.py']),
    )

    job_manager = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('job_manager'), 'launch'), '/job_manager.launch.py']),
    )
    statistics_collector = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('statistics_collector'), 'launch'), '/statistics_collector.launch.py']),
    )
    control_server = Node(
        package='mf_bringup',
        executable='control_server.py',
        name='control_server',
        output='screen'
    )

    assemble_static_tf = Node(
      package='mf_bringup',
      executable = 'assemble_robot_tf_node.py',
      name='assemble_robot_tf_node',
      output = 'screen',
      parameters=[
        {
          'robot_spec': 'fairino_robot'
        }
      ]
    )
    rosbag_recorder = Node(
      package='rosbag_recorder_server',
      executable='rosbag_recorder.py',
      output='screen'
      )

    workspace_check_node = Node(
      package='health_manager',
      executable = 'workspace_check_node.py',
      name='workspace_check_node',
      output = 'screen'
    )

    return LaunchDescription([
        GroupAction(
            actions=[
                sensors,
                endeffector,
                assemble_static_tf,
                job_manager,
                statistics_collector,
                workspace_check_node,
                control_server,
                # rosbag_recorder,
            ])
    ])
