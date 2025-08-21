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
    lidar = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/rviz_MID360_launch.py']),
    )

    mobile = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/mobile_driver.launch.py']),
    )

    lift = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/lift_driver.launch.py']),
    )
    # manipulator = IncludeLaunchDescription(
    # PythonLaunchDescriptionSource([os.path.join(
    #   get_package_share_directory('mf_robot_driver'), 'launch'), '/rb5_driver.launch.py']),
    # )

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
    # health_manager = IncludeLaunchDescription(
    # PythonLaunchDescriptionSource([os.path.join(
    #   get_package_share_directory('health_manager'), 'launch'), '/health_manager.launch.py']),
    # )
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
          'robot_spec': 'rb_robot'
        }
      ]
    )

    # robot_base_static_tf = Node(package="tf2_ros",
    #         executable="static_transform_publisher",
    #         arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "manipulator_base_link", "link0"])
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
                lidar,
                sensors,
                endeffector,
                assemble_static_tf,
                mobile,
                lift,
                job_manager,
                statistics_collector,
                workspace_check_node,
                control_server,
                # rosbag_recorder,
            ])
    ])
