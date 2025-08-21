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

    mobile = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/mobile_md200.launch.py']),
    )

    manipulator = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/manipulator_ur.launch.py']),
    )

    endeffector = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/endeffector.launch.py']),
    )

    sensors = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([os.path.join(
      get_package_share_directory('mf_bringup'), 'launch'), '/sensors.launch.py']),
    )

    control_server = Node(
        package='mf_bringup',
        executable='dx6e_control_server.py',
        name='control_server',
        output='screen'
    )

    ARGUMENTS = [
        DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0',
                             description='Joystick device path'),
        DeclareLaunchArgument('joy_deadzone', default_value='0.05',
                             description='Joystick deadzone value'),
    ]
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': LaunchConfiguration('joy_dev'),
            'deadzone': LaunchConfiguration('joy_deadzone'),
            'autorepeat_rate': 20.0,
        }],
        output='screen'
    )
    assemble_static_tf = Node(
      package='mf_bringup',
      executable = 'assemble_robot_tf_node.py',
      name='assemble_robot_tf_node',
      output = 'screen'
    )
    return LaunchDescription(
        ARGUMENTS + [
            GroupAction(
                actions=[
                    assemble_static_tf,
                    control_server,
                    joy_node,
                    # mobile,
                    # manipulator,
                    # endeffector,
                    # sensors,
                ]
            )
        ]
    )