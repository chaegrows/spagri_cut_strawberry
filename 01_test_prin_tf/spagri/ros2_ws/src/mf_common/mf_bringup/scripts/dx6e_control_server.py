#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from sensor_msgs.msg import Joy
from mf_msgs.msg import LiftControl, LiftStatus, LiftCommand, MD200Control
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import String
# Supports 3 control modes: ['Estop', 'Manual', 'Auto']
# Handle ['joy', 'lift', 'ground_vehicle]
# Todo - cloud mode, multi-robot mode?
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2, MfBehaviorTreeContext
from config.common.leaf_actions import leaf_action_impl as la
from typing import Optional
# Button Modes
mode = {
    'estop': 0,
    'md_lift_manual': 1,
    'md_lift_auto': 2,
    'md_harvesting': 3,
    'md_pollinating': 4,
}

JOY_IDX = {
    'gv_rotate': 0,
    'gv_straight': 1,
    'nouse1': 2,
    'lift_height': 3,
    'mode_selection': 4,
    'job_selection': 5, # if mode_selection is auto & job_selection < 0, then harvesting || job_selection > 0, then pollinating
}
LIFT_PID_MODE = {
    'estop': 10,
    'abs': 219,
    'rel': 220,
}

LIFT_CMD_DATA = {
    'estop': 4,
}
# Command Modes
CMD_MODE_LIFTMOBILE = 1
CMD_MODE_EMERGENCY_STOP = 4

# Mobile Control Constants
LINEAR_X_JOY_AXIS = 1
ANGULAR_Z_JOY_AXIS = 0

LINEAR_X_JOY_AXIS_MIN = -0.64
LINEAR_X_JOY_AXIS_MAX = 0.62
ANGULAR_Z_JOY_AXIS_MIN = -0.65
ANGULAR_Z_JOY_AXIS_MAX = 0.63

LINEAR_MAX_X = 0.4
ANGULAR_MAX_Z = 0.1

# Lift Control Constants
LIFT_JOY_AXIS_MIN = -0.65
LIFT_JOY_AXIS_MAX = 0.65
LIFT_MARGIN = 0.1

JOINT_CONTROL_SCALE = 10
LIFT_MIN_HEIGHT = 0
LIFT_MAX_HEIGHT = 1400
i_M_TO_MM = 1000

class DX6EJoyControllerNode(BehaviorTreeServerNodeV2):
    def __init__(self, node_name, action_list: list[la.LeafAction], context: Optional[MfBehaviorTreeContext] = None):
        super().__init__(node_name, action_list, context)


        # gv publishers/subscribers
        self.current_mode_status_pub = self.create_publisher(String, '/btc/target_work_joystick', 1)
        self.cmd_vel_mobile_pub = self.create_publisher(MD200Control, '/md200/in', 1)
        self.last_gv_auto_cmd_vel = None
        self.gv_auto_cmd_vel_sub = self.create_subscription(
            TwistStamped, '/mf_gv/auto_gv_cmd_vel', self.gv_auto_cmd_vel_cb, 1)

        # Lift publishers/subscribers
        self.lift_status_sub = self.create_subscription(LiftStatus, '/lift/out', self.lift_status_callback, 1)
        self.lift_cmd_pub = self.create_publisher(LiftCommand, '/lift/command', 1)
        self.lift_ctrl_pub = self.create_publisher(LiftControl, '/lift/in', 1)
        self.lift_auto_cmd_sub = self.create_subscription(Bool, '/mf_lift/command', self.lift_auto_cmd_cb, 1)
        self.lift_auto_ctrl_sub = self.create_subscription(Float32, '/mf_lift/auto_lift_cmd_height', self.lift_auto_ctrl_cb, 1)
        # Joy subscriber
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 1)

        # Initialize variables
        self.current_mode_status = mode['md_lift_manual']

        self.last_lift_auto_height = 0.0
        self.auto_lift_target_pid = LIFT_PID_MODE['abs']
        self.lift_auto_cmd = False
        self.lift_auto_ctrl = None
        self.forward_twist_cmd = None

        self.get_logger().info("Joy Controller Node initialized")

    def joy_callback(self, msg):
        axes = msg.axes
        buttons = msg.buttons

        # [ESTOP, MD_LIFT, UR] selection logic
        if axes[JOY_IDX['mode_selection']] < 0:
            self.current_mode_status = mode['estop']
            self.publish_lift_command_estop()
            self.publish_md200_estop()
            return
        elif axes[JOY_IDX['mode_selection']] == 0:
            self.current_mode_status = mode['md_lift_manual']
        # mode_selection > 0 : auto mode
        elif axes[JOY_IDX['mode_selection']] > 0 and axes[JOY_IDX['job_selection']] == 0:
            self.current_mode_status = mode['md_lift_auto']
            self.overwrite_mode_clear()
        elif axes[JOY_IDX['mode_selection']] > 0 and axes[JOY_IDX['job_selection']] < 0:
            self.current_mode_status = mode['md_harvesting']
        elif axes[JOY_IDX['mode_selection']] > 0 and axes[JOY_IDX['job_selection']] > 0:
            self.current_mode_status = mode['md_pollinating']

        # self.current_mode_status_pub.publish(Int32(data=self.current_mode_status))
        if self.current_mode_status == mode['md_lift_manual']:
            # gv, lift
            # self.get_logger().info(f"current_mode_status: {self.current_mode_status}")
            self.control_gv_manual(axes, buttons)
            self.control_lift_manual(axes, buttons)
            return
        elif self.current_mode_status == mode['md_lift_auto']:
            self.publish_func_ctrl_lift_mobile()
        elif self.current_mode_status == mode['md_harvesting']:
            self.overwrite_mode_harvesting()
            self.publish_func_ctrl_lift_mobile()
        elif self.current_mode_status == mode['md_pollinating']:
            self.overwrite_mode_pollinating()
            self.publish_func_ctrl_lift_mobile()
        else:
            raise NotImplementedError('impossible control flow reached')
    def publish_func_ctrl_lift_mobile(self,):
        if self.last_gv_auto_cmd_vel is not None:
            self.publish_md200(False, False, self.last_gv_auto_cmd_vel.twist.linear.x, self.last_gv_auto_cmd_vel.twist.angular.z)
        if self.lift_auto_ctrl is not None:
            self.publish_lift_height(self.lift_auto_ctrl)

    def overwrite_mode_clear(self,):
        self.current_mode_status_pub.publish(String(data='clear'))
    def overwrite_mode_harvesting(self,):
        self.current_mode_status_pub.publish(String(data='harvesting_generic'))
    def overwrite_mode_pollinating(self,):
        self.current_mode_status_pub.publish(String(data='pollination_generic'))

    def control_lift_manual(self, axes, buttons):
        axis_3_val = axes[JOY_IDX['lift_height']]
        # self.get_logger().info(f"axis_3_val: {axes[JOY_IDX['lift_height']]}")
        if abs(axis_3_val) > 1e-6:  # Lift control active
            current_height = self.last_lift_auto_height
            movement = 10 if axis_3_val < 0 else -10
            new_height = max(LIFT_MIN_HEIGHT, min(LIFT_MAX_HEIGHT, current_height + movement))

            self.get_logger().info(
                f"Current Height={current_height}, Movement={movement}, New Height={new_height}"
            )
            self.publish_lift_height(new_height)
    def control_gv_manual(self, axes, buttons):
        lin_x_joy_value = axes[JOY_IDX['gv_straight']]
        direction = 'forward' if lin_x_joy_value < 0 else 'backward'

        if direction == 'forward':
            lin_x_joy_value = min(lin_x_joy_value, LINEAR_X_JOY_AXIS_MAX)
            linear_x_ratio = lin_x_joy_value / LINEAR_X_JOY_AXIS_MAX
            linear_x_velocity = linear_x_ratio * LINEAR_MAX_X

            ang_z_joy_val = min(axes[JOY_IDX['gv_rotate']], ANGULAR_Z_JOY_AXIS_MAX)
            angular_z_ratio = ang_z_joy_val / ANGULAR_Z_JOY_AXIS_MAX
            angular_z_velocity = -angular_z_ratio * ANGULAR_MAX_Z
        else:  # backward
            lin_x_joy_value = max(lin_x_joy_value, LINEAR_X_JOY_AXIS_MIN)
            linear_x_ratio = lin_x_joy_value / LINEAR_X_JOY_AXIS_MIN
            linear_x_velocity = -linear_x_ratio * LINEAR_MAX_X

            ang_z_joy_val = max(axes[JOY_IDX['gv_rotate']], ANGULAR_Z_JOY_AXIS_MIN)
            angular_z_ratio = ang_z_joy_val / ANGULAR_Z_JOY_AXIS_MIN
            angular_z_velocity = angular_z_ratio * ANGULAR_MAX_Z

        self.publish_md200(False, False, linear_x_velocity, angular_z_velocity)

    def publish_md200(self, enable_estop, do_free_wheel, linear_x, angular_z):
        msg = MD200Control()
        msg.enable_estop = enable_estop
        msg.do_free_wheel = do_free_wheel
        msg.twist.linear.x = np.clip(linear_x, -LINEAR_MAX_X, LINEAR_MAX_X)
        msg.twist.angular.z = np.clip(angular_z, -ANGULAR_MAX_Z, ANGULAR_MAX_Z)
        # self.get_logger().info(f"linear_x: {msg.twist.linear.x}, angular_z: {msg.twist.angular.z}")
        self.cmd_vel_mobile_pub.publish(msg)


    def publish_md200_estop(self):
        self.get_logger().info('Emergency stop activated!')
        msg = MD200Control()
        msg.enable_estop = True
        msg.do_free_wheel = False
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_vel_mobile_pub.publish(msg)

    def publish_lift_height(self, height_mm):
        self.get_logger().info(f"[call publish_lift_height] {height_mm}")
        msg = LiftControl()
        msg.target_pid = LIFT_PID_MODE['abs']
        msg.height_mm = int(height_mm)
        self.get_logger().info(f"[LiftControl-server] {msg.height_mm}")
        self.lift_ctrl_pub.publish(msg)
        self.get_logger().info(f"[LiftControl-server] target_pid={msg.target_pid}, height_mm={msg.height_mm}")

    def publish_lift_command_estop(self):
        msg = LiftCommand()
        msg.target_pid = LIFT_PID_MODE['estop']
        msg.cmd_data = LIFT_CMD_DATA['estop']
        self.lift_cmd_pub.publish(msg)
        self.get_logger().warn("[E-STOP] Emergency Stop Command Published")

    def lift_status_callback(self, msg):
        self.last_lift_auto_height = msg.status_height_mm

    def lift_auto_cmd_cb(self, msg):
        self.lift_auto_cmd = msg.data
        if self.lift_auto_cmd:
            self.publish_lift_command_estop()
            return
        self.get_logger().info(f"[LiftCommand] {msg}")

    def lift_auto_ctrl_cb(self, msg):
        self.lift_auto_ctrl = msg.data * i_M_TO_MM
        self.get_logger().info(f"[LiftControl-listen auto ctrl] {msg} ,{self.lift_auto_ctrl}")

    def gv_auto_cmd_vel_cb(self, msg):
        self.last_gv_auto_cmd_vel = msg



if __name__ == '__main__':
  import time
  rclpy.init()
  action_list = []
  node = DX6EJoyControllerNode('control_server', action_list)
  node.start_ros_thread(async_spin=True)

  try:
      while rclpy.ok():
          time.sleep(0.1)
  except KeyboardInterrupt:
      pass

  rclpy.shutdown()
