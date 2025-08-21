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
from std_msgs.msg import String
# Global Variables
EPSILON = 0.0001

# Supports 3 control modes: ['Estop', 'Manual', 'Auto']
# Handle ['joy', 'lift', 'ground_vehicle]
# Todo - cloud mode, multi-robot mode?

# Button Modes
mode = {
    'estop': 0,
    'md_lift_manual': 1,
    'md_lift_auto': 2,
}

JOY_IDX = {
    'gv_rotate': 0,
    'gv_straight': 1,
    'nouse1': 2,
    'lift_height': 3,
    'mode_selection': 4,
    'nouse2': 5,
}
LIFT_PID_MODE = {
    'cmd': 10,
    'abs': 219,
    'rel': 220,
}

LIFT_CMD_DATA = {
    'estop': 4,
    'home' : 90
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

class DX6EJoyControllerNode(Node):
    def __init__(self):
        super().__init__('joy_controller_node')

        # gv publishers/subscribers
        self.current_mode_status_pub = self.create_publisher(Int32, '/control_mode', 1)


        # Lift publishers/subscribers
        self.lift_status_sub = self.create_subscription(LiftStatus, '/lift/out', self.lift_status_callback, 1)
        self.lift_cmd_pub = self.create_publisher(LiftCommand, '/lift/command', 1)
        self.lift_ctrl_pub = self.create_publisher(LiftControl, '/lift/in', 1)
        self.lift_auto_cmd_sub = self.create_subscription(String, '/mf_lift/command', self.lift_auto_cmd_cb, 1)
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
            return
        self.current_mode_status = mode['md_lift_manual']
  

        self.current_mode_status_pub.publish(Int32(data=self.current_mode_status))
        # gv, lift
        # self.get_logger().info(f"current_mode_status: {self.current_mode_status}")
        self.control_lift_manual(axes, buttons)


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
    def publish_lift_height(self, height_mm):
        self.get_logger().info(f"[call publish_lift_height] {height_mm}")
        msg = LiftControl()
        msg.target_pid = LIFT_PID_MODE['abs']
        msg.height_mm = int(height_mm)
        self.get_logger().info(f"[LiftControl-server] {msg.height_mm}")
        self.lift_ctrl_pub.publish(msg)
        self.get_logger().info(f"[LiftControl-server] target_pid={msg.target_pid}, height_mm={msg.height_mm}")
    
    def publish_lift_command_home(self):
        msg = LiftCommand()
        msg.target_pid = LIFT_PID_MODE['cmd']
        msg.cmd_data = LIFT_CMD_DATA['home']
        self.lift_cmd_pub.publish(msg)
        self.get_logger().warn("[ControlServer:Lift] Home Command Published")

    def publish_lift_command_estop(self):
        msg = LiftCommand()
        msg.target_pid = LIFT_PID_MODE['cmd']
        msg.cmd_data = LIFT_CMD_DATA['estop']
        self.lift_cmd_pub.publish(msg)
        self.get_logger().warn("[ControlServer:Lift] Emergency Stop Command Published")

    def lift_status_callback(self, msg):
        self.last_lift_auto_height = msg.status_height_mm

    def lift_auto_cmd_cb(self, msg):
        self.lift_auto_cmd = msg.data
        self.get_logger().info(f"[ControlServer:Lift-auto_cmd] lift_auto_cmd: {self.lift_auto_cmd}")
        if self.lift_auto_cmd == 'estop':
            self.publish_lift_command_estop()
            return
        elif self.lift_auto_cmd == 'home':
            self.publish_lift_command_home()
            return
        self.get_logger().info(f"[LiftCommand] {msg}")

    def lift_auto_ctrl_cb(self, msg):
        self.lift_auto_ctrl = msg.data * i_M_TO_MM
        self.get_logger().info(f"[ControlServer:Lift] from lift app node {msg} ,{self.lift_auto_ctrl}")



def main(args=None):
    rclpy.init(args=args)
    node = DX6EJoyControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
