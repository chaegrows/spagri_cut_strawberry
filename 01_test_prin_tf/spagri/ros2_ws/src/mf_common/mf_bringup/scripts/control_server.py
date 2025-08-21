#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
import threading
import sys
import time
import numpy as np
from typing import Optional
from geometry_msgs.msg import TwistStamped
from mf_msgs.msg import ManipulatorControl, ManipulatorStatus

from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mf_msgs.msg import LiftControl, LiftStatus, MD200Control

CTRL_MODE = {
  'ESTOP': 0,
  'MANUAL': 1,
  'AUTO': 2,
}

JOB_MODE = {
  'ESTOP': 0,
  'HARVESTING': 1,
  'POLLINATION': 2,
  'MONITORING': 3,
}

class ControlServerNode(BehaviorTreeServerNodeV2):
    repo = 'mf_common'
    node_name = 'control_server_node'
    def __init__(self, run_mode):
        super().__init__(run_mode)

        self.ctrl_mode = CTRL_MODE['ESTOP']
        self.job_mode  = JOB_MODE['ESTOP']

        self.set_param('publish_interval', 0.1)
        self.set_param('epsilon', 0.01)

        self.set_param('gv.linear_vel_max', 0.4)
        self.set_param('gv.angular_vel_max', 0.1)
        self.set_param('lift.linear_trans_min_mm', 0)
        self.set_param('lift.linear_trans_max_mm', 1400)

        self.set_param('joystick.stick_value_range', [-0.5, 0.5])
        self.set_param('joystick.joy_timeout', 1.0)

        # Basic keymap settings
        self.set_param('keymap.buttons', {
          'A1': 0,
          'B0': 1,
          'B2': 2,
          'C0': 3,
          'C2': 4,
          'D0': 5,
          'D2': 6,
          'F2': 7,
          'F0': 8,
          'G2': 9,
          'G0': 10,
          'H1': 11,
          'I':  12,
          'RESET': 13,
          'CANCEL': 14,
          'SELECT': 15,
          'L1_LEFT': 18,
          'L1_RIGHT': 19,
          'L2_DOWN': 20,
          'L2_UP': 21,
          'R1_LEFT': 22,
          'R1_RIGHT': 23,
          'R2_DOWN': 24,
          'R2_RIGHT': 25,
        })

        self.set_param('keymap.axes', {
            'LEFT_STICK_X': 0,   # Left stick X-axis
            'LEFT_STICK_Y': 1,   # Left stick Y-axis
            'RIGHT_STICK_X': 3,  # Right stick X-axis
            'RIGHT_STICK_Y': 4,  # Right stick Y-axis
            'KNOB': 7,           # Knob
        })

        # Button combination mapping - including axis inputs
        self.set_param('combo_function_map', {
            # Control server mode
            'H1': 'mode_estop',
            'D0+!H1': 'mode_auto',
            'D2+!H1': 'mode_manual',

            # # Manual operation
            'D2+LEFT_STICK_Y': 'move_gv_manual', # linear
            'D2+LEFT_STICK_X': 'move_gv_manual', # angular
            # 'L2+R2+LEFT_CLICK': 'move_gv_lane_home',

            'D2+RIGHT_STICK_Y': 'move_lift_manual', # linear
            'SELECT+RESET': 'recover_manipulator',

            # 'D_PAD_X+RIGHT_CLICK': 'move_lift_home',
            # 'L2+R2+D_PAD_X': 'move_manipulator_search_pose',
            # 'L2+R2+D_PAD_Y': 'move_manipulator_home_pose',

            # 'L2+R2+L1': 'toggle_eef_axis1',
            # 'L2+R2+R1': 'toggle_eef_axis2',
        })
        self._auto_combo = next((k for k, v in self.param['combo_function_map'].items() if v == 'mode_auto'), None)
        self._manual_combo = next((k for k, v in self.param['combo_function_map'].items() if v == 'mode_manual'), None)
        self._estop_combo = next((k for k, v in self.param['combo_function_map'].items() if v == 'mode_estop'), None)

        # Control server control mode & job mode publisher
        self.pub_ctrl_mode = self.create_publisher(String, '/control_server/mode', 1)
        self.pub_job_mode  = self.create_publisher(String, '/control_server/job', 1)

        # Manipulator publishers & subscribers & variables
        self.manipulator_status = ManipulatorStatus()
        self.add_async_subscriber(ManipulatorControl, '/mf_manipulator/command', self.manipulator_cmd_cb, 1) # Manipulator control command from app node
        self.pub_manipulator_ctrl = self.create_publisher(ManipulatorControl, '/manipulator/in', 1) # Manipulator control command to driver node

        # Ground vehicle publishers & subscribers & variables
        self.add_async_subscriber(TwistStamped, '/mf_gv/auto_gv_cmd_vel', self.gv_auto_cmd_vel_cb, 1) # GV auto command velocity from app node
        self.pub_gv_ctrl = self.create_publisher(MD200Control, '/md200/in', 1)

        # Lift publishers & subscribers & variables
        self.add_async_subscriber(LiftControl, '/mf_lift/command', self.lift_app_cmd_cb, 1) # Lift auto control command from app node
        self.pub_lift_ctrl = self.create_publisher(LiftControl, '/lift/in', 1)

        # Create joystick subscriber
        self.last_joy_timestamp = self.get_clock().now()
        self.joy_sub = self.add_async_subscriber(Joy, '/joy', self.joy_cb, 1)

        # Set to store currently pressed buttons
        self.current_buttons = set()
        # Set to store currently active axes
        self.active_axes = set()
        # Dictionary to store previous button states
        self.prev_buttons = {}
        # Set to store previously executed button combinations
        self.prev_button_combos = set()
        # Dictionary to store previous axis values
        self.prev_axes_values = {}
        # List of discrete axes like D-PAD
        self.discrete_axes = ['D_PAD_X', 'D_PAD_Y']

        self.add_async_timer(self.param['publish_interval'], self.publish_control_server_mode)
        self.mf_logger.info('Control Server Node initialized')
        self.gv_auto_cmd_vel_cb_init_free_wheel()


    def publish_control_server_mode(self):
        # self.mf_logger.info(f'[publish_control_server_mode] last_joy_timestamp: {(self.get_clock().now() - self.last_joy_timestamp).to_msg().sec}')
        if (self.get_clock().now() - self.last_joy_timestamp).to_msg().sec > self.param['joystick.joy_timeout']:
            if self.ctrl_mode != CTRL_MODE['AUTO']:
                self.mode_auto()
        else:
            if self._estop_combo in self.current_buttons:
              if self.ctrl_mode != CTRL_MODE['ESTOP']:
                return self.mode_estop()
            elif self._auto_combo in self.current_buttons and self.ctrl_mode != CTRL_MODE['AUTO']:
                return self.mode_auto()
            elif self._manual_combo in self.current_buttons and self.ctrl_mode != CTRL_MODE['MANUAL']:
                return self.mode_manual()

        self.pub_ctrl_mode.publish(String(data=f'{self.ctrl_mode}'))
        self.pub_job_mode.publish(String(data=f'{self.job_mode}'))
        # self.mf_logger.info(f'[publish_control_server_mode] ctrl_mode: {self.ctrl_mode}, job_mode: {self.job_mode}')


    def recover_manipulator(self):
      if self.ctrl_mode != CTRL_MODE['MANUAL']:
        return
      self.mf_logger.info('[recover_manipulator] Moving robot to recovery pose...')
      self.call_service('recover_manipulator', {})


    def gv_auto_cmd_vel_cb_init_free_wheel(self):
      msg_to_driver = MD200Control()
      msg_to_driver.enable_estop = False
      msg_to_driver.do_free_wheel = True
      self.pub_gv_ctrl.publish(msg_to_driver)


    def manipulator_cmd_cb(self, msg):
      self.pub_manipulator_ctrl.publish(msg)


    def gv_auto_cmd_vel_cb(self, msg):
      if self.ctrl_mode != CTRL_MODE['AUTO']:
        return
      msg_to_driver = MD200Control()
      msg_to_driver.enable_estop = False
      msg_to_driver.do_free_wheel = False
      msg_to_driver.twist.linear.x = msg.twist.linear.x
      msg_to_driver.twist.angular.z = msg.twist.angular.z
      self.pub_gv_ctrl.publish(msg_to_driver)


    def lift_app_cmd_cb(self, msg):
      if self.ctrl_mode != CTRL_MODE['AUTO']:
        return
      if msg.cmd_type == LiftControl.CMD_ESTOP:
        self.mode_estop()
      elif msg.cmd_type == LiftControl.CMD_MOVE_HOME:
        self.move_lift_home()
      else:
        self.pub_lift_ctrl.publish(msg)


    def check_auto(self):
      if self.prev_axes_values['D_PAD_X'] > self.param['joystick.stick_value_range'][1]:
        self.mode_estop()
        self.mode_auto()
      else:
        self.mode_estop()
        self.mode_manual()


    def check_estop(self):
      self.mf_logger.info(f"[check_estop] prev_axes_values: {self.prev_axes_values['D_PAD_Y']}")
      if self.prev_axes_values['D_PAD_Y'] < self.param['joystick.stick_value_range'][0]:
        self.mode_estop()
      elif self.prev_axes_values['D_PAD_X'] > self.param['joystick.stick_value_range'][1]:
        self.mode_estop()
        self.mode_auto()
      else:
        self.mode_estop()
        self.mode_manual()


    def mode_estop(self):
      self.ctrl_mode = CTRL_MODE['ESTOP']

      # Ground vehicle (ESTOP - enable: True, free_wheel: False, linear: 0.0, angular: 0.0)
      msg = MD200Control()
      msg.enable_estop = True
      msg.do_free_wheel = False
      msg.twist.linear.x = 0.0
      msg.twist.angular.z = 0.0
      self.pub_gv_ctrl.publish(msg)

      # Lift command (ESTOP - pid: 10, data: 4)
      msg = LiftControl()
      msg.header.stamp = self.get_clock().now().to_msg()
      msg.cmd_type = LiftControl.CMD_ESTOP
      self.pub_lift_ctrl.publish(msg)
      self.mf_logger.info(f'[mode_estop] Stopping all robot actions... (ctrl_mode: {self.ctrl_mode})')


    def mode_auto(self):
        self.ctrl_mode = CTRL_MODE['AUTO']
        self.mf_logger.info(f'[mode_auto] Entering automatic mode... (ctrl_mode: {self.ctrl_mode})')

    def mode_manual(self):
        self.ctrl_mode = CTRL_MODE['MANUAL']
        self.mf_logger.info(f'[mode_manual] Entering manual mode... (ctrl_mode: {self.ctrl_mode})')

    def mode_harvesting(self):
        self.job_mode = JOB_MODE['HARVESTING']
        self.mf_logger.info(f'[mode_harvesting] Entering harvesting mode... (job_mode: {self.job_mode})')

    def mode_pollination(self):
        self.job_mode = JOB_MODE['POLLINATION']
        self.mf_logger.info(f'[mode_pollination] Entering pollination mode... (job_mode: {self.job_mode})')

    def mode_monitoring(self):
        self.job_mode = JOB_MODE['MONITORING']
        self.mf_logger.info(f'[mode_monitoring] Entering monitoring mode... (job_mode: {self.job_mode})')

    def move_gv_manual(self):
        if not self.ctrl_mode == CTRL_MODE['MANUAL']:
            return
        linear = self.prev_axes_values['LEFT_STICK_Y']
        linear = linear if abs(linear) > self.param['epsilon'] * 10.0 else 0.0
        angular = self.prev_axes_values['LEFT_STICK_X']
        angular = angular if abs(angular) > self.param['epsilon'] * 10.0 else 0.0
        # Ground vehicle manual control
        msg = MD200Control()
        msg.enable_estop = False
        msg.do_free_wheel = False

        # 선형 속도 계산
        direction = 'forward' if linear < 0 else 'backward'
        if direction == 'forward':
            linear = min(linear, self.param['joystick.stick_value_range'][1])
            linear_ratio = linear / self.param['joystick.stick_value_range'][1]
            linear_velocity = linear_ratio * self.param['gv.linear_vel_max']
        else:
            linear = max(linear, self.param['joystick.stick_value_range'][0])
            linear_ratio = linear / self.param['joystick.stick_value_range'][0]
            linear_velocity = -linear_ratio * self.param['gv.linear_vel_max']

        # 각속도 계산
        if direction == 'forward':
            angular = min(angular, self.param['joystick.stick_value_range'][1])
            angular_ratio = angular / self.param['joystick.stick_value_range'][1]
            angular_velocity = -angular_ratio * self.param['gv.angular_vel_max']
        else:
            angular = max(angular, self.param['joystick.stick_value_range'][0])
            angular_ratio = angular / self.param['joystick.stick_value_range'][0]
            angular_velocity = angular_ratio * self.param['gv.angular_vel_max']

        # 속도 제한 및 메시지 발행
        msg.twist.linear.x = np.clip(linear_velocity, -self.param['gv.linear_vel_max'], self.param['gv.linear_vel_max'])
        msg.twist.angular.z = np.clip(angular_velocity, -self.param['gv.angular_vel_max'], self.param['gv.angular_vel_max'])
        self.pub_gv_ctrl.publish(msg)
        self.mf_logger.info(f'[move_gv_manual] linear: {linear}, angular: {angular}')

    def move_gv_lane_home(self):
        self.mf_logger.info('[move_gv_lane_home] Homming ground vehicle to lane home position...')

    def move_lift_manual(self):
        if not self.ctrl_mode == CTRL_MODE['MANUAL']:
            return
        position = self.prev_axes_values['RIGHT_STICK_Y']
        self.mf_logger.info(f"move_lift_manual {position}")

        if position > self.param['epsilon'] * 8.0:
          delta_z = -10
        elif position < -self.param['epsilon'] * 8.0:
          delta_z = 10
        else:
          delta_z = 0

        msg = LiftControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cmd_type = LiftControl.CMD_MOVE_REL
        msg.height_mm = delta_z
        self.pub_lift_ctrl.publish(msg)
        self.mf_logger.info(f'[move_lift_manual] position: {position}, {delta_z}')

    def move_lift_home(self):
        msg = LiftControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cmd_type = LiftControl.CMD_MOVE_HOME
        self.pub_lift_ctrl.publish(msg)
        self.mf_logger.info('[move_lift_home] Moving lift to zero position...')

    def move_manipulator_home_pose(self):
        self.mf_logger.info('[move_manipulator_home_pose] Moving manipulator to home pose...')

    def move_manipulator_search_pose(self):
        self.mf_logger.info('[move_manipulator_search_pose] Moving manipulator to search pose...')

    def toggle_eef_axis1(self):
        self.mf_logger.info('[toggle_eef_axis1] Toggling eef axis 1...')

    def toggle_eef_axis2(self):
        self.mf_logger.info('[toggle_eef_axis2] Toggling eef axis 2...')


    def joy_cb(self, msg):
        # Clear current button and axis states
        # self.mode_manual()
        self.last_joy_timestamp = self.get_clock().now()

        self.current_buttons.clear()
        self.active_axes.clear()

        # Process button inputs
        self.process_button_inputs(msg)

        # Process axis inputs
        self.process_axis_inputs(msg)

        # Check and process button and axis combinations
        self.check_combinations()


    def process_button_inputs(self, msg):
        # Detect and process button state changes
        for name, idx in self.param['keymap.buttons'].items():
            if idx < len(msg.buttons):
                # Initialize if previous state doesn't exist
                if name not in self.prev_buttons:
                    self.prev_buttons[name] = 0

                # Check if button is pressed
                if msg.buttons[idx] == 1:
                    # Add to current pressed buttons list
                    self.current_buttons.add(name)

                # Store current state
                self.prev_buttons[name] = msg.buttons[idx]


    def execute_combo_function(self, combo):
        # Execute function mapped to button combination
        if combo in self.param['combo_function_map']:
            function_name = self.param['combo_function_map'][combo]
            # self.mf_logger.info(f"Executing function '{function_name}' for combination '{combo}'")

            # TODO: Asynchronous action call
            function = getattr(self, function_name)
            function()
        else:
            self.mf_logger.warning(f"No function mapped to combination '{combo}'")


    def process_axis_inputs(self, msg):
        # Process axis inputs
        for name, idx in self.param['keymap.axes'].items():
            if idx < len(msg.axes):
                # Initialize if previous value doesn't exist
                if name not in self.prev_axes_values:
                    self.prev_axes_values[name] = 0.0

                current_value = msg.axes[idx]

                # For discrete axes (D-PAD), activate only when value changes
                if name in self.discrete_axes:
                    # Activate only if value changed and is non-zero
                    if abs(current_value) > self.param['epsilon'] and abs(self.prev_axes_values[name] - current_value) > self.param['epsilon']:
                        self.active_axes.add(name)
                        # self.mf_logger.info(f"Discrete axis {name} value changed: {current_value:.2f}")
                # For continuous axes, activate when above threshold
                elif abs(current_value) > self.param['epsilon']:
                    self.active_axes.add(name)
                    # self.mf_logger.info(f"Axis {name} activated: {current_value:.2f}")

                # Store current value
                self.prev_axes_values[name] = current_value


    def check_combinations(self):
        # Track currently active button combinations
        current_button_combos = set()

        # Check all possible combinations
        for combo, function in self.param['combo_function_map'].items():
            # Split combination elements (e.g., 'X+!LEFT_STICK_X' -> ['X', '!LEFT_STICK_X'])
            elements = combo.split('+')

            # Check if all elements in the combination are currently active
            all_active = True
            # Check if combination includes continuous axes
            has_continuous_axis = False

            for element in elements:
                # Check if element is a button or axis with NOT operator
                is_not = element.startswith('!')
                element_name = element[1:] if is_not else element

                if element_name in self.param['keymap.buttons'].keys():
                    # For buttons, check if in current_buttons
                    button_active = element_name in self.current_buttons
                    if is_not:
                        if button_active:  # NOT condition: button should NOT be active
                            all_active = False
                            break
                    else:
                        if not button_active:  # Normal condition: button should be active
                            all_active = False
                            break
                elif element_name in self.param['keymap.axes'].keys():
                    # For axes, check if in active_axes
                    axis_active = element_name in self.active_axes
                    if is_not:
                        if axis_active:  # NOT condition: axis should NOT be active
                            all_active = False
                            break
                    else:
                        if not axis_active:  # Normal condition: axis should be active
                            all_active = False
                            break
                    # Check if it's a continuous axis
                    if element_name not in self.discrete_axes:
                        has_continuous_axis = True

            # Execute function if all elements are active
            if all_active:
                if has_continuous_axis:
                    # Execute continuous axis combinations every frame
                    self.execute_combo_function(combo)
                else:
                    # Add button-only or discrete axis combinations to current list
                    current_button_combos.add(combo)

        # Execute only newly activated button-only or discrete axis combinations
        for combo in current_button_combos:
            if combo not in self.prev_button_combos:
                self.execute_combo_function(combo)

        # Update previous button combination state
        self.prev_button_combos = current_button_combos

    def cleanup(self):
        self.mf_logger.info('##################### control server stop ##########################')
        self.mode_estop()
        self.mf_logger.info('##################### control server stop ##########################')

if __name__ == '__main__':

    run_mode = 'standalone' if len(sys.argv) > 1 and sys.argv[1] == 'standalone' else 'server'  # 기본값
    rclpy.init()
    node = ControlServerNode(run_mode)

    try:
        if run_mode == 'server':
            node.start_ros_thread(async_spin=False)
        else:
            node.start_ros_thread(async_spin=True)
    except Exception as e:
        print(f"\n예외 발생: {e}")
    finally:
        print("프로그램 종료")
