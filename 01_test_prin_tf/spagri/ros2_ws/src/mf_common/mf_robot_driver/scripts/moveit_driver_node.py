#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import copy
import requests, json
import rclpy
import numpy as np
from threading import Lock
from control_msgs.msg import DynamicJointState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from mf_msgs.msg import BehaviorTreeStatus, ManipulatorControl, ManipulatorStatus

from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mf_msgs.msg import BehaviorTreeStatus, ManipulatorControl, ManipulatorStatus
from typing import Optional, Dict

from mflib.common.tf import ROS2TF, quaternion_to_euler, normalize_joint_angle, get_euler_zyz_from_quat
from mflib.common.tf import transform_mf_base_to_vendor_base_frame, transform_mf_eef_to_vendor_tool_frame, quaternion_inverse

from mflib.common.moveit2_interface import MoveIt2Interface, MoveIt2State
from mflib.common.moveit2_servo import MoveIt2Servo

from fairino_msgs.srv import RemoteCmdInterface # SEEDLING

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
EPSILON = 0.0001

CMD_MAP = {
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_ZERO: 'posj.left.normal.zero',
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_SAFE: 'posj.left.normal.safe',
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_HOME: 'posj.left.normal.home',
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_FLIPPED_1: 'posj.left.normal.flipped1',
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_FLIPPED_2: 'posj.left.normal.flipped2',
      ManipulatorControl.CMD_MOVEJ_LEFT_NORMAL_FLIPPED_3: 'posj.left.normal.flipped3',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_SAFE: 'posj.left.flipped.safe',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_HOME: 'posj.left.flipped.home',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_ZERO: 'posj.right.normal.zero',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_SAFE: 'posj.right.normal.safe',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_HOME: 'posj.right.normal.home',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_1: 'posj.right.normal.flipped1',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_2: 'posj.right.normal.flipped2',
      ManipulatorControl.CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_3: 'posj.right.normal.flipped3',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_SAFE: 'posj.right.flipped.safe',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_HOME: 'posj.right.flipped.home',
      ManipulatorControl.CMD_MOVEJ_TEST: 'posj.test',
    }

class RBRobotNode(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'robot_driver_node'

  def __init__(self, run_mode = 'server'):
    super().__init__(run_mode)

    # self.update_lock = Lock()
    # self.test_lock = Lock()
    self.joint_state_count = 0  # 동일한 조인트 상태가 반복된 횟수
    self.last_joint_efforts = None  # 마지막 조인트 상태

    self.declare_parameter('mf_base_frame', 'manipulator_base_link')
    self.declare_parameter('mf_eef_frame', 'endeffector')

    self.declare_parameter('mf_base_to_vendor_base.translation', [])
    self.declare_parameter('mf_base_to_vendor_base.rotation', [])

    self.declare_parameter('mf_eef_to_vendor_tool.translation', [])
    self.declare_parameter('mf_eef_to_vendor_tool.rotation', [])

    self.declare_parameter('status_publish_interval', 0.01)
    self.declare_parameter('time_sleep_for_moveit', 0.1)
    self.declare_parameter('max_cnt_joint_state_error', 20)

    # 로봇 방향 판별을 위한 파라미터
    self.declare_parameter('robot_direction.base_joint_index', 0)  # base joint index
    self.declare_parameter('robot_direction.elbow_joint_index', 2)  # elbow joint index
    self.declare_parameter('robot_direction.left_angle_min', 90.0)  # left angle min
    self.declare_parameter('robot_direction.left_angle_max', 270.0)  # left angle max
    self.declare_parameter('robot_direction.flipped_by_elbow_positive', True)  # True if flipped by elbow positive
    self.declare_parameter('robot_direction.left_by_angle_range', True)  # True if left by angle range

    self.declare_parameter('base_frame', 'link0') # Need for Moveit
    self.declare_parameter('tool_frame', 'link6') # Need nfor Moveit
    self.declare_parameter('move_group', 'manipulation')
    self.declare_parameter('follow_joint_trajectory_action_name', "joint_trajectory_controller/follow_joint_trajectory")
    self.declare_parameter('joint_names', ['base',
                                    'shoulder',
                                    'elbow',
                                    'wrist1',
                                    'wrist2',
                                    'wrist3'])
    self.declare_parameter('moveit2.planner_id', 'RRTConnectkConfigDefault') # RRTstarkConfigDefault
    self.declare_parameter('moveit2.cartesian_avoid_collisions', False)
    self.declare_parameter('moveit2.cartesian_jump_threshold', 0.0)
    self.declare_parameter('moveit2.max_cartesian_speed', 1.5)
    self.declare_parameter('moveit2.max_velocity', 1.5)
    self.declare_parameter('moveit2.max_acceleration', 0.8)
    self.declare_parameter('moveit2.max_try', 3)
    self.declare_parameter('moveit2.timeout.start_motion', 3.0)
    self.declare_parameter('moveit2.timeout.end_motion', 30.0)
    self.declare_parameter('speed_factor', 1.0)

    self.declare_parameter('posj.left.normal.zero', [-90.0, -180.0, 0.0, -90.0, 0.0, 0.0])
    self.declare_parameter('posj.left.normal.safe', [-90.0, -60.0, -120.0, -90.0, 90.0, -90.0])
    self.declare_parameter('posj.left.normal.home', [-90.0, -115.0, -125.0, -30.0, 90.0, -90.0])
    self.declare_parameter('posj.left.normal.flipped1', [-80.0, -180.0, 0.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.left.normal.flipped2', [-80.0, -180.0, 120.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.left.normal.flipped3', [-90.0, -120.0, 120.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.left.flipped.safe', [-90.0, -120.0, 120.0, -90.0, -90.0, 90.0])
    self.declare_parameter('posj.left.flipped.home', [-90.0, -65.0, 125.0, -150.0, -90.0, 90.0])
    self.declare_parameter('posj.right.normal.zero', [90.0, 0.0, 0.0, -90.0, 0.0, 0.0])
    self.declare_parameter('posj.right.normal.safe', [90.0, -120.0, 120.0, -90.0, -90.0, 90.0])
    self.declare_parameter('posj.right.normal.home', [90.0, -65.0, 125.0, -150.0, -90.0, 90.0])
    self.declare_parameter('posj.right.normal.flipped1', [80.0, 0.0, 0.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.right.normal.flipped3', [90.0, -60.0, -120.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.right.normal.flipped2', [80.0, 0.0, -120.0, -90.0, 0.0, -90.0])
    self.declare_parameter('posj.right.flipped.safe', [90.0, -60.0, -120.0, -90.0, 90.0, -90.0])
    self.declare_parameter('posj.right.flipped.home', [90.0, -115.0, -125.0, -30.0, 90.0, -90.0])

    self.declare_parameter('posj.left.normal.recovery', [-90.0, -90.0, -90.0, -90.0, 0.0, 0.0])
    self.declare_parameter('posj.left.flipped.recovery', [-90.0, -90.0, 90.0, 90.0, 0.0, 0.0])
    self.declare_parameter('posj.right.normal.recovery', [90.0, -90.0, 90.0, 90.0, 0.0, 0.0])
    self.declare_parameter('posj.right.flipped.recovery', [90.0, -90.0, -90.0, -90.0, 0.0, 0.0])

    self.declare_parameter('posj.test', [-28.8, -109.4, -106.0, -50.1, 89.7, -24.9])






    #@LCY joint state health check update
    self.declare_parameter('joint_state_health_check_interval', 5.0)
    self.declare_parameter('threshold_joint_state_error', 1.0)
    self.time_last_joint_state_health_check = time.time()

    self.robot_srv_client = self.create_client(
        RemoteCmdInterface,
        '/fairino_remote_command_service',
        callback_group=self._parallel_callback_group)

    self.rosTF = ROS2TF(self)
    self.moveit2 = MoveIt2Interface(
        node=self,
        joint_names=self.param['joint_names'],
        base_link_name=self.param['base_frame'],
        end_effector_name=self.param['tool_frame'],
        group_name=self.param['move_group'],
        follow_joint_trajectory_action_name=self.param['follow_joint_trajectory_action_name'],
        callback_group=self._parallel_callback_group,
    )
    self.moveit2.planner_id = self.param['moveit2.planner_id']
    self.moveit2.max_cartesian_speed = self.param['moveit2.max_cartesian_speed']
    self.moveit2.max_velocity = self.param['moveit2.max_velocity']
    self.moveit2.max_acceleration = self.param['moveit2.max_acceleration']
    self.moveit2.cartesian_avoid_collisions = self.param['moveit2.cartesian_avoid_collisions']
    self.moveit2.cartesian_jump_threshold = self.param['moveit2.cartesian_jump_threshold']

    self.servo = MoveIt2Servo(
        node=self,
        frame_id=self.param['tool_frame'],
        callback_group=self._parallel_callback_group,
    )

    self.status = ManipulatorStatus()
    self.status.current_status = ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE
    self.status.base_frame = self.param['base_frame']
    self.status.tool_frame = self.param['tool_frame']
    self.status.move_group = self.param['move_group']
    self.update_static_tf()


    self.pub_manipulator_status = self.create_publisher(
      ManipulatorStatus,
      '/manipulator/out',
      1,
    )

    self.add_sequential_subscriber(
      ManipulatorControl,
      '/manipulator/in',
      self.control_cb,
      1,
    )

    self.add_async_subscriber(
      JointState,
      '/joint_states',
      self.jointstate_cb,
      1,
    )

    #@LCY joint state health check update
    self.add_async_timer(
      self.param['joint_state_health_check_interval'],
      self.joint_state_health_check,
    )


    self.add_async_timer(
      self.param['status_publish_interval'],
      self.publish_status,
    )
    self.set_moveit2_params()
    self.mark_heartbeat(0)


  def update_static_tf(self):
    if len(self.param['mf_base_to_vendor_base.translation']) == 3 and \
       len(self.param['mf_base_to_vendor_base.rotation']) == 4:
      return
    if len(self.param['mf_eef_to_vendor_tool.translation']) == 3 and \
       len(self.param['mf_eef_to_vendor_tool.rotation']) == 4:
      return

    trans, rot = self.rosTF.getTF(self.param['mf_base_frame'], self.param['base_frame'])
    self.mf_logger.warn(f"trans, rot type: {type(trans)}, {type(rot)} / to_list type: {type(trans.tolist())}, {type(rot.tolist())}")
    self.set_param('mf_base_to_vendor_base.translation', trans.tolist())
    self.set_param('mf_base_to_vendor_base.rotation', rot.tolist())

    trans, rot = self.rosTF.getTF(self.param['mf_eef_frame'], self.param['tool_frame'])
    self.set_param('mf_eef_to_vendor_tool.translation', trans.tolist())
    self.set_param('mf_eef_to_vendor_tool.rotation', rot.tolist())


  def publish_status(self):
    if self.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
      return

    base_joint_index = self.param['robot_direction.base_joint_index']
    elbow_joint_index = self.param['robot_direction.elbow_joint_index']
    left_angle_min = self.param['robot_direction.left_angle_min']
    left_angle_max = self.param['robot_direction.left_angle_max']
    flipped_by_elbow_positive = self.param['robot_direction.flipped_by_elbow_positive']

    theta_deg, _ = normalize_joint_angle(self.status.current_joints[base_joint_index] * RAD2DEG)
    elbow_positive = self.status.current_joints[elbow_joint_index] > 0

    # 로봇 방향 판별 로직
    if flipped_by_elbow_positive:
      if elbow_positive:
        self.status.is_flipped = True if left_angle_min < theta_deg <= left_angle_max else False
        self.status.is_left = True if not self.status.is_flipped else False
      else:
        self.status.is_flipped = False if left_angle_min <= theta_deg < left_angle_max else True
        self.status.is_left = False if not self.status.is_flipped else True
    else:
      if elbow_positive:
        self.status.is_flipped = False if left_angle_min <= theta_deg < left_angle_max else True
        self.status.is_left = True if not self.status.is_flipped else False
      else:
        self.status.is_flipped = True if left_angle_min < theta_deg <= left_angle_max else False
        self.status.is_left = False if not self.status.is_flipped else True

    self.status.is_left = self.status.is_left if self.param['robot_direction.left_by_angle_range'] else not self.status.is_left

    # self.mf_logger.info(f"[Param] flipped_by_elbow_positive: {flipped_by_elbow_positive}, left_angle_min: {left_angle_min}, left_angle_max: {left_angle_max}")
    # self.mf_logger.info(f"[Status] theta_deg: {theta_deg}, elbow_positive: {elbow_positive}")
    # self.mf_logger.info(f"[Result] is_flipped: {self.status.is_flipped}, is_left: {self.status.is_left}")

    position, orientation = self.rosTF.getTF(self.param['base_frame'], self.param['tool_frame'])

    if len(position) == 3 and len(orientation) == 4:
      self.status.current_pose.position.x = position[0]
      self.status.current_pose.position.y = position[1]
      self.status.current_pose.position.z = position[2]
      self.status.current_pose.orientation.x = orientation[0]
      self.status.current_pose.orientation.y = orientation[1]
      self.status.current_pose.orientation.z = orientation[2]
      self.status.current_pose.orientation.w = orientation[3]
      if len(self._marked_heartbeats) == 0 :
        self.pub_manipulator_status.publish(self.status)


  def control_cb(self, msg):
    if  msg.seq <= self.status.seq:
      return
    dir1 = 'left' if self.status.is_left else 'right'
    dir2 = 'flipped' if self.status.is_flipped else 'normal'
    self.mf_logger.info(f"[{dir1}.{dir2}] control_cb: {msg.cmd_type}")

    cmd_type = msg.cmd_type
    self.set_param('speed_factor', msg.speed_factor)

    if cmd_type == ManipulatorControl.CMD_MOVEJ:
      self.move_joints(msg.target_joints)

    elif cmd_type == ManipulatorControl.CMD_MOVEL_FROM_BASE:
      self.move_linear(msg.target_pose, from_base=True)

    elif cmd_type == ManipulatorControl.CMD_MOVEL_FROM_EEF:
      self.move_linear(msg.target_pose, from_base=False)

    elif cmd_type == ManipulatorControl.CMD_MOVE_RECOVERY:
      self.set_param('speed_factor', 0.1)
      self.recover_manipulator()

    elif cmd_type == ManipulatorControl.CMD_SET_DIGITAL_IO:
      cmd_io = json.loads(msg.digital_io)
      self.set_digital_io(cmd_io)

    elif cmd_type >= ManipulatorControl.CMD_MOVEJ_ID_START and cmd_type <= ManipulatorControl.CMD_MOVEJ_ID_END:
      joints_deg = self.param[CMD_MAP[cmd_type]]
      joints_rad = [joint * DEG2RAD for joint in joints_deg]
      self.move_joints(joints_rad)

    self.mf_logger.info(f"[control_cb] command executed!!! \n{msg}")
    # with self.update_lock:
    self.status.seq = msg.seq


  def set_digital_io(self, cmd_io, timeout_sec=5.0):
    pin_number, status = list(cmd_io.items())[0]
    request = RemoteCmdInterface.Request()

    request.cmd_str = f'SetToolDO({pin_number},{int(status)})'

    future = self.robot_srv_client.call_async(request)

    # Wait for the future to complete
    start_time = time.time()
    while not future.done():
      time.sleep(0.01)
      if time.time() - start_time > timeout_sec:
        response = RemoteCmdInterface.Response()
        return response

    response = future.result()
    self.mf_logger.info(f"Digital IO Set ({cmd_io})")
    return response


  #@LCY joint state health check update
  def joint_state_health_check(self):
    if self.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
      return
    if time.time() - self.time_last_joint_state_health_check > self.param['threshold_joint_state_error']:
      self.mf_logger.info(f"[Manipulator] Joint state health check failed")
      self.mark_heartbeat(1)
    else:
      self.mf_logger.info(f"[Manipulator] Joint state health check passed")

  def jointstate_cb(self, msg):
    if self.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
      self.status.current_status = ManipulatorStatus.ROBOT_STATUS_IDLE
    current_joints = []
    current_effort = []
    for joint_name in self.param['joint_names']:
      idx = msg.name.index(joint_name)
      current_joints.append(msg.position[idx])
      current_effort.append(msg.effort[idx])

    # 로봇은 살아있지만 effort가 0이면 조인트 상태가 이상함
    if self.last_joint_efforts is not None and current_effort == self.last_joint_efforts:
      self.joint_state_count += 1
      if self.joint_state_count >= self.param['max_cnt_joint_state_error']:
        self.mark_heartbeat(1)
    else:
      self.mark_heartbeat(0)
      # with self.update_lock:
      self.status.header = msg.header
      self.status.current_joints = current_joints
      self.last_joint_efforts = current_effort
      self.joint_state_count = 1

    #@LCY joint state health check update
    self.time_last_joint_state_health_check = time.time()


  def servo_local_path(self, msg):
    twist_linear = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
    twist_angular = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
    self.servo.frame_id = msg.header.frame_id
    self.servo(linear=tuple(twist_linear), angular=(twist_angular))
    self.servo.frame_id = self.param['tool_frame']


  def set_moveit2_params(self):
    # with self.update_lock:
    self.moveit2.max_velocity = self.param['speed_factor'] * self.param['moveit2.max_velocity']
    self.moveit2.max_acceleration = self.param['speed_factor'] * self.param['moveit2.max_acceleration']
    self.moveit2.max_cartesian_speed = self.param['speed_factor'] * self.param['moveit2.max_cartesian_speed']


  def move_joints(self, joints: list):
    joints = np.round(joints, decimals=4)

    self.set_moveit2_params()
    self.mf_logger.info(f"[move_joints] Start move_joints (joints: {joints})")
    n_try = 0
    t_start = time.time()
    success = False
    while not success and n_try < self.param['moveit2.max_try']:
      self.mf_logger.info(f"[move_joints] Start move_joints (n_try: {n_try}) (joints: {joints})")
      self.moveit2.move_to_configuration(joints)
      while self.moveit2.query_state() != MoveIt2State.EXECUTING and time.time() - t_start < self.param['moveit2.timeout.start_motion']:
        self.mf_logger.info(f"[move_joints] query_state(): {self.moveit2.query_state()}")
        time.sleep(0.1)
      future = self.moveit2.get_execution_future()
      if future is None:
        n_try += 1
        continue
      while not future.done():
        time.sleep(self.param['time_sleep_for_moveit'])
        if time.time() - t_start > self.param['moveit2.timeout.end_motion']:
          self.moveit2.cancel_execution()
          self.mf_logger.error(f"[move_joints] Motion failed after {time.time() - t_start} seconds")
          break
        self.mf_logger.info(f"[move_joints] future.done(): {future.done()}, future.status(): {future}")
        success = future.done()
      # success = self.moveit2.wait_until_executed()
      n_try += 1
      self.mf_logger.info(f"[move_joints] End move_joints (try: {n_try}, success: {success})")


  @BehaviorTreeServerNodeV2.available_service()
  def recover_manipulator(self, args_json: Optional[dict] = None):
    self.set_param('speed_factor', 0.1)
    # with self.update_lock:
    current_joints = copy.deepcopy(self.status.current_joints)
    is_left = self.status.is_left
    is_flipped = self.status.is_flipped

    if is_left and is_flipped:
      target_joints_deg = self.param['posj.left.flipped.recovery']
    elif is_left and not is_flipped:
      target_joints_deg = self.param['posj.left.normal.recovery']
    elif not is_left and is_flipped:
      target_joints_deg = self.param['posj.right.flipped.recovery']
    elif not is_left and not is_flipped:
      target_joints_deg = self.param['posj.right.normal.recovery']
    else:
      self.mf_logger.error(f"Invalid robot direction: {is_left}, {is_flipped}")
      raise ValueError(f"Invalid robot direction: {is_left}, {is_flipped}")

    target_joints = [joint * DEG2RAD for joint in target_joints_deg]
    cmd_joints = current_joints
    self.mf_logger.info(f"[recover_manipulator] is_left: {is_left} / is_flipped: {is_flipped}")
    self.mf_logger.info(f"[recover_manipulator] current_joints: {current_joints} / target_joints: {target_joints}")

    for i in [1, 0, 2]:
      cmd_joints[i] = target_joints[i]
      self.mf_logger.info(f"[recover_manipulator] cmd_joints[{i}]: {cmd_joints[i]}")
      self.move_joints(cmd_joints)
      time.sleep(0.5)
    self.move_joints(target_joints)


  def move_linear(self, pose: Pose, from_base=True):
    self.set_moveit2_params()
    self.update_static_tf()
    if not len(self.param['mf_eef_to_vendor_tool.rotation']) == 4 or \
        not len(self.param['mf_base_to_vendor_base.rotation']) == 4:
        self.mf_logger.error("Static TF from mf_base to robot_base is not set. Check assemble_robot.py")
        raise ValueError("Static TF from mf_base to robot_base is not set. Check assemble_robot.py")

    input_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    input_rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    if from_base: # Move from base
      ref_frame_mf = self.param['mf_base_frame']
      ref_frame_vendor = self.param['base_frame']

      rotated_pos, rotated_dir = transform_mf_base_to_vendor_base_frame(
        self.param['mf_base_to_vendor_base.rotation'],
        self.param['mf_eef_to_vendor_tool.rotation'],
        self.param['mf_eef_to_vendor_tool.translation'],
        input_pos,
        input_rot,
    )
    else: # Move from eef
      ref_frame_mf = self.param['mf_eef_frame']
      ref_frame_vendor = self.param['tool_frame']

      rotated_pos, rotated_dir = transform_mf_eef_to_vendor_tool_frame(
        self.param['mf_eef_to_vendor_tool.rotation'],
        input_pos,
        input_rot,
      )

    transformed_pose = Pose()
    transformed_pose.position.x = rotated_pos[0]
    transformed_pose.position.y = rotated_pos[1]
    transformed_pose.position.z = rotated_pos[2]
    transformed_pose.orientation.x = rotated_dir[0]
    transformed_pose.orientation.y = rotated_dir[1]
    transformed_pose.orientation.z = rotated_dir[2]
    transformed_pose.orientation.w = rotated_dir[3]

    self.rosTF.publishTF(
        ref_frame_mf, 'input_pose_mf_frame',
        input_pos, input_rot
    )

    self.rosTF.publishTF(
        ref_frame_vendor, 'target_pose_vendor_base',
        rotated_pos, rotated_dir
    )

    self.mf_logger.info(f"[move_linear] Start move_to_pose (input_pose: {input_pos}, input_rot: {input_rot})")
    n_try = 0
    t_start = time.time()
    success = False
    while not success and n_try < self.param['moveit2.max_try']:
      self.moveit2.move_to_pose(pose=transformed_pose, cartesian=True, frame_id=ref_frame_vendor)
      while self.moveit2.query_state() != MoveIt2State.EXECUTING and time.time() - t_start < self.param['moveit2.timeout.start_motion']:
        self.mf_logger.info(f"[move_linear] Moveit state: {self.moveit2.query_state()}")
        time.sleep(0.1)

      n_try += 1
      if self.moveit2.query_state() != MoveIt2State.EXECUTING:
        self.mf_logger.warning(f"[move_linear] Moveit state: {self.moveit2.query_state()}")
        continue

      future = self.moveit2.get_execution_future()
      if future is None:
        continue
      while not future.done():
        time.sleep(self.param['time_sleep_for_moveit'])
        if time.time() - t_start > self.param['moveit2.timeout.end_motion']:
          self.moveit2.cancel_execution()
          self.mf_logger.error(f"[move_to_pose] Motion failed after {time.time() - t_start} seconds")
          break
        self.mf_logger.info(f"[move_to_pose] future.done(): {future.done()}, future.status(): {future}")
        success = future.done()
      # success = self.moveit2.wait_until_executed()
      n_try += 1
      self.mf_logger.info(f"[move_to_pose] End move_joints (try: {n_try}, success: {success})")



if __name__ == '__main__':
  rclpy.init()
  node = RBRobotNode()
  node.start_ros_thread(async_spin=False)
