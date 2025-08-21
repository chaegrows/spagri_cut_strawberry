#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import math, sys
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import numpy as np
import copy
from typing import Optional
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mf_msgs.msg import BehaviorTreeStatus, ManipulatorControl, ManipulatorStatus
import rclpy
from mflib.common.tf import ROS2TF, quaternion_to_euler, quaternion_inverse, normalize_joint_angle, get_euler_zyz_from_quat
from mflib.common.tf import transform_mf_base_to_vendor_base_frame, transform_mf_eef_to_vendor_tool_frame

sys.path.append('/root/bringup_ws/src/doosan-robot2/common2/imp')
from DR_common2 import *
import DR_init
ROBOT_ID   = ""
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = "a0509"
#
from dsr_msgs2.msg import ServojStream, ServolStream, RobotState


DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
METER2MM = 1000.0
MM2METER = 1.0 / METER2MM
EPSILON = 0.0001

CMD_MAP = {
      ManipulatorControl.CMD_MOVEJ_LEFT_ZERO: 'posj.left.zero',
      ManipulatorControl.CMD_MOVEJ_LEFT_SAFE: 'posj.left.safe',
      ManipulatorControl.CMD_MOVEJ_LEFT_HOME: 'posj.left.home',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_1: 'posj.left.flipped1',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_2: 'posj.left.flipped2',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_3: 'posj.left.flipped3',
      ManipulatorControl.CMD_MOVEJ_LEFT_FLIPPED_HOME: 'posj.left.flipped.home',
      ManipulatorControl.CMD_MOVEJ_RIGHT_ZERO: 'posj.right.zero',
      ManipulatorControl.CMD_MOVEJ_RIGHT_SAFE: 'posj.right.safe',
      ManipulatorControl.CMD_MOVEJ_RIGHT_HOME: 'posj.right.home',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_1: 'posj.right.flipped1',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_2: 'posj.right.flipped2',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_3: 'posj.right.flipped3',
      ManipulatorControl.CMD_MOVEJ_RIGHT_FLIPPED_HOME: 'posj.right.flipped.home',
    }


rclpy.init()
node__ = rclpy.create_node('dsr_robot_driver', namespace=ROBOT_ID)

DR_init.__dsr__node = node__

from DSR_ROBOT2 import set_velj, set_accj, posx, posj, mwait, print_ext_result, movej, movel, movec, move_periodic, move_spiral, set_velx, set_accx, DR_BASE, DR_TOOL, DR_WORLD, DR_AXIS_X
from DSR_ROBOT2 import DR_MV_MOD_REL, DR_MV_MOD_ABS, set_tool, set_tcp

class DSRRobotNode(BehaviorTreeServerNodeV2):

  repo = 'mf_common'
  node_name = 'robot_driver_node'

  def __init__(self, run_mode = 'server'):
    super().__init__(run_mode)

    self.declare_parameter('mf_base_frame', 'manipulator_base_link')
    self.declare_parameter('mf_eef_frame', 'endeffector')

    self.declare_parameter('mf_base_to_vendor_base.translation', [])
    self.declare_parameter('mf_base_to_vendor_base.rotation', [])

    self.declare_parameter('mf_eef_to_vendor_tool.translation', [])
    self.declare_parameter('mf_eef_to_vendor_tool.rotation', [])

    self.declare_parameter('status_publish_interval', 0.01)

    # Manipulator driver parameters for ROS2 related stuffs (e.g., Moveit2, TF)
    self.declare_parameter('base_frame', 'base_link')
    self.declare_parameter('tool_frame', 'link_6')
    self.declare_parameter('move_group', 'manipulator')
    self.declare_parameter('joint_names', ['joint_1',
                                          'joint_2',
                                          'joint_3',
                                          'joint_4',
                                          'joint_5',
                                          'joint_6'])

    # Robot vendor specific parameters
    self.declare_parameter('dsr.speed.linear', 300)       # TCP linear speed (mm/s)
    self.declare_parameter('dsr.speed.angular', 100)      # TCP angular speed (deg/s)
    self.declare_parameter('dsr.speed.joint', 80)         # Joint angular speed (deg/s)
    self.declare_parameter('dsr.acceleration.linear', 150)   # TCP linear acceleration (mm/s^2)
    self.declare_parameter('dsr.acceleration.angular', 50)   # TCP angular acceleration (deg/s^2)
    self.declare_parameter('dsr.acceleration.joint', 40)     # Joint acceleration (deg/s^2)
    self.declare_parameter('dsr.acceleration.joint_max', 100)     # Joint acceleration (deg/s^2)

    self.declare_parameter('posj.left.zero', [180.0, 90.0, 0.0, 90.0, 90.0, 0.0])
    self.declare_parameter('posj.left.safe', [180.0, -30.0, 120.0, 90.0, 90.0, 0.0])
    self.declare_parameter('posj.left.home', [180.0, 0.0, 120.0, 0.0, 60.0, 0.0])
    self.declare_parameter('posj.left.flipped1', [170.0, 90.0, 0.0, 90.0, 0.0, 0.0])
    self.declare_parameter('posj.left.flipped2', [170.0, 90.0, -120.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.left.flipped3', [180.0, 30.0, -120.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.left.flipped.home', [180.0, 0.0, -120.0, 0.0, -60.0, 0.0])
    self.declare_parameter('posj.right.zero', [0.0, -90.0, 0.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.right.safe', [0.0, 30.0, -120.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.right.home', [0.0, 0.0, -120.0, 0.0, -60.0, 0.0])
    self.declare_parameter('posj.right.flipped1', [10.0, -90.0, 0.0, -90.0, 0.0, 0.0])
    self.declare_parameter('posj.right.flipped2', [10.0, -90.0, 120.0, 90.0, 90.0, 0.0])
    self.declare_parameter('posj.right.flipped3', [0.0, -30.0, 120.0, 90.0, 90.0, 0.0])
    self.declare_parameter('posj.right.flipped.home', [0.0, 0.0, 120.0, 0.0, 60.0, 0.0])

    self.declare_parameter('posj.left.recovery', [180.0, 0.0, 90.0, 90.0, 90.0, 0.0])
    self.declare_parameter('posj.left.flipped.recovery', [180.0, 0.0, -90.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.right.recovery', [0.0, 0.0, -90.0, -90.0, -90.0, 0.0])
    self.declare_parameter('posj.right.flipped.recovery', [0.0, 0.0, 90.0, 90.0, 90.0, 0.0])

    self.rosTF = ROS2TF(self)

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
      1
    )

    self.add_async_timer(
      self.param['status_publish_interval'],
      self.publish_status,
    )

    self.set_dsr_motion_params()
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

    self.mf_logger.warn(f"mf_base_to_vendor_base.translation: {self.param['mf_base_to_vendor_base.translation']}")
    self.mf_logger.warn(f"mf_base_to_vendor_base.rotation: {self.param['mf_base_to_vendor_base.rotation']}")
    self.mf_logger.warn(f"mf_eef_to_vendor_tool.translation: {self.param['mf_eef_to_vendor_tool.translation']}")
    self.mf_logger.warn(f"mf_eef_to_vendor_tool.rotation: {self.param['mf_eef_to_vendor_tool.rotation']}")


  def publish_status(self):
    if self.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
      return

    theta0_deg, _ = normalize_joint_angle(self.status.current_joints[0] * RAD2DEG)
    if self.status.current_joints[2] < 0:
      self.status.is_flipped = True if 90.0 < theta0_deg <= 270.0 else False
      self.status.is_left = True if self.status.is_flipped else False
    else:
      self.status.is_flipped = False if 90.0 <= theta0_deg < 270.0 else True
      self.status.is_left = False if self.status.is_flipped else True

    position, orientation = self.rosTF.getTF(self.param['base_frame'], self.param['tool_frame'])

    if len(position) == 3 and len(orientation) == 4:
      self.status.current_pose.position.x = position[0]
      self.status.current_pose.position.y = position[1]
      self.status.current_pose.position.z = position[2]
      self.status.current_pose.orientation.x = orientation[0]
      self.status.current_pose.orientation.y = orientation[1]
      self.status.current_pose.orientation.z = orientation[2]
      self.status.current_pose.orientation.w = orientation[3]
      self.pub_manipulator_status.publish(self.status)


  def control_cb(self, msg):
    if  msg.seq <= self.status.seq:
      return

    cmd_type = msg.cmd_type

    if cmd_type == ManipulatorControl.CMD_MOVEJ:
      self.move_joints(msg.target_joints)

    elif cmd_type == ManipulatorControl.CMD_MOVEL_FROM_BASE:
      self.move_linear_from_base(msg.target_pose)

    elif cmd_type == ManipulatorControl.CMD_MOVEL_FROM_EEF:
      self.move_linear_from_tool(msg.target_pose)

    elif cmd_type == ManipulatorControl.CMD_MOVE_RECOVERY:
      self.recover_manipulator()

    elif cmd_type >= ManipulatorControl.CMD_MOVEJ_ID_START and cmd_type <= ManipulatorControl.CMD_MOVEJ_ID_END:
      joints_deg = self.param[CMD_MAP[cmd_type]]
      joints_rad = [joint * DEG2RAD for joint in joints_deg]
      self.move_joints(joints_rad)

    self.status.seq = msg.seq


  def set_dsr_motion_params(self):
    set_velx(self.param['dsr.speed.linear'], self.param['dsr.speed.angular'])
    set_velj(self.param['dsr.speed.joint'])
    set_accx(self.param['dsr.acceleration.linear'], self.param['dsr.acceleration.angular'])
    set_accj(self.param['dsr.acceleration.joint'])
    self.get_logger().info(f'DSR motion parameters set ({self.param["dsr.speed.linear"]}, {self.param["dsr.speed.angular"]}, {self.param["dsr.speed.joint"]}, {self.param["dsr.acceleration.linear"]}, {self.param["dsr.acceleration.angular"]}, {self.param["dsr.acceleration.joint"]})')


  @BehaviorTreeServerNodeV2.available_service()
  def recover_manipulator(self, args_json: Optional[dict] = None):
    current_joints = copy.deepcopy(self.status.current_joints)

    if self.status.is_left and self.status.is_flipped:
      target_joints_deg = self.param['posj.left.flipped.recovery']
    elif self.status.is_left and not self.status.is_flipped:
      target_joints_deg = self.param['posj.left.recovery']
    elif not self.status.is_left and self.status.is_flipped:
      target_joints_deg = self.param['posj.right.flipped.recovery']
    elif not self.status.is_left and not self.status.is_flipped:
      target_joints_deg = self.param['posj.right.recovery']
    else:
      self.mf_logger.error(f"Invalid robot direction: {self.status.is_left}, {self.status.is_flipped}")
      raise ValueError(f"Invalid robot direction: {self.status.is_left}, {self.status.is_flipped}")

    target_joints = [joint * DEG2RAD for joint in target_joints_deg]
    cmd_joints = current_joints
    for i in range(len(current_joints)):
      cmd_joints[i] = target_joints[i]
      self.move_joints(cmd_joints)


  def move_joints(self, joints: list):
    pos_list = posj(*[q * RAD2DEG for q in joints])
    print("movej", pos_list)
    movej(pos_list)
    mwait(0)


  def convert_pose_to_dsr_posx(self, pose: Pose, from_base=True):
    self.update_static_tf()
    if not len(self.param['mf_eef_to_vendor_tool.rotation']) == 4 or \
        not len(self.param['mf_base_to_vendor_base.rotation']) == 4:
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
        quaternion_inverse(self.param['mf_eef_to_vendor_tool.rotation']),
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
    quat_xyzw = [
        transformed_pose.orientation.x,
        transformed_pose.orientation.y,
        transformed_pose.orientation.z,
        transformed_pose.orientation.w,
    ]
    z1_deg, y1_deg, z2_deg = get_euler_zyz_from_quat(quat_xyzw, to_deg=True)
    dsr_posx = posx(
        transformed_pose.position.x * METER2MM,
        transformed_pose.position.y * METER2MM,
        transformed_pose.position.z * METER2MM,
        z1_deg,
        y1_deg,
        z2_deg,
    )
    self.mf_logger.info(f'dsr_posx: {dsr_posx}')
    return dsr_posx

  def move_linear_from_base(self, pose: Pose):
    target_posx = self.convert_pose_to_dsr_posx(pose, from_base=True)
    self.mf_logger.info(f'move_linear_from_base: {target_posx}')
    movel(target_posx, ref=DR_BASE, mod = DR_MV_MOD_ABS)
    mwait(0)


  def move_linear_from_tool(self, pose: Pose):
    target_posx = self.convert_pose_to_dsr_posx(pose, from_base=False)
    self.mf_logger.info(f'move_linear_from_tool: {target_posx}')
    movel(target_posx, ref=DR_TOOL, mod = DR_MV_MOD_REL)
    mwait(0)


  def jointstate_cb(self, msg):
    if self.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
      self.status.current_status = ManipulatorStatus.ROBOT_STATUS_IDLE
    self.status.header = msg.header
    current_joints = []
    for joint_name in self.param['joint_names']:
      idx = msg.name.index(joint_name)
      current_joints.append(msg.position[idx])
    self.status.current_joints = current_joints



if __name__ == "__main__":
  # rclpy.init()
  node = DSRRobotNode()
  node.start_ros_thread(async_spin=False)

  # pose = Pose()
  # pose.position.x = 0.0
  # pose.position.y = 0.0
  # pose.position.z = 0.0
  # pose.orientation.x = 0.0
  # pose.orientation.y = 0.0
  # pose.orientation.z = 0.0
  # pose.orientation.w = 1.0

  # while True:
  #   time.sleep(0.01)
  #   if node.status.current_status == ManipulatorStatus.ROBOT_STATUS_NOT_AVAILABLE:
  #     continue

  #   movel(posx(0,100,0,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
  #   mwait(0)
  #   movel(posx(0,-100,0,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
  #   mwait(0)

  #   pose.position.z = 0.1
  #   node.move_linear_from_tool(pose)
  #   pose.position.z = -0.1
  #   node.move_linear_from_tool(pose)

  # movel(posx(-509.000, 202.0, 64.0, -180.0, 180.0, 180.0), ref=DR_BASE, mod = DR_MV_MOD_ABS)
  # movel(posx(-609.000, 102.0, 64.0, -180.0, 180.0, 180.0), ref=DR_BASE, mod = DR_MV_MOD_ABS)
  # mwait(0)



    # movel(posx(0,-100,0,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
    # mwait(0)
    # movel(posx(100,0,0,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
    # mwait(0)
    # movel(posx(-100,0,0,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
    # mwait(0)
    # movel(posx(0,0,100,0,0,0), ref=DR_TOOL, mod=DR_MV_MOD_REL)
