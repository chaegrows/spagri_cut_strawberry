#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg

from squaternion import Quaternion


RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


class ROS2TF():
    def __init__(self,
                 node: Node,
                 verbose=False):

        self._node = node
        self.verbose = verbose

        # TF buffer & listener
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self._node)

        # TF broadcaster (static & dynamic)
        self.broadcaster_static  = tf2_ros.StaticTransformBroadcaster(self._node)
        self.broadcaster_dynamic = tf2_ros.TransformBroadcaster(self._node)


    def getTF(self, reference_frame, target_frame, timeout=1.0):
        """
        Transform을 가져오는 함수
        Args:
            reference_frame: 기준 프레임
            target_frame: 대상 프레임
            timeout: 최대 대기 시간 (초)
        Returns:
            (translation, rotation) tuple 또는 빈 리스트들
        """
        trans = []
        rot = []
        try:
            # timeout 적용하여 transform 확인
            if self.tf_buffer.can_transform(
                reference_frame,
                target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout)
            ):
                transform = self.tf_buffer.lookup_transform(
                    reference_frame,
                    target_frame,
                    rclpy.time.Time()
                )
                trans = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ]
                rot = [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ]
                if self.verbose:
                    self._node.get_logger().debug(
                        f"Transform from {reference_frame} to {target_frame}:"
                        f"\ntrans: {trans}\nrot: {rot}"
                    )
            else:
                if self.verbose:
                    self._node.get_logger().debug(
                        f"Transform from {reference_frame} to {target_frame} not available within timeout"
                    )
        except Exception as e:
            if self.verbose:
                self._node.get_logger().warning(
                    f"Error getting transform from {reference_frame} to {target_frame}: {str(e)}"
                )

        return (np.asarray(trans), np.asarray(rot))


    def publishTF(self, reference_frame, target_frame, translation, rotation=None, static_tf=False):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = reference_frame
        t.child_frame_id = target_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])

        if rotation is None:
            quaternion = [0.0, 0.0, 0.0, 1.0] # Default identity quaternion
        elif len(rotation) == 3:
            quaternion = euler_to_quaternion(
              rx_deg=rotation[0],
              ry_deg=rotation[1],
              rz_deg=rotation[2],
            )

            # rotation[0], rotation[2] = rotation[2], rotation[0]
            # quat = transforms3d.euler.euler2quat(*rotation)
            # quaternion = [0, 0, 0, 1]
            # quaternion[0] = quat[1]
            # quaternion[1] = quat[2]
            # quaternion[2] = quat[3]
            # quaternion[3] = quat[0]
        elif len(rotation) == 4:
            quaternion = rotation

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        if static_tf:  self.broadcaster_static.sendTransform(t)
        else:          self.broadcaster_dynamic.sendTransform(t)
        return t



    def getDistanceFromTF(self, reference_frame, target_frame):
      trans, rot = self.getTF(reference_frame, target_frame)
      if len(trans) > 0 and len(rot) > 0:
        dist_trans, dist_rot = calcDistance(trans, rot)

    def getScaledTwistFromTF(self, reference_frame, target_frame, scale=1.0, upper_limit_dist=0.1):
      trans, rot = self.getTF(reference_frame, target_frame)

      if len(trans) > 0 and len(rot) > 0:
        calcScaledTwist(trans, rot, scale, upper_limit_dist)


    def get_frames(self):
        """현재 TF tree에 있는 모든 프레임 목록을 반환"""
        try:
            frames = []
            now = rclpy.time.Time()
            for _ in range(10):  # 최대 10번 시도
                try:
                    # 전체 프레임 문자열 가져오기
                    frames_str = self.tf_buffer.all_frames_as_string()
                    # 각 라인을 파싱하여 프레임 이름만 추출
                    frames = []
                    for line in frames_str.split('\n'):
                        if 'Frame' in line and 'exists with parent' in line:
                            # "Frame {frame_name} exists with parent {parent_name}" 형식에서 frame_name 추출
                            frame_name = line.split('Frame ')[1].split(' exists')[0]
                            frames.append(frame_name)
                    if frames:
                        break
                except Exception:
                    time.sleep(0.1)
            return frames
        except Exception as e:
            self._node.get_logger().error(f'Error getting frames: {str(e)}')
            return []

    def lookup_transform(self, target_frame, source_frame, timeout=1.0):
        """지정된 시간 내에 transform을 찾아 반환"""
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout))
        except Exception as e:
            if self.verbose:
                self._node.get_logger().warning(f'Transform lookup failed: {str(e)}')
            return None


def calcScaledTwist(trans_in_meter, rot_in_quat, scale=1.0, upper_limit_dist=0.1):
  scaled_trans = trans_in_meter * scale
  scaled_rot = rot_in_quat # TODO: rotation에 대한 scale 적용?

  trans_dist = np.linalg.norm(scaled_trans)
  if trans_dist > upper_limit_dist:
      scale = upper_limit_dist / trans_dist
      scaled_trans *= scale
  # print(f'Scaled twist(scale={scale}): {trans_in_meter} -> {scaled_trans}')
  return (scaled_trans, scaled_rot)


def calcDistance(trans_in_meter, rot_in_quat):
  def quaternion_distance(q1, q2):
    dot_product = np.dot(q1, q2)  # 쿼터니언 내적 계산
    return np.arccos(2 * dot_product**2 - 1) # 내적 결과를 사용하여 각도 차이 계산

  translation_distance = np.linalg.norm(trans_in_meter)
  rotation_distance = quaternion_distance(rot_in_quat, [0, 0, 0, 1])
  # print(f'Distance in translation: {translation_distance}, \nDistance in rotation: {rotation_distance}')
  return (translation_distance, rotation_distance)


def quaternion_to_euler(x, y, z, w, to_deg=False):
    quat = Quaternion(w=w, x=x, y=y, z=z)
    euler = quat.to_euler(degrees=to_deg)
    return euler

def euler_to_quaternion(rx_deg, ry_deg, rz_deg):
    quat = Quaternion.from_euler(rx_deg, ry_deg, rz_deg, degrees=True)
    quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
    return quat_xyzw

def normalize_joint_angle(angle_deg):
    n = 0
    while angle_deg < 0:
      angle_deg += 360.0
      n += 1
    angle = angle_deg % 360.0
    return angle, n

def get_euler_zyz_from_quat(quat_xyzw, to_deg=True):
  q_x = quat_xyzw[0]
  q_y = quat_xyzw[1]
  q_z = quat_xyzw[2]
  q_w = quat_xyzw[3]
  t1 = math.atan2(q_x, q_y)
  t2 = math.atan2(q_z, q_w)
  z1 = t2 - t1
  y1 = 2*math.acos(math.sqrt(q_w*q_w + q_z*q_z))
  z2 = t2 + t1
  return [RAD2DEG * z1, RAD2DEG * y1, RAD2DEG * z2]


# 새로 추가된 좌표 변환 관련 함수들 ===================
def quaternion_multiply(quat_xyzw1, quat_xyzw2):
    x1, y1, z1, w1 = quat_xyzw1
    x2, y2, z2, w2 = quat_xyzw2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]


def quaternion_inverse(quat_xyzw):
    x, y, z, w = quat_xyzw
    norm_squared = x*x + y*y + z*z + w*w
    if norm_squared < 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    return [-x/norm_squared, -y/norm_squared, -z/norm_squared, w/norm_squared]


def quaternion_to_rotation_matrix(quat):
    x, y, z, w = quat

    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz, wx, wy, wz = x*y, x*z, y*z, w*x, w*y, w*z

    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2*(y2 + z2)
    R[0, 1] = 2*(xy - wz)
    R[0, 2] = 2*(xz + wy)
    R[1, 0] = 2*(xy + wz)
    R[1, 1] = 1 - 2*(x2 + z2)
    R[1, 2] = 2*(yz - wx)
    R[2, 0] = 2*(xz - wy)
    R[2, 1] = 2*(yz + wx)
    R[2, 2] = 1 - 2*(x2 + y2)

    return R



def transform_mf_base_to_vendor_base_frame(mf_base_to_base_rot,
                                           mf_eef_to_tool_rot,
                                           mf_eef_to_tool_trans,
                                           input_pos,
                                           input_rot):
    """MF base_frame 기준 pose를 DSR base_frame 기준으로 변환

    Parameters:
    -----------
    mf_base_to_base_rot: 쿼터니언 [x,y,z,w]
        mf_base_frame에서 base_frame으로의 회전
    mf_eef_to_tool_rot: 쿼터니언 [x,y,z,w]
        mf_eef_frame에서 tool_frame으로의 회전
    mf_eef_to_tool_trans: numpy.ndarray [x,y,z]
        mf_eef_frame에서 tool_frame으로의 이동
    input_pos: numpy.ndarray [x,y,z]
        mf_base_frame 기준 위치
    input_rot: 쿼터니언 [x,y,z,w]
        mf_base_frame 기준 회전

    Returns:
    --------
    rotated_pos: numpy.ndarray
        base_frame 기준 변환된 위치 (소수점 2째자리까지 반올림)
    base_frame_dir: 쿼터니언 [x,y,z,w]
        base_frame 기준 변환된 회전 (소수점 2째자리까지 반올림)
    """
    # 1. 위치 변환: mf_base -> base
    vendor_base_to_mf_base_rot = quaternion_inverse(mf_base_to_base_rot)
    rot_matrix = quaternion_to_rotation_matrix(vendor_base_to_mf_base_rot)
    rotated_pos = np.dot(rot_matrix, input_pos)

    # 2. 회전 변환: mf_base -> base
    # 단계 1: mf_base에서 base로 직접 회전 변환
    mf_base_to_base_rotation = quaternion_multiply(vendor_base_to_mf_base_rot, input_rot)

    # 단계 2: tool-eef 자세 차이 반영
    # eef->tool 방향 회전을 적용하여 tool 기준으로 회전 변환
    # 여기서 mf_eef_to_tool_rot를 적용해야 올바른 orientation이 계산됨
    base_frame_dir = quaternion_multiply(mf_base_to_base_rotation, mf_eef_to_tool_rot)

    # 3. tool->eef translation 반영 (필요한 경우)
    # mf_eef_to_tool_trans가 있는 경우만 처리
    if len(mf_eef_to_tool_trans) == 3:
        # 변환된 translation을 적용
        tool_to_eef_trans_base = np.dot(rot_matrix, mf_eef_to_tool_trans)
        rotated_pos = np.array(rotated_pos) + tool_to_eef_trans_base

    # 소수점 4째자리까지 반올림
    rotated_pos = np.round(rotated_pos, decimals=4).tolist()
    base_frame_dir = np.round(base_frame_dir, decimals=4)

    return rotated_pos, base_frame_dir

def transform_mf_eef_to_vendor_tool_frame(mf_eef_to_tool_rot, input_pos, input_rot):
    """MF eef_frame 기준 pose를 DSR tool_frame 기준으로 변환

    Parameters:
    -----------
    tool_to_mf_eef_rot: 쿼터니언 [x,y,z,w]
        tool_frame에서 mf_eef_frame으로의 회전
    input_pos: numpy.ndarray [x,y,z]
        mf_eef_frame 기준 위치
    input_rot: 쿼터니언 [x,y,z,w]
        mf_eef_frame 기준 회전

    Returns:
    --------
    rotated_pos: numpy.ndarray
        tool_frame 기준 변환된 위치
    tool_frame_dir: 쿼터니언 [x,y,z,w]
        tool_frame 기준 변환된 회전
    """
    try:
        # 1. 회전 변환: 완전한 좌표계 변환을 위한 3단계 회전 적용
        # tool->mf_eef의 역변환 (mf_eef->tool)
        tool_to_mf_eef_rot = quaternion_inverse(mf_eef_to_tool_rot)

        # 결합된 회전: tool->mf_eef, input_rot, mf_eef->tool
        tool_frame_dir = quaternion_multiply(
            quaternion_multiply(tool_to_mf_eef_rot, input_rot),
            mf_eef_to_tool_rot
        )

        # 2. 위치 변환: mf_eef -> tool
        # 회전 행렬 계산
        rot_matrix = quaternion_to_rotation_matrix(mf_eef_to_tool_rot)

        # 위치 변환: 회전 행렬을 사용하여 위치를 변환
        # 이때 회전의 역행렬을 사용하여 올바른 방향으로 변환
        rotated_pos = np.dot(rot_matrix.T, input_pos)

    except Exception as e:
        # 변환 실패 시 입력값 그대로 사용
        rotated_pos = input_pos.copy()
        tool_frame_dir = input_rot.copy()

    # 소수점 2째자리까지 반올림
    rotated_pos = np.round(rotated_pos, decimals=4)
    base_frame_dir = np.round(tool_frame_dir, decimals=4)

    return rotated_pos, tool_frame_dir


if __name__ == "__main__":
    rclpy.init()
    node = Node("ros2_test_tf")

    tf = ROS2TF(node, verbose=True)


    def test_tf_functions():
        reference_frame = 'base_link'
        target_frame = 'test_link'
        translation = [1.0, 0.0, 0.0]
        rotation = [180, 0, -90]

        tf.publishTF(reference_frame, target_frame, translation, rotation)
        position, orientation = tf.getTF(reference_frame, target_frame)
        if len(position) > 0 and len(orientation) > 0:
            euler = quaternion_to_euler(
              x = orientation[0],
              y = orientation[1],
              z = orientation[2],
              w = orientation[3],
              to_deg=True
            )
            euler_zyz = get_euler_zyz_from_quat(orientation)

            print(f"position: {position}, orientation: {orientation}, euler: {euler}, euler_zyz: {euler_zyz}")

        tf.getDistanceFromTF(reference_frame, target_frame)
        tf.getScaledTwistFromTF(reference_frame, target_frame, scale=0.9)

    timer = node.create_timer(1, test_tf_functions)

    rclpy.spin(node)  # Using spin instead of MultiThreadedExecutor for simplicity



    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()
    exit(0)
