import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import quaternion
import sys
import shutil

from rclpy.node import Node
EXIT_CODE_WRONG_CONFIG = 17

def cautious_mkdir(target_dir):
  if os.path.exists(target_dir):
    if os.path.isdir(target_dir):
      prn = f'path {target_dir} exists. delete and create new one? (y / n)'
      out = input(prn)
      if out.lower() == 'y':
        shutil.rmtree(target_dir)
    elif os.path.isfile(target_dir):
      print(f"path {target_dir} is file. Check your path")
      os._exit(EXIT_CODE_WRONG_CONFIG)
  os.makedirs(target_dir)


def load_tf_from_step1_out(path):
  poses = {}
  with open(path) as f:
    lines = f.readlines()
    for line in lines: 
      tf_mat = line.split(',')[1:]
      rgb_idx = int(line.split(',')[0])
      pose = np.array([float(x) for x in tf_mat]).reshape(4, 4) #@ to_camera

      poses[rgb_idx] = pose
  return poses

def get_average_of_transforms(Ts, inlier_ratio):
  if len(Ts) == 0:
    raise None
  
  # select inliers
  len_Ts = len(Ts)
  len_Ts_to_cut = int(len_Ts * (1-inlier_ratio)/2.0)
  Ts_inliers = Ts[len_Ts_to_cut:-len_Ts_to_cut] # remove 10% from both ends since they are usually noisy due to movements
    
  translations = np.array([matrix[:3, 3] for matrix in Ts_inliers])
  mean_translation = np.mean(translations, axis=0)
  
  # 2. 회전 행렬을 쿼터니언으로 변환하여 평균 계산
  rotations = [R.from_matrix(matrix[:3, :3]) for matrix in Ts_inliers]
  quaternions = np.array([rotation.as_quat() for rotation in rotations])  # (x, y, z, w) 형식
  mean_quaternion = np.mean(quaternions, axis=0)
  mean_quaternion /= np.linalg.norm(mean_quaternion)  # 정규화
  
  # 3. 평균 쿼터니언을 회전 행렬로 변환
  mean_rotation = R.from_quat(mean_quaternion).as_matrix()
  
  # 4. 평균 Transformation Matrix 생성
  mean_matrix = np.eye(4)
  mean_matrix[:3, :3] = mean_rotation
  mean_matrix[:3, 3] = mean_translation
  return mean_matrix

def tf_msg_to_T_mat4x4(tf_msg):
  T = np.eye(4)
  translation = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z])
  T[:3, 3] = translation
  rot_mat = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w]).as_matrix()
  T[:3, :3] = rot_mat
  return T

class FarmMonitoringTfHandler(Node):
  def __init__(self, node_name):
    super().__init__(node_name)
    self.tf_broadcaster = TransformBroadcaster(self)
  
  def _T_mat4x4_to_msg(self, stamp, T, parent, child):
    t = TransformStamped()

    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child

    t.transform.translation.x = float(T[0, 3])
    t.transform.translation.y = float(T[1, 3])
    t.transform.translation.z = float(T[2, 3])

    q = R.from_matrix(T[:3, :3]).as_quat()
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t

  def broadcast_tf_mat4x4(self, stamp, T_premap_marker, T_marker_map, T_map_camera=None):
    tf_msg_marker_map = self._T_mat4x4_to_msg(stamp, T_marker_map, 'marker', 'map')
    tf_msg_premap_marker = self._T_mat4x4_to_msg(stamp, T_premap_marker, 'prebuilt_map', 'marker')
    self.tf_broadcaster.sendTransform(tf_msg_marker_map)
    self.tf_broadcaster.sendTransform(tf_msg_premap_marker)
    if T_map_camera is not None:
      tf_msg_map_camera = self._T_mat4x4_to_msg(stamp, T_map_camera, 'map', 'camera')
      self.tf_broadcaster.sendTransform(tf_msg_map_camera)
  
def query_transform(time_and_T_queue, target_time): # assume sequential function call
  assert isinstance(target_time, float) # assume time is float
  after_ele_idx = 0
  while after_ele_idx < len(time_and_T_queue) - 1 and time_and_T_queue[after_ele_idx][0] <= target_time:
    after_ele_idx += 1
  if after_ele_idx == 0:
    return None, None
  if after_ele_idx == len(time_and_T_queue) - 1:
    if target_time > time_and_T_queue[-1][0]:
      return None, None

  prev_transform = time_and_T_queue[after_ele_idx - 1][1]
  now_transform = time_and_T_queue[after_ele_idx][1]
  
  t_prev_to_target = target_time - time_and_T_queue[after_ele_idx - 1][0]
  dt = time_and_T_queue[after_ele_idx][0] - time_and_T_queue[after_ele_idx - 1][0]
  t_factor = t_prev_to_target / dt

  # interpolate pose    
  prev_pose = prev_transform[:3, 3]
  now_pose = now_transform[:3, 3]
  pose_interp = prev_pose + t_factor * (now_pose - prev_pose)

  # interpolate orientation
  prev_quat = R.from_matrix(prev_transform[:3, :3]).as_quat()
  prev_quat = quaternion.quaternion(*prev_quat)
  now_quat = R.from_matrix(now_transform[:3, :3]).as_quat()
  now_quat = quaternion.quaternion(*now_quat)

  quat = quaternion.slerp_evaluate(prev_quat, now_quat, t_factor)
  quat_interp = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.float32)
  quat_interp_mat = R.from_quat(quat_interp).as_matrix()

  ret = np.eye(4)
  ret[:3, 3] = pose_interp
  ret[:3, :3] = quat_interp_mat

  new_ary_start_idx = after_ele_idx - 1
  return ret, new_ary_start_idx