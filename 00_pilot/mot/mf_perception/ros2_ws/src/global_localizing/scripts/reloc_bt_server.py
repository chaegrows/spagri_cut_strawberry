#!/usr/bin/env python3


import rclpy
import numpy as np
import open3d as o3d
from typing import Optional, Dict
from scipy.spatial.transform import Rotation as R

from mf_msgs.msg import BehaviorTreeStatus
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mflib.common.mf_base import raise_with_log
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from sensor_msgs.msg import PointCloud2

from threading import Lock
from copy import copy
import time

class RelocBtServer(BehaviorTreeServerNodeV2):
  repo = 'mf_perception'
  node_name = 'global_localizing'
  def __init__(self, run_mode):
    super().__init__(run_mode)
    self.mark_heartbeat(0)

    # Declare parameters
    self.declare_parameter('pcd_map_path', '/default/path/to/pcd.pcd')
    self.declare_parameter('init_xyz', [0.0, 0.0, 0.0])
    self.declare_parameter('init_yaw_deg', 0.0)
    self.declare_parameter('lidar_topic', '/livox/lidar')
    self.declare_parameter('odom_topic', '/livox/imu')
    self.declare_parameter('do_reloc_pose_diff', 0.1)
    self.declare_parameter('do_reloc_scan_accumulate', 20)
    self.declare_parameter('min_dist_to_lidar', 0.1)
    self.declare_parameter('max_dist_to_lidar', 30.0)
    self.declare_parameter('voxel_size_map', 0.4)
    self.declare_parameter('voxel_size_scan', 0.1)
    self.declare_parameter('fov_degree', 300.0)

    # Get parameters
    self.param_pcd_map_path = self.get_parameter('pcd_map_path').get_parameter_value().string_value
    self.param_init_xyz = self.param['init_xyz']
    self.param_init_yaw_deg = self.get_parameter('init_yaw_deg').get_parameter_value().double_value
    self.param_lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
    self.param_odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
    self.param_do_reloc_pose_diff = self.get_parameter('do_reloc_pose_diff').get_parameter_value().double_value
    self.param_do_reloc_scan_accumulate = self.get_parameter('do_reloc_scan_accumulate').get_parameter_value().integer_value
    self.param_min_dist_to_lidar = self.get_parameter('min_dist_to_lidar').get_parameter_value().double_value
    self.param_max_dist_to_lidar = self.get_parameter('max_dist_to_lidar').get_parameter_value().double_value
    self.param_voxel_size_map = self.get_parameter('voxel_size_map').get_parameter_value().double_value
    self.param_voxel_size_scan = self.get_parameter('voxel_size_scan').get_parameter_value().double_value
    self.param_fov_degree = self.get_parameter('fov_degree').get_parameter_value().double_value

    # Internal states
    self.n_scan_accumulated = 0
    self.T_map_P = None
    self.static_tf_broadcaster = StaticTransformBroadcaster(self)

    # Load map
    self.pcd_map = o3d.io.read_point_cloud(self.param_pcd_map_path)
    self.mf_logger.info(f'n points in map: {len(self.pcd_map.points)}')

    self.mf_logger.info('RelocBtServer initialized.')

    self.lidar_lock = Lock()
    self.lidar_msg_queue = []
    self.do_lidar_subscribe = False
    self.add_async_subscriber(PointCloud2, self.param_lidar_topic, self.point_cloud_callback, 10)
    self.mf_logger.info(f'Subscribed to lidar topic: {self.param_lidar_topic}')



  def point_cloud_callback(self, msg: PointCloud2):
    if not self.do_lidar_subscribe: return
    else:
      self.lidar_msg_queue.append(msg)



  def registration_at_scale(self, acc_scan_in_L, map_in_P, initial, scale):
    acc_scan_in_L = acc_scan_in_L.voxel_down_sample(self.param_voxel_size_scan * scale)
    map_in_P = map_in_P.voxel_down_sample(self.param_voxel_size_map * scale)
    result_icp = o3d.pipelines.registration.registration_icp(
      acc_scan_in_L, map_in_P,
      1.0 * scale, initial,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=100, relative_fitness=1e-15, relative_rmse=1e-15)
    )
    return result_icp.transformation, result_icp.fitness

  def get_inverse_tf(self, tf_4x4):
    tf_inv = np.eye(4)
    tf_inv[:3, :3] = tf_4x4[:3, :3].T
    tf_inv[:3, 3] = -np.matmul(tf_4x4[:3, :3].T, tf_4x4[:3, 3])
    return tf_inv

  @BehaviorTreeServerNodeV2.available_action()
  def accumulate_and_relocate(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
    self.mf_logger.info('RelocBtServer: accumulate_and_relocate called.')


    self.lidar_msg_queue = []
    self.do_lidar_subscribe = True
    # wait queue
    lidar_queue = None
    while rclpy.ok():
      if len(self.lidar_msg_queue) > self.param_do_reloc_scan_accumulate:
        lidar_queue = copy(self.lidar_msg_queue)
        self.do_lidar_subscribe = False
        self.lidar_msg_queue = []
        break
      else:
        # self.mf_logger.info(f'Waiting for lidar data, accumulated: {len(self.lidar_msg_queue)}')
        time.sleep(0.1)

    all_points = None
    for lidar_msg in lidar_queue:
      points_raw_bytes = np.frombuffer(lidar_msg.data, dtype=np.uint8)
      points_bytes = points_raw_bytes.reshape(-1, lidar_msg.point_step)
      points_bytes = points_bytes[:, :12]
      lidar_points = points_bytes.view(np.float32).reshape(-1, 3)

      # Distance filtering
      dists = np.linalg.norm(lidar_points, axis=1)
      lidar_points = lidar_points[(dists >= self.param_min_dist_to_lidar) & (dists <= self.param_max_dist_to_lidar)]

      if all_points is None:
        all_points = lidar_points
      else:
        all_points = np.vstack([all_points, lidar_points])

    # Perform relocation
    all_points_o3d_in_L = o3d.geometry.PointCloud()
    all_points_o3d_in_L.points = o3d.utility.Vector3dVector(all_points)

    initial = np.eye(4)
    self.mf_logger.info(f'initial: {self.param_init_xyz}, {self.param_init_yaw_deg}')
    initial[:3, 3] = self.param_init_xyz
    initial[:3, :3] = R.from_euler('z', self.param_init_yaw_deg, degrees=True).as_matrix()

    transformation1, fitness1 = self.registration_at_scale(all_points_o3d_in_L, self.pcd_map, initial, 5)
    transformation2, fitness2 = self.registration_at_scale(all_points_o3d_in_L, self.pcd_map, transformation1, 1)

    self.mf_logger.info(f'Fitness1: {fitness1}, Fitness2: {fitness2}')

    T_lidar_P = transformation2
    T_lidar_map = np.eye(4)
    T_map_P = self.get_inverse_tf(T_lidar_map) @ T_lidar_P

    tf = TransformStamped()
    tf.header.stamp = self.get_clock().now().to_msg()  
    tf.header.frame_id = 'farm_pcd'
    tf.child_frame_id = 'map'
    tf.transform.translation.x = T_map_P[0, 3]
    tf.transform.translation.y = T_map_P[1, 3]
    tf.transform.translation.z = T_map_P[2, 3]
    q = R.from_matrix(T_map_P[:3, :3]).as_quat()
    tf.transform.rotation.x = q[0]
    tf.transform.rotation.y = q[1]
    tf.transform.rotation.z = q[2]
    tf.transform.rotation.w = q[3]

    self.static_tf_broadcaster.sendTransform(tf)
    self.T_map_P = T_map_P

    self.mf_logger.info('Transform broadcasted.')
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'fitness2': fitness2}

if __name__=='__main__':
  rclpy.init()

  # run_mode = 'standalone'
  run_mode = 'server'
  server = RelocBtServer(run_mode)

  if run_mode == 'standalone':
    context = BehaviorTreeContext()

    lookup_table = LeafActionLookup.build_leaf_action_lookup_instance()
    server.start_ros_thread(async_spin=True)

    la_reloc = lookup_table.find(repo='mf_perception', node_name='global_localizing', action_name='accumulate_and_relocate')[0]

    server.mf_logger.info('##################### example bt server start ##########################')
    context.set_target_leaf_action(la_reloc)
    context = server.run_leaf_action(context)

  else:
    server.start_ros_thread(async_spin=True)
    # while rclpy.ok():
    server.accumulate_and_relocate(None, None, None)
    # time.sleep(5)
    # input('wait for enter to exit')
  rclpy.shutdown()
