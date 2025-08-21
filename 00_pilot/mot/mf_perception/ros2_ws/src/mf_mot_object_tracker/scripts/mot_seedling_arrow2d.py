#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import numpy as np
import cv2
import time
import sys
import random

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from mf_perception_msgs.msg import YoloOutArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from threading import Lock
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from mflib.common.tf import ROS2TF
from geometry_msgs.msg import Point
from mf_msgs.msg import ManipulatorStatus
import seaborn
from std_msgs.msg import String


from mflib.perception.mot import common as mot_common
from mflib.perception.mot import datatypes as mot_dtypes
from mflib.perception.mot import frame_generator_v2 as mot_frame_generator
from mflib.perception.mot import seedling_algorithms
from mflib.perception.utils import manipulator_adapter
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray

from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mf_msgs.msg import BehaviorTreeStatus
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
from typing import Optional, Dict, List
from mflib.common.mf_base import raise_with_log
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


if sys.argv[1] == 'seedling_arrow2d':
  import params.mot_seedling_arrow2d.params_seedling as P
else:
  raise ValueError(f'Unknown argument: {sys.argv[1]}')

class BiMap:
  def __init__(self):
    self.a_to_b = defaultdict(set)
    self.b_to_a = defaultdict(set)

  def add(self, a, b):
    self.a_to_b[a].add(b)
    self.b_to_a[b].add(a)

  def remove_a_b(self, a_ref):
    b_ref = None
    if a_ref in self.a_to_b and len(self.a_to_b[a_ref]) == 1:
      b_ref = next(iter(self.a_to_b[a_ref]))

    if b_ref is None:
      return

    # remove all b in any a
    for a in self.a_to_b:
      if b_ref in self.a_to_b[a]:
        self.a_to_b[a].remove(b_ref)
    # remove all a from b_to_a
    for b in self.b_to_a:
      if a_ref in self.b_to_a[b]:
        self.b_to_a[b].remove(a_ref)

    # remove a and b from both maps
    del self.a_to_b[a_ref]
    del self.b_to_a[b_ref]

  def get_bs(self, a):
    return self.a_to_b.get(a, set())

  def get_as(self, b):
    return self.b_to_a.get(b, set())

  def __repr__(self):
    return f"A->B: {dict(self.a_to_b)}\nB->A: {dict(self.b_to_a)}"

class Arrow2D:
  def __init__(self, start_xy, end_xy):
    self.xyxy = np.array([start_xy[0], start_xy[1], end_xy[0], end_xy[1]], dtype=np.float32)


class GrowingMediumBTInterface:
  def __init__(self, ros2tf, logger):
    self.MAX_DET3DS = 1000
    self.det3ds_ls = []

    self.br_tf_enabled = False
    self.tf_target_det3ds = None

    self.ros2tf = ros2tf
    self.lock = Lock()
    self.logger = logger

  def clear(self):
    with self.lock:
      self.br_tf_enabled = False
      self.tf_target_det3ds = None
      self.det3ds_ls = []

  def feed_det3ds(self, det3ds):
    if len(det3ds) == 0:
      return
    with self.lock:
      if len(self.det3ds_ls) < self.MAX_DET3DS:
        self.det3ds_ls.append(det3ds)

  def start_tf_broadcast(self):
    # select det3ds
    if len(self.det3ds_ls) == 0:
      self.logger.info('No det3ds to broadcast')
      self.transform_msgs = []
      return 0

    # sort det3ds with its length
    det3ds_sorted = sorted(self.det3ds_ls, key=lambda x: len(x))

    # choose 80% of the longest det3ds
    index = int(len(det3ds_sorted) * 0.8)
    self.tf_target_det3ds = det3ds_sorted[index]

    self.transform_msgs = []
    with self.lock:
      self.br_tf_enabled = True
    return len(self.tf_target_det3ds)

  def br_tf_timer(self):
    if self.br_tf_enabled:
      do_buffer_tf = False
      if len(self.transform_msgs) == 0:
        do_buffer_tf = True

      with self.lock:
        for idx, det3d in enumerate(self.tf_target_det3ds):
          trans = det3d.xyzwlh[:3]

          target_frame = f'gm_{idx}'
          tr = self.ros2tf.publishTF(P.base_frame, target_frame, trans)
          if do_buffer_tf == True:
            tlwh = det3d.det2d.tlwh
            target_xy_2d = np.array([(tlwh[0] + tlwh[2] / 2), (tlwh[1] + tlwh[3] / 2)])
            image_center = np.array(P.hole_detection_params['image_size_wh'])/2 #dhdh
            dist_from_center_2d = np.linalg.norm(target_xy_2d - image_center)
            self.transform_msgs.append((tr, target_frame, dist_from_center_2d))

  def get_one_target_frame_id(self):
    if self.br_tf_enabled:
      if len(self.transform_msgs) > 0:
        closest_target = min(self.transform_msgs, key=lambda x: x[2])  # find the closest target
        return closest_target[1]  # return the target frame id
      else:
        return None

class SeedlingBTInterface:
  def __init__(self, ros2tf, logger):
    self.seedling_trajectories = []

    self.br_tf_enabled = False
    self.transform_msgs = []

    self.rosTF = ros2tf
    self.lock = Lock()
    self.logger = logger


  def clear(self):
    with self.lock:
      self.br_tf_enabled = False
    self.seedling_trajectories = []

  def set_seedling_trajectories(self, seedling_trajectories):
    if len(seedling_trajectories) == 0:
      return
    with self.lock:
      if not isinstance(seedling_trajectories, List):
        raise_with_log(ValueError, self.mf_logger, f'seedling_trajectories must be List')
      self.seedling_trajectories = seedling_trajectories

  def pub_seedling_tf(self, seedling_data):
    # Get arrow start and end points
    start_point = seedling_data['pos_root']  # [x, y, z]
    end_point = seedling_data['pos_center']    # [x, y, z]
    traj_id = seedling_data['traj_id']

    # Calculate arrow direction vector
    arrow_vector = end_point - start_point
    arrow_length = np.linalg.norm(arrow_vector)

    tr = None
    if arrow_length > 0:
      # Normalize the arrow vector
      arrow_direction =  arrow_vector / arrow_length

      # Calculate orientation quaternion from arrow direction
      # Assume arrow points along positive X-axis in its local frame
      # We need to find rotation from [1,0,0] to arrow_direction
      z_axis = np.array([0, 0, 1])  # P.base_frame z-axis
      x_axis = arrow_direction      # arrow direction
      y_axis = np.cross(z_axis, x_axis)

      # Create rotation matrix
      rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

      # Convert to quaternion
      rotation = R.from_matrix(rotation_matrix)
      quat = rotation.as_quat()  # [x, y, z, w]

      # Publish TF transform
      target_frame_id = f"seedling_{traj_id}"
      tr = self.rosTF.publishTF(
        P.base_frame,
        target_frame_id,
        start_point,
        rotation=quat,
      )
      self.transform_msgs.append((tr, target_frame_id))
    else:
      self.mf_logger.warn(f"Arrow {traj_id} has zero length, skipping TF publish")
    return tr, target_frame_id


  def start_tf_broadcast(self):
    from copy import copy
    with self.lock:
      self.br_tf_enabled = True

    self.transform_msgs = []
    with self.lock:
      self.br_tf_enabled = True

    return len(self.seedling_trajectories)

  def br_tf_timer(self):
    if self.br_tf_enabled:
      self.transform_msgs = []

      with self.lock:
        for idx, seedling_traj in enumerate(self.seedling_trajectories):
          self.pub_seedling_tf(seedling_traj)

  def get_one_target_frame_id(self):
    if len(self.transform_msgs) > 0:
      return self.transform_msgs[0][1]
    else:
      return None

class MOTseedlingArrow2D(BehaviorTreeServerNodeV2):
  repo = 'mf_perception'
  node_name = 'mot_seedling_arrow2d_node'
  def __init__(self, run_mode='server'):
    super().__init__(run_mode)

    # init queue for frame generation (data synchronizing)
    self.rgb_queue              = mot_dtypes.MfMotImageQueue(P.image_queue_size)
    self.depth_queue            = mot_dtypes.MfMotImageQueue(P.image_queue_size)
    self.idetect_array_queue    = mot_dtypes.MfMotImageBBoxDataArrayQueue(P.light_data_queue_size)
    self.odom_queue             = mot_dtypes.MfMotOdometryQueue(P.light_data_queue_size)
    self.last_odom = None
    self.synced_queue = []
    self.seedling_queue = []

    self.available_gm_num = None
    self.available_seedling_num = None

    # initialization
    self.bridge           = CvBridge()
    self.frame_generator  = mot_frame_generator.FrameGeneratorV2()
    self.rgb_K = P.rgb_K
    self.rgb_D = P.rgb_D
    self.depth_K = P.depth_K
    self.depth_D = P.depth_D
    self.rgb_info_received = None
    self.depth_info_received = None

    if P.odom_source == 'tf':
      self.tf_buffer        = Buffer()
      self.tf_listener      = TransformListener(self.tf_buffer, self)
    self.ros2TF = ROS2TF(self)

    # seedling trajectory management (association and lifecycle management)
    self.seedling_trajectory_manager = seedling_algorithms.TrajectoryArrow2DManager(
      **P.seedling_trajectory_params
    )
    self.odom_when_trajset_initialized = None
    self.result_pub = self.create_publisher(
      Float32MultiArray,
      '/mf_perception/mot_seedling_arrows_in_3d', 1
    )

    # growing medium
    self.gm_hole_detector = seedling_algorithms.Detection3DCreaterMediumHole(
      **P.hole_detection_params
    )

    # debug members
    self.distinct_rgbs = seaborn.color_palette("husl", 20)
    self.arrow_debug_img_pub = self.create_publisher(CompressedImage, '/mf_perception/seedling_arrow_debug_image/compressed', 1)
    self.tracked_arrow_debug_img_pub = self.create_publisher(CompressedImage, '/mf_perception/tracked_arrow_debug_image/compressed', 1)
    self.arrows_in_3d_pub = self.create_publisher(MarkerArray, '/mf_perception/arrows_in_3d', 1)

    self.hole_rect_pub = self.create_publisher(CompressedImage, '/mf_perception/growing_medium_hole_rects/compressed', 1)
    self.hole_bin_rect_pub = self.create_publisher(CompressedImage, '/mf_perception/growing_medium_hole_bin_rects/compressed', 1)
    self.hole_det3d_pub = self.create_publisher(MarkerArray, '/mf_perception/growing_medium_hole_det3d', 1)
    ####

    # subscribers
    if P.rgb_topic.endswith('compressed'):
      rgb_msg_type = CompressedImage
      rgb_cb = self.compressed_rgb_callback
    else:
      rgb_msg_type = Image
      rgb_cb = self.rgb_callback
    if P.depth_topic.endswith('compressedDepth'):
      depth_msg_type = CompressedImage
      depth_cb = self.compressed_depth_callback
    else:
      depth_msg_type = Image
      depth_cb = self.depth_callback
    if P.odom_source == 'topic':
      self.add_async_subscriber(Odometry, P.odom_topic, self.odometry_callback, 10)
    elif P.odom_source == 'tf':
      self.add_async_timer(P.tf_timer_period_seconds, self.tf_timer)
    self.seedling_run_mode = None # seedling or growing_medium
    # self.add_sequential_subscriber(ManipulatorStatus, '/manipulator/out', self.manipulator_status_cb, 1)
    self.add_sequential_subscriber(String, '/seedling_work_state', self.manipulator_status_cb, 1)

    self.rgb_lock = Lock()
    self.depth_lock = Lock()
    self.odom_lock = Lock()
    self.yolo_lock = Lock()
    self.mot_lock = Lock()

    self.add_async_subscriber(rgb_msg_type, P.rgb_topic, rgb_cb, 10)
    self.add_async_subscriber(depth_msg_type, P.depth_topic, depth_cb, 10)
    self.add_async_subscriber(CameraInfo, P.rgb_info_topic, self.rgb_info_callback, 10)
    self.add_async_subscriber(CameraInfo, P.depth_info_topic, self.depth_info_callback, 10)
    self.add_async_subscriber(YoloOutArray, P.bbox_topic, self.yolo_callback, 10)

    self.add_async_timer(P.dt_mot_timer, self.mot_timer)

    # health checker members
    self.health_check_lock = Lock()
    self.n_synced_data_1seconds = 0
    self.dt_update_one = []
    self.add_async_timer(1, self.health_timer)
    self.mark_heartbeat(0)

    # behavior tree interface for gm
    self.seedling_bt_interface = SeedlingBTInterface(self.ros2TF, self.mf_logger)
    self.add_async_timer(P.tf_timer_period_seconds, self.seedling_bt_interface.br_tf_timer)
    self.growing_medium_bt_interface = GrowingMediumBTInterface(self.ros2TF, self.mf_logger)
    self.add_async_timer(P.tf_timer_period_seconds, self.growing_medium_bt_interface.br_tf_timer)

    self.mf_logger.info('MOT by YOLO initialized')

  @BehaviorTreeServerNodeV2.available_action()
  def seedling_create_and_broadcast_tf(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    self.mf_logger.info('---------------dfdf---------------')
    self.mf_logger.info(f'----{self.seedling_run_mode}----')

    if self.seedling_run_mode == P.ENUM_SEEDLING:
      n_dets = self.seedling_bt_interface.start_tf_broadcast()
    if self.seedling_run_mode == P.ENUM_GROWING_MEDIUM:
      n_dets = self.growing_medium_bt_interface.start_tf_broadcast()
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'n_dets': n_dets}

  def manipulator_status_cb(self, msg):
    if msg.data == 'port':
      self.seedling_run_mode = P.ENUM_GROWING_MEDIUM
    elif msg.data == 'plant':
      self.seedling_run_mode = P.ENUM_SEEDLING
    else:
      pass
      # self.mf_logger.warning(f'seedling_run_mode is not defined {self.seedling_run_mode}')

  def create_arrows_from_seedling_bbox(self, det2ds):
    self.mf_logger.info('------------------------------')

    # decompose bbox to full / root
    bbox_seedling_full_ls = []
    bbox_seedling_root_ls = []
    for mf_mot_bbox_data in det2ds:
      if mf_mot_bbox_data.label == P.LABEL_SEEDLING_FULL:
        bbox_seedling_full_ls.append(mf_mot_bbox_data)
      elif mf_mot_bbox_data.label == P.LABEL_SEEDLING_ROOT:
        if mf_mot_bbox_data.score > P.LABEL_SEEDLING_ROOT_SCORE_THRESH: # filter with score
          bbox_seedling_root_ls.append(mf_mot_bbox_data)
    # print(f'create_arrows_from_seedling_bbox: {len(bbox_seedling_full_ls)} full, {len(bbox_seedling_root_ls)} root')

    # check if bbox of root is inside any bbox of full
    index_bimap = BiMap() # A: full. B: root
    # def is_Abox_in_Bbox(A, B):
    #   # A: full, B: root
    #   return (A.tlwh[0] >= B.tlwh[0] and
    #           A.tlwh[1] >= B.tlwh[1] and
    #           A.tlwh[0] + A.tlwh[2] <= B.tlwh[0] + B.tlwh[2] and
    #           A.tlwh[1] + A.tlwh[3] <= B.tlwh[1] + B.tlwh[3])
    def get_max_exclusive_pixels_full_root(full_bbox, root_bbox):
      # A: full, B: root
      cand_pixels = [
        full_bbox.tlwh[0] - root_bbox.tlwh[0],
        full_bbox.tlwh[1] - root_bbox.tlwh[1],
        root_bbox.tlwh[0] + root_bbox.tlwh[2] - (full_bbox.tlwh[0] + full_bbox.tlwh[2]),
        root_bbox.tlwh[1] + root_bbox.tlwh[3] - (full_bbox.tlwh[1] + full_bbox.tlwh[3])
      ]
      return max(cand_pixels)
    for idx_full, full in enumerate(bbox_seedling_full_ls):
      for idx_root, root in enumerate(bbox_seedling_root_ls):
        if get_max_exclusive_pixels_full_root(full, root) <= P.exclusive_pixel_thresh:
          index_bimap.add(idx_full, idx_root)
    # self.mf_logger.info(f'bimap: {index_bimap}')


    # create arrows from center(root) to center(full)
    arrows = []
    # exclude uniquely mapped pair until no more paires
    def get_arrow2d(root_det2d, full_det2d):
      # calculate center of bbox
      root_center = (root_det2d.tlwh[0] + root_det2d.tlwh[2] / 2, root_det2d.tlwh[1] + root_det2d.tlwh[3] / 2)
      full_center = (full_det2d.tlwh[0] + full_det2d.tlwh[2] / 2, full_det2d.tlwh[1] + full_det2d.tlwh[3] / 2)
      return Arrow2D(root_center, full_center)

    # remove unique pairs from bimap until no more unique pairs
    while rclpy.ok():
      unique_pairs = []
      # self.mf_logger.info('before removing unique pairs')
      # self.mf_logger.info(f'bimap: {index_bimap}')
      for idx_full, full in enumerate(bbox_seedling_full_ls):
        root_indicies = index_bimap.get_bs(idx_full)
        if len(root_indicies) == 1:
          unique_pairs.append((idx_full, next(iter(root_indicies))))
      for unique_pairs in unique_pairs:
        # create arrow
        root_det2d = bbox_seedling_root_ls[unique_pairs[1]]
        full_det2d = bbox_seedling_full_ls[unique_pairs[0]]
        arrow = get_arrow2d(root_det2d, full_det2d)
        length = np.linalg.norm(np.array(arrow.xyxy[:2]) - np.array(arrow.xyxy[2:]))
        if length >= P.arrow_min_pixel_length_thresh:
          arrows.append(arrow)
        # remove unique pairs from bimap
        index_bimap.remove_a_b(unique_pairs[0])
      # self.mf_logger.info(f'after removing unique pairs')
      # self.mf_logger.info(f'bimap: {index_bimap}')
      # self.mf_logger.info(f'create_arrows_from_seedling_bbox: {len(unique_pairs)} arrows created')
      if len(unique_pairs) == 0:
        break

    # select only one root
    if P.enable_select_most_likely_seedling:
      # for each full, select the root with the highest score
      index_bimap_new = BiMap()
      for idx_full, full in enumerate(bbox_seedling_full_ls):
        max_score = -1
        max_idx_root = -1
        for idx_root in index_bimap.get_bs(idx_full):
          root = bbox_seedling_root_ls[idx_root]
          if root.score > max_score:
            max_score = root.score
            max_idx_root = idx_root
        if max_idx_root != -1:
          index_bimap_new.add(idx_full, max_idx_root)
      index_bimap = index_bimap_new

    # in order to reject isolated root or full, arrows are defined from all root -> all full that are mapped
    for idx_root in index_bimap.b_to_a.keys():
      for idx_full in index_bimap.get_as(idx_root):
        root_det2d = bbox_seedling_root_ls[idx_root]
        full_det2d = bbox_seedling_full_ls[idx_full]
        # calculate center of bbox
        root_center = (root_det2d.tlwh[0] + root_det2d.tlwh[2] / 2, root_det2d.tlwh[1] + root_det2d.tlwh[3] / 2)
        full_center = (full_det2d.tlwh[0] + full_det2d.tlwh[2] / 2, full_det2d.tlwh[1] + full_det2d.tlwh[3] / 2)

        # calculate distance
        dist = np.linalg.norm(np.array(root_center) - np.array(full_center))
        if dist >= P.arrow_min_pixel_length_thresh:
          arrows.append(Arrow2D(root_center, full_center))

    return arrows

  def health_timer(self):
    with self.health_check_lock:
      dt_update_one_mean = np.mean(self.dt_update_one) if len(self.dt_update_one) > 0 else 0
      if P.do_print_health_check:
        self.mf_logger.info(f'n_synced_data_1seconds: {self.n_synced_data_1seconds}')
        self.mf_logger.info(f'dt_update_one: {dt_update_one_mean:.3f} seconds')

      self.n_synced_data_1seconds = 0
      self.dt_update_one = []

  @BehaviorTreeServerNodeV2.available_action()
  def seedling_destroy_seedling_data(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    with self.mot_lock:
      self.seedling_trajectory_manager.reset_db()
      self.seedling_bt_interface.clear()

    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

  @BehaviorTreeServerNodeV2.available_action()
  def seedling_destroy_port_data(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    with self.mot_lock:
      self.growing_medium_bt_interface.clear()

    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

  @BehaviorTreeServerNodeV2.available_action()
  def seedling_query_target_frame_id(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    with self.mot_lock:
      seedling_frame_ids = self.seedling_bt_interface.get_one_target_frame_id()
      port_frame_ids = self.growing_medium_bt_interface.get_one_target_frame_id()
    if seedling_frame_ids is None and port_frame_ids is None:
      return BehaviorTreeStatus.TASK_STATUS_FAILURE, {'target_frame_id': None}
    else:
      return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'seedling_frame_ids': seedling_frame_ids, 'port_frame_ids': port_frame_ids}



  def mot_timer(self):
    # sync data
    t_update_one_start = time.time()
    with self.rgb_lock:
      with self.depth_lock:
        with self.odom_lock:
          with self.yolo_lock:
            is_success, synced = \
              self.frame_generator.sync_rgbd_odom_det2ds(
                self.rgb_queue,
                self.depth_queue,
                self.odom_queue,
                self.idetect_array_queue,
                squeeze=True,
                max_allowed_rgbd_timediff_ms=P.max_allowed_rgbd_timediff_ms,
                max_allowed_odom_time_diff_ms=P.max_allowed_odom_time_diff_ms)
            if is_success:
              self.n_synced_data_1seconds += len(synced)
            else:
              if P.verbose_sync:
                self.mf_logger.info(f'queue length of rgb, depth, odom, yolo: {len(self.rgb_queue)}, {len(self.depth_queue)}, {len(self.odom_queue)}, {len(self.idetect_array_queue)}')
                error_code = synced
                self.mf_logger.info(f'sync_rgbd_and_interpolate_odom failed: {self.frame_generator.get_error_string(error_code)}')
              return

    # reject if run_mode is not set
    if self.seedling_run_mode is None:
      # self.mf_logger.error('seedling_run_mode is not set, cannot process data')
      return

    # clear db when robot moves
    if self.odom_when_trajset_initialized is None:
      self.odom_when_trajset_initialized = synced[-1].odom
    else:
      if len(synced) > 0:
        new_synced_odom = synced[-1].odom
        pose_diff = np.linalg.norm(new_synced_odom.pose - self.odom_when_trajset_initialized.pose)
        if pose_diff > P.reset_db_posediff_thresh:
          self.mf_logger.info(f'Reset trajectory set due to pose diff: {pose_diff:.3f} > {P.reset_db_posediff_thresh}')
          self.seedling_destroy_seedling_data(None, None, None)
          self.odom_when_trajset_initialized = new_synced_odom

    # detection + trajectory management
    if self.seedling_run_mode == P.ENUM_SEEDLING:
      self.mf_logger.info('Processing data in SEEDLING mode')
      self.process_seedling_frames(synced)
    elif self.seedling_run_mode == P.ENUM_GROWING_MEDIUM:
      self.mf_logger.info('Processing data in GROWING_MEDIUM mode')
      self.process_growing_hole_frames(synced)
    else:
      raise_with_log(ValueError, f'Unknown seedling_run_mode: {self.seedling_run_mode}')

    t_update_one_done = time.time()
    self.dt_update_one.append(t_update_one_done - t_update_one_start)

  def process_growing_hole_frames(self, frames):
    # create det3d
    det3ds = [] # it can be used for mot but not now...
    for frame in frames:
      det2ds = frame.detections

      pose = frame.odom.pose
      quat = frame.odom.orientation
      rot_mat = Rotation.from_quat(quat).as_matrix()

      transform_world_to_body = np.zeros((4, 4), np.float32)
      transform_world_to_body[:3, :3] = rot_mat
      transform_world_to_body[:3, 3] = pose
      transform_world_to_body[3, 3] = 1

      hole_rects = []
      binarized_images = []
      det3d_tmp = []
      for det2d in det2ds:
        ret = self.gm_hole_detector.create(det2d, frame)
        if ret[0] == True:
          # det3d, rect, binarized = ret[1]
          det3d, rect, _ = ret[1]

          det3d = seedling_algorithms.convert_det3d_with_tf(det3d, transform_world_to_body)
          det3d.add_metadata(det2d, None, None)

          hole_rects.append(rect)
          # binarized_images.append(binarized)
          det3d_tmp.append(det3d)
          det3ds.append(det3d)

      self.growing_medium_bt_interface.feed_det3ds(det3d_tmp)

    # visualize last frame
    if P.debug_growing_medium_holes_image:
      self.gm_visualize_rects(hole_rects, frame)
      # self.gm_visualize_binary_images(binarized_images)
      self.gm_visualize_det3ds(det3d_tmp)

  def gm_visualize_det3ds(self, det3ds):
    marker_array = MarkerArray()
    # erasing marker
    delete_all_marker = Marker()
    delete_all_marker.action = Marker.DELETEALL
    marker_array.markers.append(delete_all_marker)
    self.hole_det3d_pub.publish(marker_array)

    for idx, det3d in enumerate(det3ds):
      color = self.distinct_rgbs[(idx % len(self.distinct_rgbs))]
      color = [int(c * 255) for c in color]

      marker = Marker()
      marker.header.frame_id = P.base_frame  # base frame for visualization
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "growing_medium_hole"
      marker.id = idx
      marker.type = Marker.CUBE
      marker.action = Marker.ADD
      marker.pose.position.x = det3d.xyzwlh[0]
      marker.pose.position.y = det3d.xyzwlh[1]
      marker.pose.position.z = det3d.xyzwlh[2]
      marker.scale.x = det3d.xyzwlh[3]
      marker.scale.y = det3d.xyzwlh[4]
      marker.scale.z = det3d.xyzwlh[5]
      marker.color.r = color[0] / 255.0
      marker.color.g = color[1] / 255.0
      marker.color.b = color[2] / 255.0
      marker.color.a = 0.5  # semi-transparent
      marker.lifetime.sec = 0  # 0 = forever
      marker.lifetime.nanosec = 0
      marker_array.markers.append(marker)
    self.hole_det3d_pub.publish(marker_array)

  def gm_visualize_binary_images(self, binarized_images):
    # this function assumpes identical size of binarized images
    n_roi = len(binarized_images)
    if n_roi == 0:
      self.mf_logger.info('No binarized images to visualize')
      return
    image = np.zeros((21, n_roi*21))
    for idx, roi in enumerate(binarized_images):
      image[:, idx*21:(idx+1)*21] = roi
    msg = self.bridge.cv2_to_compressed_imgmsg(image, dst_format='jpg')
    self.hole_bin_rect_pub.publish(msg)

  def gm_visualize_rects(self, rects, frame):
    rgb = frame.rgb.image.copy()
    for idx, tlwh in enumerate(rects):
      x1, y1, w, h = tlwh
      x2, y2 = x1 + w, y1 + h

      color = self.distinct_rgbs[(idx % len(self.distinct_rgbs))]
      color = [int(c * 255) for c in reversed(color)]

      cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    msg = self.bridge.cv2_to_compressed_imgmsg(rgb, dst_format='jpg')
    self.hole_rect_pub.publish(msg)



  def process_seedling_frames(self, frames):
    arrows2d = []
    for frame in frames:
      arrows2d.append(self.create_arrows_from_seedling_bbox(frame.detections))
    if P.debug_seedling_detection:
      self.debug_arrows(arrows2d, frames[-1])

    with self.mot_lock:
      # update trajectories
      for arrow1d in arrows2d:
        ret = self.seedling_trajectory_manager.update_with_det_in_one_frame(arrow1d)
        # self.mf_logger.info(f'update_with_det_in_one_frame returned: {ret}')

      # get arrows in 3D
      tracked_arrows = self.seedling_trajectory_manager.get_tracked_arrows()
      if len(tracked_arrows) > 0:
        arrows_in_3d_cam_frame = self.arrow_on_image_to_3d(tracked_arrows, frames[-1])
        arrows_in_3d_baselink_frame = self.transform_results_to_baselink_se3(arrows_in_3d_cam_frame, frames[-1])

        # publish results
        msg = Float32MultiArray() # order: Tracking ID, x_start, y_start, z_start, x_end, y_end, z_end, ...
        tracked_seedlings = []
        for traj, arrow_in_3d in zip(tracked_arrows, arrows_in_3d_baselink_frame):
          if arrow_in_3d is None:
            self.mf_logger.info('invalid depth')
            continue
          traj_id = traj['trajectory_id_if_confirmed']
          data_ls = [float(traj_id),
                      arrow_in_3d[0][0], arrow_in_3d[0][1], arrow_in_3d[0][2],
                      arrow_in_3d[1][0], arrow_in_3d[1][1], arrow_in_3d[1][2]]
          msg.data.extend(data_ls)

          seedling_data = {
            'traj_id': traj_id,
            'pos_root': arrow_in_3d[0][:3],
            'pos_center': arrow_in_3d[1][:3],
          }
          tracked_seedlings.append(seedling_data)

        self.result_pub.publish(msg)
        self.seedling_bt_interface.set_seedling_trajectories(tracked_seedlings)

        if P.debug_arrow_in_3d:
          self.publish_3d_arrows(arrows_in_3d_baselink_frame)

    # publish trajectories in image plane
    with self.mot_lock:
      tracked_arrows = self.seedling_trajectory_manager.get_tracked_arrows()
      self.debug_tracked_arrows(tracked_arrows, frames[-1])

  def pub_seedling_tf(self, seedling_data):
    # Get arrow start and end points
    start_point = seedling_data['pos_root']  # [x, y, z]
    end_point = seedling_data['pos_center']    # [x, y, z]
    traj_id = seedling_data['traj_id']

    # Calculate arrow direction vector
    arrow_vector = end_point - start_point
    arrow_length = np.linalg.norm(arrow_vector)

    if arrow_length > 0:
      # Normalize the arrow vector
      arrow_direction = - arrow_vector / arrow_length

      # Calculate orientation quaternion from arrow direction
      # Assume arrow points along positive X-axis in its local frame
      # We need to find rotation from [1,0,0] to arrow_direction
      z_axis = np.array([0, 0, -1])  # P.base_frame z-axis
      x_axis = arrow_direction      # arrow direction
      y_axis = np.cross(z_axis, x_axis)

      # Create rotation matrix
      rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

      # Convert to quaternion
      rotation = R.from_matrix(rotation_matrix)
      quat = rotation.as_quat()  # [x, y, z, w]

      # Publish TF transform
      self.rosTF.publishTF(
        P.base_frame,
        f"seedling_{traj_id}",
        start_point,
        rotation=quat,
      )
    else:
      self.mf_logger.warn(f"Arrow {traj_id} has zero length, skipping TF publish")

  def transform_results_to_baselink_se3(self, results_3d, frame):
    orientation = frame.odom.orientation
    pose = frame.odom.pose
    # Create SE(3) matrix
    rot = R.from_quat([*orientation])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pose

    def apply_se3(p):
      p_h = np.hstack([p, 1.0])   # make homogeneous
      return (T @ p_h)[:3]        # transform and return as 3D

    transformed = []
    for pair in results_3d:
      if pair is None:
        transformed.append(None)
        continue
      p1_bl = apply_se3(pair[0])
      p2_bl = apply_se3(pair[1])
      transformed.append((p1_bl, p2_bl))

    return transformed

  def arrow_on_image_to_3d(self, tracked_arrows, frame):
    depth_img = frame.depth.image.astype(np.float32) / 1000.  # Ensure float32 for median
    K = frame.depth.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth_img.shape
    roi = P.depth_roi_size // 2

    results_3d = []

    for xyxy in tracked_arrows['xyxy']:
      x1, y1, x2, y2 = map(int, xyxy)

      def get_depth_and_project(x, y):
        # Make sure ROI is within image bounds
        xmin = max(x - roi, 0)
        xmax = min(x + roi + 1, w)
        ymin = max(y - roi, 0)
        ymax = min(y + roi + 1, h)

        roi_depth = depth_img[ymin:ymax, xmin:xmax]
        valid = roi_depth[roi_depth > 0]  # remove invalid depth

        if len(valid) == 0:
          return None  # depth unavailable

        z = np.median(valid) # assuming input is in mm
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        return np.array([X, Y, z], dtype=np.float32)

      p1_3d = get_depth_and_project(x1, y1)
      p2_3d = get_depth_and_project(x2, y2)

      if p1_3d is not None and p2_3d is not None:
        results_3d.append((p1_3d, p2_3d))
      else:
        results_3d.append(None)  # skip or mark as invalid

    return results_3d

  def publish_3d_arrows(self, results_3d):
    marker_array = MarkerArray()
    # publish deleteall marker
    delete_all_marker = Marker()
    delete_all_marker.action = Marker.DELETEALL
    marker_array.markers.append(delete_all_marker)
    self.arrows_in_3d_pub.publish(marker_array)

    # markers
    marker_array = MarkerArray()
    for idx, pair in enumerate(results_3d):
      if pair is None:
        continue
      p1, p2 = pair

      marker = Marker()
      marker.header.frame_id = P.base_frame  # base frame for visualization
      # marker.header.frame_id = P.body_frame
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "arrow_3d"
      marker.id = idx
      marker.type = Marker.ARROW
      marker.action = Marker.ADD

      # Define arrow geometry
      marker.points = [Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                        Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))]

      marker.scale.x = 0.005  # shaft diameter
      marker.scale.y = 0.01  # head diameter
      marker.scale.z = 0.005  # head length

      marker.color.r = 1.0
      marker.color.g = 0.3
      marker.color.b = 0.0
      marker.color.a = 1.0

      marker.lifetime.sec = 0  # 0 = forever
      marker_array.markers.append(marker)

    self.arrows_in_3d_pub.publish(marker_array)

  def debug_arrows(self, arrows2d, frame):
    # 2d arrows to 1d
    arrows_1d = []
    for arrow1d in arrows2d:
      arrows_1d.extend(arrow1d)

    cv_img = np.copy(frame.rgb.image)
    # draw arrows on the image
    if len(arrows_1d) > 0:
      for arrow in arrows_1d:
        start = (int(arrow.xyxy[0]), int(arrow.xyxy[1]))
        end = (int(arrow.xyxy[2]), int(arrow.xyxy[3]))
        cv2.arrowedLine(cv_img, start, end, (0, 255, 0), 2, tipLength=0.2)

    # publish debug image
    compressed = self.bridge.cv2_to_compressed_imgmsg(cv_img, dst_format='jpg')
    self.arrow_debug_img_pub.publish(compressed)

  def debug_tracked_arrows(self, arrows2d_tracked, frame):
    # 2d arrows to 1d
    cv_img = np.copy(frame.rgb.image)

    # draw
    if len(arrows2d_tracked) > 0:
      for at in arrows2d_tracked:
        # arrows
        start = (int(at['xyxy'][0]), int(at['xyxy'][1]))
        end = (int(at['xyxy'][2]), int(at['xyxy'][3]))
        cv2.arrowedLine(cv_img, start, end, (0, 0, 255), 2, tipLength=0.2)
        # ids
        text = f'ID: {at["trajectory_id_if_confirmed"]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 1
        color = (255, 0, 0)  # Blue color for text
        center_xy = at['xyxy'][:2]
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Compute bottom-left corner of the text so it appears centered below the point
        origin = (int(center_xy[0] - text_width / 2), int(center_xy[1] + text_height + 5))  # +5: small padding
        cv2.putText(cv_img, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)

    # publish debug image
    compressed = self.bridge.cv2_to_compressed_imgmsg(cv_img, dst_format='jpg')
    self.tracked_arrow_debug_img_pub.publish(compressed)

  def add_cv_image_to_queue(self, queue, sec, nsec, cv_image, K, D, lock):
    if self.rgb_K is None or self.rgb_D is None:
      self.mf_logger.info('no rgb camera info!')
      return
    if self.depth_K is None or self.depth_D is None:
      self.mf_logger.info('no depth rgb camera info!')
      return
    with lock:
      queue.add_image(sec, nsec, cv_image, K, D)

  def rgb_callback(self, msg):
    rgb_cv = self.bridge.imgmsg_to_cv2(msg)
    self.add_cv_image_to_queue(
      self.rgb_queue,
      msg.header.stamp.sec,
      msg.header.stamp.nanosec,
      rgb_cv,
      self.rgb_K,
      self.rgb_D,
      self.rgb_lock)

  def compressed_rgb_callback(self, msg):
    rgb_cv = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.add_cv_image_to_queue(
      self.rgb_queue,
      msg.header.stamp.sec,
      msg.header.stamp.nanosec,
      rgb_cv,
      self.rgb_K,
      self.rgb_D,
      self.rgb_lock)

  def rgb_info_callback(self, msg):
    if self.rgb_info_received:
      return
    self.rgb_K = np.array(msg.k, np.float32).reshape(3, 3)
    self.rgb_D = np.array(msg.d, np.float32)
    self.rgb_info_received = True

  def depth_callback(self, msg):
    depth_cv = self.bridge.imgmsg_to_cv2(msg)
    self.add_cv_image_to_queue(
      self.depth_queue,
      msg.header.stamp.sec,
      msg.header.stamp.nanosec,
      depth_cv,
      self.depth_K,
      self.depth_D,
      self.depth_lock)

  def compressed_depth_callback(self, msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    depth_cv = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)
    self.add_cv_image_to_queue(
      self.depth_queue,
      msg.header.stamp.sec,
      msg.header.stamp.nanosec,
      depth_cv,
      self.depth_K,
      self.depth_D,
      self.depth_lock)

  def depth_info_callback(self, msg):
    if self.depth_info_received:
      return
    self.depth_K = np.array(msg.k, np.float32).reshape(3, 3)
    self.depth_D = np.array(msg.d, np.float32)
    self.depth_info_received = True

  def yolo_callback(self, msg):
    sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
    detections = []
    for out in msg.yolo_out_array:
      score = out.score
      label = out.label
      tlbr = out.tlbr
      tlwh = np.array([tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]]).astype(np.float32)

      d = mot_dtypes.MfMotImageBBoxData(sec, nsec, tlwh, score, label)
      detections.append(d)

    with self.yolo_lock:
      self.idetect_array_queue.add_image_bbox_array(sec, nsec, detections)

  def tf_timer(self):
    try:
      transform = self.tf_buffer.lookup_transform(
        P.base_frame,
        P.body_frame,
        rclpy.time.Time())
    except Exception as e:
      if P.verbose_sync:
        self.mf_logger.info(f'Error in odometry_callback: {e}')
      return

    sec, nsec = transform.header.stamp.sec, transform.header.stamp.nanosec
    if len(self.odom_queue):
      ts_cur = mot_common.Timestamp(sec, nsec)
      ts_last = self.odom_queue[-1].ts
      if ts_cur == ts_last:
        return

    pose = np.array([transform.transform.translation.x,
                     transform.transform.translation.y,
                     transform.transform.translation.z]).astype(np.float32)
    orientation = np.array([transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w]).astype(np.float32)

    with self.odom_lock:
      self.odom_queue.add_odometry(sec, nsec, pose, orientation, None, None)
      self.last_odom = self.odom_queue[-1]

  def odometry_callback(self, msg):
    sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
    pose = np.array([msg.pose.pose.position.x,
                     msg.pose.pose.position.y,
                     msg.pose.pose.position.z]).astype(np.float32)
    orientation = np.array([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w]).astype(np.float32)
    velocity = np.array([msg.twist.twist.linear.x,
                          msg.twist.twist.linear.y,
                          msg.twist.twist.linear.z]).astype(np.float32)
    angular_velocity = np.array([msg.twist.twist.angular.x,
                                 msg.twist.twist.angular.y,
                                 msg.twist.twist.angular.z]).astype(np.float32)
    with self.odom_lock:
      self.odom_queue.add_odometry(sec, nsec, pose, orientation, velocity, angular_velocity)

  @BehaviorTreeServerNodeV2.available_action()
  def count_remaining_targets(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    if self.seedling_run_mode == P.ENUM_GROWING_MEDIUM:
      self.available_gm_num = input_dict['n_dets']
    if self.seedling_run_mode == P.ENUM_SEEDLING:
      self.available_seedling_num = input_dict['n_dets']
      if self.available_seedling_num == 0 or self.available_gm_num == 0:
        return BehaviorTreeStatus.TASK_STATUS_FAILURE, {'n_port': self.available_gm_num, 'seedling': self.available_seedling_num}
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'n_port': self.available_gm_num, 'seedling': self.available_seedling_num}

  @BehaviorTreeServerNodeV2.available_action()
  def clear_target_count(self,
              input_dict: Optional[Dict],
              target_sector: Sector,
              target_work: ALLOWED_WORKS):
    self.available_seedling_num = None
    self.available_gm_num = None
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'n_port': self.available_gm_num, 'seedling': self.available_seedling_num}


def main(args=None):
  rclpy.init(args=args)

  run_mode = 'server'
  mot_server = MOTseedlingArrow2D(run_mode)
  if run_mode == 'server':
    try:
      mot_server.start_ros_thread(async_spin=False)
    except KeyboardInterrupt:
      print("Shutdown requested")
    input('')
    mot_server.mf_logger.info('why do you see this msg?')
    mot_server.destroy_node()

  elif run_mode == 'standalone':
    context = BehaviorTreeContext()
    lookup_table = LeafActionLookup.build_leaf_action_lookup_instance()

    create_and_broadcast = \
      lookup_table.find(repo='mf_perception', node_name='mot_seedling_arrow2d_node', action_name='create_and_broadcast_growing_medium_tf')[0]
    destroy_db = \
      lookup_table.find(repo='mf_perception', node_name='mot_seedling_arrow2d_node', action_name='destroy_growing_medium_tf')[0]

    mot_server.start_ros_thread(async_spin=True)

    input('Press Enter to create growing medium tf...')
    context.set_target_leaf_action(create_and_broadcast)
    context = mot_server.run_leaf_action(context)

    input('Press Enter to destroy growing medium tf...')
    context.set_target_leaf_action(destroy_db)
    context = mot_server.run_leaf_action(context)

    input('Press Enter to shutdown...')
    mot_server.destroy_node()

  rclpy.shutdown()

if __name__ == '__main__':
  main()
