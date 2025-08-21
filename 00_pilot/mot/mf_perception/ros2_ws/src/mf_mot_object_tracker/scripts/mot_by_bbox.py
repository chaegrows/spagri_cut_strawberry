#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import os
import numpy as np
import cv2
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from mf_perception_msgs.msg import YoloOutArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from threading import Lock
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from mflib.common.tf import ROS2TF

from mflib.perception.brain.mot import common as mot_common
from mflib.perception.brain.mot import datatypes as mot_dtypes
from mflib.perception.brain.mot import frame_generator_v2 as mot_frame_generator
from mflib.perception.brain.mot import frame_algorithms as m_fa
from mflib.perception.brain.mot.utils import mot3d_visualizer
from mflib.perception.brain.mot.utils import manipulator_adapter
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2, MfBehaviorTreeContext

import sys
from mf_msgs.msg import BehaviorTreeStatus
from typing import Optional


# if arg == 'seedling': # in the near future...
#   import params.params_mot_by_bbox_seedling as P
# elif arg == 'cherry_tomato':
#   import params.params_mot_by_bbox_cherry_tomato as P
# else:
  # raise NotImplementedError('Unknown arg[1]: ', arg)
# elif arg == 'strawberry':
#   import params.params_mot_by_bbox_strawberry as P

if sys.argv[1] == 'strawberry_harvesting_pollination':
  import params.mot_by_bbox.strawberry_harvesting_pollination as P


class MOTbyYOLO(BehaviorTreeServerNodeV2):
  def __init__(self, node_name='mot_by_bbox', trajectory_visualizer=None):
    super().__init__(node_name, [])
    # init queue for frame generation (data synchronizing)
    self.rgb_queue              = mot_dtypes.MfMotImageQueue(P.image_queue_size)
    self.depth_queue            = mot_dtypes.MfMotImageQueue(P.image_queue_size)
    self.idetect_array_queue    = mot_dtypes.MfMotImageBBoxDataArrayQueue(P.light_data_queue_size)
    self.odom_queue             = mot_dtypes.MfMotOdometryQueue(P.light_data_queue_size)
    self.synced_queue = []

    # initialization
    self.bridge           = CvBridge()
    self.frame_generator  = mot_frame_generator.FrameGeneratorV2()
    self.rgb_K = P.rgb_K
    self.rgb_D = P.rgb_D
    self.depth_K = P.depth_K
    self.depth_D = P.depth_D
    if P.odom_source == 'tf':
      self.tf_buffer        = Buffer()
      self.tf_listener      = TransformListener(self.tf_buffer, self)
    self.rosTF = ROS2TF(self)

    # 3d object detection
    self.det3d_creator = m_fa.Detection3DCreaterBBox(**P.det3d_params)

    # trajectory management (association and lifecycle management)
    self.cost_metric = m_fa.CostMetrics(**P.cost_metric_params)
    self.trajectory_manager = m_fa.Trajectory3DManager(**P.trajectory_params)

    # debug members
    self.trajectory_visualizer = trajectory_visualizer

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

    self.rgb_lock = Lock()
    self.depth_lock = Lock()
    self.odom_lock = Lock()
    self.yolo_lock = Lock()
    self.traj_lock = Lock()

    self.add_async_subscriber(rgb_msg_type, P.rgb_topic, rgb_cb, 10)
    self.add_async_subscriber(depth_msg_type, P.depth_topic, depth_cb, 10)
    self.add_async_subscriber(CameraInfo, P.rgb_info_topic, self.rgb_info_callback, 10)
    self.add_async_subscriber(CameraInfo, P.depth_info_topic, self.depth_info_callback, 10)
    self.add_async_subscriber(YoloOutArray, P.bbox_topic, self.yolo_callback, 10)

    self.add_async_timer(P.dt_mot_timer, self.mot_timer)

    # health checker members
    self.health_check_lock = Lock()
    self.n_synced_data_1seconds = 0
    self.n_det3ds_1seconds = 0
    self.dt_trajectory_update = []
    self.dt_update_one = []
    self.add_async_timer(1, self.health_timer)
    self.set_heartbeat(0)
    print('MOT by YOLO initialized')

  @BehaviorTreeServerNodeV2.available_action('clear_db')
  def clear_trajectory_db(self, context: Optional[MfBehaviorTreeContext] = None):
    with self.traj_lock:
      self.trajectory_manager.clear_db()
    self.set_status(BehaviorTreeStatus.TASK_STATUS_SUCCESS)

  def health_timer(self):
    with self.health_check_lock:
      mean_dt = np.mean(self.dt_trajectory_update) if len(self.dt_trajectory_update) else 0
      mean_dt_mot = np.mean(self.dt_update_one) if len(self.dt_update_one) else 0
      print('n_synced_data_1seconds:', self.n_synced_data_1seconds)
      print('n_det3ds_1seconds:', self.n_det3ds_1seconds)
      print('mean dt for trajectory update:', mean_dt)
      print('mean dt for update mot ', mean_dt_mot)
      self.n_synced_data_1seconds = 0
      self.n_det3ds_1seconds = 0
      self.dt_trajectory_update = []
      self.dt_update_one = []

  def mot_timer(self):
    # sync data
    t_mot_start = time.time()
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
                max_allowed_rgbd_timediff_ms=P.max_allowed_rgbd_timediff_ms,
                max_allowed_odom_time_diff_ms=P.max_allowed_odom_time_diff_ms)
            if is_success:
              self.n_synced_data_1seconds += len(synced)
            else:
              if P.verbose_sync:
                print('queue length of rgb, depth, odom, yolo: ', 
                        len(self.rgb_queue), len(self.depth_queue),
                        len(self.odom_queue), len(self.idetect_array_queue))
                error_code = synced
                print('sync_rgbd_and_interpolate_odom failed: ', \
                      self.frame_generator.get_error_string(error_code))
              return
            
    # create 3D object detection
    det3ds_2darray = self.create_det3ds(synced)
    if len(det3ds_2darray) and P.debug_det3d:
      # publish det3d on an image
      last_det3ds = det3ds_2darray[-1]
      last_synced = synced[-1]
      rgb_cv = last_synced.rgb.image
      self.trajectory_visualizer.publish_det3d_on_image(rgb_cv, last_det3ds)
      image_size_wh = (rgb_cv.shape[1], rgb_cv.shape[0])

      # publish camera tf
      self.trajectory_visualizer.publish_camera_fov_from_frame(
        last_synced.odom, image_size_wh, 
        P.depth_max_mm/1000., last_synced.depth.K, frame_id=P.base_frame)
    
      # publish det3d as markers
      self.trajectory_visualizer.publish_det3ds_as_markers(last_det3ds, P.base_frame)

    total_det3ds = sum([len(det3ds) for det3ds in det3ds_2darray])
    self.n_det3ds_1seconds += total_det3ds
    
    # update trajectory
    self.update_trajectory(det3ds_2darray)

    # manipulator adapter
    if P.use_manipulator_adapter:
      traj_set = self.trajectory_manager.get_trajectory_set()
      traj_det3ds_label_list = traj_set.get_confirmed_traj_det3ds_label_list()
      for traj, _, label in traj_det3ds_label_list:
        prefix = 'i hope you do not see me'
        if label == 7:
          prefix='strawberry'
        elif label == 5:
          prefix='flower'
        manipulator_adapter.pub_tf_for_manipulator(self.rosTF, traj, target_prefix=prefix)
    
    # debug
    last_frame = synced[-1]
    if P.debug_trajectory_mode == 'simple':
      self.debug_trajectories(last_frame)
    elif P.debug_trajectory_mode == 'verbose':
      t1 = time.time()
      self.debug_reliable_trajectories(last_frame)
      self.debug_unreliable_trajectories(last_frame)
      print('debug_reliable_trajectories:', time.time() - t1)
    
    t_mot_done = time.time()
    self.dt_update_one.append(t_mot_done - t_mot_start)


  def debug_trajectories(self, last_frame):
    trajectory_set = self.trajectory_manager.get_trajectory_set()
    trajectories = trajectory_set.get_live_trajectories()
    if len(trajectories) == 0:
      return
    
    labels = trajectory_set.get_labels(trajectories)
    self.trajectory_visualizer.publish_trajectories_on_image(
      last_frame, trajectories, labels)
    
    self.trajectory_visualizer.publish_trajectories_as_markers(
      trajectories, labels, P.base_frame)

  def debug_reliable_trajectories(self, last_frame):
    trajectory_set = self.trajectory_manager.get_trajectory_set()
    valid_trajectories = trajectory_set.get_trajectories_by_state(
      m_fa.Trajectory3DSet.LIFE_CONFIRMED, min_count_thresh=P.debug_trajectory_min_det3d_count)
    if len(valid_trajectories) == 0:
      print('oh no!')
      return
    
    labels = trajectory_set.get_labels(valid_trajectories)
    self.trajectory_visualizer.publish_valid_trajectories_on_image(
      last_frame, valid_trajectories, labels)
    # self.trajectory_visualizer.publish_valid_trajectories_as_markers( # need to improve performance
    #   valid_trajectories, labels, P.base_frame)
    
  def debug_unreliable_trajectories(self, last_frame):
    trajectory_set = self.trajectory_manager.get_trajectory_set()
    unreliable_trajectories = trajectory_set.get_trajectories_by_state(
      m_fa.Trajectory3DSet.LIFE_ALIVE, min_count_thresh=P.debug_trajectory_min_det3d_count)
    if len(unreliable_trajectories) == 0:
      return
    
    labels = trajectory_set.get_labels(unreliable_trajectories)
    self.trajectory_visualizer.publish_unreliable_trajectories_on_image(
      last_frame, unreliable_trajectories, labels)
    # self.trajectory_visualizer.publish_unreliable_trajectories_as_markers( # need to improve performance
    #   unreliable_trajectories, labels, P.base_frame)
    

  def update_trajectory(self, det3ds_list):
    with self.traj_lock:
      for det3ds_in_a_frame in det3ds_list:
        ts = time.time()
        _, _ = self.trajectory_manager.update_with_det3ds_in_one_frame(
          det3ds_in_a_frame, self.cost_metric)
        self.dt_trajectory_update.append(time.time() - ts)


  def publish_det3d_on_image(self, cv_img, det3ds):
    pass

  def publish_det3d_as_markers(self, det3ds):
    pass

  def create_det3ds(self, synced_frames):
    detections3d_world = []
    for frame_idx, frame in enumerate(synced_frames):
      mf_depth_data   = frame.depth
      mf_det2ds_data  = frame.detections
      mf_odom_data    = frame.odom

      depth_cv = mf_depth_data.image
      depth_cam_K = np.array(mf_depth_data.K, np.float32).reshape(3, 3)

      # preprocess depth
      depth_cv_preprocessed = m_fa.preprocess_depth(
        depth_cv, P.depth_min_mm, P.depth_max_mm, P.depth_unit)
      
      # tf
      pose = mf_odom_data.pose
      quat = mf_odom_data.orientation
      rot_mat = Rotation.from_quat(quat).as_matrix()

      transform_world_to_cam = np.zeros((4, 4), np.float32)
      transform_world_to_cam[:3, :3] = rot_mat
      transform_world_to_cam[:3, 3] = pose
      transform_world_to_cam[3, 3] = 1

      # create 3D detection
      detections3d_world.append([])
      for det2d in mf_det2ds_data:
        success, det3d = self.det3d_creator.create(
          det2d, depth_cv_preprocessed, depth_cam_K, None)
        if not success: 
          if P.detection3d_verbose:
            print('det3d generation failed: ', det3d)
          continue
        det3d.add_metadata(det2d, '', -1)
        det3d = m_fa.convert_with_tf(det3d, transform_world_to_cam, P.base_frame)
        detections3d_world[-1].append(det3d)

    return detections3d_world

  def add_cv_image_to_queue(self, queue, sec, nsec, cv_image, K, D, lock):
    if self.rgb_K is None or self.rgb_D is None:
      print('no rgb camera info!')
      return
    if self.depth_K is None or self.depth_D is None:
      print('no depth rgb camera info!')
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
    self.rgb_K = np.array(msg.k, np.float32).reshape(3, 3)
    self.rgb_D = np.array(msg.d, np.float32)
    self.destroy_subscription(self.get_subscriber(P.rgb_info_topic))

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
    self.depth_K = np.array(msg.k, np.float32).reshape(3, 3)
    self.depth_D = np.array(msg.d, np.float32)
    self.destroy_subscription(self.get_subscriber(P.depth_info_topic))

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
        print('Error in odometry_callback:', e)
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

def main(args=None):
  rclpy.init(args=args)
  trajectory_visualizer =None
  if P.debug_det3d or P.debug_trajectory_update:
    trajectory_visualizer = mot3d_visualizer.TrajectoryVisualizer(P.labels_to_names_and_colors)
  frame_generator = MOTbyYOLO(trajectory_visualizer=trajectory_visualizer)
  rclpy.spin(frame_generator)

  frame_generator.destroy_node()

if __name__ == '__main__':
  main()
