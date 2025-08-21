from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from cv_bridge import CvBridge

from geometry_msgs.msg import Point
from rclpy.node import Node
import cv2
from scipy.spatial.transform import Rotation as R
import os
import pickle
import mflib.perception.mot.common as common
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
from sensor_msgs.msg import CompressedImage
import seaborn

def quaternion_from_euler(ai, aj, ak):
  ai /= 2.0
  aj /= 2.0
  ak /= 2.0
  ci = math.cos(ai)
  si = math.sin(ai)
  cj = math.cos(aj)
  sj = math.sin(aj)
  ck = math.cos(ak)
  sk = math.sin(ak)
  cc = ci*ck
  cs = ci*sk
  sc = si*ck
  ss = si*sk

  q = np.empty((4, ))
  q[0] = cj*sc - sj*cs
  q[1] = cj*ss + sj*cc
  q[2] = cj*cs - sj*sc
  q[3] = cj*cc + sj*ss

  return q


def visualize_det2ds_on_cv_image(im, det2ds, labels_to_names_and_colors):
  if len(im.shape) == 2:
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
  for det2d in det2ds:
    color = labels_to_names_and_colors[det2d.label]['color']
    label_str = labels_to_names_and_colors[det2d.label]['name']
    im = det2d.draw_on_image(im, color=color, thickness=2, text=label_str)
  return im

def visualize_det3ds_on_image_offline(detection3d_output_dir):
  # sync frames and det3ds
  N_PROCESS = 500
  det3ds_names = os.listdir(detection3d_output_dir)
  det3ds_names.sort()

  det3ds = []
  for name in det3ds_names:
    det3ds_tmp = pickle.load(open(f'{detection3d_output_dir}/{name}', 'rb'))
    det3ds.extend(det3ds_tmp)

  # divide deections3d by frame
  det3ds_in_a_frame = []
  det3ds_divided = []

  # debug
  frames = []
  frame_getter = common.FrameGetter()

  last_det3d_ts = None
  for det3d in det3ds:
    if len(det3ds_divided) >= N_PROCESS:
      break
    if last_det3d_ts is None:
      last_det3d_ts = det3d.ts
      frame = frame_getter.get_frame(
        det3d.framearray_file,
        det3d.framearray_index
        )
      frames.append(frame)
    elif det3d.ts != last_det3d_ts:
      det3ds_divided.append(det3ds_in_a_frame)
      det3ds_in_a_frame = []
      last_det3d_ts = det3d.ts
      frame = frame_getter.get_frame(
        det3d.framearray_file,
        det3d.framearray_index
        )
      frames.append(frame)
    det3ds_in_a_frame.append(det3d)
    # image

  det3ds_divided.append(det3ds_in_a_frame)
  print('length of images: ', len(frames))
  print('det3ds: ', len(det3ds_divided))
  frames_and_det3ds = list(zip(frames, det3ds_divided))

  for frame, det3ds in frames_and_det3ds:
    rgb_cv = frame.rgb.image

    # draw det3d
    det3d_drawn = np.copy(rgb_cv)
    for det3d in det3ds:
      det2d = det3d.det2d
      det3d_drawn = det2d.draw_on_image(det3d_drawn)
    cv2.putText(det3d_drawn, f'3D detections (filtered 2D detections)', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
    # draw raw det2d
    det2ds = frame.detections

    det2d_drawn = np.copy(rgb_cv)
    for det2d in det2ds:
      det2d_drawn = det2d.draw_on_image(det2d_drawn)
    cv2.putText(det2d_drawn, f'2D detections', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

    depth_cv = frame.depth.image
    # convert with jet
    depth_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_cv, alpha=0.03), cv2.COLORMAP_JET)
    # draw det2ds
    depth_cv_2ddrawn = np.copy(depth_cv)
    for det2d in det2ds:
      depth_cv_2ddrawn = det2d.draw_on_image(depth_cv_2ddrawn)
    cv2.putText(depth_cv_2ddrawn, f'2D detections', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

    depth_cv_3ddrawn = np.copy(depth_cv)
    for det3d in det3ds:
      det2d = det3d.det2d
      depth_cv_3ddrawn = det2d.draw_on_image(depth_cv_3ddrawn)
    cv2.putText(depth_cv_3ddrawn, f'3D detections (filtered 2D detections)', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

    H, W = rgb_cv.shape[:2]
    im_to_show = np.zeros((H*2, W*2, 3), dtype=np.uint8)

    im_to_show[:H, :W] = det2d_drawn
    im_to_show[H:, :W] = det3d_drawn
    im_to_show[:H, W:] = depth_cv_2ddrawn
    im_to_show[H:, W:] = depth_cv_3ddrawn

    cv2.imshow('det2d', im_to_show)
    cv2.waitKey(0)

class TrajectoryVisualizer(Node):
  def __init__(self, labels_to_names_and_colors, node_name='trajectory_visualizer'):
    super().__init__(node_name)
    self.labels_to_names_and_colors = labels_to_names_and_colors

    self.det3d_on_image_pub = self.create_publisher(CompressedImage, '~/det3d_on_image/compressed', 10)
    self.det3d_cube_pub = self.create_publisher(MarkerArray, '~/det3d_cubes', 10)

    self.valid_trajectory_image_pub = self.create_publisher(CompressedImage, '~/valid_trajectory_image/compressed', 10)
    self.valid_trajectory_cube_pub = self.create_publisher(MarkerArray, '~/valid_trajectory_cubes', 10)

    self.unreliable_trajectory_image_pub = self.create_publisher(CompressedImage, '~/unreliable_trajectory_image/compressed', 10)
    self.unreliable_trajectory_cube_pub = self.create_publisher(MarkerArray, '~/unreliable_trajectory_cubes', 10)

    self.trajectory_image_pub = self.create_publisher(CompressedImage, '~/trajectory_image/compressed', 10)
    self.trajectory_cube_pub = self.create_publisher(MarkerArray, '~/trajectory_cubes', 10)

    self.camera_fov_pub = self.create_publisher(MarkerArray, '~/camera_fov', 10)

    self.tf_broadcaster = TransformBroadcaster(self)
    self.cv_bridge = CvBridge()

    self.label_palette = seaborn.color_palette('Paired', 50)

  def draw_det3d_on_image(self, im, det3ds):
    det2ds = [det3d.det2d for det3d in det3ds]
    im = visualize_det2ds_on_cv_image(np.copy(im), det2ds, self.labels_to_names_and_colors)
    return im

  def publish_det3d_on_image(self, im, det3ds):
    im = self.draw_det3d_on_image(im, det3ds)
    msg = self.cv_bridge.cv2_to_compressed_imgmsg(im)
    self.det3d_on_image_pub.publish(msg)

  def publish_det3ds_as_markers(self, det3ds, frame_id='map'):
    m_det3d_array = MarkerArray()
    # check type of det2ds
    if len(det3ds) > 0:
      for idx, det3d in enumerate(det3ds):
        # rgb = self.labels_to_names_and_colors[det3d.det2d.label]['color']
        rgb = (1, 0, 0)
        rgba = (*rgb, 0.5)
        marker = self.det3d_to_marker(det3d, idx, rgba, frame_id)
        m_det3d_array.markers.append(marker)
      self.det3d_cube_pub.publish(m_det3d_array)

  # def draw_trajectories_on_image(self, frame, trajectories, labels):
  #   im = frame.rgb.image
  #   if len(trajectories) == 0:
  #     return im
  #   intrinsics = frame.depth.K

  #   extrinsic4x4 = np.eye(4)
  #   rot_mat = R.from_quat(frame.odom.orientation).as_matrix()
  #   extrinsic4x4[:3, :3] = rot_mat
  #   extrinsic4x4[:3, 3] = frame.odom.pose

  #   points_3ds_list = []
  #   for traj in trajectories:
  #     xyz = traj['xyzwlh'][:3]
  #     wlh = traj['xyzwlh'][3:]

  #     points_3ds = [
  #       [xyz[0] - wlh[0] / 2, xyz[1] - wlh[1] / 2, xyz[2] - wlh[2] / 2],
  #       [xyz[0] + wlh[0] / 2, xyz[1] - wlh[1] / 2, xyz[2] - wlh[2] / 2],
  #       [xyz[0] + wlh[0] / 2, xyz[1] + wlh[1] / 2, xyz[2] - wlh[2] / 2],
  #       [xyz[0] - wlh[0] / 2, xyz[1] + wlh[1] / 2, xyz[2] - wlh[2] / 2],
  #       [xyz[0] - wlh[0] / 2, xyz[1] - wlh[1] / 2, xyz[2] + wlh[2] / 2],
  #       [xyz[0] + wlh[0] / 2, xyz[1] - wlh[1] / 2, xyz[2] + wlh[2] / 2],
  #       [xyz[0] + wlh[0] / 2, xyz[1] + wlh[1] / 2, xyz[2] + wlh[2] / 2],
  #       [xyz[0] - wlh[0] / 2, xyz[1] + wlh[1] / 2, xyz[2] + wlh[2] / 2],
  #     ]
  #     points_3ds_list.append(points_3ds)
  #   points_3ds_list = np.array(points_3ds_list)

  #   points_2ds_np = np.zeros((len(trajectories), 8, 2))
  #   for idx, points_3ds in enumerate(points_3ds_list):
  #     points_3d_homo = np.hstack((points_3ds, np.ones((points_3ds.shape[0], 1))))
  #     points_camera = np.linalg.inv(extrinsic4x4) @ points_3d_homo.T
  #     points_camera /= points_camera[2, :]  # Normalize by z-axis (depth)
  #     points_image = intrinsics @ points_camera[:3, :]
  #     points_image /= points_image[2, :]
  #     points_2ds_np[idx, ...] = points_image[:2, :].T
  #   points_2ds_np = points_2ds_np.astype(np.int32)

  #   max_conf = 0.7
  #   im = (im.astype(np.float32) - 50) # increase alpha
  #   im = np.clip(im, 0, 255).astype(np.uint8)
  #   for pts2d, traj, label in zip(points_2ds_np, trajectories, labels):
  #     # if not only_one_label:
  #     #   color = self.labels_to_names_and_colors[label]['color']
  #     #   color = np.array(color, np.float32)
  #     # else:
  #     color_idx = traj['trajectory_id_if_confirmed'] % 20
  #     color = self.label_palette[color_idx]
  #     color = tuple(int(c * 255) for c in color[::-1])  # RGB를 BGR로 변환

  #     name = f"{traj['trajectory_id_if_confirmed']}"

  #     factor = min(traj['score'], max_conf) / max_conf
  #     alpha_int = int(factor*255)
  #     x_min = np.min(pts2d[:, 0])
  #     x_max = np.max(pts2d[:, 0])
  #     y_min = np.min(pts2d[:, 1])
  #     y_max = np.max(pts2d[:, 1])
  #     if x_min < 0 or y_min < 0 or x_max >= im.shape[1] or y_max >= im.shape[0]:
  #       continue

  #     im = cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color + (alpha_int,), -1)

  #     # # putText
  #     cv2.putText(im, f'{name}', (x_min, y_max), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)

  #   return im

  def draw_trajectories_on_image(self, frame, trajectories, labels, monotone=False):
    im = np.copy(frame.rgb.image)
    # im = (im.astype(np.float32) - 50) # decrease brightness
    # im = np.clip(im, 0, 255).astype(np.uint8)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    if len(trajectories) == 0:
      return im
    intrinsics = frame.depth.K

    extrinsic4x4 = np.eye(4)
    extrinsic4x4[:3, :3] = R.from_quat(frame.odom.orientation).as_matrix()
    extrinsic4x4[:3, 3] = frame.odom.pose
    extrinsic_inv = np.linalg.inv(extrinsic4x4)

    box_offsets = np.array([
      [-1, -1, -1],
      [ 1, -1, -1],
      [ 1,  1, -1],
      [-1,  1, -1],
      [-1, -1,  1],
      [ 1, -1,  1],
      [ 1,  1,  1],
      [-1,  1,  1]
    ]) * 0.5

    # Compute 3D bounding box corners (N, 8, 3)
    points_3ds = trajectories['xyzwlh'][:, None, :3] + box_offsets * trajectories['xyzwlh'][:, None, 3:]
    # Convert to homogeneous coordinates (N, 8, 4)
    points_3ds_h = np.concatenate([points_3ds, np.ones((*points_3ds.shape[:2], 1))], axis=-1)

    # Transform to camera coordinates
    points_cam_h = points_3ds_h @ extrinsic_inv.T
    points_cam = points_cam_h[..., :3] / points_cam_h[..., 3:4]  # Normalize
    # Project to image plane
    points_2d_h = points_cam @ intrinsics.T
    points_2d = points_2d_h[..., :2] / points_2d_h[..., 2:3]  # Normalize by z
    points_2d = points_2d.astype(np.int32)

    # check minmax
    x_min = np.min(points_2d[:, :, 0], axis=1)  # n_traj
    y_min = np.min(points_2d[:, :, 1], axis=1)  # n_traj
    x_max = np.max(points_2d[:, :, 0], axis=1)  # n_traj
    y_max = np.max(points_2d[:, :, 1], axis=1)  # n_traj
    tlbr = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.int32)  # (n_traj, 4)

    image_h, image_w = im.shape[:2]
    invalid1 = tlbr[..., 0] < 0
    invalid2 = tlbr[..., 1] < 0
    invalid3 = tlbr[..., 2] >= image_w
    invalid4 = tlbr[..., 3] >= image_h
    invalid = np.logical_or(invalid1, invalid2)
    invalid = np.logical_or(invalid, invalid3)
    invalid = np.logical_or(invalid, invalid4)
    if np.all(invalid):
      return im

    valid = np.logical_not(invalid)
    valid_tlbr = tlbr[valid]
    valid_trajectories = trajectories[valid]

    max_conf = 0.7
    for (tl_x, tl_y, br_x, br_y), traj in zip(valid_tlbr, valid_trajectories):
      # if not only_one_label:
      #   color = self.labels_to_names_and_colors[label]['color']
      #   color = np.array(color, np.float32)
      # else:
      color_idx = traj['trajectory_id_if_confirmed'] % len(self.label_palette)
      color = self.label_palette[color_idx]
      color = tuple(int(c * 255) for c in color[::-1])  # RGB를 BGR로 변환
      if traj['trajectory_id_if_confirmed'] == -1:
        color = (139, 0, 139)
      name = f"{traj['trajectory_id_if_confirmed']}"

      im = cv2.rectangle(im, (tl_x, tl_y), (br_x, br_y), color, 3)

      if monotone == False:
        cv2.putText(im, f'{name}', (tl_x, br_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)

    return im

  def publish_trajectories_on_image(self, frame, trajectories, labels):
    self._publish_trajectories_on_image(frame, trajectories, labels, pub_type='all')

  def publish_valid_trajectories_on_image(self, frame, trajectories, labels):
    self._publish_trajectories_on_image(frame, trajectories, labels, pub_type='valid')

  def publish_unreliable_trajectories_on_image(self, frame, trajectories, labels):
    self._publish_trajectories_on_image(frame, trajectories, labels, pub_type='unreliable')

  def _publish_trajectories_on_image(self, frame, trajectories, labels, pub_type='valid'):
    im = self.draw_trajectories_on_image(frame, trajectories, labels)
    msg = self.cv_bridge.cv2_to_compressed_imgmsg(im)
    if pub_type == 'valid':
      self.valid_trajectory_image_pub.publish(msg)
    elif pub_type == 'unreliable':
      self.unreliable_trajectory_image_pub.publish(msg)
    elif pub_type == 'all':
      self.trajectory_image_pub.publish(msg)

  def publish_camera_fov_from_frame(self, odom, image_wh, depth_of_interest_meter, intrinsic, frame_id='map'):
    # algorithm: calculate 5 point of camera FOV with fixed_depth:=P.det3d_params['max_dist_from_camera']
    # then, connect these points to draw FOV

    # get extrinsic R, T
    rotation_mat = R.from_quat(odom.orientation).as_matrix().T
    # translation = odom.pose.T
    translation = -rotation_mat @ odom.pose

    # calculate 4 points of FOV square
    pts_uv_homogeneous = np.array([
      [0, 0, 1],
      [image_wh[0]-1, 0, 1],
      [0, image_wh[1]-1, 1],
      [image_wh[0]-1, image_wh[1]-1, 1],
    ]).T
    # compose mat [translation transtion translation transtion]
    translation_extended = np.tile(translation, (4, 1)).T # 4 from pts_uv_homogeneous
    pts_fov_camera = depth_of_interest_meter * np.linalg.inv(intrinsic) @ pts_uv_homogeneous
    pts_fov_world = rotation_mat.T @ (pts_fov_camera - translation_extended)
    pts_fov_world = pts_fov_world.T
    pts_fov_tl = pts_fov_world[0]
    pts_fov_tr = pts_fov_world[1]
    pts_fov_bl = pts_fov_world[2]
    pts_fov_br = pts_fov_world[3]
    pts_fov_orig = -rotation_mat.T @ translation
    # compose marker
    m = Marker()
    m.header.frame_id = frame_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.01
    m.color.r = 1.0
    m.color.g = 0.0
    m.color.b = 0.0
    m.color.a = 1.0
    m.points = [
      self.numpy_to_point(pts_fov_orig),
      self.numpy_to_point(pts_fov_tl),
      self.numpy_to_point(pts_fov_orig),
      self.numpy_to_point(pts_fov_tr),
      self.numpy_to_point(pts_fov_orig),
      self.numpy_to_point(pts_fov_bl),
      self.numpy_to_point(pts_fov_orig),
      self.numpy_to_point(pts_fov_br),
      self.numpy_to_point(pts_fov_tl),
      self.numpy_to_point(pts_fov_tr),
      self.numpy_to_point(pts_fov_tr),
      self.numpy_to_point(pts_fov_br),
      self.numpy_to_point(pts_fov_br),
      self.numpy_to_point(pts_fov_bl),
      self.numpy_to_point(pts_fov_bl),
      self.numpy_to_point(pts_fov_tl),
    ]
    m_camera_fov_array = MarkerArray()
    m_camera_fov_array.markers.append(m)
    self.camera_fov_pub.publish(m_camera_fov_array)

    t = TransformStamped()

    # Read message content and assign it to
    # corresponding tf variables
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = 'map'
    t.child_frame_id = 'camera'

    # Turtle only exists in 2D, thus we get x and y translation
    # coordinates from the message and set the z coordinate to 0
    t.transform.translation.x = float(odom.pose[0])
    t.transform.translation.y = float(odom.pose[1])
    t.transform.translation.z = float(odom.pose[2])

    # For the same reason, turtle can only rotate around one axis
    # and this why we set rotation in x and y to 0 and obtain
    # rotation in z axis from the message
    t.transform.rotation.x = float(odom.orientation[0])
    t.transform.rotation.y = float(odom.orientation[1])
    t.transform.rotation.z = float(odom.orientation[2])
    t.transform.rotation.w = float(odom.orientation[3])

    # Send the transformation
    self.tf_broadcaster.sendTransform(t)

  def numpy_to_point(self, np_array):
    p = Point()
    p.x = np_array[0]
    p.y = np_array[1]
    p.z = np_array[2]
    return p

  def publish_trajectories_as_markers(self, trajectories, labels, frame_id):
    self._publish_trajectory(trajectories, labels, frame_id, pub_type='all')

  def publish_valid_trajectories_as_markers(self, trajectories, labels, frame_id):
    self._publish_trajectory(trajectories, labels, frame_id, pub_type='valid')

  def publish_unreliable_trajectories_as_markers(self, trajectories, labels, frame_id):
    self._publish_trajectory(trajectories, labels, frame_id, pub_type='unreliable')

  def _publish_trajectory(self, trajectories, labels, frame_id, pub_type='valid'):
    m_trajectory_array = MarkerArray()
    # add deleteall marker
    m = Marker()
    m.header.frame_id = 'map'
    m.action = Marker.DELETEALL
    m_trajectory_array.markers.append(m)

    for traj, label in zip(trajectories, labels):
      marker, text_marker = self.trajectory_to_marker(
        traj, label, traj['trajectory_id_if_confirmed'], frame_id)
      m_trajectory_array.markers.append(marker)
      m_trajectory_array.markers.append(text_marker)

    if pub_type == 'valid':
      self.valid_trajectory_cube_pub.publish(m_trajectory_array)
    elif pub_type == 'unreliable':
      self.unreliable_trajectory_cube_pub.publish(m_trajectory_array)
    elif pub_type == 'all':
      self.trajectory_cube_pub.publish(m_trajectory_array)



  def trajectory_to_marker(self, traj, label, mid, frame_id='map'):
    traj_id = traj['trajectory_id_if_confirmed']
    name = f"{traj_id}"
    # if not only_one_label:
    #   rgb = self.labels_to_names_and_colors[label]['color']
    #   rgb = np.array(rgb, dtype=np.float32)/255
    # else:
    color_idx = traj_id % len(self.label_palette)
    rgb = self.label_palette[color_idx]
    rgb = tuple(c for c in rgb)

    marker = Marker()
    marker.ns = 'trajectories'
    marker.id = int(mid)
    marker.header.frame_id = frame_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(traj['xyzwlh'][0])
    marker.pose.position.y = float(traj['xyzwlh'][1])
    marker.pose.position.z = float(traj['xyzwlh'][2])
    marker.scale.x = float(traj['xyzwlh'][3])
    marker.scale.y = float(traj['xyzwlh'][4])
    marker.scale.z = float(traj['xyzwlh'][5])
    marker.color.r = float(rgb[0])
    marker.color.g = float(rgb[1])
    marker.color.b = float(rgb[2]) # reversed
    marker.color.a = 1.0

    text_marker = Marker()
    scale = np.sum(traj['xyzwlh'][3:]) / 3 * 1.5
    text_marker.ns = 'trajectories_text'
    text_marker.id = int(mid) + 100000000 # I hope this id become unique...
    text_marker.header.frame_id = frame_id
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = float(traj['xyzwlh'][0])
    text_marker.pose.position.y = float(traj['xyzwlh'][1])
    text_marker.pose.position.z = float(traj['xyzwlh'][2] + scale)
    text_marker.scale.z = float(scale)
    text_marker.color.r = float(rgb[0])
    text_marker.color.g = float(rgb[1])
    text_marker.color.b = float(rgb[2])
    text_marker.color.a = 1.0
    text_marker.text = name

    return marker, text_marker

  def det3d_to_marker(self, det3d, mid, rgba, frame_id='map'):
    marker = Marker()
    marker.ns = 'det3d'
    marker.id = mid
    marker.header.frame_id = frame_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(det3d.xyzwlh[0])
    marker.pose.position.y = float(det3d.xyzwlh[1])
    marker.pose.position.z = float(det3d.xyzwlh[2])
    marker.scale.x = float(det3d.xyzwlh[3])
    marker.scale.y = float(det3d.xyzwlh[4])
    marker.scale.z = float(det3d.xyzwlh[5])
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])

    return marker

if __name__ == '__main__':
  # visualize_det2ds_on_image()
  # visualize_det3ds_on_image()
  vis = TrajectoryVisualizer()
