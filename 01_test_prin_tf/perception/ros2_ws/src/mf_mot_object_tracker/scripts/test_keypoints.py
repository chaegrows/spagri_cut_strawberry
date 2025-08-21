#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import numpy as np
import cv2
import os
from datetime import datetime
import math
from ultralytics import SAM
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf_transformations import quaternion_from_euler
import tf2_geometry_msgs
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut

class StableFruitKeypointTFNode(Node):
  def __init__(self, node_name='stable_fruit_keypoint_tf_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    self.save_counter = 0
    self.save_dir = "/workspace/ripe_fruit_crops"
    
    # Camera parameters (from params file)
    self.base_frame = 'camera_hand_link'
    self.depth_frame = 'camera_hand_depth_optical_frame'
    self.depth_min_mm = 70
    self.depth_max_mm = 700
    
    # Camera intrinsics (will be updated from camera_info)
    self.fx = 525.0  # Default values
    self.fy = 525.0
    self.cx = 320.0
    self.cy = 240.0
    self.camera_info_received = False
    
    # Stability parameters
    self.declare_parameter('smoothing_window', 5)  # Temporal smoothing window size
    self.declare_parameter('confidence_threshold', 0.7)  # Minimum confidence for PCA
    self.declare_parameter('angle_change_threshold', 15.0)  # Max angle change per frame (degrees)
    self.declare_parameter('position_change_threshold', 0.05)  # Max position change per frame (meters)
    self.declare_parameter('outlier_rejection_factor', 2.0)  # Factor for outlier rejection
    
    # Tracking parameters
    self.declare_parameter('tracking_distance_threshold', 0.1)  # Max distance to associate fruits (meters)
    self.declare_parameter('max_missing_frames', 10)  # Max frames before removing a track
    
    # Get parameters
    self.smoothing_window = self.get_parameter('smoothing_window').get_parameter_value().integer_value
    self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
    self.angle_change_threshold = self.get_parameter('angle_change_threshold').get_parameter_value().double_value
    self.position_change_threshold = self.get_parameter('position_change_threshold').get_parameter_value().double_value
    self.outlier_rejection_factor = self.get_parameter('outlier_rejection_factor').get_parameter_value().double_value
    self.tracking_distance_threshold = self.get_parameter('tracking_distance_threshold').get_parameter_value().double_value
    self.max_missing_frames = self.get_parameter('max_missing_frames').get_parameter_value().integer_value
    
    # History tracking for each fruit
    self.fruit_history = {}  # fruit_id -> history data
    
    # Tracking system
    self.fruit_tracks = {}  # track_id -> track info
    self.next_track_id = 0
    self.frame_count = 0
    
    # TF broadcaster
    self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    # Publisher for enhanced debug images
    self.enhanced_debug_pub = self.create_publisher(
      CompressedImage, 
      '/mf_perception/yolo_debug_enhanced/compressed', 
      1
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(self.save_dir, exist_ok=True)
    
    # Subscribe to keypoint output, RGB image, depth image, and YOLO debug
    self.keypoint_subscription = self.create_subscription(
      KeypointOutArray,
      '/mf_perception/keypoint_out',
      self.keypoint_callback,
      10
    )
    
    # Also subscribe to bbox for additional info
    self.bbox_subscription = self.create_subscription(
      YoloOutArray,
      '/mf_perception/bbox_out',
      self.bbox_callback,
      10
    )
    
    self.image_subscription = self.create_subscription(
      CompressedImage,
      '/camera/camera_hand/color/image_rect_raw/compressed',
      self.image_callback,
      10
    )
    
    # Subscribe to depth image
    self.depth_subscription = self.create_subscription(
      Image,
      '/camera/camera_hand/depth/image_rect_raw',
      self.depth_callback,
      10
    )
    
    # Subscribe to camera info for accurate intrinsics
    self.camera_info_subscription = self.create_subscription(
      CameraInfo,
      '/camera/camera_hand/color/camera_info',
      self.camera_info_callback,
      10
    )
    
    # Subscribe to YOLO debug image for enhancement
    self.yolo_debug_subscription = self.create_subscription(
      CompressedImage,
      '/mf_perception/yolo_debug/compressed',
      self.yolo_debug_callback,
      10
    )
    
    self.latest_image = None
    self.latest_depth = None
    self.latest_header = None
    self.latest_yolo_debug = None
    self.latest_keypoints = None
    self.latest_bboxes = None
    self.latest_keypoint_analysis = {}  # Store keypoint analysis results
    
    print('Stable fruit keypoint TF node is initialized')
    print(f'Smoothing window: {self.smoothing_window}')
    print(f'Confidence threshold: {self.confidence_threshold}')
    print(f'Angle change threshold: {self.angle_change_threshold}°')
    print(f'Position change threshold: {self.position_change_threshold}m')
    print(f'Tracking distance threshold: {self.tracking_distance_threshold}m')
    print(f'Max missing frames: {self.max_missing_frames}')
    print(f'Initial camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
    print('Waiting for camera_info to update intrinsics...')

  def image_callback(self, msg):
    """Store the latest RGB image for processing"""
    self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.latest_header = msg.header

  def depth_callback(self, msg):
    """Store the latest depth image"""
    self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

  def camera_info_callback(self, msg):
    """Update camera intrinsic parameters from camera_info"""
    if not self.camera_info_received:
      self.fx = msg.k[0]  # K[0,0]
      self.fy = msg.k[4]  # K[1,1]  
      self.cx = msg.k[2]  # K[0,2]
      self.cy = msg.k[5]  # K[1,2]
      self.camera_info_received = True
      
      print(f"Camera intrinsics updated:")
      print(f"  fx: {self.fx:.2f}, fy: {self.fy:.2f}")
      print(f"  cx: {self.cx:.2f}, cy: {self.cy:.2f}")
      print(f"  Image size: {msg.width}x{msg.height}")
      
      # Check for distortion
      if any(abs(d) > 0.01 for d in msg.d[:4]):  # Check first 4 distortion coefficients
        print(f"  Distortion coefficients: {msg.d[:5]}")
        print("  WARNING: Camera has significant distortion!")
      else:
        print("  Camera distortion is minimal")

  def keypoint_callback(self, msg):
    """Store the latest keypoint detection results"""
    self.latest_keypoints = msg
    print(f"Received {len(msg.keypoints)} keypoint detections")
    # Process keypoints if both bbox and keypoints are available
    if self.latest_bboxes is not None:
      self.process_keypoint_detections()

  def yolo_debug_callback(self, msg):
    """Process YOLO debug image and add keypoint analysis visualization"""
    self.latest_yolo_debug = self.bridge.compressed_imgmsg_to_cv2(msg)
    
    # Enhance the debug image with keypoint analysis if available
    if self.latest_keypoint_analysis:
      enhanced_img = self.draw_keypoint_analysis_on_debug_image(
        self.latest_yolo_debug.copy(), 
        self.latest_keypoint_analysis
      )
      
      # Publish enhanced debug image
      enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(enhanced_img, dst_format='jpg')
      enhanced_msg.header = msg.header
      self.enhanced_debug_pub.publish(enhanced_msg)

  # def extract_depth_from_bbox_robust(self, depth_image, bbox):
  #   """Extract reliable depth value with enhanced robustness"""
  #   if depth_image is None or bbox is None:
  #     return None
    
  #   x1, y1, x2, y2 = bbox
    
  #   # Crop depth region
  #   depth_roi = depth_image[y1:y2, x1:x2]
    
  #   if depth_roi.size == 0:
  #     return None
    
  #   # Convert to mm and filter valid range
  #   depth_mm = depth_roi.astype(np.float32)
    
  #   # Filter depth values within valid range
  #   valid_mask = (depth_mm >= self.depth_min_mm) & (depth_mm <= self.depth_max_mm)
  #   valid_depths = depth_mm[valid_mask]
    
  #   if len(valid_depths) == 0:
  #     return None
    
  #   # Enhanced outlier removal using multiple methods
  #   # 1. IQR method
  #   q1 = np.percentile(valid_depths, 25)
  #   q3 = np.percentile(valid_depths, 75)
  #   iqr = q3 - q1
    
  #   lower_bound = q1 - 1.5 * iqr
  #   upper_bound = q3 + 1.5 * iqr
    
  #   iqr_filtered = valid_depths[
  #     (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
  #   ]
    
  #   # 2. Z-score method for additional filtering
  #   if len(iqr_filtered) > 3:
  #     mean_depth = np.mean(iqr_filtered)
  #     std_depth = np.std(iqr_filtered)
  #     z_scores = np.abs((iqr_filtered - mean_depth) / (std_depth + 1e-8))
  #     z_filtered = iqr_filtered[z_scores < 2.0]
      
  #     if len(z_filtered) > 0:
  #       final_depths = z_filtered
  #     else:
  #       final_depths = iqr_filtered
  #   else:
  #     final_depths = iqr_filtered
    
  #   if len(final_depths) == 0:
  #     return None
    
  #   # Use median as the representative depth (more robust than mean)
  #   median_depth_mm = np.median(final_depths)
    
  #   return {
  #     'depth_mm': median_depth_mm,
  #     'depth_m': median_depth_mm / 1000.0,
  #     'valid_pixel_count': len(valid_depths),
  #     'filtered_pixel_count': len(final_depths),
  #     'depth_std': np.std(final_depths),
  #     'confidence': min(1.0, len(final_depths) / len(valid_depths))  # Confidence based on filtering ratio
  #   }

  def extract_depth_from_keypoints_line(self, depth_image, keypoint1, keypoint2):
    """
    Extract depth values from pixels along the line between two keypoints
    Args:
      depth_image: Depth image
      keypoint1: (x, y) coordinates of first keypoint (yellow)
      keypoint2: (x, y) coordinates of second keypoint (red)
    Returns:
      Dictionary with depth information or None
    """
    if depth_image is None or keypoint1 is None or keypoint2 is None:
      return None
    
    x1, y1 = int(keypoint1[0]), int(keypoint1[1])
    x2, y2 = int(keypoint2[0]), int(keypoint2[1])
    
    # Get image dimensions
    h, w = depth_image.shape
    
    # Generate line pixels using Bresenham's line algorithm
    line_pixels = []
    
    # Calculate line parameters
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx == 0 and dy == 0:
      # Same point
      if 0 <= x1 < w and 0 <= y1 < h:
        line_pixels = [(x1, y1)]
      else:
        return None
    else:
      # Use numpy linspace to get line points
      num_points = max(dx, dy, 10)  # At least 10 points
      x_coords = np.linspace(x1, x2, num_points)
      y_coords = np.linspace(y1, y2, num_points)
      
      for x, y in zip(x_coords, y_coords):
        px, py = int(round(x)), int(round(y))
        if 0 <= px < w and 0 <= py < h:
          line_pixels.append((px, py))
    
    if len(line_pixels) == 0:
      return None
    
    # Extract depth values from line pixels
    depth_values = []
    for px, py in line_pixels:
      depth_val = depth_image[py, px]
      depth_values.append(depth_val)
    
    depth_values = np.array(depth_values, dtype=np.float32)
    
    # Filter depth values within valid range
    valid_mask = (depth_values >= self.depth_min_mm) & (depth_values <= self.depth_max_mm)
    valid_depths = depth_values[valid_mask]
    
    if len(valid_depths) == 0:
      return None
    
    # Outlier removal using IQR method
    if len(valid_depths) > 3:
      q1 = np.percentile(valid_depths, 25)
      q3 = np.percentile(valid_depths, 75)
      iqr = q3 - q1
      
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      
      iqr_filtered = valid_depths[
        (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
      ]
      
      if len(iqr_filtered) > 0:
        final_depths = iqr_filtered
      else:
        final_depths = valid_depths
    else:
      final_depths = valid_depths
    
    if len(final_depths) == 0:
      return None
    
    # Use median as the representative depth
    median_depth_mm = np.median(final_depths)
    
    return {
      'depth_mm': median_depth_mm,
      'depth_m': median_depth_mm / 1000.0,
      'line_pixel_count': len(line_pixels),
      'valid_pixel_count': len(valid_depths),
      'filtered_pixel_count': len(final_depths),
      'depth_std': np.std(final_depths),
      'confidence': min(1.0, len(final_depths) / len(valid_depths))
    }

  def pixel_to_3d_point(self, u, v, depth_m):
    """Convert pixel coordinates and depth to 3D point in camera frame"""
    # Use updated camera intrinsic parameters
    fx = self.fx
    fy = self.fy
    cx = self.cx
    cy = self.cy
    
    # Convert to 3D coordinates in camera frame
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    
    return x, y, z

  def mask_principal_axis_robust(self, mask):
    """Enhanced PCA with better robustness"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:  # Increased minimum points for better stability
      return None
    
    pts = np.vstack([xs, ys]).T.astype(np.float32)  # (N,2)

    # Remove outlier points using RANSAC-like approach
    if len(pts) > 20:
      # Sample-based outlier removal
      center_estimate = np.median(pts, axis=0)
      distances = np.linalg.norm(pts - center_estimate, axis=1)
      distance_threshold = np.percentile(distances, 85)  # Keep 85% of points
      inlier_mask = distances <= distance_threshold
      pts = pts[inlier_mask]
    
    if len(pts) < 10:
      return None

    # Compute PCA
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean).T)
    
    # Add small regularization to avoid numerical issues
    C += np.eye(2) * 1e-6
    
    evals, evecs = np.linalg.eig(C)
    i_max = np.argmax(evals)
    v = evecs[:, i_max]
    v = v / (np.linalg.norm(v) + 1e-8)
    
    # Calculate confidence based on eigenvalue ratio
    confidence = evals[i_max] / (evals.sum() + 1e-8)
    
    return (float(mean[0]), float(mean[1])), (float(v[0]), float(v[1])), (float(evals[i_max]), float(evals[1 - i_max])), float(confidence)

  def smooth_angle(self, fruit_id, new_angle, confidence):
    """Apply temporal smoothing to angle measurements"""
    if fruit_id not in self.fruit_history:
      self.fruit_history[fruit_id] = {
        'angles': deque(maxlen=self.smoothing_window),
        'positions': deque(maxlen=self.smoothing_window),
        'confidences': deque(maxlen=self.smoothing_window),
        'last_valid_angle': new_angle,
        'last_valid_position': None
      }
    
    history = self.fruit_history[fruit_id]
    
    # Check for outliers based on previous measurements
    if len(history['angles']) > 0:
      last_angle = history['last_valid_angle']
      angle_diff = abs(new_angle - last_angle)
      
      # Handle angle wraparound
      if angle_diff > 180:
        angle_diff = 360 - angle_diff
      
      # Reject outliers if change is too large and confidence is low
      if angle_diff > self.angle_change_threshold and confidence < self.confidence_threshold:
        print(f"  Rejecting angle outlier: {new_angle:.1f}° (diff: {angle_diff:.1f}°, conf: {confidence:.3f})")
        return history['last_valid_angle']
    
    # Add to history
    history['angles'].append(new_angle)
    history['confidences'].append(confidence)
    
    # Weighted average based on confidence
    if len(history['angles']) > 1:
      angles = np.array(history['angles'])
      confidences = np.array(history['confidences'])
      
      # Handle angle wraparound for averaging
      angles_rad = np.radians(angles)
      avg_x = np.average(np.cos(angles_rad), weights=confidences)
      avg_y = np.average(np.sin(angles_rad), weights=confidences)
      smoothed_angle = math.degrees(math.atan2(avg_y, avg_x))
      
      # Normalize to [0, 360)
      if smoothed_angle < 0:
        smoothed_angle += 360
    else:
      smoothed_angle = new_angle
    
    history['last_valid_angle'] = smoothed_angle
    return smoothed_angle

  def smooth_position(self, fruit_id, new_position, confidence):
    """Apply temporal smoothing to position measurements"""
    if fruit_id not in self.fruit_history:
      return new_position
    
    history = self.fruit_history[fruit_id]
    
    # Check for position outliers
    if history['last_valid_position'] is not None:
      last_pos = np.array(history['last_valid_position'])
      new_pos = np.array(new_position)
      position_diff = np.linalg.norm(new_pos - last_pos)
      
      # Reject outliers if change is too large and confidence is low
      if position_diff > self.position_change_threshold and confidence < self.confidence_threshold:
        print(f"  Rejecting position outlier: diff={position_diff:.3f}m, conf={confidence:.3f}")
        return history['last_valid_position']
    
    # Add to history
    history['positions'].append(new_position)
    
    # Weighted average
    if len(history['positions']) > 1:
      positions = np.array(history['positions'])
      # Convert deque to list for slicing
      confidences_list = list(history['confidences'])
      confidences = np.array(confidences_list[-len(positions):])  # Match lengths
      
      smoothed_position = np.average(positions, axis=0, weights=confidences)
      smoothed_position = tuple(smoothed_position)
    else:
      smoothed_position = new_position
    
    history['last_valid_position'] = smoothed_position
    return smoothed_position

  def calculate_fruit_orientation_from_white_yellow_keypoints(self, keypoints, bbox):
    """
    Calculate fruit orientation using white and yellow keypoints (keypoint 0 and 1)
    Simple direct line from white to yellow keypoint as Z-axis
    
    Args:
      keypoints: KeypointOut message with x, y, conf arrays (expecting 3 keypoints)
      bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
      Dictionary with orientation analysis or None
    """
    try:
      if len(keypoints.x) < 2:
        print(f"  Need at least 2 keypoints, got {len(keypoints.x)}")
        return None
      
      # Use keypoint 0 (white) and keypoint 1 (yellow)
      white_x, white_y = keypoints.x[0], keypoints.y[0]
      yellow_x, yellow_y = keypoints.x[1], keypoints.y[1]
      red_x, red_y = keypoints.x[2], keypoints.y[2]
      white_conf = keypoints.conf[0]
      yellow_conf = keypoints.conf[1]
      red_conf = keypoints.conf[2]
      
      # Check confidence
      min_confidence = 0.3
      if white_conf < min_confidence or yellow_conf < min_confidence:
        print(f"  Low confidence: white={white_conf:.2f}, yellow={yellow_conf:.2f}")
      
      # Calculate direction vector from white to yellow
      direction_vector = np.array([yellow_x - white_x, yellow_y - white_y])
      direction_length = np.linalg.norm(direction_vector)
      
      if direction_length < 1e-6:
        print("  White and yellow keypoints are too close")
        return None
      
      # Normalize direction vector
      fruit_axis_2d = direction_vector / direction_length
      
      # Calculate midpoint as centroid
      centroid = np.array([(white_x + yellow_x) / 2, (white_y + yellow_y) / 2])
      
      # Overall confidence is average of the two keypoints
      overall_confidence = (white_conf + yellow_conf) / 2
      
      # Calculate angle for visualization
      angle_rad = np.arctan2(fruit_axis_2d[1], fruit_axis_2d[0])

      angle_deg = np.degrees(angle_rad)
      # if angle_deg < 0:
      #   angle_deg += 360
      
      return {
        'fruit_axis_2d': fruit_axis_2d,     # 2D direction vector from white to yellow
        'centroid': centroid,               # Midpoint between white and yellow
        'confidence': overall_confidence,
        'white_conf': white_conf,
        'yellow_conf': yellow_conf,
        'red_conf': red_conf,
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'white_point': np.array([white_x, white_y]),
        'yellow_point': np.array([yellow_x, yellow_y]),
        'direction_length': direction_length
      }
      
    except Exception as e:
      print(f"Error calculating fruit orientation from white-yellow keypoints: {e}")
      return None

  def create_fruit_oriented_tf_from_3_keypoints(self, fruit_bbox, depth_info, keypoint_orientation, fruit_id):
    """
    Create TF with fruit orientation from 3 keypoints
    Position: Yellow keypoint (keypoints[1]) as starting point
    Z-axis: White→Yellow direction (fruit main axis)
    X-axis: Camera depth direction projected onto plane perpendicular to Z
    Y-axis: Completes right-handed coordinate system (Y = Z × X)
    """
    if depth_info is None or keypoint_orientation is None:
      return False
    
    x1, y1, x2, y2 = fruit_bbox
    
    # 1. Yellow 키포인트의 2D 픽셀 좌표
    yellow_point = keypoint_orientation['yellow_point']
    center_u = int(yellow_point[0])  # u (픽셀 x 좌표)
    center_v = int(yellow_point[1])  # v (픽셀 y 좌표)

    # 2. Yellow-Red 라인에서 추출한 깊이 값
    depth_m = depth_info['depth_m']  # 깊이 (미터)

    # 3. 3D 좌표 계산 (camera frame 기준)
    x = (center_u - self.cx) * depth_m / self.fx  # X축: Yellow의 u좌표 + 깊이로 계산
    y = (center_v - self.cy) * depth_m / self.fy  # Y축: Yellow의 v좌표 + 깊이로 계산  
    z = depth_m                  # Z축: Yellow-Red 라인의 깊이값
    
    # Get fruit axis direction from keypoint analysis
    fruit_axis_2d = keypoint_orientation['fruit_axis_2d']
    raw_angle = keypoint_orientation['angle_deg']
    keypoint_confidence = keypoint_orientation['confidence']
    
    # Check if confidence is sufficient
    overall_confidence = min(depth_info.get('confidence', 1.0), keypoint_confidence)
    
    if overall_confidence < self.confidence_threshold:
      print(f"  Low confidence for fruit {fruit_id}: {overall_confidence:.3f}")
      # Still proceed but with more conservative smoothing
    
    # Apply temporal smoothing
    smoothed_angle = self.smooth_angle(fruit_id, raw_angle, overall_confidence)
    smoothed_position = self.smooth_position(fruit_id, (x, y, z), overall_confidence)
    
    # Convert smoothed angle back to 2D direction vector
    smoothed_angle_rad = math.radians(smoothed_angle)
    smoothed_fruit_axis_2d = np.array([math.cos(smoothed_angle_rad), math.sin(smoothed_angle_rad)])
    
    # Create 3D coordinate system
    # Camera frame: X-right, Y-down, Z-forward
    # Fruit frame: Z-axis = white→yellow direction (fruit axis)
    
    # Z-axis: White→Yellow direction (fruit main axis)
    # Map 2D fruit axis to 3D (assuming it lies in the image plane)
    z_axis = np.array([smoothed_fruit_axis_2d[0], smoothed_fruit_axis_2d[1], 0.0])
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    # X-axis: Camera depth direction (forward into scene) projected onto plane perpendicular to Z
    camera_forward = np.array([0.0, 0.0, 1.0])  # Z direction in camera frame
    
    # Project camera forward onto plane perpendicular to fruit axis
    x_axis = camera_forward - np.dot(camera_forward, z_axis) * z_axis
    x_axis_norm = np.linalg.norm(x_axis)
    
    if x_axis_norm < 1e-6:
      # If Z and camera forward are parallel, choose alternative X
      print(f"  Warning: Fruit axis parallel to camera forward for fruit {fruit_id}")
      # Use camera right direction as alternative
      alternative_x = np.array([1.0, 0.0, 0.0])  # X direction in camera frame
      x_axis = alternative_x - np.dot(alternative_x, z_axis) * z_axis
      x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    else:
      x_axis = x_axis / x_axis_norm
    
    # Y-axis: Complete right-handed system (Y = Z × X)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Create rotation matrix [X Y Z] (column vectors)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Verify orthogonality
    det = np.linalg.det(rotation_matrix)
    if abs(det - 1.0) > 0.1:
      print(f"  Warning: Rotation matrix determinant = {det:.3f} for fruit {fruit_id}")
    
    # Convert to quaternion
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]
    
    # Create transform
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = self.depth_frame
    t.child_frame_id = f'fruit_{fruit_id}_oriented'
    
    # Set smoothed translation
    t.transform.translation.x = smoothed_position[0]
    t.transform.translation.y = smoothed_position[1]
    t.transform.translation.z = smoothed_position[2]
    
    # Set rotation
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    
    # Broadcast transform
    self.tf_broadcaster.sendTransform(t)
    
    print(f"  Fruit TF published: fruit_{fruit_id}_oriented")
    print(f"  Position: ({smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}, {smoothed_position[2]:.3f}) m (at yellow keypoint)")
    print(f"  Fruit axis angle: {raw_angle:.1f}° → {smoothed_angle:.1f}° (conf: {overall_confidence:.3f})")
    print(f"  Coordinate system: Origin=yellow, Z=white→yellow(fruit_axis), X=depth_projected, Y=perpendicular")
    
    return True

  def draw_keypoint_analysis_on_debug_image(self, debug_img, keypoint_analysis):
    """Draw white-yellow keypoint fruit orientation analysis results on the YOLO debug image"""
    for fruit_id, analysis in keypoint_analysis.items():
      bbox = analysis['bbox']
      keypoint_orientation = analysis['keypoint_orientation']
      depth_info = analysis['depth_info']
      keypoints_data = analysis.get('keypoints_data', None)
      
      if keypoint_orientation is None:
        continue
      
      x1, y1, x2, y2 = bbox
      
      # Get orientation information
      fruit_axis_2d = keypoint_orientation['fruit_axis_2d']
      angle_deg = keypoint_orientation['angle_deg']
      confidence = keypoint_orientation['confidence']
      centroid = keypoint_orientation['centroid']
      white_conf = keypoint_orientation['white_conf']
      yellow_conf = keypoint_orientation['yellow_conf']
      red_conf = keypoint_orientation['red_conf']
      white_point = keypoint_orientation['white_point']
      yellow_point = keypoint_orientation['yellow_point']
      
      # Draw all keypoints with index labels for debugging
      if keypoints_data is not None:
        for i, (kx, ky, kconf) in enumerate(zip(keypoints_data.x, keypoints_data.y, keypoints_data.conf)):
          if i == 0:  # White keypoint
            color = (255, 255, 255)
            radius = max(8, int(12 * kconf))
          elif i == 1:  # Yellow keypoint  
            color = (0, 255, 255)
            radius = max(8, int(12 * kconf))
          elif i == 2:  # Red keypoint
            color = (0, 0, 255)
            radius = max(8, int(12 * kconf))
          else:  # Other keypoints (gray)
            color = (128, 128, 128)
            radius = max(6, int(8 * kconf))
          
          # Draw keypoint with thick border
          cv2.circle(debug_img, (int(kx), int(ky)), radius, color, -1)
          cv2.circle(debug_img, (int(kx), int(ky)), radius+2, (0, 0, 0), 3)  # Thick black border
          
          # Draw keypoint index and confidence with larger text
          label = f"KP{i}({kconf:.2f})"
          cv2.putText(debug_img, label, (int(kx)+10, int(ky)-10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 3)
          cv2.putText(debug_img, label, (int(kx)+10, int(ky)-10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
      
      # Draw line from white to yellow (this is the Z-axis direction)
      white_pt = (int(white_point[0]), int(white_point[1]))
      yellow_pt = (int(yellow_point[0]), int(yellow_point[1]))
      
      # Draw thick white-to-yellow line
      cv2.line(debug_img, white_pt, yellow_pt, (255, 255, 255), 6)  # Thick white base
      cv2.line(debug_img, white_pt, yellow_pt, (0, 255, 255), 4)    # Yellow overlay
      
      # Draw direction arrow (extends the line)
      center_x, center_y = centroid
      scale = 40.0
      vx, vy = fruit_axis_2d
      
      p1 = (int(center_x - vx * scale), int(center_y - vy * scale))
      p2 = (int(center_x + vx * scale), int(center_y + vy * scale))
      
      # Color based on confidence: red (low) to green (high)
      if confidence > 0.8:
        axis_color = (0, 255, 0)  # Green - high confidence
      elif confidence > 0.6:
        axis_color = (0, 255, 255)  # Yellow - medium confidence
      else:
        axis_color = (0, 0, 255)  # Red - low confidence
      
      # Draw extended axis arrow
      cv2.arrowedLine(debug_img, p1, p2, axis_color, 3, cv2.LINE_AA, tipLength=0.3)
      
      # Draw TF position marker (yellow keypoint = starting point)
      tf_center_u = int(yellow_point[0])
      tf_center_v = int(yellow_point[1])
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 12, (255, 0, 255), 4, cv2.LINE_AA)  # Larger marker
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 6, (255, 0, 255), -1, cv2.LINE_AA)
      
      # Add TF label
      tf_label = f"TF_{fruit_id}"
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 3)
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
      
      # Add stability indicators
      stability_text = "STABLE" if confidence > 0.5 else "UNSTABLE"
      
      # Add simple text information
      text_lines = [
        f"Fruit {fruit_id} [{stability_text}]",
        f"Angle: {angle_deg:.1f}deg",
        f"White: {white_conf:.2f} | Yellow: {yellow_conf:.2f} | Red: {red_conf:.2f}",
        f"Confidence: {confidence:.2f}",
        f"Depth(Y-R line): {depth_info['depth_mm']:.0f}mm",
        f"Line pixels: {depth_info['filtered_pixel_count']}/{depth_info['line_pixel_count']}",
      ]
      
      text_x = max(5, x1 + 60)
      text_y = max(25, y1 + 60)
      
      for i, line in enumerate(text_lines):
        y_pos = text_y + i * 16
        
        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (0, 0, 0), -1)
        
        # Draw text
        text_color = (255, 255, 255) if confidence > 0.5 else (100, 100, 255)
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
    
    return debug_img

  def bbox_callback(self, msg):
    """Store bbox information for matching with keypoints"""
    self.latest_bboxes = msg
    # Process keypoints if both bbox and keypoints are available
    if self.latest_keypoints is not None:
      self.process_keypoint_detections()

  def process_keypoint_detections(self):
    """Process keypoint detections and create stable TFs"""
    if (self.latest_image is None or self.latest_depth is None or 
        self.latest_keypoints is None or self.latest_bboxes is None):
      return
    
    # Get keypoint detections
    keypoint_detections = self.latest_keypoints.keypoints
    bbox_detections = self.latest_bboxes.yolo_out_array
    
    if not keypoint_detections:
      # Clear keypoint analysis if no detections
      self.latest_keypoint_analysis = {}
      return
    
    print(f"Processing {len(keypoint_detections)} keypoint detections")
    
    # First pass: collect all valid detections with their 3D positions
    valid_detections = []
    
    # Match keypoints with bboxes (assume same order and timing)
    for i, keypoints in enumerate(keypoint_detections):
      # Get corresponding bbox if available
      bbox = None
      if i < len(bbox_detections):
        bbox_obj = bbox_detections[i]
        # Filter for specific label if needed (e.g., label 7 for ripe fruits)
        if hasattr(bbox_obj, 'label') and bbox_obj.label in [0, 1, 2]:  # Adjust labels as needed
          bbox = bbox_obj.tlbr  # [x1, y1, x2, y2]
      
      if bbox is None:
        # Create bbox from keypoints if not available
        if len(keypoints.x) > 0:
          x_coords = np.array(keypoints.x)
          y_coords = np.array(keypoints.y)
          conf_coords = np.array(keypoints.conf)
          
          # Filter confident keypoints for bbox estimation
          valid_mask = conf_coords > 0.3
          if np.sum(valid_mask) > 0:
            x_valid = x_coords[valid_mask]
            y_valid = y_coords[valid_mask]
            
            margin = 20
            bbox = [
              int(np.min(x_valid) - margin), int(np.min(y_valid) - margin),
              int(np.max(x_valid) + margin), int(np.max(y_valid) + margin)
            ]
          else:
            print(f"Keypoint {i}: No confident keypoints for bbox estimation")
            continue
      
      # Calculate fruit orientation from white and yellow keypoints first
      keypoint_orientation = self.calculate_fruit_orientation_from_white_yellow_keypoints(keypoints, bbox)
      
      if keypoint_orientation is None:
        print(f"Keypoint {i}: Failed to calculate fruit orientation from white-yellow keypoints")
        continue
      
      # Extract depth information from yellow-red keypoints line
      yellow_point = keypoint_orientation['yellow_point'] if 'yellow_point' in keypoint_orientation else [keypoints.x[1], keypoints.y[1]]
      red_point = [keypoints.x[2], keypoints.y[2]] if len(keypoints.x) > 2 else yellow_point
      
      depth_info = self.extract_depth_from_keypoints_line(self.latest_depth, yellow_point, red_point)
      
      if depth_info is None:
        print(f"Keypoint {i}: No valid depth data from keypoints line")
        continue
      
      # Calculate 3D position for tracking - use yellow keypoint as reference
      yellow_point = keypoint_orientation['yellow_point']
      center_u = int(yellow_point[0])
      center_v = int(yellow_point[1])
      position_3d = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
      
      # Store valid detection
      valid_detections.append({
        'detection_idx': i,
        'bbox': bbox,
        'keypoint_orientation': keypoint_orientation,
        'depth_info': depth_info,
        'keypoints_data': keypoints,
        'position_3d': position_3d
      })
      
      print(f"Detection {i}: Valid keypoint detection at 3D position ({position_3d[0]:.3f}, {position_3d[1]:.3f}, {position_3d[2]:.3f})")
    
    # Second pass: Apply tracking to associate detections with consistent IDs
    print(f"Found {len(valid_detections)} valid detections, applying tracking...")
    associated_pairs = self.associate_detections_with_tracks(valid_detections)
    
    # Third pass: Process tracked detections and create TFs
    keypoint_analysis = {}
    
    for track_id, detection in associated_pairs:
      bbox = detection['bbox']
      keypoint_orientation = detection['keypoint_orientation']
      depth_info = detection['depth_info']
      keypoints_data = detection['keypoints_data']
      
      # Store analysis results for visualization (use track_id as key)
      keypoint_analysis[track_id] = {
        'bbox': bbox,
        'keypoint_orientation': keypoint_orientation,
        'depth_info': depth_info,
        'keypoints_data': keypoints_data
      }
      
      # Create fruit-oriented TF with consistent track_id
      tf_created = self.create_fruit_oriented_tf_from_3_keypoints(bbox, depth_info, keypoint_orientation, track_id)
      
      print(f"Fruit {track_id}: BBox: {bbox}")
      print(f"  Depth from Y-R line: {depth_info['depth_mm']:.1f}mm ± {depth_info['depth_std']:.1f}mm")
      print(f"  Line pixels: {depth_info['line_pixel_count']}, Valid: {depth_info['valid_pixel_count']}, Filtered: {depth_info['filtered_pixel_count']}")
      print(f"  Depth confidence: {depth_info.get('confidence', 1.0):.3f}")
      print(f"  White→Yellow axis: {keypoint_orientation['angle_deg']:.1f}deg (conf: {keypoint_orientation['confidence']:.3f})")
      print(f"  White conf: {keypoint_orientation['white_conf']:.3f}, Yellow conf: {keypoint_orientation['yellow_conf']:.3f}")
      print("---")
    
    # Update keypoint analysis for visualization
    self.latest_keypoint_analysis = keypoint_analysis
    self.save_counter += 1

  def associate_detections_with_tracks(self, detections):
    """
    Associate new detections with existing tracks based on position similarity
    Args:
      detections: List of detection dictionaries with 'position_3d' key
    Returns:
      List of (track_id, detection) tuples
    """
    self.frame_count += 1
    
    # Update existing tracks (mark as not updated)
    for track_id in self.fruit_tracks:
      self.fruit_tracks[track_id]['updated'] = False
      self.fruit_tracks[track_id]['missing_frames'] += 1
    
    associated_pairs = []
    unassociated_detections = list(range(len(detections)))
    
    # Try to associate each detection with existing tracks
    for det_idx, detection in enumerate(detections):
      if det_idx not in unassociated_detections:
        continue
        
      det_pos = np.array(detection['position_3d'])
      best_track_id = None
      best_distance = float('inf')
      
      # Find closest track
      for track_id, track_info in self.fruit_tracks.items():
        if track_info['updated']:  # Already associated
          continue
          
        track_pos = np.array(track_info['last_position'])
        distance = np.linalg.norm(det_pos - track_pos)
        
        if distance < self.tracking_distance_threshold and distance < best_distance:
          best_distance = distance
          best_track_id = track_id
      
      if best_track_id is not None:
        # Associate detection with existing track
        self.fruit_tracks[best_track_id]['last_position'] = det_pos
        self.fruit_tracks[best_track_id]['updated'] = True
        self.fruit_tracks[best_track_id]['missing_frames'] = 0
        associated_pairs.append((best_track_id, detection))
        unassociated_detections.remove(det_idx)
        print(f"  Associated detection {det_idx} with existing track {best_track_id} (dist: {best_distance:.3f}m)")
    
    # Create new tracks for unassociated detections
    for det_idx in unassociated_detections:
      detection = detections[det_idx]
      new_track_id = self.next_track_id
      self.next_track_id += 1
      
      self.fruit_tracks[new_track_id] = {
        'last_position': np.array(detection['position_3d']),
        'updated': True,
        'missing_frames': 0,
        'created_frame': self.frame_count
      }
      
      associated_pairs.append((new_track_id, detection))
      print(f"  Created new track {new_track_id} for detection {det_idx}")
    
    # Remove tracks that have been missing for too long
    tracks_to_remove = []
    for track_id, track_info in self.fruit_tracks.items():
      if track_info['missing_frames'] > self.max_missing_frames:
        tracks_to_remove.append(track_id)
        print(f"  Removing track {track_id} (missing for {track_info['missing_frames']} frames)")
    
    for track_id in tracks_to_remove:
      del self.fruit_tracks[track_id]
      # Also clean up history for removed tracks
      if track_id in self.fruit_history:
        del self.fruit_history[track_id]
    
    print(f"  Frame {self.frame_count}: {len(associated_pairs)} associations, {len(self.fruit_tracks)} active tracks")
    
    return associated_pairs

if __name__ == '__main__':
  rclpy.init()
  node = StableFruitKeypointTFNode()
  
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Shutting down stable fruit keypoint TF node...")
  finally:
    node.destroy_node()
    rclpy.shutdown() 