#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
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
from mf_perception_msgs.msg import YoloOutArray, YoloOut

class StableFruitTFNode(Node):
  def __init__(self, node_name='stable_fruit_tf_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    self.save_counter = 0
    self.save_dir = "/workspace/ripe_fruit_crops"
    
    # Camera parameters (from params file)
    self.base_frame = 'camera_hand_link'
    self.depth_frame = 'camera_hand_depth_optical_frame'
    self.depth_min_mm = 70
    self.depth_max_mm = 700
    
    # Stability parameters
    self.declare_parameter('smoothing_window', 5)  # Temporal smoothing window size
    self.declare_parameter('confidence_threshold', 0.7)  # Minimum confidence for PCA
    self.declare_parameter('angle_change_threshold', 15.0)  # Max angle change per frame (degrees)
    self.declare_parameter('position_change_threshold', 0.05)  # Max position change per frame (meters)
    self.declare_parameter('outlier_rejection_factor', 2.0)  # Factor for outlier rejection
    
    # Get parameters
    self.smoothing_window = self.get_parameter('smoothing_window').get_parameter_value().integer_value
    self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
    self.angle_change_threshold = self.get_parameter('angle_change_threshold').get_parameter_value().double_value
    self.position_change_threshold = self.get_parameter('position_change_threshold').get_parameter_value().double_value
    self.outlier_rejection_factor = self.get_parameter('outlier_rejection_factor').get_parameter_value().double_value
    
    # History tracking for each fruit
    self.fruit_history = {}  # fruit_id -> history data
    
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
    
    # Initialize MobileSAM model
    try:
      self.sam_model = SAM("/workspace/ai_models/mobile_sam.pt")
      print("MobileSAM model loaded successfully")
    except Exception as e:
      print(f"Failed to load MobileSAM: {e}")
      self.sam_model = None
    
    # Create save directory if it doesn't exist
    os.makedirs(self.save_dir, exist_ok=True)
    
    # Subscribe to bbox output, RGB image, depth image, and YOLO debug
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
    self.latest_stem_analysis = {}  # Store stem analysis results
    
    print('Stable fruit TF node is initialized')
    print(f'Smoothing window: {self.smoothing_window}')
    print(f'Confidence threshold: {self.confidence_threshold}')
    print(f'Angle change threshold: {self.angle_change_threshold}°')
    print(f'Position change threshold: {self.position_change_threshold}m')

  def image_callback(self, msg):
    """Store the latest RGB image for processing"""
    self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.latest_header = msg.header

  def depth_callback(self, msg):
    """Store the latest depth image"""
    self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

  def yolo_debug_callback(self, msg):
    """Process YOLO debug image and add stem analysis visualization"""
    self.latest_yolo_debug = self.bridge.compressed_imgmsg_to_cv2(msg)
    
    # Enhance the debug image with stem analysis if available
    if self.latest_stem_analysis:
      enhanced_img = self.draw_stem_analysis_on_debug_image(
        self.latest_yolo_debug.copy(), 
        self.latest_stem_analysis
      )
      
      # Publish enhanced debug image
      enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(enhanced_img, dst_format='jpg')
      enhanced_msg.header = msg.header
      self.enhanced_debug_pub.publish(enhanced_msg)

  def extract_depth_from_bbox_robust(self, depth_image, bbox):
    """Extract reliable depth value with enhanced robustness"""
    if depth_image is None:
      return None
    
    x1, y1, x2, y2 = bbox
    
    # Crop depth region
    depth_roi = depth_image[y1:y2, x1:x2]
    
    if depth_roi.size == 0:
      return None
    
    # Convert to mm and filter valid range
    depth_mm = depth_roi.astype(np.float32)
    
    # Filter depth values within valid range
    valid_mask = (depth_mm >= self.depth_min_mm) & (depth_mm <= self.depth_max_mm)
    valid_depths = depth_mm[valid_mask]
    
    if len(valid_depths) == 0:
      return None
    
    # Enhanced outlier removal using multiple methods
    # 1. IQR method
    q1 = np.percentile(valid_depths, 25)
    q3 = np.percentile(valid_depths, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    iqr_filtered = valid_depths[
      (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
    ]
    
    # 2. Z-score method for additional filtering
    if len(iqr_filtered) > 3:
      mean_depth = np.mean(iqr_filtered)
      std_depth = np.std(iqr_filtered)
      z_scores = np.abs((iqr_filtered - mean_depth) / (std_depth + 1e-8))
      z_filtered = iqr_filtered[z_scores < 2.0]
      
      if len(z_filtered) > 0:
        final_depths = z_filtered
      else:
        final_depths = iqr_filtered
    else:
      final_depths = iqr_filtered
    
    if len(final_depths) == 0:
      return None
    
    # Use median as the representative depth (more robust than mean)
    median_depth_mm = np.median(final_depths)
    
    return {
      'depth_mm': median_depth_mm,
      'depth_m': median_depth_mm / 1000.0,
      'valid_pixel_count': len(valid_depths),
      'filtered_pixel_count': len(final_depths),
      'depth_std': np.std(final_depths),
      'confidence': min(1.0, len(final_depths) / len(valid_depths))  # Confidence based on filtering ratio
    }

  def pixel_to_3d_point(self, u, v, depth_m):
    """Convert pixel coordinates and depth to 3D point in camera frame"""
    # Camera intrinsic parameters
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 320.0  # principal point x
    cy = 240.0  # principal point y
    
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

  def segment_stem_region_with_sam(self, image, fruit_bbox):
    """Use MobileSAM to segment the stem region above the fruit"""
    if self.sam_model is None:
      return None, None, None
    
    try:
      x1, y1, x2, y2 = fruit_bbox
      
      # Define stem region bbox
      stem_y1 = max(0, y1 - 40)
      stem_y2 = y1
      stem_x1 = x1
      stem_x2 = x2
      
      stem_bbox = [stem_x1, stem_y1, stem_x2, stem_y2]
      
      # Run MobileSAM prediction
      results = self.sam_model.predict(image, bboxes=[stem_bbox])
      
      if results and len(results) > 0:
        masks = results[0].masks
        if masks is not None and len(masks.data) > 0:
          mask = masks.data[0].cpu().numpy().astype(np.uint8)
          
          # Crop the stem region from original image and mask
          stem_region = image[stem_y1:stem_y2, stem_x1:stem_x2]
          stem_mask = mask[stem_y1:stem_y2, stem_x1:stem_x2]
          
          return stem_region, stem_mask, (stem_x1, stem_y1, stem_x2, stem_y2)
      
      return None, None, None
      
    except Exception as e:
      print(f"MobileSAM stem segmentation failed: {e}")
      return None, None, None

  def create_stable_stem_oriented_tf(self, fruit_bbox, depth_info, stem_mask, fruit_id):
    """Create stable TF with temporal smoothing"""
    if depth_info is None or stem_mask is None:
      return False
    
    x1, y1, x2, y2 = fruit_bbox
    
    # TF position
    center_u = (x1 + x2) // 2
    center_v = y1
    
    # Convert to 3D point
    raw_position = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
    
    # Enhanced PCA with confidence
    pca_result = self.mask_principal_axis_robust(stem_mask)
    
    if pca_result is None or len(pca_result) < 4:
      print(f"  Failed to analyze stem direction for fruit {fruit_id}")
      return False
    
    center, v, evals, pca_confidence = pca_result
    
    # Check if confidence is sufficient
    overall_confidence = min(depth_info.get('confidence', 1.0), pca_confidence)
    
    if overall_confidence < self.confidence_threshold:
      print(f"  Low confidence for fruit {fruit_id}: {overall_confidence:.3f}")
      # Still proceed but with more conservative smoothing
    
    vx, vy = v
    raw_angle = math.degrees(math.atan2(vy, vx))
    if raw_angle < 0:
      raw_angle += 360
    
    # Apply temporal smoothing
    smoothed_angle = self.smooth_angle(fruit_id, raw_angle, overall_confidence)
    smoothed_position = self.smooth_position(fruit_id, raw_position, overall_confidence)
    
    # Convert smoothed angle back to direction vector
    smoothed_angle_rad = math.radians(smoothed_angle)
    smoothed_vx = math.cos(smoothed_angle_rad)
    smoothed_vy = math.sin(smoothed_angle_rad)
    
    # Create TF with smoothed values
    stem_direction_3d = np.array([smoothed_vx, smoothed_vy, 0.0])
    stem_direction_3d = stem_direction_3d / np.linalg.norm(stem_direction_3d)
    
    z_axis = stem_direction_3d
    x_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Create rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Convert to quaternion
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]
    
    # Create transform
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = self.depth_frame
    t.child_frame_id = f'fruit_{fruit_id}_stem_oriented'
    
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
    
    print(f"  Stable TF published: fruit_{fruit_id}_stem_oriented")
    print(f"  Position: ({smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}, {smoothed_position[2]:.3f}) m")
    print(f"  Angle: {raw_angle:.1f}° → {smoothed_angle:.1f}° (conf: {overall_confidence:.3f})")
    
    return True

  def draw_stem_analysis_on_debug_image(self, debug_img, stem_analysis):
    """Draw stem analysis results on the YOLO debug image"""
    for fruit_id, analysis in stem_analysis.items():
      bbox = analysis['bbox']
      stem_bbox = analysis['stem_bbox']
      pca_result = analysis['pca_result']
      depth_info = analysis['depth_info']
      
      if pca_result is None or len(pca_result) < 4:
        continue
      
      x1, y1, x2, y2 = bbox
      stem_x1, stem_y1, stem_x2, stem_y2 = stem_bbox
      center, v, evals, confidence = pca_result
      
      # Convert stem region coordinates to full image coordinates
      stem_center_x = stem_x1 + center[0]
      stem_center_y = stem_y1 + center[1]
      
      vx, vy = v
      
      # Draw stem region bbox
      cv2.rectangle(debug_img, (stem_x1, stem_y1), (stem_x2, stem_y2), (0, 255, 255), 2)
      
      # Draw principal axis with confidence-based color
      scale = 30.0
      p1 = (int(round(stem_center_x - vx * scale)), int(round(stem_center_y - vy * scale)))
      p2 = (int(round(stem_center_x + vx * scale)), int(round(stem_center_y + vy * scale)))
      
      # Color based on confidence: red (low) to green (high)
      if confidence > 0.8:
        color = (0, 255, 0)  # Green - high confidence
      elif confidence > 0.6:
        color = (0, 255, 255)  # Yellow - medium confidence
      else:
        color = (0, 0, 255)  # Red - low confidence
      
      cv2.line(debug_img, p1, p2, color, 3, cv2.LINE_AA)
      cv2.circle(debug_img, (int(round(stem_center_x)), int(round(stem_center_y))), 
                5, color, -1, cv2.LINE_AA)
      
      # Draw TF position marker
      tf_center_u = (x1 + x2) // 2
      tf_center_v = y1
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 8, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 3, (0, 255, 0), -1, cv2.LINE_AA)
      
      # Add stability indicators
      stability_text = "STABLE" if confidence > 0.7 else "UNSTABLE"
      
      # Add text information
      # Calculate angle from vertical up (0° = vertical up, clockwise positive)
      angle_from_vertical_up = math.degrees(math.atan2(vx, -vy))  # -vy makes vertical up = 0°
      text_lines = [
        f"Fruit {fruit_id} [{stability_text}]",
        f"Stem: {angle_from_vertical_up:.1f}deg",
        f"Conf: {confidence:.2f}",
        f"Depth: {depth_info['depth_mm']:.0f}mm"
      ]
      
      text_x = max(5, x1 + 50)
      text_y = max(25, y1 + 50)
      
      for i, line in enumerate(text_lines):
        y_pos = text_y + i * 20
        
        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (0, 0, 0), -1)
        
        # Draw text
        text_color = (255, 255, 255) if confidence > 0.6 else (100, 100, 255)
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    return debug_img

  def bbox_callback(self, msg):
    """Process bounding box detections and create stable TFs"""
    if self.latest_image is None or self.latest_depth is None:
      return
    
    # Filter for label 7
    ripe_fruits = [yolo_out for yolo_out in msg.yolo_out_array if yolo_out.label == 7]
    
    if not ripe_fruits:
      # Clear stem analysis if no fruits detected
      self.latest_stem_analysis = {}
      return
    
    print(f"Found {len(ripe_fruits)} ripe fruits")
    
    # Store stem analysis results for visualization
    stem_analysis = {}
    
    for i, fruit in enumerate(ripe_fruits):
      bbox = fruit.tlbr  # [x1, y1, x2, y2]
      
      # Extract robust depth information
      depth_info = self.extract_depth_from_bbox_robust(self.latest_depth, bbox)
      
      if depth_info is None:
        print(f"Fruit {i}: No valid depth data")
        continue
      
      # Segment stem region using SAM
      stem_region, stem_mask, stem_bbox = self.segment_stem_region_with_sam(self.latest_image, bbox)
      
      if stem_region is None or stem_mask is None:
        print(f"Fruit {i}: Failed to segment stem region")
        continue
      
      # Analyze principal axis with enhanced robustness
      pca_result = self.mask_principal_axis_robust(stem_mask)
      
      # Store analysis results for visualization
      stem_analysis[i] = {
        'bbox': bbox,
        'stem_bbox': stem_bbox,
        'pca_result': pca_result,
        'depth_info': depth_info
      }
      
      # Create stable stem-oriented TF
      tf_created = self.create_stable_stem_oriented_tf(bbox, depth_info, stem_mask, i)
      
      print(f"Fruit {i}: Score {fruit.score:.3f}, BBox: {bbox}")
      print(f"  Depth: {depth_info['depth_mm']:.1f}mm ± {depth_info['depth_std']:.1f}mm (conf: {depth_info.get('confidence', 1.0):.3f})")
      if pca_result is not None and len(pca_result) >= 4:
        center, v, evals, confidence = pca_result
        angle_from_vertical_up = math.degrees(math.atan2(v[1], -v[0]))
        print(f"  Stem direction: {angle_from_vertical_up:.1f}deg (PCA conf: {confidence:.3f})")
      print("---")
    
    # Update stem analysis for visualization
    self.latest_stem_analysis = stem_analysis
    self.save_counter += 1

if __name__ == '__main__':
  rclpy.init()
  node = StableFruitTFNode()
  
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Shutting down stable fruit TF node...")
  finally:
    node.destroy_node()
    rclpy.shutdown() 