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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from mf_perception_msgs.msg import YoloOutArray, YoloOut

class RipeFruitCropperNode(Node):
  def __init__(self, node_name='ripe_fruit_cropper_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    self.save_counter = 0
    self.save_dir = "/workspace/ripe_fruit_crops"
    
    # Camera parameters (from params file)
    self.base_frame = 'camera_hand_link'
    self.depth_frame = 'camera_hand_depth_optical_frame'
    self.depth_min_mm = 70
    self.depth_max_mm = 700
    
    # Target labels for processing (can be configured)
    self.declare_parameter('target_labels', [7])  # Default: label 7
    self.declare_parameter('sam_model_path', '/workspace/ai_models/mobile_sam.pt')
    self.declare_parameter('stem_region_height', 40)  # pixels above fruit bbox
    self.declare_parameter('gpu_device', '3')
    
    # Get parameters
    self.target_labels = self.get_parameter('target_labels').get_parameter_value().integer_array_value
    self.sam_model_path = self.get_parameter('sam_model_path').get_parameter_value().string_value
    self.stem_region_height = self.get_parameter('stem_region_height').get_parameter_value().integer_value
    gpu_device = self.get_parameter('gpu_device').get_parameter_value().string_value
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    
    print(f"Target labels for processing: {self.target_labels}")
    print(f"SAM model path: {self.sam_model_path}")
    print(f"Stem region height: {self.stem_region_height} pixels")
    print(f"GPU device: {gpu_device}")
    
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
      self.sam_model = SAM(self.sam_model_path)
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
    
    print('Ripe fruit cropper node is initialized')
    print(f'Saving cropped images to: {self.save_dir}')
    print('Publishing enhanced debug images to: /mf_perception/yolo_debug_enhanced/compressed')

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

  def extract_depth_from_bbox(self, depth_image, bbox):
    """Extract reliable depth value from bounding box region"""
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
    
    # Remove outliers using IQR method
    q1 = np.percentile(valid_depths, 25)
    q3 = np.percentile(valid_depths, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter outliers
    filtered_depths = valid_depths[
      (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
    ]
    
    if len(filtered_depths) == 0:
      return None
    
    # Use median as the representative depth
    median_depth_mm = np.median(filtered_depths)
    
    return {
      'depth_mm': median_depth_mm,
      'depth_m': median_depth_mm / 1000.0,
      'valid_pixel_count': len(valid_depths),
      'filtered_pixel_count': len(filtered_depths),
      'depth_std': np.std(filtered_depths)
    }

  def pixel_to_3d_point(self, u, v, depth_m):
    """Convert pixel coordinates and depth to 3D point in camera frame"""
    # Camera intrinsic parameters (you may need to get these from camera_info topic)
    # For now, using typical values - you should replace with actual camera parameters
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 320.0  # principal point x
    cy = 240.0  # principal point y
    
    # Convert to 3D coordinates in camera frame
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    
    return x, y, z

  def get_sam_fitted_bbox(self, stem_mask):
    """
    SAM segmentation 결과에 대해 fit한 bounding box를 생성
    Returns: (x1, y1, x2, y2) in mask coordinates, center point
    """
    ys, xs = np.where(stem_mask > 0)
    if len(xs) == 0:
      return None, None
    
    # Tight bounding box
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    
    # Center point of SAM fitted bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    return (x1, y1, x2, y2), (center_x, center_y)

  def segment_stem_region_with_sam(self, image, fruit_bbox):
    """Use MobileSAM to segment the stem region above the fruit"""
    if self.sam_model is None:
      return None, None, None
    
    try:
      x1, y1, x2, y2 = fruit_bbox
      
      # Define stem region bbox: 가로는 동일, 세로는 y1 위로 설정된 픽셀만큼
      stem_y1 = max(0, y1 - self.stem_region_height)
      stem_y2 = y1  # y1 높이까지
      stem_x1 = x1  # 가로는 동일
      stem_x2 = x2  # 가로는 동일
      
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

  def create_stem_oriented_tf_with_vector(self, fruit_bbox, depth_info, stem_mask, stem_bbox_coords, fruit_id):
    """Create TF with stem direction from YOLO bbox center to SAM fitted bbox center"""
    if depth_info is None or stem_mask is None:
      return False, None, None, None
    
    # Get YOLO bbox center (시작점)
    x1, y1, x2, y2 = fruit_bbox
    yolo_center_u = (x1 + x2) / 2.0
    yolo_center_v = (y1 + y2) / 2.0
    
    # Get SAM fitted bbox and its center (끝점)
    sam_fitted_bbox, sam_center = self.get_sam_fitted_bbox(stem_mask)
    
    if sam_fitted_bbox is None or sam_center is None:
      print(f"  Failed to get SAM fitted bbox for fruit {fruit_id}")
      return False, None, None, None
    
    # Convert SAM center to full image coordinates
    stem_x1, stem_y1, stem_x2, stem_y2 = stem_bbox_coords
    sam_center_full_u = stem_x1 + sam_center[0]
    sam_center_full_v = stem_y1 + sam_center[1]
    
    # Calculate direction vector from YOLO center to SAM center (YOLO → SAM)
    direction_u = sam_center_full_u - yolo_center_u
    direction_v = sam_center_full_v - yolo_center_v
    
    # Normalize direction vector
    direction_length = math.sqrt(direction_u**2 + direction_v**2)
    if direction_length < 1e-6:
      print(f"  Direction vector too small for fruit {fruit_id}")
      return False, None, None, None
    
    direction_u_norm = direction_u / direction_length
    direction_v_norm = direction_v / direction_length
    
    # TF 중점: YOLO bbox 중심의 x좌표, y1 높이, depth 정보
    center_u = (x1 + x2) // 2
    center_v = y1  # y1 높이에 맞춤
    
    # Convert to 3D point
    x, y, z = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
    
    # TF 좌표계 정의:
    # z축: stem 방향 (YOLO center → SAM center)
    # x축: camera depth 방향 (카메라 z축과 동일)
    # y축: z × x (오른손 법칙)
    
    # 방향 벡터를 3D로 확장 (z 성분은 0으로 설정)
    stem_direction_3d = np.array([direction_u_norm, direction_v_norm, 0.0])
    stem_direction_3d = stem_direction_3d / np.linalg.norm(stem_direction_3d)
    
    z_axis = stem_direction_3d  # stem 방향
    x_axis = np.array([0.0, 0.0, 1.0])  # camera depth 방향
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # 회전 행렬 구성
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # 회전 행렬을 쿼터니언으로 변환
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]
    
    # Create transform
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = self.depth_frame
    t.child_frame_id = f'fruit_{fruit_id}_stem_oriented'
    
    # Set translation
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    
    # Set rotation
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    
    # Broadcast transform
    self.tf_broadcaster.sendTransform(t)
    
    # Calculate angle for display
    angle_deg = math.degrees(math.atan2(direction_v, direction_u))
    
    print(f"  Stem-oriented TF published: fruit_{fruit_id}_stem_oriented")
    print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"  YOLO center (start): ({yolo_center_u:.1f}, {yolo_center_v:.1f})")
    print(f"  SAM center (end): ({sam_center_full_u:.1f}, {sam_center_full_v:.1f})")
    print(f"  Direction vector (YOLO→SAM): ({direction_u_norm:.3f}, {direction_v_norm:.3f})")
    print(f"  Stem angle: {angle_deg:.1f}°")
    
    return True, sam_fitted_bbox, (sam_center_full_u, sam_center_full_v), angle_deg

  def draw_stem_analysis_on_debug_image(self, debug_img, stem_analysis):
    """Draw stem analysis results on the YOLO debug image"""
    for fruit_id, analysis in stem_analysis.items():
      bbox = analysis['bbox']
      stem_bbox = analysis['stem_bbox']
      sam_fitted_bbox = analysis.get('sam_fitted_bbox')
      sam_center_full = analysis.get('sam_center_full')
      stem_angle = analysis.get('stem_angle')
      depth_info = analysis['depth_info']
      
      x1, y1, x2, y2 = bbox
      stem_x1, stem_y1, stem_x2, stem_y2 = stem_bbox
      
      # Draw stem region bbox (yellow)
      cv2.rectangle(debug_img, (stem_x1, stem_y1), (stem_x2, stem_y2), (0, 255, 255), 2)
      
      # Draw SAM fitted bbox if available (cyan)
      if sam_fitted_bbox is not None:
        sam_x1, sam_y1, sam_x2, sam_y2 = sam_fitted_bbox
        # Convert to full image coordinates
        full_sam_x1 = stem_x1 + sam_x1
        full_sam_y1 = stem_y1 + sam_y1
        full_sam_x2 = stem_x1 + sam_x2
        full_sam_y2 = stem_y1 + sam_y2
        cv2.rectangle(debug_img, (full_sam_x1, full_sam_y1), (full_sam_x2, full_sam_y2), (255, 255, 0), 2)
      
      # Draw YOLO center point (red - 시작점)
      yolo_center_u = int((x1 + x2) // 2)
      yolo_center_v = int((y1 + y2) // 2)
      cv2.circle(debug_img, (yolo_center_u, yolo_center_v), 8, (0, 0, 255), -1, cv2.LINE_AA)
      cv2.putText(debug_img, "START", (yolo_center_u - 20, yolo_center_v - 15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
      
      # Draw SAM center point (blue - 끝점)
      if sam_center_full is not None:
        sam_center_x = int(round(sam_center_full[0]))
        sam_center_y = int(round(sam_center_full[1]))
        cv2.circle(debug_img, (sam_center_x, sam_center_y), 8, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(debug_img, "END", (sam_center_x - 15, sam_center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
      
      # Draw direction vector line from YOLO center to SAM center (thick green line)
      if sam_center_full is not None:
        sam_center_x = int(round(sam_center_full[0]))
        sam_center_y = int(round(sam_center_full[1]))
        cv2.line(debug_img, 
                (yolo_center_u, yolo_center_v),
                (sam_center_x, sam_center_y), 
                (0, 255, 0), 4, cv2.LINE_AA)
        
        # Draw arrow head at SAM center (end point)
        arrow_length = 15
        if stem_angle is not None:
          angle_rad = math.radians(stem_angle)
          # Arrow pointing towards SAM center
          arrow_x1 = int(sam_center_x - arrow_length * math.cos(angle_rad - 2.8))
          arrow_y1 = int(sam_center_y - arrow_length * math.sin(angle_rad - 2.8))
          cv2.line(debug_img, (sam_center_x, sam_center_y), 
                  (arrow_x1, arrow_y1), (0, 255, 0), 3, cv2.LINE_AA)
          
          arrow_x2 = int(sam_center_x - arrow_length * math.cos(angle_rad + 2.8))
          arrow_y2 = int(sam_center_y - arrow_length * math.sin(angle_rad + 2.8))
          cv2.line(debug_img, (sam_center_x, sam_center_y), 
                  (arrow_x2, arrow_y2), (0, 255, 0), 3, cv2.LINE_AA)
      
      # Draw TF position marker at fruit center (green circle)
      tf_center_u = int((x1 + x2) // 2)
      tf_center_v = int(y1)
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 8, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 3, (0, 255, 0), -1, cv2.LINE_AA)
      cv2.putText(debug_img, "TF", (tf_center_u - 10, tf_center_v - 15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
      
      # Add text information
      text_lines = [
        f"Fruit {fruit_id}",
        f"Stem: {stem_angle:.1f}°" if stem_angle is not None else "Stem: N/A",
        f"Depth: {depth_info['depth_mm']:.0f}mm",
        f"YOLO→SAM"
      ]
      
      text_x = max(5, x1)
      text_y = max(25, y1 - 10)
      
      for i, line in enumerate(text_lines):
        y_pos = text_y + i * 20
        
        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return debug_img

  def bbox_callback(self, msg):
    """Process bounding box detections and create TFs"""
    if self.latest_image is None or self.latest_depth is None:
      return
    
    # Filter for target labels
    target_fruits = [yolo_out for yolo_out in msg.yolo_out_array if yolo_out.label in self.target_labels]
    
    if not target_fruits:
      # Clear stem analysis if no target fruits detected
      self.latest_stem_analysis = {}
      return
    
    print(f"Found {len(target_fruits)} target fruits (labels: {[f.label for f in target_fruits]})")
    
    # Store stem analysis results for visualization
    stem_analysis = {}
    
    for i, fruit in enumerate(target_fruits):
      bbox = fruit.tlbr  # [x1, y1, x2, y2]
      
      # Extract depth information from fruit bbox
      depth_info = self.extract_depth_from_bbox(self.latest_depth, bbox)
      
      if depth_info is None:
        print(f"Fruit {i} (label {fruit.label}): No valid depth data")
        continue
      
      # Segment stem region using SAM
      stem_region, stem_mask, stem_bbox_coords = self.segment_stem_region_with_sam(self.latest_image, bbox)
      
      if stem_region is None or stem_mask is None:
        print(f"Fruit {i} (label {fruit.label}): Failed to segment stem region")
        continue
      
      # Create stem-oriented TF using vector approach
      tf_result = self.create_stem_oriented_tf_with_vector(bbox, depth_info, stem_mask, stem_bbox_coords, i)
      
      # Store analysis results for visualization
      if tf_result[0]:  # Check if TF creation was successful
        tf_created, sam_fitted_bbox, sam_center_full, stem_angle = tf_result
        stem_analysis[i] = {
          'bbox': bbox,
          'stem_bbox': stem_bbox_coords,
          'sam_fitted_bbox': sam_fitted_bbox,
          'sam_center_full': sam_center_full,
          'stem_angle': stem_angle,
          'depth_info': depth_info,
          'label': fruit.label
        }
        print(f"Fruit {i} (label {fruit.label}): TF created successfully")
      else:
        stem_analysis[i] = {
          'bbox': bbox,
          'stem_bbox': stem_bbox_coords,
          'sam_fitted_bbox': None,
          'sam_center_full': None,
          'stem_angle': None,
          'depth_info': depth_info,
          'label': fruit.label
        }
        print(f"Fruit {i} (label {fruit.label}): Failed to create TF")
      
      print(f"Fruit {i}: Label {fruit.label}, Score {fruit.score:.3f}, BBox: {bbox}")
      print(f"  Depth: {depth_info['depth_mm']:.1f}mm ± {depth_info['depth_std']:.1f}mm")
      print("---")
    
    # Update stem analysis for visualization
    self.latest_stem_analysis = stem_analysis
    self.save_counter += 1

if __name__ == '__main__':
  rclpy.init()
  node = RipeFruitCropperNode()
  
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Shutting down ripe fruit cropper node...")
  finally:
    node.destroy_node()
    rclpy.shutdown()
