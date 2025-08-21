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
    
    # TF broadcaster
    self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    # Initialize MobileSAM model
    try:
      self.sam_model = SAM("/workspace/ai_models/mobile_sam.pt")
      print("MobileSAM model loaded successfully")
    except Exception as e:
      print(f"Failed to load MobileSAM: {e}")
      self.sam_model = None
    
    # Create save directory if it doesn't exist
    os.makedirs(self.save_dir, exist_ok=True)
    
    # Subscribe to bbox output, RGB image, and depth image
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
    
    self.latest_image = None
    self.latest_depth = None
    self.latest_header = None
    
    print('Ripe fruit cropper node is initialized')
    print(f'Saving cropped images to: {self.save_dir}')

  def image_callback(self, msg):
    """Store the latest RGB image for processing"""
    self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.latest_header = msg.header

  def depth_callback(self, msg):
    """Store the latest depth image"""
    self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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

  def create_fruit_tf(self, bbox, depth_info, fruit_id):
    """Create TF for detected fruit"""
    if depth_info is None:
      return False
    
    x1, y1, x2, y2 = bbox
    
    # Use center of bounding box
    center_u = (x1 + x2) // 2
    center_v = (y1 + y2) // 2
    
    # Convert to 3D point
    x, y, z = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
    
    # Create transform message
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = self.depth_frame  # camera_hand_depth_optical_frame
    t.child_frame_id = f'fruit_{fruit_id}'
    
    # Set translation
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    
    # Set rotation (identity for now, could be modified based on stem direction)
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0
    
    # Broadcast transform
    self.tf_broadcaster.sendTransform(t)
    
    print(f"  TF published: fruit_{fruit_id} at ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"  Depth stats: {depth_info['depth_mm']:.1f}mm, std: {depth_info['depth_std']:.1f}mm")
    print(f"  Pixels used: {depth_info['filtered_pixel_count']}/{depth_info['valid_pixel_count']}")
    
    return True

  def create_stem_oriented_tf(self, bbox, depth_info, stem_angle, fruit_id):
    """Create TF with orientation based on stem direction"""
    if depth_info is None or stem_angle is None:
      return self.create_fruit_tf(bbox, depth_info, fruit_id)
    
    x1, y1, x2, y2 = bbox
    center_u = (x1 + x2) // 2
    center_v = (y1 + y2) // 2
    
    # Convert to 3D point
    x, y, z = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
    
    # Create transform with stem orientation
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = self.depth_frame
    t.child_frame_id = f'fruit_{fruit_id}_oriented'
    
    # Set translation
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    
    # Convert stem angle to quaternion
    # Assuming stem_angle is in degrees and represents rotation around z-axis
    stem_rad = math.radians(stem_angle)
    quat = quaternion_from_euler(0, 0, stem_rad)
    
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    
    # Broadcast transform
    self.tf_broadcaster.sendTransform(t)
    
    print(f"  Oriented TF published: fruit_{fruit_id}_oriented")
    print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}) m, Stem angle: {stem_angle:.1f}°")
    
    return True

  def segment_stem_region_with_sam(self, image, fruit_bbox):
    """Use MobileSAM to segment the stem region above the fruit"""
    if self.sam_model is None:
      return None, None
    
    try:
      x1, y1, x2, y2 = fruit_bbox
      
      # Define stem region bbox
      stem_y1 = max(0, y1 - 40)
      stem_y2 = y1 + 5
      stem_x1 = max(0, x1 - 15)
      stem_x2 = min(image.shape[1], x2 + 15)
      
      stem_bbox = [stem_x1, stem_y1, stem_x2, stem_y2]
      
      # Run MobileSAM prediction
      results = self.sam_model.predict(image, bboxes=[stem_bbox])
      
      if results and len(results) > 0:
        masks = results[0].masks
        if masks is not None and len(masks.data) > 0:
          mask = masks.data[0].cpu().numpy().astype(np.uint8)
          
          stem_region = image[stem_y1:stem_y2, stem_x1:stem_x2]
          stem_mask = mask[stem_y1:stem_y2, stem_x1:stem_x2]
          
          return stem_region, stem_mask
      
      return None, None
      
    except Exception as e:
      print(f"MobileSAM stem segmentation failed: {e}")
      return None, None

  def analyze_stem_direction_simple(self, stem_region, stem_mask):
    """Simple stem direction analysis using SAM mask"""
    if stem_region is None or stem_mask is None:
      return None
    
    binary_mask = (stem_mask > 0).astype(np.uint8) * 255
    masked_pixels = np.column_stack(np.where(binary_mask > 0))
    
    if len(masked_pixels) < 20:
      return None
    
    # PCA analysis
    mean = np.mean(masked_pixels, axis=0)
    centered_points = masked_pixels - mean
    
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    pca_angle = math.degrees(math.atan2(principal_component[0], principal_component[1]))
    
    if pca_angle < 0:
      pca_angle += 180
    
    confidence = np.max(eigenvalues) / np.sum(eigenvalues)
    
    if confidence > 0.6:  # Only return if confident
      return pca_angle
    
    return None

  def bbox_callback(self, msg):
    """Process bounding box detections and create TFs"""
    if self.latest_image is None or self.latest_depth is None:
      return
    
    # Filter for label 7 (adjust based on your model)
    ripe_fruits = [yolo_out for yolo_out in msg.yolo_out_array if yolo_out.label == 7]
    
    if not ripe_fruits:
      return
    
    print(f"Found {len(ripe_fruits)} ripe fruits")
    
    for i, fruit in enumerate(ripe_fruits):
      bbox = fruit.tlbr  # [x1, y1, x2, y2]
      
      # Extract depth information
      depth_info = self.extract_depth_from_bbox(self.latest_depth, bbox)
      
      if depth_info is None:
        print(f"Fruit {i}: No valid depth data")
        continue
      
      # Create basic TF
      tf_created = self.create_fruit_tf(bbox, depth_info, i)
      
      if not tf_created:
        continue
      
      # Analyze stem direction for oriented TF
      stem_region, stem_mask = self.segment_stem_region_with_sam(self.latest_image, bbox)
      stem_angle = self.analyze_stem_direction_simple(stem_region, stem_mask)
      
      if stem_angle is not None:
        self.create_stem_oriented_tf(bbox, depth_info, stem_angle, i)
      
      # Save debug images
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      
      if stem_region is not None:
        filename = f"fruit_tf_{timestamp}_{self.save_counter:04d}_{i:02d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, stem_region)
      
      print(f"Fruit {i}: Score {fruit.score:.3f}, BBox: {bbox}")
      if stem_angle is not None:
        print(f"  Stem angle: {stem_angle:.1f}°")
      print("---")
    
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
