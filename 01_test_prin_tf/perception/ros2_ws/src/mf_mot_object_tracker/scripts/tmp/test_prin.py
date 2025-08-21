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

  def mask_principal_axis(self, mask):
    """
    주성분축 분석: 중심(cx,cy), 주축 단위벡터 v(2,), 고유값(λ1, λ2)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
      return None
    
    pts = np.vstack([xs, ys]).T.astype(np.float32)  # (N,2)

    # 중심화
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eig(C)  # evecs[:,i]가 i번째 고유벡터
    i_max = np.argmax(evals)
    v = evecs[:, i_max]  # (2,)
    v = v / (np.linalg.norm(v) + 1e-8)

    return (float(mean[0]), float(mean[1])), (float(v[0]), float(v[1])), (float(evals[i_max]), float(evals[1 - i_max]))

  def segment_stem_region_with_sam(self, image, fruit_bbox):
    """Use MobileSAM to segment the stem region above the fruit"""
    if self.sam_model is None:
      return None, None
    
    try:
      x1, y1, x2, y2 = fruit_bbox
      
      # Define stem region bbox: 가로는 동일, 세로는 y1 위로 40pixel
      stem_y1 = max(0, y1 - 40)
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

  def create_stem_oriented_tf(self, fruit_bbox, depth_info, stem_mask, fruit_id):
    """Create TF with stem direction as z-axis and camera depth as x-axis"""
    if depth_info is None or stem_mask is None:
      return False
    
    x1, y1, x2, y2 = fruit_bbox
    
    # TF 중점: bbox 중심의 x좌표, y1 높이, depth 정보
    center_u = (x1 + x2) // 2
    center_v = y1  # y1 높이에 맞춤
    
    # Convert to 3D point
    x, y, z = self.pixel_to_3d_point(center_u, center_v, depth_info['depth_m'])
    
    # 주성분 분석으로 stem 방향 계산
    pca_result = self.mask_principal_axis(stem_mask)
    
    if pca_result is None:
      print(f"  Failed to analyze stem direction for fruit {fruit_id}")
      return False
    
    center, v, evals = pca_result
    vx, vy = v
    
    # 이미지 좌표계에서 카메라 좌표계로 변환
    # 이미지 좌표계: x=오른쪽, y=아래
    # 카메라 좌표계: x=오른쪽, y=아래, z=앞쪽(depth)
    
    # 주성분 벡터를 3D로 확장 (z 성분은 0으로 설정)
    stem_direction_3d = np.array([vx, vy, 0.0])
    stem_direction_3d = stem_direction_3d / np.linalg.norm(stem_direction_3d)
    
    # TF 좌표계 정의:
    # z축: stem 방향 (주성분 방향)
    # x축: camera depth 방향 (카메라 z축과 동일)
    # y축: z × x (오른손 법칙)
    
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
    
    print(f"  Stem-oriented TF published: fruit_{fruit_id}_stem_oriented")
    print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"  Stem direction (image): ({vx:.3f}, {vy:.3f})")
    
    return True

  def draw_stem_analysis_on_debug_image(self, debug_img, stem_analysis):
    """Draw stem analysis results on the YOLO debug image"""
    for fruit_id, analysis in stem_analysis.items():
      bbox = analysis['bbox']
      stem_bbox = analysis['stem_bbox']
      pca_result = analysis['pca_result']
      depth_info = analysis['depth_info']
      
      if pca_result is None:
        continue
      
      x1, y1, x2, y2 = bbox
      stem_x1, stem_y1, stem_x2, stem_y2 = stem_bbox
      center, v, evals = pca_result
      
      # Convert stem region coordinates to full image coordinates
      stem_center_x = stem_x1 + center[0]
      stem_center_y = stem_y1 + center[1]
      
      vx, vy = v
      
      # Draw stem region bbox
      cv2.rectangle(debug_img, (stem_x1, stem_y1), (stem_x2, stem_y2), (0, 255, 255), 2)
      
      # Draw principal axis
      scale = 30.0  # Fixed scale for visibility
      p1 = (int(round(stem_center_x - vx * scale)), int(round(stem_center_y - vy * scale)))
      p2 = (int(round(stem_center_x + vx * scale)), int(round(stem_center_y + vy * scale)))
      
      # Draw stem direction line (thick blue line)
      cv2.line(debug_img, p1, p2, (255, 0, 0), 3, cv2.LINE_AA)
      
      # Draw center point
      cv2.circle(debug_img, (int(round(stem_center_x)), int(round(stem_center_y))), 
                5, (255, 0, 0), -1, cv2.LINE_AA)
      
      # Draw TF position marker at fruit center
      tf_center_u = (x1 + x2) // 2
      tf_center_v = y1
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 8, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.circle(debug_img, (tf_center_u, tf_center_v), 3, (0, 255, 0), -1, cv2.LINE_AA)
      
      # Add text information
      angle_deg = math.degrees(math.atan2(vy, vx))
      confidence = evals[0] / sum(evals)
      
      # Text background for better visibility
      text_lines = [
        f"Fruit {fruit_id}",
        f"Stem: {angle_deg:.1f}°",
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
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return debug_img

  def bbox_callback(self, msg):
    """Process bounding box detections and create TFs"""
    if self.latest_image is None or self.latest_depth is None:
      return
    
    # Filter for label 7 (adjust based on your model)
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
      
      # Extract depth information from fruit bbox
      depth_info = self.extract_depth_from_bbox(self.latest_depth, bbox)
      
      if depth_info is None:
        print(f"Fruit {i}: No valid depth data")
        continue
      
      # Segment stem region using SAM
      stem_region, stem_mask, stem_bbox = self.segment_stem_region_with_sam(self.latest_image, bbox)
      
      if stem_region is None or stem_mask is None:
        print(f"Fruit {i}: Failed to segment stem region")
        continue
      
      # Analyze principal axis
      pca_result = self.mask_principal_axis(stem_mask)
      
      # Store analysis results for visualization
      stem_analysis[i] = {
        'bbox': bbox,
        'stem_bbox': stem_bbox,
        'pca_result': pca_result,
        'depth_info': depth_info
      }
      
      # Create stem-oriented TF
      tf_created = self.create_stem_oriented_tf(bbox, depth_info, stem_mask, i)
      
      print(f"Fruit {i}: Score {fruit.score:.3f}, BBox: {bbox}")
      print(f"  Depth: {depth_info['depth_mm']:.1f}mm ± {depth_info['depth_std']:.1f}mm")
      if pca_result is not None:
        center, v, evals = pca_result
        angle_deg = math.degrees(math.atan2(v[1], v[0]))
        print(f"  Stem direction: {angle_deg:.1f}° (confidence: {evals[0]/sum(evals):.3f})")
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
