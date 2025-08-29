#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Camera depth direction as TF X-axis
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import numpy as np
import cv2
import os
from datetime import datetime
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
import sys

# Import parameter file based on command line argument
if len(sys.argv) > 1 and sys.argv[1] == 'strawberry_harvesting_pollination':
  import params.mot_by_bbox.strawberry_harvesting_pollination as P
else:
  import utils.keypoint_tf_generation as P

# Set GPU device from parameters
os.environ["CUDA_VISIBLE_DEVICES"] = getattr(P, 'cuda_visible_devices', "3")

from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut

# Import utility classes
from utils.keypoint_processor import KeypointProcessor
from utils.depth_processor import DepthProcessor
from utils.tf_generator import TFGenerator
from utils.object_tracker import ObjectTracker
from utils.image_visualizer import ImageVisualizer
from utils.data_saver import DataSaver

class StableFruitKeypointTFNode(Node):
  def __init__(self, node_name='stable_fruit_keypoint_tf_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    self.save_counter = 0
    self.verbose_yolo_predict = getattr(P, 'verbose_yolo_predict', True)
    
    # Camera parameters (from params file)
    self.base_frame = getattr(P, 'base_frame', 'camera_link')
    self.depth_frame = getattr(P, 'depth_frame', 'camera_depth_optical_frame')
    self.depth_min_mm = getattr(P, 'depth_min_mm', 0)
    self.depth_max_mm = getattr(P, 'depth_max_mm', 1000)
    
    # Camera intrinsics (will be updated from camera_info)
    self.fx = None
    self.fy = None
    self.cx = None
    self.cy = None
    self.rgb_width = None
    self.rgb_height = None
    self.camera_info_received = False
    
    # Initialize utility classes
    self.keypoint_processor = KeypointProcessor(
      smoothing_window=getattr(P, 'smoothing_window', 5),
      confidence_threshold=getattr(P, 'confidence_threshold', 0.7),
      angle_change_threshold=getattr(P, 'angle_change_threshold', 15.0),
      position_change_threshold=getattr(P, 'position_change_threshold', 0.05)
    )
    
    self.depth_processor = DepthProcessor(
      depth_min_mm=self.depth_min_mm,
      depth_max_mm=self.depth_max_mm
    )
    
    self.tf_generator = TFGenerator(depth_frame=self.depth_frame)
    
    self.object_tracker = ObjectTracker(
      tracking_distance_threshold=getattr(P, 'tracking_distance_threshold', 0.1),
      max_missing_frames=getattr(P, 'max_missing_frames', 10)
    )
    
    self.image_visualizer = ImageVisualizer()
    
    # Enhanced image saving parameters (from params file)
    enhanced_images_folder = getattr(P, 'enhanced_images_folder', '/root/ros2_ws/src/mf_mot_object_tracker/scripts/enhanced_images')
    save_enhanced_images = getattr(P, 'save_enhanced_images', True)
    save_tf_data_csv = getattr(P, 'save_tf_data_csv', True)
    
    self.data_saver = DataSaver(
      enhanced_images_folder=enhanced_images_folder,
      save_enhanced_images=save_enhanced_images,
      save_tf_data_csv=save_tf_data_csv
    )
    
    # TF information storage for reprojection with timestamp tracking
    self.latest_tf_info = {}
    self.tf_info_timestamp = None
    
    # TF broadcaster
    self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    # Publisher for enhanced debug images
    enhanced_debug_topic = getattr(P, 'enhanced_debug_topic', P.yolo_debug_topic)
    self.enhanced_debug_pub = self.create_publisher(
      CompressedImage, 
      enhanced_debug_topic, 
      1
    )
    
    # Subscribe to keypoint output, RGB image, depth image, and YOLO debug (using params)
    keypoint_topic = getattr(P, 'keypoint_topic', P.keypoint_topic)
    self.keypoint_subscription = self.create_subscription(
      KeypointOutArray,
      keypoint_topic,
      self.keypoint_callback,
      10
    )
    
    # Also subscribe to bbox for additional info
    bbox_topic = getattr(P, 'bbox_topic', P.bbox_topic)
    self.bbox_subscription = self.create_subscription(
      YoloOutArray,
      bbox_topic,
      self.bbox_callback,
      10
    )
    
    # RGB image subscription (using params)
    rgb_topic = getattr(P, 'rgb_topic', P.rgb_topic)
    self.image_subscription = self.create_subscription(
      CompressedImage,
      rgb_topic,
      self.image_callback,
      10
    )
    
    # Subscribe to depth image (using params)
    depth_topic = getattr(P, 'depth_topic', P.depth_topic)
    self.depth_subscription = self.create_subscription(
      CompressedImage,
      depth_topic,
      self.depth_callback,
      10
    )

    # Subscribe to camera info for accurate intrinsics (using params)
    rgb_info_topic = getattr(P, 'rgb_info_topic', P.rgb_info_topic)
    self.camera_info_subscription = self.create_subscription(
      CameraInfo,
      rgb_info_topic,
      self.camera_info_callback,
      10
    )
    
    # Subscribe to YOLO debug image for enhancement (using params)
    yolo_debug_topic = getattr(P, 'yolo_debug_topic', P.yolo_debug_topic)
    self.yolo_debug_subscription = self.create_subscription(
      CompressedImage,
      yolo_debug_topic,
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
    self.keypoint_analysis_timestamp = None  # Track when analysis was last updated
    self.keypoint_analysis_image = None  # Store the image used for keypoint analysis

    # Print loaded parameters for verification
    print("====================================================================")
    print("Loaded parameters:")
    print(f"  RGB topic: {rgb_topic}")
    print(f"  Depth topic: {depth_topic}")
    print(f"  Camera info topic: {rgb_info_topic}")
    print(f"  Keypoint topic: {keypoint_topic}")
    print(f"  BBox topic: {bbox_topic}")
    print(f"  Enhanced debug topic: {enhanced_debug_topic}")
    print(f"  Base frame: {self.base_frame}")
    print(f"  Depth frame: {self.depth_frame}")
    print(f"  Depth range: {self.depth_min_mm}-{self.depth_max_mm} mm")
    print(f"  Save enhanced images: {save_enhanced_images}")
    print(f"  Save TF data CSV: {save_tf_data_csv}")
    print("====================================================================")

  def image_callback(self, msg):
    """Store the latest RGB image for processing and create enhanced debug image"""
    self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.latest_header = msg.header
    
    # Check camera intrinsics are available for reprojection
    if not self.camera_info_received:
      print("  Debug: Camera intrinsics not yet received - cannot perform TF reprojection")
      # Publish original image
      enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(self.latest_image, dst_format='jpg')
      enhanced_msg.header = msg.header
      self.enhanced_debug_pub.publish(enhanced_msg)
      return
    
    # Create enhanced debug image with TF reprojection if keypoint analysis is available
    if self.latest_keypoint_analysis and self.keypoint_analysis_timestamp is not None:
      # Check timing synchronization (within 100ms)
      current_time = self.get_clock().now()
      time_diff = abs((current_time.nanoseconds - self.keypoint_analysis_timestamp) / 1e9)
      
      if time_diff > 0.1:  # 100ms threshold
        print(f"  Warning: Image and keypoint analysis timing mismatch: {time_diff:.3f}s")
      
      # Use the image that was used for keypoint analysis (for proper synchronization)
      analysis_image = self.keypoint_analysis_image if self.keypoint_analysis_image is not None else self.latest_image
      
      if analysis_image is None:
        print(f"  Debug: No analysis image available - skipping visualization")
        # Publish original RGB image if no analysis image available
        enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(self.latest_image, dst_format='jpg')
        enhanced_msg.header = msg.header
        self.enhanced_debug_pub.publish(enhanced_msg)
        return
      
      detection_count = len(self.latest_keypoint_analysis)
      tf_count = len([k for k in self.latest_keypoint_analysis.keys() if k > 0])
      
      # Only proceed if there are actual TFs to save
      if tf_count > 0:
        print(f"  Debug: Drawing enhanced keypoint analysis on synchronized image ({detection_count} detections, {tf_count} TFs)")
        print(f"  Debug: Using stored analysis image shape: {analysis_image.shape}")
        print(f"  Debug: Camera intrinsics - fx:{self.fx:.1f}, fy:{self.fy:.1f}, cx:{self.cx:.1f}, cy:{self.cy:.1f}")
        
        # Get TF reprojections for visualization with error handling
        tf_reprojections = {}
        if self.latest_tf_info:
          try:
            tf_reprojections = self.tf_generator.reproject_tf_to_image(
              self.latest_tf_info, self.fx, self.fy, self.cx, self.cy
            )
            print(f"  Debug: TF reprojection successful - {len(tf_reprojections)} TFs reprojected")
            
            # Debug: Print reprojection details
            for fruit_id, points in tf_reprojections.items():
              origin = points['origin']
              print(f"    Fruit {fruit_id}: Origin at ({origin[0]}, {origin[1]})")
              
          except Exception as e:
            print(f"  Error: TF reprojection failed: {e}")
            tf_reprojections = {}
        else:
          print(f"  Debug: No TF info available for reprojection")
        
        # Create detailed enhanced image (with all information) using synchronized image
        detailed_enhanced_img = self.image_visualizer.draw_keypoint_analysis_on_debug_image(
          analysis_image.copy(), 
          self.latest_keypoint_analysis,
          tf_reprojections,
          self.save_counter,
          self.object_tracker.get_active_tracks_count()
        )
        
        # Create TF-only reprojection image (clean) using synchronized image
        tf_only_img = self.image_visualizer.draw_tf_reprojection_only(
          analysis_image.copy(),
          self.latest_keypoint_analysis,
          tf_reprojections,
          self.save_counter,
          self.latest_tf_info
        )
        
        print(f"  Debug: Enhanced images created - detailed: {detailed_enhanced_img.shape}, TF-only: {tf_only_img.shape}")
        
        # Save both images and get filenames
        detailed_filename = self.data_saver.save_enhanced_image(detailed_enhanced_img, detection_count, "detailed")
        tf_only_filename = self.data_saver.save_enhanced_image(tf_only_img, tf_count, "tf_only")
        
        # Save TF data to CSV (only if both images were saved successfully)
        if detailed_filename is not None and tf_only_filename is not None:
          self.data_saver.save_tf_data_to_csv(self.latest_keypoint_analysis, detailed_filename, tf_only_filename, self.latest_tf_info)
        
        # Increment counter after saving both
        self.data_saver.increment_counter()
        
        # Publish the TF-only image (cleaner for reprojection visualization)
        enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(tf_only_img, dst_format='jpg')
        enhanced_msg.header = msg.header
        self.enhanced_debug_pub.publish(enhanced_msg)
        print(f"  Debug: TF-only debug image published to enhanced debug topic")
      else:
        print(f"  Debug: No TFs detected ({detection_count} detections but 0 TFs) - skipping image save")
        # Still publish original image for debugging
        enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(self.latest_image, dst_format='jpg')
        enhanced_msg.header = msg.header
        self.enhanced_debug_pub.publish(enhanced_msg)
    else:
      print(f"  Debug: No keypoint analysis available - publishing original RGB image")
      # Publish original RGB image if no analysis available
      enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(self.latest_image, dst_format='jpg')
      enhanced_msg.header = msg.header
      self.enhanced_debug_pub.publish(enhanced_msg)

  def depth_callback(self, msg):
    """Store the latest depth image"""
    try:
      # ROS compressedDepth format handling
      if 'compressedDepth' in msg.format:
        # compressedDepth has a special header format
        # Skip the first 12 bytes (header) and decode the PNG data
        png_data = msg.data[12:]  # Skip header
        
        # Decode PNG data
        np_arr = np.frombuffer(png_data, np.uint8)
        self.latest_depth = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
        
        if self.latest_depth is not None:
          print(f"  Debug: Depth image received - shape: {self.latest_depth.shape}, dtype: {self.latest_depth.dtype}")
          print(f"  Debug: Depth range: {np.min(self.latest_depth):.1f} - {np.max(self.latest_depth):.1f}")
          print(f"  Debug: Format: {msg.format}")
        else:
          print("  Debug: Depth image decode failed after header skip")
      else:
        # Try regular compressed image decoding
        self.latest_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if self.latest_depth is not None:
          print(f"  Debug: Depth image received - shape: {self.latest_depth.shape}, dtype: {self.latest_depth.dtype}")
          print(f"  Debug: Depth range: {np.min(self.latest_depth):.1f} - {np.max(self.latest_depth):.1f}")
        else:
          print("  Debug: Regular compressed image decode failed")
          
    except Exception as e:
      print(f"  Debug: Error converting depth image: {e}")
      self.latest_depth = None

  def camera_info_callback(self, msg):
    """Update camera intrinsic parameters from camera_info"""
    if not self.camera_info_received:
      self.fx = msg.k[0]  # K[0,0]
      self.fy = msg.k[4]  # K[1,1]  
      self.cx = msg.k[2]  # K[0,2]
      self.cy = msg.k[5]  # K[1,2]
      self.camera_info_received = True
      self.rgb_width = msg.width
      self.rgb_height = msg.height
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
    if self.verbose_yolo_predict:
      # Debug: Print detailed keypoint information
      if len(msg.keypoints) > 1:
        for i, keypoint in enumerate(msg.keypoints):
          print(f"  Keypoint detection {i}:")
          print(f"    n_keypoints: {keypoint.n_keypoints}")
          print(f"    x array length: {len(keypoint.x)}")
          print(f"    y array length: {len(keypoint.y)}")
          print(f"    conf array length: {len(keypoint.conf)}")
          if len(keypoint.x) > 0:
            print(f"    x values: {keypoint.x}")
            print(f"    y values: {keypoint.y}")
            print(f"    conf values: {keypoint.conf}")
          else:
            print(f"    WARNING: Empty keypoint arrays!")
    
    # Process keypoints if both bbox and keypoints are available
    if self.latest_bboxes is not None:
      self.process_keypoint_detections()

  def yolo_debug_callback(self, msg):
    """Process YOLO debug image - now just store it, enhanced image is created in image_callback"""
    self.latest_yolo_debug = self.bridge.compressed_imgmsg_to_cv2(msg)
    print(f"  Debug: YOLO debug image received - shape: {self.latest_yolo_debug.shape}")

  def bbox_callback(self, msg):
    """Store bbox information for matching with keypoints"""
    self.latest_bboxes = msg
    # Process keypoints if both bbox and keypoints are available
    if self.latest_keypoints is not None:
      self.process_keypoint_detections()

  def process_keypoint_detections(self):
    """Process keypoint detections and create stable TFs"""
    # Only require keypoints and bboxes - depth and image are optional
    if (self.latest_keypoints is None or self.latest_bboxes is None):
      print("  Debug: Missing required data for keypoint processing")
      print(f"  Debug: latest_keypoints: {self.latest_keypoints is None}")
      print(f"  Debug: latest_bboxes: {self.latest_bboxes is None}")
      return
    
    # Store the current image for this analysis cycle (for proper synchronization)
    current_analysis_image = self.latest_image.copy() if self.latest_image is not None else None
    
    # Optional data for enhanced processing
    print(f"  Debug: latest_image available: {self.latest_image is not None}")
    print(f"  Debug: latest_depth available: {self.latest_depth is not None}")
    
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
      
      # Calculate fruit orientation from keypoints
      keypoint_orientation = self.keypoint_processor.calculate_fruit_orientation_from_keypoints(
        keypoints, bbox, self.verbose_yolo_predict
      )
      
      if keypoint_orientation is None:
        print(f"Keypoint {i}: Failed to calculate fruit orientation from keypoints")
        continue
      
      # Extract depth information from 꼭지-중간 keypoints line if depth is available
      depth_info = None
      position_3d = None
      
      if self.latest_depth is not None:
        # Extract depth information from 꼭지-중간 keypoints line (keypoints[1]-[0])
        tip_point = keypoint_orientation['tip_point']  # 꼭지
        middle_point = keypoint_orientation['middle_point']        # 중간
        
        rgb_width = self.rgb_width if self.rgb_width is not None else 1280
        rgb_height = self.rgb_height if self.rgb_height is not None else 720
        
        depth_info = self.depth_processor.extract_depth_from_keypoints_line(
          self.latest_depth, tip_point, middle_point, rgb_width, rgb_height
        )
        
        if depth_info is not None:
          # Calculate 3D position for tracking - use 꼭지 point (keypoints[1])
          tip_u = tip_point[0]  # 꼭지 u coordinate
          tip_v = tip_point[1]  # 꼭지 v coordinate
          
          position_3d = self.depth_processor.pixel_to_3d_point(
            tip_u, tip_v, depth_info['depth_m'], self.fx, self.fy, self.cx, self.cy
          )
          print(f"Detection {i}: Valid keypoint detection with depth at 3D position ({position_3d[0]:.3f}, {position_3d[1]:.3f}, {position_3d[2]:.3f}) (꼭지 기준)")
        else:
          print(f"Detection {i}: Keypoint detected but no valid depth data from 꼭지-중간 line")
      else:
        print(f"Detection {i}: Keypoint detected but no depth image available")
      
      # Store detection results
      valid_detections.append({
        'detection_idx': i,
        'bbox': bbox,
        'keypoint_orientation': keypoint_orientation,
        'depth_info': depth_info,
        'keypoints_data': keypoints,
        'position_3d': position_3d
      })
    
    # Second pass: Apply tracking to associate detections with consistent IDs
    print(f"Found {len(valid_detections)} valid detections, applying tracking...")
    associated_pairs = self.object_tracker.associate_detections_with_tracks(valid_detections)
    if len(associated_pairs) == 0:
      print("  Debug: No valid detections after tracking")
      # Clear keypoint analysis if no detections
      self.latest_keypoint_analysis = {}
      return
    
    # Third pass: Process tracked detections and create TFs
    keypoint_analysis = {}
    current_tf_info = {}  # Store TF info for this processing cycle
    
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
      
      # Only create TF for tracked detections with depth information
      tf_created = False
      if track_id > 0 and depth_info is not None:  # Positive track_id means it's being tracked
        # Apply temporal smoothing
        z_axis_angle_deg = keypoint_orientation['z_axis_angle_deg']
        keypoint_confidence = keypoint_orientation['confidence']
        overall_confidence = min(depth_info.get('confidence', 1.0), keypoint_confidence)
        
        # Calculate 3D position - use 꼭지 point (keypoints[1])
        tip_point = keypoint_orientation['tip_point']
        x = (tip_point[0] - self.cx) * depth_info['depth_m'] / self.fx
        y = (tip_point[1] - self.cy) * depth_info['depth_m'] / self.fy  
        z = depth_info['depth_m']
        
        smoothed_angle = self.keypoint_processor.smooth_angle(track_id, z_axis_angle_deg, overall_confidence)
        smoothed_position = self.keypoint_processor.smooth_position(track_id, (x, y, z), overall_confidence)
        
        tf_created, tf_info = self.tf_generator.create_fruit_oriented_tf_from_keypoints(
          bbox, depth_info, keypoint_orientation, track_id, 
          smoothed_angle, smoothed_position, self.tf_broadcaster, self.get_clock()
        )
        
        if tf_created and tf_info:
          current_tf_info[track_id] = tf_info
          print(f"  TF info stored for fruit {track_id}: position=({tf_info['position'][0]:.3f}, {tf_info['position'][1]:.3f}, {tf_info['position'][2]:.3f})")
      
      # Print detection information
      if track_id > 0:
        print(f"Fruit {track_id} (TRACKED): BBox: {bbox}")
      else:
        print(f"Detection {abs(track_id)} (UNTRACKED - no depth): BBox: {bbox}")
      
      if depth_info is not None:
        print(f"  Depth from 꼭지-중간 line: {depth_info['depth_mm']:.1f}mm ± {depth_info['depth_std']:.1f}mm")
        print(f"  Line pixels: {depth_info['line_pixel_count']}, Valid: {depth_info['valid_pixel_count']}, Filtered: {depth_info['filtered_pixel_count']}")
        print(f"  Depth confidence: {depth_info.get('confidence', 1.0):.3f}")
        if tf_created:
          print(f"  TF created: fruit_{track_id}_oriented")
        else:
          print(f"  TF not created (track_id: {track_id})")
      else:
        print(f"  No depth information available - visualization only")
      
      print(f"  Z-axis angle (꼭지→끝): {keypoint_orientation['z_axis_angle_deg']:.1f}deg (conf: {keypoint_orientation['confidence']:.3f})")
      print(f"  끝 conf: {keypoint_orientation['end_conf']:.3f}, 꼭지 conf: {keypoint_orientation['tip_conf']:.3f}, 중간 conf: {keypoint_orientation['middle_conf']:.3f}")
      print("---")
    
    # Update keypoint analysis and TF info atomically with timestamp and image
    self.latest_keypoint_analysis = keypoint_analysis
    self.latest_tf_info = current_tf_info  # Update TF info
    self.keypoint_analysis_timestamp = self.get_clock().now().nanoseconds  # Set timestamp
    self.tf_info_timestamp = self.keypoint_analysis_timestamp  # Sync TF timestamp
    self.keypoint_analysis_image = current_analysis_image  # Store the image used for this analysis
    
    print(f"Keypoint analysis updated with {len(keypoint_analysis)} detections, {len(current_tf_info)} TFs")
    self.save_counter += 1

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