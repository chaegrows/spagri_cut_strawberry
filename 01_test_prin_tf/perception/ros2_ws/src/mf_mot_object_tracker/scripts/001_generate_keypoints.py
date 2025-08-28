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
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from collections import deque
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut

class StableFruitKeypointTFNode(Node):
  def __init__(self, node_name='stable_fruit_keypoint_tf_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    self.save_counter = 0
    self.verbose_yolo_predict = True
    # Camera parameters (from params file)
    self.base_frame = 'camera_link'
    self.depth_frame = 'camera_depth_optical_frame'
    self.depth_min_mm = 0
    self.depth_max_mm = 1000
    
    # Camera intrinsics (will be updated from camera_info)
    self.fx = None  # Default values
    self.fy = None
    self.cx = None
    self.cy = None
    self.rgb_width = None  # RGB image width from camera_info
    self.rgb_height = None  # RGB image height from camera_info
    self.camera_info_received = False
    
    # TF information storage for reprojection
    self.latest_tf_info = {}
    
    # Enhanced image saving parameters
    self.declare_parameter('save_enhanced_images', True)  # Enable/disable saving
    self.declare_parameter('enhanced_images_folder', '/root/ros2_ws/src/mf_mot_object_tracker/scripts/enhanced_images')  # Save folder path
    self.declare_parameter('save_tf_data_csv', True)  # Enable/disable CSV saving
    
    self.save_enhanced_images = self.get_parameter('save_enhanced_images').get_parameter_value().bool_value
    self.enhanced_images_folder = self.get_parameter('enhanced_images_folder').get_parameter_value().string_value
    self.save_tf_data_csv = self.get_parameter('save_tf_data_csv').get_parameter_value().bool_value
    
    # Create enhanced images folder if it doesn't exist
    if self.save_enhanced_images or self.save_tf_data_csv:
      import os
      os.makedirs(self.enhanced_images_folder, exist_ok=True)
      print(f"Enhanced images will be saved to: {self.enhanced_images_folder}")
      
      if self.save_tf_data_csv:
        # Initialize CSV file with headers
        print("====================================================================")
        self.csv_filepath = os.path.join(self.enhanced_images_folder, "tf_data.csv")
        self.initialize_csv_file()
    
    # Image save counter for sequential naming
    self.enhanced_image_counter = 0
    
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
      # '/camera/camera_hand/color/image_rect_raw/compressed',
      '/camera/color/image_raw/compressed',
      self.image_callback,
      10
    )
    
    # Subscribe to depth image
    self.depth_subscription = self.create_subscription(
      CompressedImage,  # 다시 CompressedImage로 변경
      '/camera/depth/image_raw/compressedDepth',
      self.depth_callback,
      10
    )
    
    # Subscribe to camera info for accurate intrinsics
    self.camera_info_subscription = self.create_subscription(
      CameraInfo,
      # '/camera/camera_hand/color/camera_info',
      '/camera/color/camera_info',
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
    self.latest_tf_info = {} # Store TF reprojections for enhanced image

  def initialize_csv_file(self):
    """Initialize CSV file with headers"""
    try:
      import csv
      import os
      
      # CSV headers
      headers = [
        'timestamp',
        'frame_counter',
        'image_filename_detailed',
        'image_filename_tf_only',
        'fruit_id',
        'track_id',
        'bbox_label',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'keypoint_0_x', 'keypoint_0_y', 'keypoint_0_conf',  # 끝 (white)
        'keypoint_1_x', 'keypoint_1_y', 'keypoint_1_conf',  # 꼭지 (yellow)
        'keypoint_2_x', 'keypoint_2_y', 'keypoint_2_conf',  # 중간 (red)
        'tf_position_x', 'tf_position_y', 'tf_position_z',
        'tf_rotation_qx', 'tf_rotation_qy', 'tf_rotation_qz', 'tf_rotation_qw',
        'tf_z_axis_angle_deg',
        'depth_mm', 'depth_confidence',
        'keypoint_confidence',
        'overall_confidence',
        'tracking_status'
      ]
      
      # Write headers if file doesn't exist
      if not os.path.exists(self.csv_filepath):
        with open(self.csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(headers)
        print(f"  CSV file initialized: {self.csv_filepath}")
      else:
        print(f"  CSV file exists: {self.csv_filepath}")
        
    except Exception as e:
      print(f"  Error initializing CSV file: {e}")

  def save_tf_data_to_csv(self, keypoint_analysis, detailed_filename, tf_only_filename):
    """Save TF data to CSV file - only for tracked fruits with TF"""
    if not self.save_tf_data_csv:
      return
    
    try:
      import csv
      from datetime import datetime
      from scipy.spatial.transform import Rotation as R
      
      timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # milliseconds
      
      # Count how many TFs will actually be saved
      tf_count = 0
      
      with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for fruit_id, analysis in keypoint_analysis.items():
          # Only save data for tracked fruits with TF (positive IDs)
          if fruit_id <= 0:
            continue
            
          bbox = analysis['bbox']
          keypoint_orientation = analysis['keypoint_orientation']
          depth_info = analysis['depth_info']
          keypoints_data = analysis.get('keypoints_data', None)
          
          if keypoint_orientation is None or depth_info is None:
            continue
          
          # Get TF information - only proceed if TF exists
          tf_info = self.latest_tf_info.get(fruit_id, None)
          if tf_info is None:
            print(f"  Skipping fruit {fruit_id}: No TF information available")
            continue
          
          # Extract bbox information
          x1, y1, x2, y2 = bbox
          bbox_label = "fruit"  # Default label, could be extracted from bbox_obj if available
          
          # Extract keypoint information
          kp_data = [0, 0, 0] * 3  # Default values
          if keypoints_data is not None:
            for i in range(min(3, len(keypoints_data.x))):
              kp_data[i*3] = keypoints_data.x[i]      # x coordinate
              kp_data[i*3+1] = keypoints_data.y[i]    # y coordinate  
              kp_data[i*3+2] = keypoints_data.conf[i] # confidence
          
          # Extract TF position and rotation
          position = tf_info['position']
          rotation_matrix = tf_info['rotation_matrix']
          z_axis_angle = tf_info['z_axis_angle']
          
          # Convert rotation matrix to quaternion
          r = R.from_matrix(rotation_matrix)
          quat = r.as_quat()  # [x, y, z, w]
          
          # Calculate confidences
          keypoint_confidence = keypoint_orientation['confidence']
          depth_confidence = depth_info.get('confidence', 1.0)
          overall_confidence = min(depth_confidence, keypoint_confidence)
          
          # Determine tracking status
          tracking_status = "TRACKED"  # Only tracked fruits reach this point
          
          # Write row data
          row_data = [
            timestamp,
            self.enhanced_image_counter,
            detailed_filename,
            tf_only_filename,
            abs(fruit_id),  # fruit_id (absolute value)
            fruit_id,       # track_id (positive for tracked)
            bbox_label,
            x1, y1, x2, y2,
            kp_data[0], kp_data[1], kp_data[2],   # keypoint 0 (끝/white)
            kp_data[3], kp_data[4], kp_data[5],   # keypoint 1 (꼭지/yellow)
            kp_data[6], kp_data[7], kp_data[8],   # keypoint 2 (중간/red)
            position[0], position[1], position[2], # TF position
            quat[0], quat[1], quat[2], quat[3],   # TF rotation quaternion
            z_axis_angle,                         # Z-axis angle in degrees
            depth_info['depth_mm'],               # Depth in mm
            depth_confidence,                     # Depth confidence
            keypoint_confidence,                  # Keypoint confidence
            overall_confidence,                   # Overall confidence
            tracking_status                       # Tracking status
          ]
          
          writer.writerow(row_data)
          tf_count += 1
          print(f"  TF data saved to CSV: fruit_id={fruit_id}, pos=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})")
      
      if tf_count > 0:
        print(f"  CSV: Saved {tf_count} TF records for frame {self.enhanced_image_counter}")
      else:
        print(f"  CSV: No TF data to save for frame {self.enhanced_image_counter}")
          
    except Exception as e:
      print(f"  Error saving TF data to CSV: {e}")

  def save_enhanced_image(self, enhanced_img, detection_count, image_type="enhanced"):
    """
    Save enhanced image to folder with sequential naming
    Args:
      enhanced_img: Enhanced OpenCV image
      detection_count: Number of detections in this frame
      image_type: Type of image ("detailed" or "tf_only")
    Returns:
      filename: Saved filename for CSV logging
    """
    if not self.save_enhanced_images or enhanced_img is None:
      return None
    
    try:
      from datetime import datetime
      import os
      
      # Create timestamp
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      
      # Sequential filename with detection count and type
      filename = f"{image_type}_{self.enhanced_image_counter:06d}_{timestamp}_det{detection_count}.jpg"
      filepath = os.path.join(self.enhanced_images_folder, filename)
      
      # Save image
      success = cv2.imwrite(filepath, enhanced_img)
      
      if success:
        print(f"  {image_type} image saved: {filename} ({detection_count} detections)")
        return filename
      else:
        print(f"  Failed to save {image_type} image: {filename}")
        return None
        
    except Exception as e:
      print(f"  Error saving {image_type} image: {e}")
      return None

  def image_callback(self, msg):
    """Store the latest RGB image for processing and create enhanced debug image"""
    self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.latest_header = msg.header
    
    # Create enhanced debug image with TF reprojection if keypoint analysis is available
    if self.latest_keypoint_analysis:
      detection_count = len(self.latest_keypoint_analysis)
      tf_count = len([k for k in self.latest_keypoint_analysis.keys() if k > 0])
      
      # Only proceed if there are actual TFs to save
      if tf_count > 0:
        print(f"  Debug: Drawing enhanced keypoint analysis on RGB image ({detection_count} detections, {tf_count} TFs)")
        
        # Create detailed enhanced image (with all information)
        detailed_enhanced_img = self.draw_keypoint_analysis_on_debug_image(
          self.latest_image.copy(), 
          self.latest_keypoint_analysis
        )
        
        # Create TF-only reprojection image (clean)
        tf_only_img = self.draw_tf_reprojection_only(
          self.latest_image.copy(),
          self.latest_keypoint_analysis
        )
        
        print(f"  Debug: Enhanced images created - detailed: {detailed_enhanced_img.shape}, TF-only: {tf_only_img.shape}")
        
        # Save both images and get filenames
        detailed_filename = self.save_enhanced_image(detailed_enhanced_img, detection_count, "detailed")
        tf_only_filename = self.save_enhanced_image(tf_only_img, tf_count, "tf_only")
        
        # Save TF data to CSV (only if both images were saved successfully)
        if detailed_filename is not None and tf_only_filename is not None:
          self.save_tf_data_to_csv(self.latest_keypoint_analysis, detailed_filename, tf_only_filename)
        
        # Increment counter after saving both
        self.enhanced_image_counter += 1
        
        # Publish the TF-only image (cleaner for reprojection visualization)
        enhanced_msg = self.bridge.cv2_to_compressed_imgmsg(tf_only_img, dst_format='jpg')
        enhanced_msg.header = msg.header
        self.enhanced_debug_pub.publish(enhanced_msg)
        print(f"  Debug: TF-only debug image published to /mf_perception/yolo_debug_enhanced/compressed")
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
      # Format: 16UC1; compressedDepth
      # Data structure: [header bytes] + [PNG compressed depth data]
      
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

  def extract_depth_from_keypoints_line(self, depth_image, keypoint1, keypoint2):
    """
    Extract depth values from pixels along the line between two keypoints
    Args:
      depth_image: Depth image
      keypoint1: (x, y) coordinates of first keypoint (꼭지) - in RGB image coordinates
      keypoint2: (x, y) coordinates of second keypoint (중간) - in RGB image coordinates
    Returns:
      Dictionary with depth information or None
    """
    if depth_image is None or keypoint1 is None or keypoint2 is None:
      print(f"  Debug: depth_image is None: {depth_image is None}, keypoint1 is None: {keypoint1 is None}, keypoint2 is None: {keypoint2 is None}")
      return None
    
    # Get image dimensions
    h, w = depth_image.shape
    print(f"  Debug: Depth image shape: {h}x{w}")
    
    # Original keypoint coordinates (from RGB image)
    x1_rgb, y1_rgb = int(keypoint1[0]), int(keypoint1[1])  # 꼭지
    x2_rgb, y2_rgb = int(keypoint2[0]), int(keypoint2[1])  # 중간
    print(f"  Debug: Original keypoints - 꼭지: ({x1_rgb}, {y1_rgb}), 중간: ({x2_rgb}, {y2_rgb})")
    
    # Scale keypoints to depth image coordinates
    # Use camera_info values if available, otherwise fallback to default
    rgb_width = self.rgb_width if self.rgb_width is not None else 1280
    rgb_height = self.rgb_height if self.rgb_height is not None else 720
    scale_x = w / rgb_width
    scale_y = h / rgb_height
    
    x1 = int(x1_rgb * scale_x)
    y1 = int(y1_rgb * scale_y)
    x2 = int(x2_rgb * scale_x)
    y2 = int(y2_rgb * scale_y)
    
    print(f"  Debug: Scaled keypoints - 꼭지: ({x1}, {y1}), 중간: ({x2}, {y2})")
    print(f"  Debug: Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
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
        print(f"  Debug: Keypoints are same and out of bounds")
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
      print(f"  Debug: No valid line pixels found")
      return None
    
    print(f"  Debug: Line pixels count: {len(line_pixels)}")
    
    # Extract depth values from line pixels
    depth_values = []
    for px, py in line_pixels:
      depth_val = depth_image[py, px]
      depth_values.append(depth_val)
    
    depth_values = np.array(depth_values, dtype=np.float32)
    print(f"  Debug: Raw depth values range: {np.min(depth_values):.1f} - {np.max(depth_values):.1f}")
    print(f"  Debug: Depth range filter: {self.depth_min_mm} - {self.depth_max_mm} mm")
    
    # Filter depth values within valid range
    valid_mask = (depth_values >= self.depth_min_mm) & (depth_values <= self.depth_max_mm)
    valid_depths = depth_values[valid_mask]
    
    print(f"  Debug: Valid depths count: {len(valid_depths)} / {len(depth_values)}")
    
    if len(valid_depths) == 0:
      print(f"  Debug: No valid depth values in range {self.depth_min_mm}-{self.depth_max_mm}mm")
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
    
    print(f"  Debug: Final depths: {final_depths}")
    # Use median as the representative depth
    median_depth_mm = np.median(final_depths)
    
    return {
      'depth_mm': median_depth_mm,
      'depth_m': median_depth_mm / 1000.0,
      'line_pixel_count': len(line_pixels),
      'valid_pixel_count': len(valid_depths),
      'filtered_pixel_count': len(final_depths),
      'depth_std': np.std(final_depths),
      'confidence': min(1.0, len(final_depths) / len(valid_depths)),
      'scaled_center_u': (x1 + x2) / 2,  # Center point in depth image coordinates
      'scaled_center_v': (y1 + y2) / 2   # Center point in depth image coordinates
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
      # Convert deque to list for slicing and ensure lengths match
      confidences_list = list(history['confidences'])
      
      # Make sure both arrays have the same length
      min_length = min(len(positions), len(confidences_list))
      positions = positions[-min_length:]  # Take last min_length positions
      confidences = np.array(confidences_list[-min_length:])  # Take last min_length confidences
      
      if len(confidences) == len(positions):
        smoothed_position = np.average(positions, axis=0, weights=confidences)
        smoothed_position = tuple(smoothed_position)
      else:
        # Fallback to simple average if lengths still don't match
        print(f"  Warning: Length mismatch in smoothing (pos: {len(positions)}, conf: {len(confidences)}), using simple average")
        smoothed_position = tuple(np.mean(positions, axis=0))
    else:
      smoothed_position = new_position
    
    history['last_valid_position'] = smoothed_position
    return smoothed_position

  def calculate_fruit_orientation_from_keypoints(self, keypoints, bbox):
    """
    Calculate fruit orientation using keypoints
    keypoints[0] (white): 과일 끝
    keypoints[1] (yellow): 과일 꼭지  
    keypoints[2] (red): 과일 중간
    
    Main axis: 꼭지 → 중간 (keypoints[1] → keypoints[2])
    Z-axis angle: 끝 → 꼭지 (keypoints[0] → keypoints[1]) direction in -y axis
    
    Args:
      keypoints: KeypointOut message with x, y, conf arrays (expecting 3 keypoints)
      bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
      Dictionary with orientation analysis or None
    """
    try:
      # Debug: Print keypoint information
      if self.verbose_yolo_predict:
        print(f"    Keypoint debug info:")
        print(f"      n_keypoints field: {getattr(keypoints, 'n_keypoints', 'NOT_SET')}")
        print(f"      x array: {keypoints.x} (length: {len(keypoints.x)})")
        print(f"      y array: {keypoints.y} (length: {len(keypoints.y)})")
        print(f"      conf array: {keypoints.conf} (length: {len(keypoints.conf)})")
      
      if len(keypoints.x) < 3:
        print(f"  Need at least 3 keypoints, got {len(keypoints.x)}")
        return None
      
      # keypoints 정의 (수정됨)
      white_x, white_y = keypoints.x[0], keypoints.y[0]     # 과일 끝
      yellow_x, yellow_y = keypoints.x[1], keypoints.y[1]   # 과일 꼭지
      red_x, red_y = keypoints.x[2], keypoints.y[2]         # 과일 중간
      
      white_conf = keypoints.conf[0]   # 과일 끝 신뢰도
      yellow_conf = keypoints.conf[1]  # 과일 꼭지 신뢰도
      red_conf = keypoints.conf[2]     # 과일 중간 신뢰도
      
      # Check confidence
      min_confidence = 0.3
      if yellow_conf < min_confidence or red_conf < min_confidence:
        print(f"  Low confidence: yellow(꼭지)={yellow_conf:.2f}, red(중간)={red_conf:.2f}")
      
      # 1. Main axis direction: 꼭지 → 중간 (keypoints[1] → keypoints[2])
      main_axis_vector = np.array([red_x - yellow_x, red_y - yellow_y])
      main_axis_length = np.linalg.norm(main_axis_vector)
      
      if main_axis_length < 1e-6:
        print("  꼭지와 중간 keypoints are too close")
        return None
      
      # Normalize main axis direction
      fruit_main_axis_2d = main_axis_vector / main_axis_length
      
      # 2. Z-axis angle: 끝 → 꼭지 (keypoints[0] → keypoints[1]) direction
      z_axis_vector = np.array([yellow_x - white_x, yellow_y - white_y])
      z_axis_length = np.linalg.norm(z_axis_vector)
      
      if z_axis_length < 1e-6:
        print("  끝과 꼭지 keypoints are too close")
        return None
      
      # Z축 방향 각도 계산 (-y축 방향 기준)
      z_axis_angle_rad = np.arctan2(-z_axis_vector[1], z_axis_vector[0])  # -y축 방향
      z_axis_angle_deg = np.degrees(z_axis_angle_rad)
      
      # Overall confidence is average of key keypoints (꼭지, 중간)
      overall_confidence = (yellow_conf + red_conf) / 2
      
      return {
        'fruit_main_axis_2d': fruit_main_axis_2d,     # 꼭지 → 중간 방향벡터
        'z_axis_angle_deg': z_axis_angle_deg,         # Z축 각도 (끝→꼭지 방향, -y축 기준)
        'z_axis_angle_rad': z_axis_angle_rad,
        'confidence': overall_confidence,
        'white_conf': white_conf,    # 끝 신뢰도
        'yellow_conf': yellow_conf,  # 꼭지 신뢰도
        'red_conf': red_conf,        # 중간 신뢰도
        'white_point': np.array([white_x, white_y]),    # 끝 점
        'yellow_point': np.array([yellow_x, yellow_y]), # 꼭지 점 (TF 원점)
        'red_point': np.array([red_x, red_y]),          # 중간 점
        'main_axis_length': main_axis_length,
        'z_axis_length': z_axis_length
      }
      
    except Exception as e:
      print(f"Error calculating fruit orientation from keypoints: {e}")
      return None

  def create_fruit_oriented_tf_from_keypoints(self, fruit_bbox, depth_info, keypoint_orientation, fruit_id):
    """
    Create TF with fruit orientation from keypoints
    Position: keypoints[1] (꼭지) as origin point
    Z-axis rotation: keypoints[0] → keypoints[1] direction in camera z,y plane
    """
    if depth_info is None or keypoint_orientation is None:
      return False
    
    # 1. Use keypoints[1] (꼭지) as TF origin
    yellow_point = keypoint_orientation['yellow_point']  # 꼭지
    
    # Convert to RGB coordinates if needed
    rgb_width = self.rgb_width if self.rgb_width is not None else 1280
    rgb_height = self.rgb_height if self.rgb_height is not None else 720
    
    # 2. Depth value from keypoints[1]-[2] line (꼭지-중간 라인)
    depth_m = depth_info['depth_m']

    # 3. Calculate 3D coordinates (camera frame) - using keypoints[1] (꼭지)
    x = (yellow_point[0] - self.cx) * depth_m / self.fx
    y = (yellow_point[1] - self.cy) * depth_m / self.fy  
    z = depth_m
    
    # Get Z-axis angle from keypoint analysis
    z_axis_angle_rad = keypoint_orientation['z_axis_angle_rad']
    keypoint_confidence = keypoint_orientation['confidence']
    
    # Check if confidence is sufficient
    overall_confidence = min(depth_info.get('confidence', 1.0), keypoint_confidence)
    
    if overall_confidence < self.confidence_threshold:
      print(f"  Low confidence for fruit {fruit_id}: {overall_confidence:.3f}")
    
    # Apply temporal smoothing
    z_axis_angle_deg = keypoint_orientation['z_axis_angle_deg']
    smoothed_angle = self.smooth_angle(fruit_id, z_axis_angle_deg, overall_confidence)
    smoothed_position = self.smooth_position(fruit_id, (x, y, z), overall_confidence)
    
    # Convert smoothed angle back to radians
    smoothed_angle_rad = math.radians(smoothed_angle)
    
    # Create 3D coordinate system
    # Camera frame: X-right, Y-down, Z-forward
    # Fruit frame: Origin at 꼭지, Z-axis rotated in camera z,y plane
    
    # Z축을 camera z,y 평면에서 회전
    # 기본 Z축 [0, 0, 1]에서 z,y 평면으로 각도만큼 회전
    cos_angle = math.cos(smoothed_angle_rad)
    sin_angle = math.sin(smoothed_angle_rad)
    
    # Z축: z,y 평면에서 회전된 방향
    z_axis = np.array([0.0, -sin_angle, cos_angle])  # -y축 방향 고려
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    # X축: 카메라 right 방향 유지
    x_axis = np.array([1.0, 0.0, 0.0])
    
    # Y축: 우수 좌표계 완성 (Y = Z × X)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # X축 재계산 (직교성 보장)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
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
    
    # Set smoothed translation (꼭지 위치)
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
    
    # Store TF info for reprojection
    self.latest_tf_info[fruit_id] = {
      'position': smoothed_position,
      'rotation_matrix': rotation_matrix,
      'z_axis_angle': smoothed_angle
    }
    
    print(f"  Fruit TF published: fruit_{fruit_id}_oriented")
    print(f"  Position: ({smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}, {smoothed_position[2]:.3f}) m (at 꼭지)")
    print(f"  Z-axis angle: {z_axis_angle_deg:.1f}° → {smoothed_angle:.1f}° (conf: {overall_confidence:.3f})")
    print(f"  Coordinate system: Origin=꼭지, Z=rotated in z,y plane")
    
    return True

  def draw_keypoint_analysis_on_debug_image(self, debug_img, keypoint_analysis):
    """Draw keypoint fruit orientation analysis results with TF reprojection on the debug image"""
    
    # Initialize TF info storage if not exists
    if not hasattr(self, 'latest_tf_info'):
      self.latest_tf_info = {}
    
    # Get TF reprojections for visualization
    tf_reprojections = self.reproject_tf_to_image(self.latest_tf_info, None) if hasattr(self, 'latest_tf_info') else {}
    
    # Add frame information overlay
    frame_info = f"Frame: {self.save_counter:06d} | Detections: {len(keypoint_analysis)} | Active Tracks: {len(self.fruit_tracks)}"
    cv2.putText(debug_img, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(debug_img, timestamp, (10, debug_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for fruit_id, analysis in keypoint_analysis.items():
      bbox = analysis['bbox']
      keypoint_orientation = analysis['keypoint_orientation']
      depth_info = analysis['depth_info']
      keypoints_data = analysis.get('keypoints_data', None)
      
      if keypoint_orientation is None:
        continue
      
      x1, y1, x2, y2 = bbox
      
      # Get orientation information
      z_axis_angle_deg = keypoint_orientation['z_axis_angle_deg']
      confidence = keypoint_orientation['confidence']
      white_conf = keypoint_orientation['white_conf']    # 끝
      yellow_conf = keypoint_orientation['yellow_conf']  # 꼭지
      red_conf = keypoint_orientation['red_conf']        # 중간
      white_point = keypoint_orientation['white_point']  # 끝
      yellow_point = keypoint_orientation['yellow_point']# 꼭지
      red_point = keypoint_orientation['red_point']      # 중간
      
      # Draw bounding box with different colors for tracked/untracked
      if fruit_id > 0:  # Tracked
        bbox_color = (0, 255, 0)  # Green for tracked
        bbox_thickness = 2  # 더 얇게
      else:  # Untracked
        bbox_color = (128, 128, 128)  # Gray for untracked
        bbox_thickness = 1  # 더 얇게
      
      cv2.rectangle(debug_img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
      
      # Draw all keypoints with updated colors and labels (더 작은 원)
      if keypoints_data is not None:
        for i, (kx, ky, kconf) in enumerate(zip(keypoints_data.x, keypoints_data.y, keypoints_data.conf)):
          if i == 0:  # 끝 (White keypoint)
            color = (255, 255, 255)  # White
            radius = max(3, int(6 * kconf))  # 더 작게
            label = f"end({kconf:.2f})"
          elif i == 1:  # 꼭지 (Yellow keypoint)  
            color = (0, 255, 255)    # Yellow
            radius = max(3, int(6 * kconf))  # 더 작게
            label = f"tip({kconf:.2f})"
          elif i == 2:  # 중간 (Red keypoint)
            color = (0, 0, 255)      # Red
            radius = max(3, int(6 * kconf))  # 더 작게
            label = f"middle({kconf:.2f})"
          else:  # Other keypoints (gray)
            color = (128, 128, 128)
            radius = max(2, int(4 * kconf))  # 더 작게
            label = f"KP{i}({kconf:.2f})"
          
          # Draw keypoint with thin border (더 작은 원과 얇은 테두리)
          cv2.circle(debug_img, (int(kx), int(ky)), radius, color, -1)
          cv2.circle(debug_img, (int(kx), int(ky)), radius+1, (0, 0, 0), 1)  # 얇은 검은 테두리
      
      # Draw line from 꼭지 to 중간 (main axis direction) - 더 얇고 짧게
      yellow_pt = (int(yellow_point[0]), int(yellow_point[1]))  # 꼭지
      red_pt = (int(red_point[0]), int(red_point[1]))           # 중간
      
      # 라인을 더 짧게 만들기 위해 중점에서 일정 거리만 그리기
      center_x = (yellow_pt[0] + red_pt[0]) / 2
      center_y = (yellow_pt[1] + red_pt[1]) / 2
      
      # 방향 벡터 계산
      dx = red_pt[0] - yellow_pt[0]
      dy = red_pt[1] - yellow_pt[1]
      length = math.sqrt(dx*dx + dy*dy)
      
      if length > 0:
        # 정규화
        dx /= length
        dy /= length
        
        # 짧은 라인 길이 (원래 길이의 60%)
        short_length = min(length * 0.6, 30)  # 최대 30픽셀
        
        # 새로운 시작점과 끝점
        start_x = int(center_x - dx * short_length / 2)
        start_y = int(center_y - dy * short_length / 2)
        end_x = int(center_x + dx * short_length / 2)
        end_y = int(center_y + dy * short_length / 2)
        
        # Draw thin 꼭지-중간 line (main axis)
        cv2.line(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)  # 얇은 노란 라인
        cv2.line(debug_img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)    # 얇은 빨간 오버레이
      
      # Draw Z-axis direction line (끝 → 꼭지) - 더 얇고 짧게
      white_pt = (int(white_point[0]), int(white_point[1]))     # 끝
      
      # Z축 라인도 더 짧게
      dx_z = yellow_pt[0] - white_pt[0]
      dy_z = yellow_pt[1] - white_pt[1]
      length_z = math.sqrt(dx_z*dx_z + dy_z*dy_z)
      
      if length_z > 0:
        dx_z /= length_z
        dy_z /= length_z
        
        # 짧은 Z축 라인 길이
        short_length_z = min(length_z * 0.7, 25)  # 최대 25픽셀
        
        z_end_x = int(white_pt[0] + dx_z * short_length_z)
        z_end_y = int(white_pt[1] + dy_z * short_length_z)
        
        cv2.line(debug_img, white_pt, (z_end_x, z_end_y), (255, 255, 255), 2)  # 얇은 흰 라인
        cv2.arrowedLine(debug_img, white_pt, (z_end_x, z_end_y), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)
      
      # Draw TF reprojection if available (더 얇고 짧게)
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # TF 축들도 더 짧게 만들기
        def shorten_axis(origin, axis_end, max_length=20):
          dx = axis_end[0] - origin[0]
          dy = axis_end[1] - origin[1]
          length = math.sqrt(dx*dx + dy*dy)
          if length > max_length:
            scale = max_length / length
            new_x = int(origin[0] + dx * scale)
            new_y = int(origin[1] + dy * scale)
            return (new_x, new_y)
          return axis_end
        
        # 축들을 더 짧게
        short_x_axis = shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = shorten_axis(origin, tf_proj['z_axis'], 20)
        
        # Draw coordinate axes (더 얇게)
        cv2.arrowedLine(debug_img, origin, short_x_axis, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.4)    # X-axis: Red
        cv2.arrowedLine(debug_img, origin, short_y_axis, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.4)   # Y-axis: Green  
        cv2.arrowedLine(debug_img, origin, short_z_axis, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.4)   # Z-axis: Blue
        
        # Draw origin point (꼭지 위치) - 더 작게
        cv2.circle(debug_img, origin, 4, (255, 255, 255), -1)  # 더 작은 원점
        cv2.circle(debug_img, origin, 5, (0, 0, 0), 1)         # 얇은 테두리
        
        # Add axis labels (더 작은 폰트)
        cv2.putText(debug_img, 'X', (short_x_axis[0]+3, short_x_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(debug_img, 'Y', (short_y_axis[0]+3, short_y_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_img, 'Z', (short_z_axis[0]+3, short_z_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
      
      # Add TF label - different for tracked vs untracked
      tf_center_u = int(yellow_point[0])  # 꼭지가 TF 원점
      tf_center_v = int(yellow_point[1])
      
      if fruit_id > 0:  # Tracked
        tf_label = f"TF_{fruit_id}"
        tf_color = (255, 0, 255)  # Magenta for tracked
      else:  # Untracked
        tf_label = f"D{abs(fruit_id)}"
        tf_color = (128, 128, 128)  # Gray for untracked
        
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf_color, 2)  # 얇은 텍스트
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
      
      # Add stability and tracking indicators
      stability_text = "STABLE" if confidence > 0.5 else "UNSTABLE"
      tracking_text = "TRACKED" if fruit_id > 0 else "UNTRACKED"
      
      # Add updated text information with better formatting
      text_lines = [
        f"Fruit {abs(fruit_id)} [{tracking_text}] [{stability_text}]",
        f"Z-axis angle (end-tip): {z_axis_angle_deg:.1f}deg",
        f"end: {white_conf:.2f} | tip: {yellow_conf:.2f} | middle: {red_conf:.2f}",
        f"Confidence: {confidence:.2f}",
      ]
      
      # Add depth info only if available
      if depth_info is not None:
        text_lines.extend([
          f"Depth(line): {depth_info['depth_mm']:.0f}mm",
          f"Line pixels: {depth_info['filtered_pixel_count']}/{depth_info['line_pixel_count']}",
        ])
      else:
        text_lines.append("No depth data")
      
      text_x = max(5, x1 + 60)
      text_y = max(25, y1 + 60)
      
      for i, line in enumerate(text_lines):
        y_pos = text_y + i * 16  # 줄 간격도 조금 줄임
        
        # Draw text background with better visibility (더 얇은 테두리)
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (0, 0, 0), -1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (255, 255, 255), 1)
        
        # Draw text (더 작은 폰트)
        text_color = (255, 255, 255) if confidence > 0.5 else (100, 100, 255)
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
    
    return debug_img

  def reproject_tf_to_image(self, tf_info, camera_matrix):
    """
    Reproject TF coordinate system back to image for visualization
    """
    reprojected_points = {}
    
    for fruit_id, tf_data in tf_info.items():
      position = tf_data['position']  # 3D position
      rotation_matrix = tf_data['rotation_matrix']  # 3x3 rotation matrix
      
      # Define coordinate system axes in 3D (더 짧은 길이)
      axis_length = 0.02  # 5cm에서 2cm로 줄임
      origin_3d = np.array(position)
      x_axis_3d = origin_3d + rotation_matrix[:, 0] * axis_length  # X axis (red)
      y_axis_3d = origin_3d + rotation_matrix[:, 1] * axis_length  # Y axis (green)  
      z_axis_3d = origin_3d + rotation_matrix[:, 2] * axis_length  # Z axis (blue)
      
      # Project to image coordinates
      def project_3d_to_image(point_3d):
        x, y, z = point_3d
        if z <= 0:  # 유효하지 않은 깊이 체크
          return None
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return (int(u), int(v))
      
      if position[2] > 0:  # Valid depth
        origin_2d = project_3d_to_image(origin_3d)
        x_axis_2d = project_3d_to_image(x_axis_3d)
        y_axis_2d = project_3d_to_image(y_axis_3d)
        z_axis_2d = project_3d_to_image(z_axis_3d)
        
        # 모든 투영점이 유효한지 확인
        if all(pt is not None for pt in [origin_2d, x_axis_2d, y_axis_2d, z_axis_2d]):
          reprojected_points[fruit_id] = {
            'origin': origin_2d,
            'x_axis': x_axis_2d,
            'y_axis': y_axis_2d,
            'z_axis': z_axis_2d
          }
    
    return reprojected_points

  #def draw_enhanced_debug_image(self, debug_img, keypoint_analysis, tf_reprojections):
    """
    Draw enhanced debug image with TF reprojections
    """
    for fruit_id, analysis in keypoint_analysis.items():
      # ... existing keypoint drawing code ...
      
      # Draw TF reprojection if available
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # Draw coordinate axes
        cv2.arrowedLine(debug_img, origin, tf_proj['x_axis'], (0, 0, 255), 3)    # X-axis: Red
        cv2.arrowedLine(debug_img, origin, tf_proj['y_axis'], (0, 255, 0), 3)   # Y-axis: Green  
        cv2.arrowedLine(debug_img, origin, tf_proj['z_axis'], (255, 0, 0), 3)   # Z-axis: Blue
        
        # Draw origin point
        cv2.circle(debug_img, origin, 8, (255, 255, 255), -1)
        cv2.circle(debug_img, origin, 10, (0, 0, 0), 2)
        
        # Add axis labels
        cv2.putText(debug_img, 'X', tf_proj['x_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(debug_img, 'Y', tf_proj['y_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(debug_img, 'Z', tf_proj['z_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return debug_img

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
      
      # Calculate fruit orientation from keypoints (수정된 함수 사용)
      keypoint_orientation = self.calculate_fruit_orientation_from_keypoints(keypoints, bbox)
      
      if keypoint_orientation is None:
        print(f"Keypoint {i}: Failed to calculate fruit orientation from keypoints")
        continue
      
      # Extract depth information from 꼭지-중간 keypoints line if depth is available
      depth_info = None
      position_3d = None
      
      if self.latest_depth is not None:
        # Extract depth information from 꼭지-중간 keypoints line (keypoints[1]-[2])
        yellow_point = keypoint_orientation['yellow_point']  # 꼭지
        red_point = keypoint_orientation['red_point']        # 중간
        
        depth_info = self.extract_depth_from_keypoints_line(self.latest_depth, yellow_point, red_point)
        
        if depth_info is not None:
          # Calculate 3D position for tracking - use 꼭지 point (keypoints[1])
          yellow_u = yellow_point[0]  # 꼭지 u coordinate
          yellow_v = yellow_point[1]  # 꼭지 v coordinate
          
          position_3d = self.pixel_to_3d_point(yellow_u, yellow_v, depth_info['depth_m'])
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
    associated_pairs = self.associate_detections_with_tracks(valid_detections)
    if len(associated_pairs) == 0:
      print("  Debug: No valid detections after tracking")
      # Clear keypoint analysis if no detections
      self.latest_keypoint_analysis = {}
      return
    
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
      
      # Only create TF for tracked detections with depth information
      tf_created = False
      if track_id > 0 and depth_info is not None:  # Positive track_id means it's being tracked
        tf_created = self.create_fruit_oriented_tf_from_keypoints(bbox, depth_info, keypoint_orientation, track_id)
      
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
      
      print(f"  Z-axis angle (끝→꼭지): {keypoint_orientation['z_axis_angle_deg']:.1f}deg (conf: {keypoint_orientation['confidence']:.3f})")
      print(f"  끝 conf: {keypoint_orientation['white_conf']:.3f}, 꼭지 conf: {keypoint_orientation['yellow_conf']:.3f}, 중간 conf: {keypoint_orientation['red_conf']:.3f}")
      print("---")
    
    # Update keypoint analysis for visualization
    self.latest_keypoint_analysis = keypoint_analysis
    self.save_counter += 1

  def associate_detections_with_tracks(self, detections):
    """
    Associate new detections with existing tracks based on position similarity
    Only tracks detections that have valid 3D position information
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
    
    # Filter detections - only process those with valid 3D position
    trackable_detections = []
    non_trackable_detections = []
    
    for det_idx, detection in enumerate(detections):
      if detection['position_3d'] is not None:
        trackable_detections.append((det_idx, detection))
      else:
        non_trackable_detections.append((det_idx, detection))
    
    print(f"  Tracking: {len(trackable_detections)} detections with depth, {len(non_trackable_detections)} without depth")
    
    unassociated_detections = list(range(len(trackable_detections)))
    
    # Try to associate each trackable detection with existing tracks
    for i, (det_idx, detection) in enumerate(trackable_detections):
      if i not in unassociated_detections:
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
        unassociated_detections.remove(i)
        print(f"  Associated detection {det_idx} with existing track {best_track_id} (dist: {best_distance:.3f}m)")
    
    # Create new tracks for unassociated trackable detections
    for i in unassociated_detections:
      det_idx, detection = trackable_detections[i]
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
    
    # Handle non-trackable detections (without depth) - assign temporary IDs for visualization only
    for det_idx, detection in non_trackable_detections:
      # Use negative IDs for non-tracked detections to distinguish them
      temp_id = -(det_idx + 1)  # -1, -2, -3, etc.
      associated_pairs.append((temp_id, detection))
      print(f"  Detection {det_idx}: No depth - assigned temporary ID {temp_id} (visualization only)")
    
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
    
    print(f"  Frame {self.frame_count}: {len([p for p in associated_pairs if p[0] > 0])} tracked, {len([p for p in associated_pairs if p[0] < 0])} untracked, {len(self.fruit_tracks)} active tracks")
    
    return associated_pairs

  def draw_tf_reprojection_only(self, debug_img, keypoint_analysis):
    """Draw only TF reprojection on clean image - minimal visualization"""
    
    # Initialize TF info storage if not exists
    if not hasattr(self, 'latest_tf_info'):
      self.latest_tf_info = {}
    
    # Get TF reprojections for visualization
    tf_reprojections = self.reproject_tf_to_image(self.latest_tf_info, None) if hasattr(self, 'latest_tf_info') else {}
    
    # Count actual TFs
    actual_tf_count = len([k for k in keypoint_analysis.keys() if k > 0 and k in self.latest_tf_info])
    
    # Add minimal frame information overlay
    # frame_info = f"Frame: {self.save_counter:06d} | TFs: {actual_tf_count}"
    # cv2.putText(debug_img, frame_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Only proceed if there are actual TFs to draw
    if actual_tf_count == 0:
      # Add "No TFs" message
      no_tf_msg = "No TFs available for visualization"
      cv2.putText(debug_img, no_tf_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      return debug_img
    
    for fruit_id, analysis in keypoint_analysis.items():
      # Only draw for tracked fruits (positive IDs) with TF
      if fruit_id <= 0:
        continue
        
      keypoint_orientation = analysis['keypoint_orientation']
      depth_info = analysis['depth_info']
      
      if keypoint_orientation is None or depth_info is None:
        continue
      
      # Check if TF actually exists
      if fruit_id not in self.latest_tf_info:
        continue
      
      yellow_point = keypoint_orientation['yellow_point']  # 꼭지 (TF 원점)
      
      # Draw TF reprojection if available
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # TF 축들을 적당한 길이로
        def shorten_axis(origin, axis_end, max_length=25):
          dx = axis_end[0] - origin[0]
          dy = axis_end[1] - origin[1]
          length = math.sqrt(dx*dx + dy*dy)
          if length > max_length:
            scale = max_length / length
            new_x = int(origin[0] + dx * scale)
            new_y = int(origin[1] + dy * scale)
            return (new_x, new_y)
          return axis_end
        
        # 축들을 적당한 길이로
        short_z_axis = shorten_axis(origin, tf_proj['z_axis'], 25)
        
        # Draw coordinate axes - clean and clear (Z축만)
        cv2.arrowedLine(debug_img, origin, short_z_axis, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)   # Z-axis: Blue
        
        # Draw origin point (꼭지 위치) - clear and visible
        cv2.circle(debug_img, origin, 3, (255, 255, 255), -1)  # White center
        cv2.circle(debug_img, origin, 4, (0, 0, 0), 2)         # Black border
        
        # Add axis labels - clear and readable
        # cv2.putText(debug_img, 'Z', (short_z_axis[0]+5, short_z_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add TF ID label - simple and clean
        tf_label = f"TF_{fruit_id}"
        cv2.putText(debug_img, tf_label, (origin[0]+10, origin[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)  # White text with thick outline
        cv2.putText(debug_img, tf_label, (origin[0]+10, origin[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)        # Black outline
        
        # Add minimal TF info
        tf_info_text = f"Pos: ({analysis['depth_info']['depth_mm']:.0f}mm)"
        cv2.putText(debug_img, tf_info_text, (origin[0]+10, origin[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(debug_img, tf_info_text, (origin[0]+10, origin[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return debug_img

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