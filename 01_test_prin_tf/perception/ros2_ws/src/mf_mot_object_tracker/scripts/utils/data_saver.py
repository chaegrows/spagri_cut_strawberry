import cv2
import csv
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R

class DataSaver:
  """Utility class for saving enhanced images and TF data to CSV"""
  
  def __init__(self, enhanced_images_folder, save_enhanced_images=True, save_tf_data_csv=True):
    self.enhanced_images_folder = enhanced_images_folder
    self.save_enhanced_images = save_enhanced_images
    self.save_tf_data_csv = save_tf_data_csv
    self.enhanced_image_counter = 0
    
    # Create enhanced images folder if it doesn't exist
    if self.save_enhanced_images or self.save_tf_data_csv:
      os.makedirs(self.enhanced_images_folder, exist_ok=True)
      print(f"Enhanced images will be saved to: {self.enhanced_images_folder}")
      
      if self.save_tf_data_csv:
        # Initialize CSV file with headers
        print("====================================================================")
        self.csv_filepath = os.path.join(self.enhanced_images_folder, "tf_data.csv")
        self.initialize_csv_file()

  def initialize_csv_file(self):
    """Initialize CSV file with headers"""
    try:
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

  def save_tf_data_to_csv(self, keypoint_analysis, detailed_filename, tf_only_filename, latest_tf_info):
    """Save TF data to CSV file - only for tracked fruits with TF"""
    if not self.save_tf_data_csv:
      return
    
    try:
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
          tf_info = latest_tf_info.get(fruit_id, None)
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

  def increment_counter(self):
    """Increment the enhanced image counter"""
    self.enhanced_image_counter += 1

  def get_counter(self):
    """Get current counter value"""
    return self.enhanced_image_counter 