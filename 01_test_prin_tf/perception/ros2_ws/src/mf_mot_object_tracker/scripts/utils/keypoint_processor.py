import numpy as np
import math
from collections import deque

class KeypointProcessor:
  """Utility class for processing keypoint data and calculating fruit orientation"""
  
  def __init__(self, smoothing_window=5, confidence_threshold=0.7, 
               angle_change_threshold=15.0, position_change_threshold=0.05):
    self.smoothing_window = smoothing_window
    self.confidence_threshold = confidence_threshold
    self.angle_change_threshold = angle_change_threshold
    self.position_change_threshold = position_change_threshold
    
    # History tracking for each fruit
    self.fruit_history = {}  # fruit_id -> history data
    
  def calculate_fruit_orientation_from_keypoints(self, keypoints, bbox, verbose=False):
    """
    Calculate fruit orientation using keypoints
    keypoints[0] (white): 중간 (middle)
    keypoints[1] (yellow): 꼭지 (tip)  
    keypoints[2] (red): 끝 (end)
    
    Main axis: 꼭지 → 중간 (keypoints[1] → keypoints[0])
    Z-axis angle: 꼭지 → 끝 (keypoints[1] → keypoints[2]) direction in -y axis
    
    Args:
      keypoints: KeypointOut message with x, y, conf arrays (expecting 3 keypoints)
      bbox: Bounding box [x1, y1, x2, y2]
      verbose: Print debug information
    
    Returns:
      Dictionary with orientation analysis or None
    """
    try:
      # Debug: Print keypoint information
      if verbose:
        print(f"    Keypoint debug info:")
        print(f"      n_keypoints field: {getattr(keypoints, 'n_keypoints', 'NOT_SET')}")
        print(f"      x array: {keypoints.x} (length: {len(keypoints.x)})")
        print(f"      y array: {keypoints.y} (length: {len(keypoints.y)})")
        print(f"      conf array: {keypoints.conf} (length: {len(keypoints.conf)})")
      
      if len(keypoints.x) < 3:
        print(f"  Need at least 3 keypoints, got {len(keypoints.x)}")
        return None
      
      # keypoints 정의 (올바른 라벨링으로 수정)
      middle_x, middle_y = keypoints.x[0], keypoints.y[0]   # 중간 (middle)
      tip_x, tip_y = keypoints.x[1], keypoints.y[1]         # 꼭지 (tip)
      end_x, end_y = keypoints.x[2], keypoints.y[2]         # 끝 (end)
      
      middle_conf = keypoints.conf[0]  # 중간 신뢰도
      tip_conf = keypoints.conf[1]     # 꼭지 신뢰도
      end_conf = keypoints.conf[2]     # 끝 신뢰도
      
      # Debug: Check keypoint positions with correct labeling
      if verbose:
        print(f"    Corrected keypoint positions:")
        print(f"      keypoints[0] (white): 중간 ({middle_x:.1f}, {middle_y:.1f}) conf: {middle_conf:.3f}")
        print(f"      keypoints[1] (yellow): 꼭지 ({tip_x:.1f}, {tip_y:.1f}) conf: {tip_conf:.3f}")
        print(f"      keypoints[2] (red): 끝 ({end_x:.1f}, {end_y:.1f}) conf: {end_conf:.3f}")
      
      # Check confidence
      min_confidence = 0.3
      if tip_conf < min_confidence or middle_conf < min_confidence:
        print(f"  Low confidence: tip(꼭지)={tip_conf:.2f}, middle(중간)={middle_conf:.2f}")
      
      # 1. Main axis direction: 꼭지 → 중간 (keypoints[1] → keypoints[0])
      main_axis_vector = np.array([middle_x - tip_x, middle_y - tip_y])
      main_axis_length = np.linalg.norm(main_axis_vector)
      
      if main_axis_length < 1e-6:
        print("  꼭지와 중간 keypoints are too close")
        return None
      
      # Normalize main axis direction
      fruit_main_axis_2d = main_axis_vector / main_axis_length
      
      # 2. Z-axis angle: 꼭지 → 끝 (keypoints[1] → keypoints[2]) direction
      z_axis_vector = np.array([end_x - tip_x, end_y - tip_y])
      z_axis_length = np.linalg.norm(z_axis_vector)
      
      if z_axis_length < 1e-6:
        print("  꼭지와 끝 keypoints are too close")
        return None
      
      # Z축 방향 각도 계산 (-y축 방향 기준)
      z_axis_angle_rad = np.arctan2(-z_axis_vector[1], z_axis_vector[0])  # -y축 방향
      z_axis_angle_deg = np.degrees(z_axis_angle_rad)
      
      # Overall confidence is average of key keypoints (꼭지, 중간)
      overall_confidence = (tip_conf + middle_conf) / 2
      
      return {
        'fruit_main_axis_2d': fruit_main_axis_2d,     # 꼭지 → 중간 방향벡터
        'z_axis_angle_deg': z_axis_angle_deg,         # Z축 각도 (끝→꼭지 방향, -y축 기준)
        'z_axis_angle_rad': z_axis_angle_rad,
        'confidence': overall_confidence,
        'middle_conf': middle_conf,    # 중간 신뢰도
        'tip_conf': tip_conf,  # 꼭지 신뢰도
        'end_conf': end_conf,        # 끝 신뢰도
        'middle_point': np.array([middle_x, middle_y]),    # 중간 점
        'tip_point': np.array([tip_x, tip_y]), # 꼭지 점 (TF 원점)
        'end_point': np.array([end_x, end_y]),          # 끝 점
        'main_axis_length': main_axis_length,
        'z_axis_length': z_axis_length
      }
      
    except Exception as e:
      print(f"Error calculating fruit orientation from keypoints: {e}")
      return None

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

  def cleanup_history(self, fruit_id):
    """Clean up history for removed tracks"""
    if fruit_id in self.fruit_history:
      del self.fruit_history[fruit_id] 