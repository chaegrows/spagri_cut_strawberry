import numpy as np
import cv2

class DepthProcessor:
  """Utility class for processing depth images and extracting depth information"""
  
  def __init__(self, depth_min_mm=0, depth_max_mm=1000):
    self.depth_min_mm = depth_min_mm
    self.depth_max_mm = depth_max_mm
    
  def extract_depth_from_keypoints_line(self, depth_image, keypoint1, keypoint2, 
                                       rgb_width=1280, rgb_height=720):
    """
    Extract depth values from pixels along the line between two keypoints
    Args:
      depth_image: Depth image
      keypoint1: (x, y) coordinates of first keypoint (꼭지) - in RGB image coordinates
      keypoint2: (x, y) coordinates of second keypoint (중간) - in RGB image coordinates
      rgb_width: RGB image width for scaling
      rgb_height: RGB image height for scaling
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
    # Use max as the representative depth (closest valid point)
    median_depth_mm = np.max(final_depths)
    
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

  def pixel_to_3d_point(self, u, v, depth_m, fx, fy, cx, cy):
    """Convert pixel coordinates and depth to 3D point in camera frame"""
    # Convert to 3D coordinates in camera frame
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    
    return x, y, z 