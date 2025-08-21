import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def draw_rectangle_on_mask(mask, bbox, color=(0, 0, 255), thickness=2):
  """
  Draw rectangle on mask with proper coordinate adjustment
  
  Args:
    mask: Input mask (grayscale or BGR)
    bbox: Bounding box coordinates [x1, y1, x2, y2]
    color: Rectangle color (B, G, R)
    thickness: Rectangle line thickness
  
  Returns:
    mask_with_rectangle: Mask with rectangle drawn
  """
  mask_height, mask_width = mask.shape[:2]
  
  # Ensure coordinates are within mask bounds
  x1 = max(0, min(bbox[0], mask_width-1))
  y1 = max(0, min(bbox[1], mask_height-1))
  x2 = max(0, min(bbox[2], mask_width-1))
  y2 = max(0, min(bbox[3], mask_height-1))
  
  # Convert to BGR if mask is grayscale
  if len(mask.shape) == 2:
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  else:
    mask_vis = mask.copy()
  
  cv2.rectangle(mask_vis, (x1, y1), (x2, y2), color, thickness)
  return mask_vis

def scale_bbox_to_mask(bbox, original_size, mask_size):
  """
  Scale bounding box coordinates to match mask size
  
  Args:
    bbox: Original bounding box [x1, y1, x2, y2]
    original_size: (width, height) of original image
    mask_size: (width, height) of mask image
  
  Returns:
    scaled_bbox: Scaled bounding box coordinates
  """
  orig_width, orig_height = original_size
  mask_width, mask_height = mask_size
  
  scale_x = mask_width / orig_width
  scale_y = mask_height / orig_height
  
  scaled_bbox = [
    int(bbox[0] * scale_x),  # x1
    int(bbox[1] * scale_y),  # y1
    int(bbox[2] * scale_x),  # x2
    int(bbox[3] * scale_y)   # y2
  ]
  
  return scaled_bbox

def analyze_stem_direction(mask_path, box_coords=None):
  """
  Analyze strawberry stem direction from mask image
  
  Args:
    mask_path: Path to the mask image
    box_coords: Bounding box coordinates [x1, y1, x2, y2]
  
  Returns:
    direction_vector: Unit vector indicating stem direction
    center_point: Center point of the stem
    angle_degrees: Angle in degrees
  """
  
  # Load mask image
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  if mask is None:
    raise ValueError(f"Could not load mask image: {mask_path}")
  
  # If bounding box is provided, crop the mask
  if box_coords:
    x1, y1, x2, y2 = box_coords
    mask = mask[y1:y2, x1:x2]
  
  # Find contours in the mask
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  if not contours:
    raise ValueError("No contours found in mask")
  
  # Get the largest contour (assuming it's the stem)
  largest_contour = max(contours, key=cv2.contourArea)
  
  # Get all points from the contour
  points = largest_contour.reshape(-1, 2)
  
  # Method 1: Use PCA to find the principal direction
  pca = PCA(n_components=2)
  pca.fit(points)
  
  # First principal component gives the main direction
  direction_vector = pca.components_[0]
  
  # Calculate center point
  center_point = np.mean(points, axis=0)
  
  # Calculate angle in degrees
  angle_degrees = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
  
  return direction_vector, center_point, angle_degrees, points

def find_stem_endpoints(points, direction_vector, center_point):
  """
  Find the two endpoints of the stem along the principal direction
  """
  # Project all points onto the principal direction
  projections = np.dot(points - center_point, direction_vector)
  
  # Find the points with maximum and minimum projections
  max_idx = np.argmax(projections)
  min_idx = np.argmin(projections)
  
  endpoint1 = points[max_idx]
  endpoint2 = points[min_idx]
  
  return endpoint1, endpoint2

def determine_stem_tip_direction(points, direction_vector, center_point):
  """
  Determine which end of the stem is the tip (thinner end)
  """
  # Project points onto the principal direction
  projections = np.dot(points - center_point, direction_vector)
  
  # Divide points into two halves along the principal direction
  median_proj = np.median(projections)
  
  half1_points = points[projections <= median_proj]
  half2_points = points[projections > median_proj]
  
  # Calculate average distance from center line for each half
  # The tip should be thinner (smaller average distance)
  def avg_distance_from_line(pts, center, direction):
    # Vector perpendicular to direction
    perp_vector = np.array([-direction[1], direction[0]])
    # Distance from center line
    distances = np.abs(np.dot(pts - center, perp_vector))
    return np.mean(distances)
  
  dist1 = avg_distance_from_line(half1_points, center_point, direction_vector)
  dist2 = avg_distance_from_line(half2_points, center_point, direction_vector)
  
  # The end with smaller average distance is likely the tip
  if dist1 < dist2:
    # Tip is in the direction of negative projection
    tip_direction = -direction_vector
    tip_center = np.mean(half1_points, axis=0)
  else:
    # Tip is in the direction of positive projection
    tip_direction = direction_vector
    tip_center = np.mean(half2_points, axis=0)
  
  return tip_direction, tip_center

def visualize_stem_direction(mask_path, box_coords=None, output_path=None):
  """
  Visualize stem direction with arrow overlay
  """
  # Analyze stem direction
  direction_vector, center_point, angle_degrees, points = analyze_stem_direction(mask_path, box_coords)
  
  # Find stem tip direction
  tip_direction, tip_center = determine_stem_tip_direction(points, direction_vector, center_point)
  
  # Load original mask for visualization
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  if box_coords:
    x1, y1, x2, y2 = box_coords
    mask = mask[y1:y2, x1:x2]
    # Adjust coordinates for cropped image
    center_point_adjusted = center_point
    tip_center_adjusted = tip_center
  else:
    center_point_adjusted = center_point
    tip_center_adjusted = tip_center
  
  # Create visualization
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  
  # Plot 1: Original mask with direction analysis
  ax1.imshow(mask, cmap='gray')
  ax1.scatter(points[:, 0], points[:, 1], c='red', s=1, alpha=0.5, label='Stem points')
  ax1.scatter(center_point_adjusted[0], center_point_adjusted[1], c='blue', s=100, marker='o', label='Center')
  
  # Draw principal direction line
  line_length = 50
  start_point = center_point_adjusted - direction_vector * line_length
  end_point = center_point_adjusted + direction_vector * line_length
  ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', linewidth=2, label='Principal axis')
  
  ax1.set_title(f'Stem Analysis\nAngle: {angle_degrees:.1f}°')
  ax1.legend()
  ax1.axis('equal')
  
  # Plot 2: Stem direction arrow
  ax2.imshow(mask, cmap='gray')
  
  # Draw arrow pointing towards the tip
  arrow_length = 40
  arrow_start = tip_center_adjusted - tip_direction * arrow_length * 0.3
  arrow_end = tip_center_adjusted + tip_direction * arrow_length * 0.7
  
  ax2.annotate('', xy=arrow_end, xytext=arrow_start,
              arrowprops=dict(arrowstyle='->', color='red', lw=3, 
                            connectionstyle="arc3,rad=0"))
  
  ax2.scatter(tip_center_adjusted[0], tip_center_adjusted[1], c='yellow', s=100, marker='*', label='Stem tip area')
  ax2.set_title(f'Stem Direction\nTip Direction: {np.degrees(np.arctan2(tip_direction[1], tip_direction[0])):.1f}°')
  ax2.legend()
  ax2.axis('equal')
  
  plt.tight_layout()
  
  if output_path:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
  
  plt.show()
  
  return {
    'direction_vector': direction_vector,
    'tip_direction': tip_direction,
    'center_point': center_point,
    'tip_center': tip_center,
    'angle_degrees': angle_degrees,
    'tip_angle_degrees': np.degrees(np.arctan2(tip_direction[1], tip_direction[0]))
  }

# Example usage
if __name__ == "__main__":
  # Example with the provided bounding box
  mask_path = "sam_out/straw_color_A_000007_mask.png"
  box_coords = [343, 330, 371, 370]  # From the box.txt file
  
  try:
    result = visualize_stem_direction(mask_path, box_coords, "stem_direction_analysis.png")
    print("Stem direction analysis completed!")
    print(f"Principal direction angle: {result['angle_degrees']:.1f}°")
    print(f"Stem tip direction angle: {result['tip_angle_degrees']:.1f}°")
    print(f"Stem tip center: ({result['tip_center'][0]:.1f}, {result['tip_center'][1]:.1f})")
    
  except Exception as e:
    print(f"Error: {e}")
    print("Please check if the mask image exists and contains valid stem data.")
