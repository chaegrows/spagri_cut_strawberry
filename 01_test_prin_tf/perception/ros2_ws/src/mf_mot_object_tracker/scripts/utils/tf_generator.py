import numpy as np
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class TFGenerator:
  """Utility class for generating TF transforms from keypoint data"""
  
  def __init__(self, depth_frame='camera_depth_optical_frame'):
    self.depth_frame = depth_frame
    
  def create_fruit_oriented_tf_from_keypoints(self, fruit_bbox, depth_info, keypoint_orientation, 
                                            fruit_id, smoothed_angle, smoothed_position, 
                                            tf_broadcaster, clock):
    """
    Create TF with fruit orientation from keypoints
    Position: keypoints[1] (꼭지) as origin point
    X-axis: Camera depth direction (toward object)
    Z-axis: Fruit orientation based on keypoints
    """
    if depth_info is None or keypoint_orientation is None:
      return False, None
    
    # Convert smoothed angle back to radians
    smoothed_angle_rad = math.radians(smoothed_angle)
    
    # Create 3D coordinate system
    # Camera frame: X-right, Y-down, Z-forward (depth direction)
    # TF frame: X-forward (toward object), Y and Z based on fruit orientation
    
    # TF X축: 카메라에서 물체로의 방향 (depth 방향)
    tf_x_axis = np.array([0.0, 0.0, 1.0])  # Camera Z-axis (forward/depth)
    
    # Z축을 camera z,y 평면에서 회전하여 과일의 방향성 표현
    cos_angle = math.cos(smoothed_angle_rad)
    sin_angle = math.sin(smoothed_angle_rad)
    
    # TF Z축: z,y 평면에서 회전된 방향 (fruit orientation)
    tf_z_axis = np.array([0.0, -sin_angle, cos_angle])  # -y축 방향 고려
    tf_z_axis = tf_z_axis / (np.linalg.norm(tf_z_axis) + 1e-8)
    
    # TF Y축: 우수 좌표계 완성 (Y = Z × X)
    tf_y_axis = np.cross(tf_z_axis, tf_x_axis)
    tf_y_axis = tf_y_axis / (np.linalg.norm(tf_y_axis) + 1e-8)
    
    # TF X축 재계산 (직교성 보장) - 카메라에서 물체로의 방향
    tf_x_axis = np.cross(tf_y_axis, tf_z_axis)
    tf_x_axis = tf_x_axis / (np.linalg.norm(tf_x_axis) + 1e-8)
    
    # Create rotation matrix [X Y Z] (column vectors)
    rotation_matrix = np.column_stack([tf_x_axis, tf_y_axis, tf_z_axis])
    
    # Verify orthogonality
    det = np.linalg.det(rotation_matrix)
    if abs(det - 1.0) > 0.1:
      print(f"  Warning: Rotation matrix determinant = {det:.3f} for fruit {fruit_id}")
    
    # Convert to quaternion
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]
    
    # Create transform
    t = TransformStamped()
    t.header.stamp = clock.now().to_msg()
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
    tf_broadcaster.sendTransform(t)
    
    # Store TF info for reprojection
    tf_info = {
      'position': smoothed_position,
      'rotation_matrix': rotation_matrix,
      'z_axis_angle': smoothed_angle
    }
    
    print(f"  Fruit TF published: fruit_{fruit_id}_oriented")
    print(f"  Position: ({smoothed_position[0]:.3f}, {smoothed_position[1]:.3f}, {smoothed_position[2]:.3f}) m (at 꼭지)")
    print(f"  Z-axis angle: {keypoint_orientation['z_axis_angle_deg']:.1f}° → {smoothed_angle:.1f}°")
    print(f"  Coordinate system: Origin=꼭지, X=toward object, Z=fruit orientation")
    
    return True, tf_info

  def reproject_tf_to_image(self, tf_info_dict, fx, fy, cx, cy):
    """
    Reproject TF coordinate system back to image for visualization
    """
    if not tf_info_dict:
      print("  Debug: No TF info provided for reprojection")
      return {}
    
    if fx is None or fy is None or cx is None or cy is None:
      print("  Error: Camera intrinsics not available for TF reprojection")
      return {}
    
    print(f"  Debug: Reprojecting {len(tf_info_dict)} TFs with camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    reprojected_points = {}
    
    for fruit_id, tf_data in tf_info_dict.items():
      try:
        position = tf_data['position']  # 3D position
        rotation_matrix = tf_data['rotation_matrix']  # 3x3 rotation matrix
        
        print(f"  Debug: Fruit {fruit_id} - Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        
        # Check if position is valid
        if position[2] <= 0:
          print(f"  Warning: Fruit {fruit_id} has invalid depth {position[2]:.3f} - skipping")
          continue
        
        # Define coordinate system axes in 3D (짧은 길이)
        axis_length = 0.02  # 2cm
        origin_3d = np.array(position)
        x_axis_3d = origin_3d + rotation_matrix[:, 0] * axis_length  # X axis (red)
        y_axis_3d = origin_3d + rotation_matrix[:, 1] * axis_length  # Y axis (green)  
        z_axis_3d = origin_3d + rotation_matrix[:, 2] * axis_length  # Z axis (blue)
        
        # Project to image coordinates
        def project_3d_to_image(point_3d):
          x, y, z = point_3d
          if z <= 0:  # 유효하지 않은 깊이 체크
            print(f"    Warning: Invalid depth {z:.3f} for point ({x:.3f}, {y:.3f}, {z:.3f})")
            return None
          u = fx * x / z + cx
          v = fy * y / z + cy
          
          # Check if projection is within reasonable image bounds (with margin)
          if u < -100 or u > 2000 or v < -100 or v > 2000:
            print(f"    Warning: Projection ({u:.1f}, {v:.1f}) is outside reasonable image bounds")
            return None
            
          return (int(u), int(v))
        
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
          print(f"  Debug: Fruit {fruit_id} reprojected successfully - Origin: {origin_2d}")
        else:
          print(f"  Warning: Fruit {fruit_id} reprojection failed - some points are invalid")
          print(f"    Origin: {origin_2d}, X-axis: {x_axis_2d}, Y-axis: {y_axis_2d}, Z-axis: {z_axis_2d}")
          
      except Exception as e:
        print(f"  Error: Failed to reproject fruit {fruit_id}: {e}")
        continue
    
    print(f"  Debug: Successfully reprojected {len(reprojected_points)} out of {len(tf_info_dict)} TFs")
    return reprojected_points 