import cv2
import numpy as np
import math
from datetime import datetime

class ImageVisualizer:
  """Utility class for drawing keypoint analysis and TF visualization on images"""
  
  def __init__(self):
    pass
    
  def shorten_axis(self, origin, axis_end, max_length=25):
    """Helper function to shorten axis visualization lines"""
    dx = axis_end[0] - origin[0]
    dy = axis_end[1] - origin[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length > max_length:
      scale = max_length / length
      new_x = int(origin[0] + dx * scale)
      new_y = int(origin[1] + dy * scale)
      return (new_x, new_y)
    return axis_end

  def draw_keypoint_analysis_on_debug_image(self, debug_img, keypoint_analysis, 
                                          tf_reprojections, save_counter, fruit_tracks_count):
    """Draw keypoint fruit orientation analysis results with TF reprojection on the debug image"""
    
    # Add frame information overlay
    frame_info = f"Frame: {save_counter:06d} | Detections: {len(keypoint_analysis)} | Active Tracks: {fruit_tracks_count}"
    cv2.putText(debug_img, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add timestamp
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
      
      # Extract keypoint information
      middle_conf = keypoint_orientation['middle_conf']    # 중간
      tip_conf = keypoint_orientation['tip_conf']  # 꼭지
      end_conf = keypoint_orientation['end_conf']        # 끝
      middle_point = keypoint_orientation['middle_point']  # 중간
      tip_point = keypoint_orientation['tip_point']# 꼭지
      end_point = keypoint_orientation['end_point']      # 끝
      
      # Draw bounding box with different colors for tracked/untracked
      if fruit_id > 0:  # Tracked
        bbox_color = (0, 255, 0)  # Green for tracked
        bbox_thickness = 2
      else:  # Untracked
        bbox_color = (128, 128, 128)  # Gray for untracked
        bbox_thickness = 1
      
      cv2.rectangle(debug_img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
      
      # Draw all keypoints with updated colors and labels
      if keypoints_data is not None:
        for i, (kx, ky, kconf) in enumerate(zip(keypoints_data.x, keypoints_data.y, keypoints_data.conf)):
          if i == 0:  # 중간 (White keypoint)
            color = (255, 255, 255)  # White
            radius = max(3, int(6 * kconf))
            label = f"middle({kconf:.2f})"
          elif i == 1:  # 꼭지 (Yellow keypoint)  
            color = (0, 255, 255)    # Yellow
            radius = max(3, int(6 * kconf))
            label = f"tip({kconf:.2f})"
          elif i == 2:  # 끝 (Red keypoint)
            color = (0, 0, 255)      # Red
            radius = max(3, int(6 * kconf))
            label = f"end({kconf:.2f})"
          else:  # Other keypoints (gray)
            color = (128, 128, 128)
            radius = max(2, int(4 * kconf))
            label = f"KP{i}({kconf:.2f})"
          
          # Draw keypoint with thin border
          cv2.circle(debug_img, (int(kx), int(ky)), radius, color, -1)
          cv2.circle(debug_img, (int(kx), int(ky)), radius+1, (0, 0, 0), 1)
      
      # Draw line from 꼭지 to 중간 (main axis direction)
      tip_pt = (int(tip_point[0]), int(tip_point[1]))  # 꼭지
      middle_pt = (int(middle_point[0]), int(middle_point[1]))           # 중간
      
      # 라인을 더 짧게 만들기 위해 중점에서 일정 거리만 그리기
      center_x = (tip_pt[0] + middle_pt[0]) / 2
      center_y = (tip_pt[1] + middle_pt[1]) / 2
      
      # 방향 벡터 계산
      dx = middle_pt[0] - tip_pt[0]
      dy = middle_pt[1] - tip_pt[1]
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
        cv2.line(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.line(debug_img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
      
      # Draw Z-axis direction line (꼭지 → 끝)
      end_pt = (int(end_point[0]), int(end_point[1]))     # 끝
      
      # Z축 라인도 더 짧게
      dx_z = end_pt[0] - tip_pt[0]
      dy_z = end_pt[1] - tip_pt[1]
      length_z = math.sqrt(dx_z*dx_z + dy_z*dy_z)
      
      if length_z > 0:
        dx_z /= length_z
        dy_z /= length_z
        
        # 짧은 Z축 라인 길이
        short_length_z = min(length_z * 0.7, 25)  # 최대 25픽셀
        
        z_end_x = int(tip_pt[0] + dx_z * short_length_z)
        z_end_y = int(tip_pt[1] + dy_z * short_length_z)
        
        cv2.line(debug_img, tip_pt, (z_end_x, z_end_y), (255, 255, 255), 2)
        cv2.arrowedLine(debug_img, tip_pt, (z_end_x, z_end_y), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)
      
      # Draw TF reprojection if available
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # TF 축들을 더 짧게 만들기
        short_x_axis = self.shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = self.shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = self.shorten_axis(origin, tf_proj['z_axis'], 20)
        
        # Draw coordinate axes
        cv2.arrowedLine(debug_img, origin, short_x_axis, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.4)    # X-axis: Red
        cv2.arrowedLine(debug_img, origin, short_y_axis, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.4)   # Y-axis: Green  
        cv2.arrowedLine(debug_img, origin, short_z_axis, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.4)   # Z-axis: Blue
        
        # Draw origin point (꼭지 위치)
        cv2.circle(debug_img, origin, 4, (255, 255, 255), -1)
        cv2.circle(debug_img, origin, 5, (0, 0, 0), 1)
        
        # Add axis labels
        cv2.putText(debug_img, 'X', (short_x_axis[0]+3, short_x_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(debug_img, 'Y', (short_y_axis[0]+3, short_y_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_img, 'Z', (short_z_axis[0]+3, short_z_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
      
      # Add TF label - different for tracked vs untracked
      tf_center_u = int(tip_point[0])  # 꼭지가 TF 원점
      tf_center_v = int(tip_point[1])
      
      if fruit_id > 0:  # Tracked
        tf_label = f"TF_{fruit_id}"
        tf_color = (255, 0, 255)  # Magenta for tracked
      else:  # Untracked
        tf_label = f"D{abs(fruit_id)}"
        tf_color = (128, 128, 128)  # Gray for untracked
        
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf_color, 2)
      cv2.putText(debug_img, tf_label, (tf_center_u+15, tf_center_v+15), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
      
      # Add stability and tracking indicators
      stability_text = "STABLE" if end_conf > 0.5 else "UNSTABLE"
      tracking_text = "TRACKED" if fruit_id > 0 else "UNTRACKED"
      
      # Add updated text information with better formatting
      text_lines = [
        f"Fruit {abs(fruit_id)} [{tracking_text}] [{stability_text}]",
        f"Z-axis angle (꼭지→끝): {keypoint_orientation['z_axis_angle_deg']:.1f}deg",
        f"end: {end_conf:.2f} | tip: {tip_conf:.2f} | middle: {middle_conf:.2f}",
        f"Confidence: {end_conf:.2f}",
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
        y_pos = text_y + i * 16
        
        # Draw text background with better visibility
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (0, 0, 0), -1)
        cv2.rectangle(debug_img, (text_x - 2, y_pos - text_h - 2), 
                     (text_x + text_w + 2, y_pos + 2), (255, 255, 255), 1)
        
        # Draw text
        text_color = (255, 255, 255) if end_conf > 0.5 else (100, 100, 255)
        cv2.putText(debug_img, line, (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
    
    return debug_img

  def draw_tf_reprojection_only(self, debug_img, keypoint_analysis, tf_reprojections, 
                               save_counter, latest_tf_info):
    """Draw only TF reprojection on clean image - minimal visualization"""
    
    # Count actual TFs
    actual_tf_count = len([k for k in keypoint_analysis.keys() if k > 0 and k in latest_tf_info])
    
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
      if fruit_id not in latest_tf_info:
        continue
      
      tip_point = keypoint_orientation['tip_point']  # 꼭지 (TF 원점)
      
      # Draw TF reprojection if available
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # 축들을 적당한 길이로
        short_z_axis = self.shorten_axis(origin, tf_proj['z_axis'], 25)
        
        # Draw coordinate axes - clean and clear (Z축만)
        cv2.arrowedLine(debug_img, origin, short_z_axis, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)   # Z-axis: White
        
        # Draw origin point (꼭지 위치) - clear and visible
        cv2.circle(debug_img, origin, 3, (255, 255, 255), -1)  # White center
        cv2.circle(debug_img, origin, 4, (0, 0, 0), 2)         # Black border
        
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