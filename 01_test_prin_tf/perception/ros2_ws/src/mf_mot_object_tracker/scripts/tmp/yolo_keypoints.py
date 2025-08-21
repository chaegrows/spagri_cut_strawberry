#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2

from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2

from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut
# import params_mot_by_yolo as P

import os
import torch

# Force CPU usage for stability (avoid CUDA compatibility issues)
device = 'cpu'

# Check PyTorch and CUDA status
try:
  if torch.cuda.is_available():
    print("CUDA is available but using CPU for compatibility")
  else:
    print("CUDA not available, using CPU")
except Exception as e:
  print(f"Error checking CUDA: {e}")
  
print(f"Using device: {device}")

import sys
# import Metafarmers.app.ros2_ws.src.mf_mot_object_tracker.scripts.params_mot_by_keypoint_seedling as P
if sys.argv[1] == 'strawberry_harvesting_pollination':
  import params.mot_by_bbox.strawberry_harvesting_pollination as P

class YoloDetectionNode(BehaviorTreeServerNodeV2):
  repo = 'mf_perception'
  node_name = 'yolo_detection_node'
  def __init__(self, run_mode = 'server'):
    super().__init__(run_mode)

    # Load YOLO model with device specification
    self.model = YOLO(P.model_path)
    self.model.to(device)
    
    # Print model information
    print(f"Model loaded from: {P.model_path}")
    print(f"Model type: {self.model.task}")
    print(f"Model names: {self.model.names}")
    print(f"Device: {device}")
    
    self.bridge = CvBridge()

    if not P.rgb_topic.endswith('compressed'):
      msg_type = Image
      cb = self.rgb_callback
    else:
      msg_type = CompressedImage
      cb = self.compressed_rgb_callback


    self.pubs = {}
    self.pubs['yolo_debug'] = self.create_publisher(CompressedImage, P.yolo_debug_topic, 1)
    
    # Always create both publishers for keypoint mode
    if 'bbox' in P.inference_output_type or 'keypoint' in P.inference_output_type:
      self.pubs['bbox'] = self.create_publisher(YoloOutArray, P.bbox_topic, 1)
    if 'keypoint' in P.inference_output_type:
      self.pubs['keypoint'] = self.create_publisher(KeypointOutArray, P.keypoint_topic, 1)
    self.add_sequential_subscriber(msg_type, P.rgb_topic, cb, 1)
    
    
    if P.debug_on_depth:
      self.last_result = None
      self.pubs['yolo_depth_debug'] = self.create_publisher(CompressedImage, P.yolo_depth_debug_topic, 1)
      if not P.depth_topic.endswith('compressedDepth'):
        msg_type = Image
        cb = self.depth_callback
      else:
        msg_type = CompressedImage
        cb = self.compressed_depth_callback
      self.add_sequential_subscriber(msg_type, P.depth_topic, cb, 1)
    print('YOLOv11 keypoint detection node is initialized')
    print(f"Inference output type: {P.inference_output_type}")
    if 'keypoint' in P.inference_output_type:
      print("Keypoint mode: Both bbox and keypoint will be published")

  def rgb_callback(self, msg):
    cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    self.predict(cv_img, msg.header)

  def compressed_rgb_callback(self, msg):
    cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
    self.predict(cv_img, msg.header)

  def depth_callback(self, msg):
    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv_img = cv_img.astype(np.float32) / 1000
    cv_img[cv_img >= P.max_depth_meter] = P.max_depth_meter
    cv_img = (cv_img / P.max_depth_meter * 255).astype(np.uint8)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    self.predict(cv_img, msg.header, depth=True)

  def draw_keypoint_results(self, img, result):
    """
    Draw keypoint detection results on image
    
    Args:
      img: Input image (BGR format)
      result: YOLO detection result
    
    Returns:
      Annotated image with keypoints, boxes, and connections
    """
    try:
      # Define colors for different parts
      bbox_color = (0, 255, 0)  # Green for bbox
      keypoint_colors = [
        (255, 0, 0),    # Blue for keypoint 0
        (0, 255, 0),    # Green for keypoint 1  
        (0, 0, 255),    # Red for keypoint 2
        (255, 255, 0),  # Cyan for keypoint 3
        (255, 0, 255),  # Magenta for keypoint 4
        (0, 255, 255),  # Yellow for keypoint 5
        (128, 0, 128),  # Purple for keypoint 6
        (255, 165, 0),  # Orange for keypoint 7
      ]
      
      # Draw bounding boxes if available
      if hasattr(result, 'boxes') and result.boxes is not None:
        for idx, box in enumerate(result.boxes):
          # Draw bounding box
          x1, y1, x2, y2 = map(int, box.xyxy[0])
          cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)
          
          # Draw confidence and class
          conf = float(box.conf[0])
          cls = int(box.cls[0])
          class_name = self.model.names.get(cls, f'class_{cls}')
          label = f'{class_name}: {conf:.2f}'
          
          # Put text with background
          (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
          cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), bbox_color, -1)
          cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
      
      # Draw keypoints if available
      if hasattr(result, 'keypoints') and result.keypoints is not None and \
         result.keypoints.data is not None:
        
        keypoints_data = result.keypoints.data  # n_det, n_keypoints, 3
        
        for det_idx, keypoints in enumerate(keypoints_data):
          # Draw keypoints
          for kpt_idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:  # Draw all keypoints with minimal confidence
              x, y = int(x), int(y)
              color = keypoint_colors[kpt_idx % len(keypoint_colors)]
              
              # Adjust size and opacity based on confidence
              radius = max(2, int(8 * conf))  # Size based on confidence
              alpha = min(1.0, max(0.3, conf))  # Transparency based on confidence
              
              # Create overlay for transparency effect
              overlay = img.copy()
              
              # Draw keypoint circle with confidence-based styling
              if conf > 0.7:  # High confidence - solid circle
                cv2.circle(overlay, (x, y), radius, color, -1)
                cv2.circle(overlay, (x, y), radius, (255, 255, 255), 2)  # White border
              elif conf > 0.4:  # Medium confidence - thinner border
                cv2.circle(overlay, (x, y), radius, color, -1)
                cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)
              else:  # Low confidence - just outline
                cv2.circle(overlay, (x, y), radius, color, 2)
              
              # Blend with original image for transparency
              cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
              
              # Draw keypoint index and confidence
              label = f"{kpt_idx}({conf:.2f})"
              cv2.putText(img, label, (x+radius+2, y-radius-2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
              cv2.putText(img, label, (x+radius+2, y-radius-2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
          
          # Draw connections between keypoints (if you have skeleton definition)
          # This is a simple example - adjust based on your keypoint model
          self.draw_keypoint_connections(img, keypoints, keypoint_colors)
      
      # Add info text
      info_text = f"Device: {device} | Objects: "
      if hasattr(result, 'boxes') and result.boxes is not None:
        info_text += f"Boxes: {len(result.boxes)} | "
      if hasattr(result, 'keypoints') and result.keypoints is not None:
        info_text += f"Keypoints: {len(result.keypoints.data)}"
      
      cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
      cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
      
      return img
      
    except Exception as e:
      print(f"Error drawing keypoint results: {e}")
      return img

  def draw_keypoint_connections(self, img, keypoints, colors):
    """
    Draw connections between keypoints (skeleton)
    
    This function draws connections between keypoints based on your model's skeleton.
    Common keypoint models:
    - COCO Pose (17 keypoints): Human pose estimation
    - Custom models: Adjust connections based on your specific use case
    
    For your strawberry/fruit model, you might want to connect:
    - Stem to fruit body
    - Different parts of the fruit
    """
    try:
      # Get number of keypoints to determine connection strategy
      n_keypoints = len(keypoints)
      
      if n_keypoints >= 3:
        # Example for fruit/object keypoints - adjust based on your model
        # These are just examples - modify based on your specific keypoint meanings
        connections = []
        
        if n_keypoints == 3:  # Simple 3-point model
          connections = [(0, 1), (1, 2)]
        elif n_keypoints >= 4:  # 4+ point model
          connections = [(0, 1), (1, 2), (2, 3)]
          if n_keypoints >= 5:
            connections.extend([(0, 4), (3, 4)])  # Star pattern
        
        # Draw connections
        for start_idx, end_idx in connections:
          if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]
            
            # Only draw if both keypoints are visible (confidence > 0.3)
            if start_kpt[2] > 0.3 and end_kpt[2] > 0.3:
              start_pos = (int(start_kpt[0]), int(start_kpt[1]))
              end_pos = (int(end_kpt[0]), int(end_kpt[1]))
              
              # Draw connection line with gradient thickness based on confidence
              avg_conf = (start_kpt[2] + end_kpt[2]) / 2
              thickness = max(1, int(3 * avg_conf))
              
              cv2.line(img, start_pos, end_pos, (255, 255, 255), thickness + 1)  # White border
              cv2.line(img, start_pos, end_pos, (0, 255, 255), thickness)       # Yellow line
            
    except Exception as e:
      print(f"Error drawing keypoint connections: {e}")

  def compressed_depth_callback(self, msg):
    if self.last_result is None: return
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)
    cv_img = cv_img.astype(np.float32) / 1000
    cv_img[cv_img >= P.max_depth_meter] = P.max_depth_meter
    cv_img = (cv_img / P.max_depth_meter * 255).astype(np.uint8)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    self.predict(cv_img, msg.header, depth=True)

  def predict(self, cv_img, header, depth=False): # https://docs.ultralytics.com/ko/modes/predict/#inference-sources
    if depth:
      result = self.last_result
      if result is None:
        return
      # Use the same enhanced visualization for depth debug
      annotated_img = self.draw_keypoint_results(cv_img, result)
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_depth_debug'].publish(msg)
      return

    try:
      imgsz = P.det3d_params['image_size_wh'][1], P.det3d_params['image_size_wh'][0]
      # Run inference with device specification
      result = self.model.predict(cv_img, imgsz=imgsz, show_labels=False,
                                conf=P.conf_thresh, iou=P.iou_thresh, 
                                verbose=P.verbose_yolo_predict, device=device)[0]
      
      # Move result to CPU for processing
      if hasattr(result, 'cpu'):
        result = result.cpu()
      
      self.last_result = result
    except Exception as e:
      print(f"Error during prediction: {e}")
      return

    # publish debug image with enhanced keypoint visualization
    if P.publish_yolo_debug:
      annotated_img = self.draw_keypoint_results(cv_img.copy(), result)
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_debug'].publish(msg)

    # Publish bbox - for both bbox and keypoint modes
    if ('bbox' in P.inference_output_type or 'keypoint' in P.inference_output_type) and \
       'bbox' in self.pubs and self.pubs['bbox'].get_subscription_count() > 0:
      
      # For keypoint detection, extract bbox from keypoints if available
      if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        n_det = len(result.boxes)
        msg_to_pub = YoloOutArray()
        msg_to_pub.header = header
        
        for idx in range(n_det):
          yolo_out = YoloOut()
          yolo_out.header = header
          yolo_out.tlbr = [int(ele) for ele in result.boxes[idx].xyxy[0]]
          yolo_out.score = float(result.boxes[idx].conf[0])
          yolo_out.label = int(result.boxes[idx].cls[0])
          msg_to_pub.yolo_out_array.append(yolo_out)
        
        self.pubs['bbox'].publish(msg_to_pub)
        print(f"Published {n_det} bboxes")
      else:
        # Publish empty bbox array
        msg_to_pub = YoloOutArray()
        msg_to_pub.header = header
        self.pubs['bbox'].publish(msg_to_pub)

    # Publish keypoints
    if 'keypoint' in P.inference_output_type and 'keypoint' in self.pubs and \
       self.pubs['keypoint'].get_subscription_count() > 0:
      
      if hasattr(result, 'keypoints') and result.keypoints is not None and \
         result.keypoints.data is not None and len(result.keypoints.data) > 0:
        
        xyconf_array = result.keypoints.data  # n_det, n_keypoints, 3
        n_det, n_keypoints, _ = xyconf_array.shape
        
        msg_to_pub = KeypointOutArray()
        msg_to_pub.header = header
        
        for idx in range(n_det):
          keypoint_out = KeypointOut()
          keypoint_out.header = header
          keypoint_out.n_keypoints = n_keypoints
          
          for idx_k in range(n_keypoints):
            keypoint_out.x.append(float(xyconf_array[idx, idx_k, 0]))
            keypoint_out.y.append(float(xyconf_array[idx, idx_k, 1]))
            keypoint_out.conf.append(float(xyconf_array[idx, idx_k, 2]))
          
          msg_to_pub.keypoints.append(keypoint_out)
        
        self.pubs['keypoint'].publish(msg_to_pub)
        print(f"Published {n_det} objects with {n_keypoints} keypoints each")
      else:
        # Publish empty keypoint array
        msg_to_pub = KeypointOutArray()
        msg_to_pub.header = header
        self.pubs['keypoint'].publish(msg_to_pub)
        print("No keypoints detected")
      

if __name__ == '__main__':
  rclpy.init()
  node = YoloDetectionNode()
  node.mark_heartbeat(0)
  node.start_ros_thread(async_spin=False)
