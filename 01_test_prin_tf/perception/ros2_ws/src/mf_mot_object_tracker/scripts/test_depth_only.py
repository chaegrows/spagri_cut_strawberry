#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2

class DepthTestNode(Node):
  def __init__(self, node_name='depth_test_node'):
    super().__init__(node_name)
    
    self.bridge = CvBridge()
    
    # Subscribe to compressed depth image
    self.depth_subscription = self.create_subscription(
      CompressedImage,
      '/camera/depth/image_raw/compressedDepth',
      self.depth_callback,
      10
    )
    
    print('Depth test node initialized')
    print('Subscribing to: /camera/depth/image_raw/compressedDepth')

  def depth_callback(self, msg):
    """Test depth image conversion"""
    print(f"\n=== Depth Callback ===")
    print(f"Message format: {msg.format}")
    print(f"Data length: {len(msg.data)} bytes")
    print(f"Header frame_id: {msg.header.frame_id}")
    
    try:
      # Method 1: Try cv_bridge compressed_imgmsg_to_cv2
      print("Method 1: Using cv_bridge.compressed_imgmsg_to_cv2...")
      try:
        depth_img_1 = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth_img_1 is not None:
          print(f"  SUCCESS - Shape: {depth_img_1.shape}, dtype: {depth_img_1.dtype}")
          print(f"  Depth range: {np.min(depth_img_1):.1f} - {np.max(depth_img_1):.1f}")
        else:
          print("  FAILED - Returned None")
      except Exception as e:
        print(f"  FAILED - Exception: {e}")
      
      # Method 2: Try manual cv2.imdecode
      print("Method 2: Using cv2.imdecode...")
      try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_img_2 = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
        if depth_img_2 is not None:
          print(f"  SUCCESS - Shape: {depth_img_2.shape}, dtype: {depth_img_2.dtype}")
          print(f"  Depth range: {np.min(depth_img_2):.1f} - {np.max(depth_img_2):.1f}")
        else:
          print("  FAILED - Returned None")
      except Exception as e:
        print(f"  FAILED - Exception: {e}")
      
      # Method 3: Try different cv2 flags
      print("Method 3: Using cv2.imdecode with different flags...")
      try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        flags_to_try = [
          (cv2.IMREAD_UNCHANGED, "IMREAD_UNCHANGED"),
          (cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR, "ANYDEPTH|ANYCOLOR"),
          (cv2.IMREAD_GRAYSCALE, "IMREAD_GRAYSCALE"),
          (-1, "flag=-1")
        ]
        
        for flag, flag_name in flags_to_try:
          try:
            depth_img_3 = cv2.imdecode(np_arr, flag)
            if depth_img_3 is not None:
              print(f"  {flag_name} SUCCESS - Shape: {depth_img_3.shape}, dtype: {depth_img_3.dtype}")
              if len(depth_img_3.shape) >= 2:
                print(f"    Range: {np.min(depth_img_3):.1f} - {np.max(depth_img_3):.1f}")
            else:
              print(f"  {flag_name} FAILED - Returned None")
          except Exception as e:
            print(f"  {flag_name} FAILED - Exception: {e}")
      except Exception as e:
        print(f"  Method 3 FAILED - Exception: {e}")
      
      # Method 4: NEW - Handle compressedDepth format correctly
      print("Method 4: Handle compressedDepth format (skip header)...")
      try:
        if 'compressedDepth' in msg.format:
          # Skip the first 12 bytes (header) and decode the PNG data
          png_data = msg.data[12:]
          np_arr = np.frombuffer(png_data, np.uint8)
          depth_img_4 = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
          
          if depth_img_4 is not None:
            print(f"  SUCCESS - Shape: {depth_img_4.shape}, dtype: {depth_img_4.dtype}")
            print(f"  Depth range: {np.min(depth_img_4):.1f} - {np.max(depth_img_4):.1f}")
          else:
            print("  FAILED - Returned None after header skip")
        else:
          print("  SKIPPED - Not compressedDepth format")
      except Exception as e:
        print(f"  FAILED - Exception: {e}")
      
      # Method 5: Check raw data
      print("Method 5: Raw data analysis...")
      try:
        print(f"  First 20 bytes: {msg.data[:20]}")
        print(f"  Last 20 bytes: {msg.data[-20:]}")
        
        # Check PNG header location
        png_signature = b'\x89PNG\r\n\x1a\n'
        png_pos = msg.data.find(png_signature)
        if png_pos >= 0:
          print(f"  PNG signature found at position: {png_pos}")
        
        # Check if it's PNG or other format
        if msg.data[:8] == b'\x89PNG\r\n\x1a\n':
          print("  Detected PNG format at start")
        elif msg.data[:2] == b'\xff\xd8':
          print("  Detected JPEG format")
        else:
          print(f"  Unknown format, header: {msg.data[:8]}")
      except Exception as e:
        print(f"  Raw data analysis FAILED - Exception: {e}")
        
    except Exception as e:
      print(f"Overall callback FAILED - Exception: {e}")
    
    print("=== End Depth Callback ===\n")

if __name__ == '__main__':
  rclpy.init()
  node = DepthTestNode()
  
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Shutting down depth test node...")
  finally:
    node.destroy_node()
    rclpy.shutdown() 