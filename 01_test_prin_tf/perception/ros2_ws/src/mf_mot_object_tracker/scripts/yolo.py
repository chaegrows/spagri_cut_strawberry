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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
# import Metafarmers.app.ros2_ws.src.mf_mot_object_tracker.scripts.params_mot_by_keypoint_seedling as P
if sys.argv[1] == 'strawberry_harvesting_pollination':
  import params.mot_by_bbox.strawberry_harvesting_pollination as P

class YoloDetectionNode(BehaviorTreeServerNodeV2):
  repo = 'mf_perception'
  node_name = 'yolo_detection_node'
  def __init__(self, run_mode = 'server'):
    super().__init__(run_mode)

    self.model = YOLO(P.model_path)
    if P.use_cuda:
      if P.model_path.endswith('.pt'):
        self.model.eval()
        self.model.cuda()
    self.bridge = CvBridge()

    if not P.rgb_topic.endswith('compressed'):
      msg_type = Image
      cb = self.rgb_callback
    else:
      msg_type = CompressedImage
      cb = self.compressed_rgb_callback


    self.pubs = {}
    self.pubs['yolo_debug'] = self.create_publisher(CompressedImage, P.yolo_debug_topic, 1)
    if 'bbox' in P.inference_output_type:
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
    print('yolo detection node is initialized')

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
      annotated_img = result.plot(labels=P.print_labels, img=cv_img)
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_depth_debug'].publish(msg)
      return

    imgsz = P.det3d_params['image_size_wh'][1], P.det3d_params['image_size_wh'][0]
    result = self.model.predict(cv_img, imgsz=imgsz, show_labels=False,
                              conf=P.conf_thresh, iou=P.iou_thresh, verbose=P.verbose_yolo_predict)[0].cpu()
    self.last_result = result
    
    # publish debug image
    if P.publish_yolo_debug:
      annotated_img = result.plot(labels=P.print_labels, img=cv_img)
      # convert to bgr8 format 
      # palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_debug'].publish(msg)

    if 'bbox' in P.inference_output_type and self.pubs['bbox'].get_subscription_count() > 0:
      n_det = len(result.boxes)
      msg_to_pub = YoloOutArray()
      msg_to_pub.header = header
      if n_det != 0:
        for idx in range(n_det):
          yolo_out = YoloOut()
          yolo_out.header = header
          yolo_out.tlbr = [int(ele) for ele in result.boxes[idx].xyxy[0]]
          yolo_out.score = float(result.boxes[idx].conf[0])
          yolo_out.label = int(result.boxes[idx].cls[0])
          msg_to_pub.yolo_out_array.append(yolo_out)
      self.pubs['bbox'].publish(msg_to_pub)

    if 'keypoint' in P.inference_output_type and self.pubs['keypoint'].get_subscription_count() > 0:
      n_det = len(result.keypoints)
      msg_to_pub = KeypointOutArray()
      msg_to_pub.header = header
      xyconf_array = result.keypoints.data # n_det, n_keypoints, 3
      _, n_keypoints, _ = xyconf_array.shape
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
      

if __name__ == '__main__':
  rclpy.init()
  node = YoloDetectionNode()
  node.mark_heartbeat(0)
  node.start_ros_thread(async_spin=False)
