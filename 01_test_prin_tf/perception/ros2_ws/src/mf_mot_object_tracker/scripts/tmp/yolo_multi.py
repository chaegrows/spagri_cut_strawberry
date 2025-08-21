#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2

from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut


# import params_mot_by_yolo as P
from params import params_mot_by_bbox_seedling as P

class YoloDetectionNode(BehaviorTreeServerNodeV2):
  def __init__(self, node_name='yolo_multi'):
    super().__init__(node_name, [])

    self.models = []
    for p in P.model_paths:
      self.models.append(YOLO(p))
    if P.use_cuda:
      for m in self.models:
        m.eval()
        m.cuda()
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
      self.last_results = None
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
    if self.last_results is None: return
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)
    cv_img = cv_img.astype(np.float32) / 1000
    cv_img[cv_img >= P.max_depth_meter] = P.max_depth_meter
    cv_img = (cv_img / P.max_depth_meter * 255).astype(np.uint8)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    self.predict(cv_img, msg.header, depth=True)

  def predict(self, cv_img, header, depth=False): # https://docs.ultralytics.com/ko/modes/predict/#inference-sources
    if depth:
      results = self.last_results
      if results is None:
        return
      annotated_imgs = []
      for re in results:
        annotated_img = re.plot(labels=P.print_labels, img=cv_img)
        annotated_imgs.append(annotated_img)
      annotated_img_concat = np.concatenate(annotated_imgs, axis=1)
      
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img_concat, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_depth_debug'].publish(msg)
      return

    results = []
    for idx, model in enumerate(self.models):
      result = model.predict(cv_img, show_labels=False,
                              conf=P.conf_thresh[idx], iou=P.iou_thresh[idx], verbose=P.verbose_yolo_predict)[0].cpu()
      results.append(result)
    self.last_results = results

    # publish debug image
    if P.publish_yolo_debug:
      annotated_imgs = []
      for result in results:
        annotated_img = result.plot(labels=P.print_labels, img=cv_img)
        annotated_imgs.append(annotated_img)
      annotated_img_concat = np.concatenate(annotated_imgs, axis=1)
      # convert to bgr8 format 
      # palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors
      msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img_concat, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_debug'].publish(msg)

    del result

    if 'bbox' in P.inference_output_type and self.pubs['bbox'].get_subscription_count() > 0:
      msg_to_pub = YoloOutArray()
      msg_to_pub.header = header
      for idx_result, re in enumerate(results):
        n_det = len(re.boxes)
        if n_det != 0:
          for idx in range(n_det):
            label_in_model = int(re.boxes[idx].cls[0])
            new_label = P.model_labels_to_newlabel[idx_result][label_in_model]

            yolo_out = YoloOut()
            yolo_out.header = header
            yolo_out.tlbr = [int(ele) for ele in re.boxes[idx].xyxy[0]]
            yolo_out.score = float(re.boxes[idx].conf[0])
            yolo_out.label = new_label
            msg_to_pub.yolo_out_array.append(yolo_out)
      self.pubs['bbox'].publish(msg_to_pub)


    # if 'keypoint' in P.inference_output_type and self.pubs['keypoint'].get_subscription_count() > 0:
    #   n_det = len(result.keypoints)
    #   msg_to_pub = KeypointOutArray()
    #   msg_to_pub.header = header

    #   xyconf_array = result.keypoints.data # n_det, n_keypoints, 3
    #   _, n_keypoints, _ = xyconf_array.shape
    #   for idx in range(n_det):
    #     keypoint_out = KeypointOut()
    #     keypoint_out.header = header
    #     keypoint_out.n_keypoints = n_keypoints
    #     for idx_k in range(n_keypoints):
    #       keypoint_out.x.append(float(xyconf_array[idx, idx_k, 0]))
    #       keypoint_out.y.append(float(xyconf_array[idx, idx_k, 1]))
    #       keypoint_out.conf.append(float(xyconf_array[idx, idx_k, 2]))
    #     msg_to_pub.keypoints.append(keypoint_out)
    #   self.pubs['keypoint'].publish(msg_to_pub)
      

if __name__ == '__main__':
  rclpy.init()
  node = YoloDetectionNode()
  node.set_heartbeat(0)
  node.start_ros_thread(async_spin=False)
