#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from cv_bridge import CvBridge
import rclpy
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2

from std_msgs.msg import String
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mf_msgs.msg import BehaviorTreeStatus
from mf_msgs.msg import ManipulatorStatus

from mf_perception_msgs.msg import YoloOutArray, YoloOut, KeypointOutArray, KeypointOut
import seaborn
from mflib.common.mf_base import raise_with_log

import sys
# import Metafarmers.app.ros2_ws.src.mf_mot_object_tracker.scripts.params_mot_by_keypoint_seedling as P
if sys.argv[1] == 'strawberry_harvesting_pollination':
  import params.mot_by_bbox.strawberry_harvesting_pollination as P
elif sys.argv[1] == 'seedling_arrow2d':
  import params.mot_seedling_arrow2d.params_seedling as P
else:
  raise ValueError(f'Unknown argument: {sys.argv[1]}')

class YoloDetectionNode(BehaviorTreeServerNodeV2):
  repo = 'mf_perception'
  node_name = 'yolo_detection_node'
  def __init__(self, run_mode = 'server'):
    super().__init__(run_mode)

    self.models = {}
    self.models[P.ENUM_GROWING_MEDIUM] = YOLO(P.model_pathes[P.ENUM_GROWING_MEDIUM])
    self.models[P.ENUM_SEEDLING] = YOLO(P.model_pathes[P.ENUM_SEEDLING])
    if P.use_cuda:
      for m_name in self.models:
        if m_name.endswith('.pt') or m_name.endswith('.pth'):
          self.models[m_name].to('cuda')
          self.models[m_name].eval()
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

    # seedling run mode
    self.seedling_run_mode = None # seedling or growing_medium
    self.add_sequential_subscriber(String, '/seedling_work_state', self.manipulator_status_cb, 1)



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

    # debug
    self.distinct_rgbs = seaborn.color_palette("husl", 20)

    print('yolo detection node is initialized')

  def manipulator_status_cb(self, msg):
    if msg.data == 'port':
      self.seedling_run_mode = P.ENUM_GROWING_MEDIUM
    elif msg.data == 'plant':
      self.seedling_run_mode = P.ENUM_SEEDLING
    else:
      pass
      # self.mf_logger.warning('seedling_run_mode is not defined')


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

  def annotate_img(self, cv_img, result_lytics):
    # get unique color mapping for each label
    for re in result_lytics.boxes:
      if len(re.cls) != 1:
        raise_with_log(self.mf_logger, f'Label {label} exceeds the number of distinct colors available ({len(self.distinct_rgbs)}). Please increase the number of colors in the seaborn palette.')
      label = int(re.cls[0])
      color = self.distinct_rgbs[(label % len(self.distinct_rgbs))]
      color = [int(c * 255) for c in color]

      # bbox
      tlbr = re.xyxy[0].cpu().numpy().astype(int)
      cv2.rectangle(cv_img, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), color, 3)
      # text
      conf = float(re.conf[0])
      label_text = f'{label}_{conf:.2f}'
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.7
      thickness = 2
      (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
      # draw background rectangle for text
      cv2.rectangle(cv_img, (tlbr[0], tlbr[1] - text_h - 5),
                    (tlbr[0] + text_w, tlbr[1]), color, -1)
      cv2.putText(cv_img, label_text, (tlbr[0], tlbr[1] - 5),
                  font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
      # text done
    return cv_img

  def predict(self, cv_img, header, depth=False): # https://docs.ultralytics.com/ko/modes/predict/#inference-sources
    if self.seedling_run_mode is None: # not subscribed yet
      self.mf_logger.warning('Run mode is not set yet, skipping prediction')
      return

    if depth:
      img_to_draw = cv_img
      if self.last_result is not None:
        img_to_draw = self.annotate_img(cv_img, self.last_result)
      msg = self.bridge.cv2_to_compressed_imgmsg(img_to_draw, dst_format='jpg')
      msg.header = header
      self.pubs['yolo_depth_debug'].publish(msg)
      return

    imgsz = list(cv_img.shape[:2]) # H, W, (C)
    imgsz[0] = imgsz[0] // 32 * 32 + 32 # make sure height is divisible by 32
    imgsz[1] = imgsz[1] // 32 * 32 + 32 # make sure width is divisible by 32

    model = self.models[self.seedling_run_mode]
    result = model.predict(cv_img, imgsz=imgsz, show_labels=False,
                              conf=P.conf_thresholds[self.seedling_run_mode], iou=P.iou_thresh, verbose=P.verbose_yolo_predict)[0].cpu()

    # 2. 결과 가져오기
    boxes = result.boxes  # ultralytics YOLOv8 format

    # 3. 클래스별 confidence threshold dict 예시
    # 예: class 0: 0.5, class 1: 0.3, class 2: 0.7 ...
    class_conf_thresholds = {
        0: 0.3,
        1: 0.2,
    }

    # 필터링 인덱스 수집
    filtered_idxs = [
        i for i in range(len(boxes.cls))
        if boxes.conf[i] >= class_conf_thresholds.get(int(boxes.cls[i]), 0.5)
    ]

    # 필터링된 결과 추출
    filtered_boxes = boxes[filtered_idxs]
    result.boxes = filtered_boxes  # 이후 로직과 연결 유지 가능


    # self.mf_logger.info(f'Number of detections: {len(result.boxes)}')

    # publish debug image
    if P.publish_yolo_debug:
      annotated_img = self.annotate_img(cv_img, result)
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


'''
ros2 topic pub -r 100 /manipulator/out mf_msgs/msg/ManipulatorStatus "header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
base_frame: ''
tool_frame: ''
move_group: ''
joint_names: []
seq: 0
current_status: 0
is_left: true
is_flipped: false
current_joints: []
current_pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0"

'''
