#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional, Dict
import rclpy

from mf_msgs.msg import BehaviorTreeStatus
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mflib.common import mf_base
from typing import Optional, Dict
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
import os
import time
from datetime import datetime
import sys

from config.common.rosbag_recorder.topics import RosbagRecorderConfig
from rclpy.serialization import serialize_message

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from std_msgs.msg import String, Int32, Float32, Bool
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, Imu, JointState
from tf2_msgs.msg import TFMessage

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped


def create_basemodel_from_path(yaml_file_path, basemodel_class):
  from ruamel.yaml import YAML
  if not os.path.exists(yaml_file_path):
    raise FileNotFoundError(f"yaml file not found: {yaml_file_path}")
  with open(yaml_file_path, "r") as f:
    return basemodel_class.model_validate(YAML().load(f))

class TopicInfo:
  def __init__(self, topic_name):
    self.topic_name = topic_name
    self.topic_type_str = None
    self.topic_type = None
    self.topic_queue = []

  def __repr__(self):
    return f"topic_name={self.topic_name}, type_str={self.topic_type}"

class RosbagRecorder(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'rosbag_recorder'

  topic_type_str_to_type = {
    'std_msgs/String': String,
    'std_msgs/Int32': Int32,
    'std_msgs/Float32': Float32,
    'std_msgs/Bool': Bool,
    'sensor_msgs/Image': Image,
    'sensor_msgs/msg/CompressedImage': CompressedImage,
    'sensor_msgs/msg/CameraInfo': CameraInfo,
    'sensor_msgs/PointCloud2': PointCloud2,
    'sensor_msgs/Imu': Imu,
    'nav_msgs/Odometry': Odometry,
    'geometry_msgs/PoseStamped': PoseStamped,
    'geometry_msgs/TwistStamped': TwistStamped,
    'tf2_msgs/msg/TFMessage': TFMessage,
    'sensor_msgs/msg/JointState': JointState,
  }
  def __init__(self, run_mode):
    super().__init__(run_mode)
    self.mark_heartbeat(0)

    yaml_path = '/opt/config/common/rosbag_recorder/topic_list.yaml'
    recorder_config = create_basemodel_from_path(yaml_path, RosbagRecorderConfig)
    mf_base.yaml_print(recorder_config)

    # variables
    self._recording_enabled = False

    # initialize the topic name and subscriptions
    self.topic_data = {topic_name: TopicInfo(topic_name) for topic_name in recorder_config.topics}
    while rclpy.ok():
      time.sleep(1)
      rclpy.spin_once(self)
      # check if all topics are defined, by checking if topic_type is None
      all_topic_found = True
      for topic_name in recorder_config.topics:
        if self.topic_data[topic_name].topic_type is None:
          all_topic_found = False
      if all_topic_found:
        break


      # search topic info in the topic_names_and_types
      self.mf_logger.info("-------------------------------")
      topic_names_and_types = self.get_topic_names_and_types()

      for topic_name in recorder_config.topics:
        associated_topic_info = None
        for topic_info in topic_names_and_types:
          if topic_name == topic_info[0]:
            associated_topic_info = topic_info
            break
        if associated_topic_info is None:
          self.mf_logger.error(f"Topic {topic_name} not found")
          break

        topic_type_str = associated_topic_info[1]
        if len(topic_type_str) > 1:
          if recorder_config.force_1to1_connection:
            self.mf_logger.error(f"Topic {topic_name} has multiple types: {topic_type_str}")
            self.mf_logger.error(f"This is not supported in the current version")
            mf_base.raise_with_log(self.mf_logger, ValueError, f"Topic {topic_name} has multiple types: {topic_type_str}")
        else:
          topic_type_str = topic_type_str[0]
        self.topic_data[topic_name].topic_type_str = topic_type_str
        
        # check if type is defined or not
        if topic_type_str not in self.topic_type_str_to_type:
          self.mf_logger.warning(f"Topic type {topic_type_str} not found in topic_type_str_to_type")
          continue
        self.topic_data[topic_name].topic_type = self.topic_type_str_to_type[topic_type_str]
      
    self.mf_logger.info('All topics found')
    for info in self.topic_data.values():
      self.mf_logger.info(info)

    ########### init rosbag writer ###########
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    bag_name = f"/opt/data/rosbag2_{timestamp}"

    self.bag_writer = SequentialWriter()
    storage_options = StorageOptions(uri=bag_name, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    self.bag_writer.open(storage_options, converter_options)

    for topic_name, topic_data in self.topic_data.items():
      topic_meta = TopicMetadata(name=topic_name, type=topic_data.topic_type_str,
                                     serialization_format='cdr')
      self.bag_writer.create_topic(topic_meta)

    # generate callbacks
    for topic_name, info in self.topic_data.items():
      self.add_async_subscriber(info.topic_type, topic_name, self._gen_sub_callback(topic_name), recorder_config.queue_len)
      self.mf_logger.info(f"Subscribed to topic {topic_name} with type {info.topic_type}")

  def _gen_sub_callback(self, topic_name):
    def callback(msg):
      if not self._recording_enabled:
        return
      timestamp = self.get_clock().now().to_msg()
      timestamp = int(timestamp.sec * 1e9 + timestamp.nanosec)
      self.bag_writer.write(topic_name, serialize_message(msg), timestamp)
    return callback
  
  @BehaviorTreeServerNodeV2.available_action()
  def enable_bag_recording(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
    self._recording_enabled = True
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

  @BehaviorTreeServerNodeV2.available_action()
  def disable_bag_recording(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
    self._recording_enabled = False
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

def main(args=None):
  rclpy.init(args=args)

  run_mode = 'server'
  node = RosbagRecorder(run_mode)
  if run_mode == 'standalone':
    context = BehaviorTreeContext()
    lookup_table = LeafActionLookup.build_leaf_action_lookup_instance()
    node.start_ros_thread(async_spin=True)

    enable_bag_recording = lookup_table.find(action_name='enable_bag_recording')[0]
    disable_bag_recording = lookup_table.find(action_name='disable_bag_recording')[0]

    input("Press Enter to start recording...")
    context.set_target_leaf_action(enable_bag_recording)
    context = node.run_leaf_action(context)
    node.mf_logger.info(f"Recording for 5 seconds")
    time.sleep(5)

    node.mf_logger.info(f"Stopping recording")
    context.set_target_leaf_action(disable_bag_recording)
    context = node.run_leaf_action(context)
    time.sleep(5)
    
    node.destroy_node()
  else:
    node.start_ros_thread(async_spin=False)
    node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
    main()
