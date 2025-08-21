from pydantic import BaseModel
from typing import List, Literal
from ruamel.yaml import YAML
from collections import Counter
from rclpy.clock import Clock


import os
import sys

from mf_msgs.msg import Heartbeat
import config.common.leaves.leaf_actions as la


# Represents a single heartbeat status from the endeffector_node
class HeartbeatDef(BaseModel):
  value: int  # Unique heartbeat code
  description: str  # Description of the condition
  severity: Literal["warning", "error"]  # Severity level of the heartbeat
  
  class Config:
    frozen = True

# Represents the full configuration for the endeffector node
class HeartbeatsInNode(BaseModel):
  node_name: str  # Node name
  heartbeats: List[HeartbeatDef]  # List of heartbeat definitions

  @staticmethod
  def create_by_filename(filename):
    yaml_path = os.path.join(filename)
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Farm spec not found: {yaml_path}")
    with open(yaml_path, "r") as f:
      return HeartbeatsInNode.model_validate(YAML().load(f))
    
  class Config:
    frozen = True
    
class HeartbeatHandler:
  leaf_action_lookup_table = la.LeafActionLookup.build_leaf_action_lookup_instance()
  def __init__(self):
    self.hb_dict = {}
    self._load_all_heartbeat()

  def _load_all_heartbeat(self):
    BASE_PATH = '/opt/config/common/heartbeat'
    files = os.listdir(BASE_PATH)
    for filename in files:
      if filename.endswith('.yaml'):
        hbs_in_node = HeartbeatsInNode.create_by_filename(os.path.join(BASE_PATH, filename))
        all_heartbeats_list = [hb.value for hb in hbs_in_node.heartbeats]
        
        # check duplicates
        counts = Counter(all_heartbeats_list)
        duplicates = {value: count for value, count in counts.items() if count > 1}
        if duplicates:
          for value, count in duplicates.items():
            raise ValueError(f"  - value {value} happen {count} times in the {filename}")
        all_heartbeats_set = set(all_heartbeats_list)

        self.hb_dict[hbs_in_node.node_name] = {
          'hbs_info': hbs_in_node,
          'all_hbs': all_heartbeats_set
        }

  def to_msg(self, node_name, hb_list, stamp):
    hb_value = 0
    for val in hb_list:
      # check if val is listed in the definition
      if val not in self.hb_dict[node_name]['all_hbs']:
        raise ValueError(f"Heartbeat value {val} not found in the definition for {node_name}")

      hb_value |= (1 << val)

    msg = Heartbeat()
    msg.header.stamp = stamp
    msg.heartbeat = hb_value
    return msg
  
  def to_hb_list(self, node_name, msg):
    val = msg.heartbeat
    hb_list = []
    for i in range(64):
      if val & (1 << i):
        hb_list.append(i)

    # check if the bit is defined
    for hb in hb_list:
      if hb not in self.hb_dict[node_name]['all_hbs']:
        raise ValueError(f"Heartbeat value {hb} not found in the definition for {node_name}")
    
    return hb_list

  def to_hb_list_with_value(self, node_name, value):
    hb_list = []
    for i in range(64):
      if value & (1 << i):
        hb_list.append(i)

    # check if the bit is defined
    for hb in hb_list:
      if hb not in self.hb_dict[node_name]['all_hbs']:
        raise ValueError(f"Heartbeat value {hb} not found in the definition for {node_name}")
    
    return hb_list
  
  def to_error_list(self, node_name, msg):
    hb_list = self.to_hb_list(node_name, msg)
    error_list = []

    if node_name not in self.hb_dict:
      raise ValueError(f"Heartbeat definition not found for {node_name}")
    
    hbs_in_node = self.hb_dict[node_name]['hbs_info']
    for hb_msg in hb_list:
      for hb_def in hbs_in_node.heartbeats:
        if hb_msg == hb_def.value:
          error_list.append(hb_def.description)

    return error_list
  
  def to_error_list_with_value(self, node_name, value):
    hb_list = self.to_hb_list_with_value(node_name, value)
    error_list = []

    if node_name not in self.hb_dict:
      raise ValueError(f"Heartbeat definition not found for {node_name}")
    
    hbs_in_node = self.hb_dict[node_name]['hbs_info']
    for hb_msg in hb_list:
      for hb_def in hbs_in_node.heartbeats:
        if hb_msg == hb_def.value:
          error_list.append(hb_def.description)

    return error_list


  
if __name__ == "__main__":
  # Example usage
  # hbs_in_node = HeartbeatsInNode.create_by_filename('/opt/config/common/heartbeat/example_heartbeats.yaml')

  # yaml = YAML()
  # yaml.default_flow_style = False

  # yaml.dump(hbs_in_node.model_dump(), sys.stdout) 
  heartbeat_handler = HeartbeatHandler()

  # to_msg()
  import rclpy
  hb_list = [1, 2]
  msg = heartbeat_handler.to_msg('example_bt_server', hb_list, Clock().now().to_msg())
  print('your msg: ', msg)

  # to_hb_list()
  hb_list = heartbeat_handler.to_hb_list('example_bt_server', msg)
  print('hb list: ', hb_list)

  # to error list
  error_list = heartbeat_handler.to_error_list('example_bt_server', msg)
  print('error list: ', error_list)

  hb_list = heartbeat_handler.to_hb_list_with_value('example_bt_server', 6)
  print('hb list with value: ', hb_list)

  error_list = heartbeat_handler.to_error_list_with_value('example_bt_server', 6)
  print('error list with value: ', error_list)
