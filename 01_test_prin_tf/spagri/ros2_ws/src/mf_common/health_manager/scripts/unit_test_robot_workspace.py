#!/usr/bin/env python3

import rclpy
import time
from typing import Optional, Dict
import json
import sys

from mf_msgs.msg import BehaviorTreeStatus
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mflib.common.mf_base import raise_with_log
from mflib.common.workspace import WorkspaceManager
from mflib.common.tf import ROS2TF
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
import numpy as np

# leaf actions are listed in /opt/common/leaves/example_bt_servers
class UnitTestRobotWorkspace(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'unit_test_robot_workspace'
  def __init__(self, run_mode):
    super().__init__(run_mode)
    self.mark_heartbeat(0)
    self.ws_manager = WorkspaceManager(self)
    self.ros2tf = ROS2TF(self)
    self.declare_parameter('global_fixed_frame', 'manipulator_base_link')

    self.declare_parameter('workspaces', [
      [0.6, -0.5, -0.2, 0.4, 0.4, 0.4], # bottom left 1 (normal)
      [0.6, -0.5, 0.2, 0.4, 0.4, 0.4], # top left 1 (normal)

      [0.6, 0.5, -0.2, 0.4, 0.4, 0.4], # bottom right 1 (normal)
      [0.6, 0.5, 0.2, 0.4, 0.4, 0.4], # top right 1 (normal)

      [-0.6, -0.5, -0.2, 0.4, 0.4, 0.4], # bottom left 1 (flipped)
      [-0.6, -0.5, 0.2, 0.4, 0.4, 0.4], # top left 1 (flipped)

      [-0.6, 0.5, -0.2, 0.4, 0.4, 0.4], # bottom right 1 (flipped)
      [-0.6, 0.5, 0.2, 0.4, 0.4, 0.4], # top right 1 (flipped)
    ])

    for ws in self.param['workspaces']:
      self.ws_manager.add_ws_cube(self.param['global_fixed_frame'],
                                  ws[:3],
                                  ws[3:6])

    self.publish_markers_timer = self.add_async_timer(1.0, self.publish_markers)


  def publish_markers(self):
      """작업 공간 마커 발행"""
      self.ws_manager.publish_workspace_markers('robot_workspace')

      for i in range(10):
        time.sleep(0.5)
        flower_id = f'flower_{i}'
        try:
          _, result = self.call_service('check_tf_inside_farm_workspace', {'frame_name': flower_id})
          is_inside_farm_ws = result['is_inside']
          is_inside_robot_ws = self.ws_manager.check_tf_inside_ws(flower_id)
          if is_inside_farm_ws and is_inside_robot_ws:
            self.mf_logger.error(f"flower_{i} inside: {is_inside_farm_ws} / {is_inside_robot_ws}")
        except Exception as e:
          pass



if __name__ == '__main__':
  rclpy.init()

  run_mode = 'server' # 'standalone' or 'server'
  server = UnitTestRobotWorkspace(run_mode)

  if run_mode == 'standalone':
      server.start_ros_thread(async_spin=True)
      while True:
        time.sleep(0.5)
        test = {
          'test1': [1.78600327, 2.6160973,  1.73209251],
          'test2': [1.78427428, 2.60533092, 1.7552053],
          'test3':  [2.3514959,  2.67880236, 1.77958295]
        }
        for frame_id, trans in test.items():
          server.ros2tf.publishTF('farm_pcd', frame_id, trans, [0,0,0])
          time.sleep(3)
          is_inside = server.ws_manager.check_tf_inside_ws(frame_id)
          print(f"frame({frame_id}) trans: {trans} -> {is_inside}")


  elif run_mode == 'server':
    server.start_ros_thread(async_spin=False)
  else:
    NotImplementedError(f'run_mode: {run_mode} is not implemented') # no plan now but maybe..?

  rclpy.shutdown()

