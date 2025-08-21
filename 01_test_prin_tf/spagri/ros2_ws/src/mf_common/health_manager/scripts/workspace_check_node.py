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
class WorkspaceCheckNode(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'workspace_check_node'
  def __init__(self, run_mode):
    super().__init__(run_mode)
    self.mark_heartbeat(0)
    self.ws_manager = WorkspaceManager(self)
    self.ros2tf = ROS2TF(self)
    self.declare_parameter('global_fixed_frame', 'farm_pcd')
    self.publish_markers_timer = self.add_async_timer(60.0, self.publish_markers)
    # example
    # self.ws_manager.add_ws_cube(
    #     ref_frame=self.param['global_fixed_frame'],
    #     ws_center_xyz=[-0.5, -0.5, -0.5],
    #     ws_size_wlh=[1.0, 1.0, 1.0],
    #     ws_orientation=[0.0, 0.0, 0.0],
    # )
    self.initialize_workspace(None, None, None)
  # self.ws_manager.add_ws_sphere(
  #     ref_frame=self.param['global_fixed_frame'],
  #     ws_center_xyz=[0.5, 0.5, 0.0],
  #     ws_radius=0.3,
  #     ws_orientation=[0.0, 0.0, 0.0],
  # )


  @BehaviorTreeServerNodeV2.available_action()
  def initialize_workspace(self,
                input_dict: Optional[Dict],
                target_sector: Sector,
                target_work: ALLOWED_WORKS):

    farm_spec = self.job_manager.specifications.farm_spec
    ws_list_xyzwlh = [rack.ws_list_xyzwlh for rack in farm_spec.racks]
    # print('ws_list_xyzwlh: ', ws_list_xyzwlh)

    for idx, ws_xyzwlh in enumerate(ws_list_xyzwlh):
      for cx, cy, cz, w, l, h in ws_xyzwlh:
        self.ws_manager.add_ws_cube(
          ref_frame=self.param['global_fixed_frame'],
          ws_center_xyz=[cx, cy, cz],
          ws_size_wlh=[w, l, h],
        )

    robot_work_sectors = farm_spec.sectors
    for sector in robot_work_sectors:
      x, y, rz = sector.global_gv_pose
      z = sector.global_height
      frame_id = f'rack{sector.rack_id}_dir{sector.required_viewpoint}_horizon{sector.horizon_id}_height{sector.height_id}'
      # self.ros2tf.publishTF(self.param['global_fixed_frame'], frame_id, [x, y, z], [0.0, 0.0, rz])

    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'status': 'good'}


  def publish_markers(self):
      """작업 공간 마커 발행"""
      self.ws_manager.publish_workspace_markers('farm_workspace_markers')


  @BehaviorTreeServerNodeV2.available_service()
  def check_tf_inside_farm_workspace(self, args_json: Optional[dict] = None):
    frame_name = args_json['frame_name']
    is_inside = self.ws_manager.check_tf_inside_ws(frame_name)
    self.mf_logger.info(f"frame({frame_name}) inside the workspace -> {is_inside}")
    return {'is_inside': is_inside}


if __name__ == '__main__':
  rclpy.init()

  run_mode = 'server' # 'standalone' or 'server'
  server = WorkspaceCheckNode(run_mode)

  if run_mode == 'standalone':
      server.start_ros_thread(async_spin=True)
      while True:
        time.sleep(0.5)
        test = {
          'test1': [1.78600327, 2.6160973,  1.73209251],
          'test2': [1.78427428, 2.60533092, 1.7552053],
          'test3':  [2.3514959,  2.67880236, 1.77958295],
          'test4':  [2.04915427, 2.50683888, 1.15179347],
          'test5': [2.0082981,  0.73775728, 0.98100306],
          'test6': [1.78032756, 0.80153258, 0.9551633 ],
          'test7': [2.21969126, 2.46506916, 1.60198566],
          'test7': [1.94754103, 0.76771464, 1.0081997 ],
        }
        for frame_id, trans in test.items():
          server.ros2tf.publishTF('farm_pcd', frame_id, trans, [0,0,0])
          time.sleep(0.5)
          is_inside = server.ws_manager.check_tf_inside_ws(frame_id)
          print(f"frame({frame_id}) trans: {trans} -> {is_inside}")


  elif run_mode == 'server':
    server.start_ros_thread(async_spin=False)
  else:
    NotImplementedError(f'run_mode: {run_mode} is not implemented') # no plan now but maybe..?

  rclpy.shutdown()

