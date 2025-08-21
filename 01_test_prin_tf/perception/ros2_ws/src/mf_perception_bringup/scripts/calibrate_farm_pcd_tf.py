#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
from typing import Optional
import rclpy
from mf_msgs.msg import BehaviorTreeStatus
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mf_msgs.msg import BehaviorTreeStatus
from mflib.common.mf_base import raise_with_log
from config.common.specs.work_def import ALLOWED_WORKS
from config.common.specs.farm_def import Sector
from typing import Optional, Dict
from mflib.common.tf import ROS2TF
from config.common.specs.robot_def import RobotSpec, StaticTransforms

class CalibrateFarmPcdTf(BehaviorTreeServerNodeV2):
    repo = 'mf_perception'
    node_name = 'calibrate_farm_pcd_tf'

    def __init__(self, run_mode = 'server'):
        super().__init__(run_mode)
        self.tf_list = []
        self.declare_parameter('farm_frame_name', 'farm')
        self.declare_parameter('farm_pcd_frame_name', 'farm_pcd')
        self.declare_parameter('translation', '[0.0, 0.0, 0.0]')
        self.declare_parameter('rotation', '[0.0, 0.0, 0.0]')

        self.tf_handler = ROS2TF(self, verbose=True)

        self.tf_list = []

        self.set_static_tf(self.param['farm_frame_name'],
                          self.param['farm_pcd_frame_name'],
                          self.param['translation'],
                          self.param['rotation'])

        self.static_tf_timer = self.create_timer(5.0, self.broadcast_static_tf)

    def broadcast_static_tf(self):
        for frame_prefix in self.tf_list:
            self.tf_handler.publishTF(self.param[f"{frame_prefix}.parent_frame"],
                                    self.param[f"{frame_prefix}.child_frame"],
                                    self.param[f"{frame_prefix}.position"],
                                    self.param[f"{frame_prefix}.rotation"],
                                    static_tf=self.param[f"{frame_prefix}.static_tf"])

    def on_params_changed(self):
        try:
            for frame_prefix in self.tf_list:
                self.tf_handler.publishTF(self.param[f"{frame_prefix}.parent_frame"],
                                        self.param[f"{frame_prefix}.child_frame"],
                                        self.param[f"{frame_prefix}.position"],
                                        self.param[f"{frame_prefix}.rotation"],
                                        static_tf=self.param[f"{frame_prefix}.static_tf"])
        except Exception as e:
            pass

    def set_static_tf(self, parent_frame, child_frame, position, rotation):
        frame_prefix = f"{parent_frame}_to_{child_frame}"
        if frame_prefix not in self.tf_list:
          self.tf_handler.publishTF(parent_frame, child_frame, position, rotation, static_tf=True)
          self.tf_list.append(frame_prefix)
          self.set_param(f"{frame_prefix}.parent_frame", parent_frame)
          self.set_param(f"{frame_prefix}.child_frame", child_frame)
          self.set_param(f"{frame_prefix}.position", position)
          self.set_param(f"{frame_prefix}.rotation", rotation)
          self.set_param(f"{frame_prefix}.static_tf", True)



if __name__ == '__main__':

    rclpy.init()
    action_list = []
    server = CalibrateFarmPcdTf(run_mode='server')
    server.start_ros_thread(async_spin=False)
