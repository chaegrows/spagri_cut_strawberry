#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import json
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2

class FakeGUI(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'fake_gui'

  def __init__(self, run_mode='standalone', **kwargs):
    super().__init__(run_mode=run_mode, **kwargs)
    self.mark_heartbeat(0)
    self.remote_node = 'job_manager_node'

if __name__=='__main__':
  rclpy.init()

  # Create server instance
  server = FakeGUI('standalone')

  # Test all service calls
  result = server.call_service('do_schedule', {})
  print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

  # Cleanup
  rclpy.shutdown()
