#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import time
import json

from mf_msgs.msg import BehaviorTreeStatus
from mf_msgs.srv import MfService
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from rclpy.node import Node

class SampleBtServer2(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'example_bt_server2'

  def __init__(self, run_mode='standalone', **kwargs):
    super().__init__(run_mode=run_mode, **kwargs)
    self.mark_heartbeat(0)
    # Set remote node name
    self.remote_node = 'example_bt_server'

if __name__=='__main__':
  # Initialize ROS
  rclpy.init()

  # Create server instance
  server = SampleBtServer2('standalone')
  server.start_ros_thread()

  print('\n##################### REMOTE SERVICE CALL TEST ##########################')

  # List of services to call
  services_to_call = [
    # ('get_server_info', {}),
    # ('calculate', {'operation': 'add', 'a': 10, 'b': 5}),
    # ('calculate', {'operation': 'multiply', 'a': 7, 'b': 3}),
    # ('get_robot_status', {}),
    # ('get_robot_status', {'fields': ['power', 'position']}),
    # ('update_robot_status', {'power': 75, 'is_moving': True}),
    ('check_tf_inside_farm_workspace', {'frame_name': 'test1'}),
    ('check_tf_inside_farm_workspace', {'frame_name': 'test2'}),
    ('check_tf_inside_farm_workspace', {'frame_name': 'test3'}),
    # ('recover_manipulator', {}),
  ]

  # Test all service calls
  while True:
    time.sleep(1)
    results = {}
    for service_name, args in services_to_call:
      full_service_name = f"{service_name}"
      print(f"\nCalling: {service_name} - Args: {args}")
      result = server.call_service(full_service_name, args)
      print("DONE", result)
      # print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
      # results[service_name] = {'success': True, 'data': result}

    print('\n##################### REMOTE SERVICE CALL COMPLETED ##########################')
    print(f"Result Summary: {len([r for r in results.values() if r['success']])}/{len(results)} successful")

  # Cleanup
  rclpy.shutdown()
