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
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS

# leaf actions are listed in /opt/common/leaves/example_bt_servers
class SampleBtServer(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'example_bt_server'
  def __init__(self, run_mode):
    super().__init__(run_mode)
    self.mark_heartbeat(0)
    self.try_count = 0
    self.robot_status = {
      "power": 100,
      "position": [10.5, 20.3, 0.5],
      "is_moving": False,
      "errors": []
    }

  # def cleanup(self):
  #   self.mf_logger.info('cleanup!!!')

  @BehaviorTreeServerNodeV2.available_action()
  def wait5seconds(self,
                   input_dict: Optional[Dict],
                   target_sector: Sector,
                   target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: wait5seconds')

    time.sleep(5)
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'wait_time': 5}

  @BehaviorTreeServerNodeV2.available_action()
  def return_random_boolean(self,
                            input_dict: Optional[Dict],
                            target_sector: Sector,
                            target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: return_random_boolean')

    ret = BehaviorTreeStatus.TASK_STATUS_SUCCESS if time.time_ns() % 2 else BehaviorTreeStatus.TASK_STATUS_FAILURE
    return ret, {'random_value': ret}

  @BehaviorTreeServerNodeV2.available_action()
  def return_true(self,
                  input_dict: Optional[Dict],
                  target_sector: Sector,
                  target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: return_true')

    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'random_value': True}

  @BehaviorTreeServerNodeV2.available_action()
  def return_false(self,
                   input_dict: Optional[Dict],
                   target_sector: Sector,
                   target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: return_false')

    return BehaviorTreeStatus.TASK_STATUS_FAILURE, {'random_value': False}

  @BehaviorTreeServerNodeV2.available_action()
  def try5times(self,
                input_dict: Optional[Dict],
                target_sector: Sector,
                target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: try5times')

    self.try_count += 1
    if self.try_count < 5:
      return BehaviorTreeStatus.TASK_STATUS_RUNNING, {'try_count': self.try_count}
    else:
      return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'try_count': self.try_count}

  @BehaviorTreeServerNodeV2.available_action()
  def save_pose(self,
                input_dict: Optional[Dict],
                target_sector: Sector,
                target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: save_pose')

    # Simulate saving a pose
    pose = (1.0, 2.0, 3.0)
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'strawberry_pose': pose, 'frame_id': 'base_link'}

  @BehaviorTreeServerNodeV2.available_action()
  def print_inputs(self,
                   input_dict: Optional[Dict],
                   target_sector: Sector,
                   target_work: ALLOWED_WORKS):
    self.mf_logger.info('leaf_action: print_inputs')

    self.mf_logger.info(f'input_dict: {input_dict}')
    self.mf_logger.info(f'target_sector: {target_sector}')
    self.mf_logger.info(f'target_work: {target_work}')
    return BehaviorTreeStatus.TASK_STATUS_SUCCESS, None

  # 서비스 예시 시작
  @BehaviorTreeServerNodeV2.available_service()
  def get_server_info(self, args_json: Optional[dict] = None):
    print('leaf_service: get_server_info')

    response = {
      "server_name": "SampleBtServer",
      "repo": self.repo,
      "node_name": self.node_name,
      "uptime": time.time(),
      "version": "1.0.0"
    }

    return response

  @BehaviorTreeServerNodeV2.available_service()
  def calculate(self, args_json: Optional[dict] = None):
    print('leaf_service: calculate')
    time.sleep(2)

    if not args_json or 'operation' not in args_json:
      return {"error": "Missing required parameter: operation"}

    operation = args_json.get('operation')
    a = args_json.get('a', 0)
    b = args_json.get('b', 0)

    result = None
    if operation == 'add':
      result = a + b
    elif operation == 'subtract':
      result = a - b
    elif operation == 'multiply':
      result = a * b
    elif operation == 'divide':
      if b == 0:
        return {"error": "Division by zero"}
      result = a / b
    else:
      return {"error": f"Unknown operation: {operation}"}

    return {
      "operation": operation,
      "a": a,
      "b": b,
      "result": result
    }

  @BehaviorTreeServerNodeV2.available_service()
  def get_robot_status(self, args_json: Optional[dict] = None):
    print('leaf_service: get_robot_status')

    # 필터를 적용하여 요청된 정보만 반환
    if args_json and 'fields' in args_json:
      fields = args_json.get('fields', [])
      filtered_status = {k: v for k, v in self.robot_status.items() if k in fields}
      return filtered_status

    # 필터 없으면 전체 상태 반환
    return self.robot_status

  @BehaviorTreeServerNodeV2.available_service()
  def update_robot_status(self, args_json: Optional[dict] = None):
    print('leaf_service: update_robot_status')

    if not args_json:
      return {"error": "No update parameters provided"}

    # 상태 업데이트
    for key, value in args_json.items():
      if key in self.robot_status:
        self.robot_status[key] = value

    return {"status": "success", "updated": list(args_json.keys())}


if __name__=='__main__':

    run_mode = 'standalone' if len(sys.argv) > 1 and sys.argv[1] == 'standalone' else 'server'  # 기본값
    rclpy.init()
    server = SampleBtServer(run_mode)

    try:

        if run_mode == 'server':  # server mode
            server.start_ros_thread(async_spin=False)

        elif run_mode == 'standalone':
            context = BehaviorTreeContext()
            lookup_table = LeafActionLookup.build_leaf_action_lookup_instance()
            server.start_ros_thread(async_spin=True)

            la_wait5seconds = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='wait5seconds')[0]
            la_return_random_boolean = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='return_random_boolean')[0]
            la_return_true = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='return_true')[0]
            la_return_false = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='return_false')[0]
            la_try5times = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='try5times')[0]
            la_save_pose = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='save_pose')[0]
            la_print_inputs = lookup_table.find(repo='mf_common', node_name='example_bt_server', action_name='print_inputs')[0]

            server.mf_logger.info('##################### example bt server start ##########################')
            # context.set_target_leaf_action(la_wait5seconds)
            # context = server.run_leaf_action(context)

            context.set_target_leaf_action(la_return_random_boolean)
            context = server.run_leaf_action(context)

            context.set_target_leaf_action(la_return_true)
            context = server.run_leaf_action(context)

            context.set_target_leaf_action(la_return_false)
            context = server.run_leaf_action(context)

            context.set_target_leaf_action(la_try5times)
            context = server.run_leaf_action(context)

            # mimic other leaf_action's output
            context.set_memory_dict(la_save_pose, 'strawberry_pose', (1.0, 2.0, 3.0))
            context.set_memory_dict(la_save_pose, 'frame_id', 'base_link')
            context.set_target_leaf_action(la_print_inputs)
            context = server.run_leaf_action(context)

            server.mf_logger.info('\n##################### example bt server service test start ##########################')

            server_info = server.call_service('get_server_info', {})
            server.mf_logger.info(json.dumps(server_info, indent=2))

            calc_result = server.call_service('calculate', {
                'operation': 'add',
                'a': 10,
                'b': 5
            })
            server.mf_logger.info(json.dumps(calc_result, indent=2))

            calc_result = server.call_service('calculate', {
                'operation': 'divide',
                'a': 10,
                'b': 2
            })
            server.mf_logger.info(json.dumps(calc_result, indent=2))

            robot_status = server.call_service('get_robot_status', {})
            server.mf_logger.info(json.dumps(robot_status, indent=2))

            robot_status = server.call_service('get_robot_status', {
                'fields': ['power', 'position', 'is_moving']
            })
            server.mf_logger.info(json.dumps(robot_status, indent=2))

            update_result = server.call_service('update_robot_status', {
                'power': 80,
                'is_moving': True
            })
            server.mf_logger.info(json.dumps(update_result, indent=2))

            robot_status = server.call_service('get_robot_status', {})
            server.mf_logger.info(json.dumps(robot_status, indent=2))

            server.mf_logger.info('\n##################### example bt service server test done ##########################')



    except Exception as e:
        print(f"\n예외 발생: {e}")
    finally:
        print("프로그램 종료")
