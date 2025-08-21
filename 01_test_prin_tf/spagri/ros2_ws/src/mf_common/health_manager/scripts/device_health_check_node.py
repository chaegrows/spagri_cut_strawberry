#!/usr/bin/env python3
import rclpy
import psutil
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import numpy as np
import time
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeContext

from mf_msgs.msg import BehaviorTreeStatus
from mf_msgs.msg import LiftStatus
from mf_msgs.msg import MD200Status
from mf_msgs.msg import HealthManager
from mf_msgs.msg import ManipulatorStatus

from mf_msgs.msg import Endeffector
from typing import Optional, Dict
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
from mflib.common.mf_base import raise_with_log
import time
import psutil


class DeviceHealthCheckNode(BehaviorTreeServerNodeV2):
    repo = "mf_common"
    node_name = "device_health_check_node"

    def __init__(self, run_mode = 'server'):
        super().__init__(run_mode)
        self.declare_parameter('unit_device', '') # ['LIDAR', 'GV', 'CAM_HAND', 'LIFT', 'MANIPULATOR', 'EEF']
        self.declare_parameter('cpu_timeout', 10)
        self.declare_parameter('cpu_threshold', 80)
        self.declare_parameter('heartbeat_timeout', 3.0)

        self.mark_heartbeat(0)

        self.topics = {
            'LIDAR': { # 10hz
                'topic_name': '/livox/lidar',
                'msg_type': PointCloud2,
                'error_code': 1,
                'last_message_time': None,
                'category': 'sensor'
            },
            'CAM_HAND': { # 15
                'topic_name': '/camera/camera_hand/color/image_rect_raw',
                'msg_type': Image,
                'error_code': 2,
                'last_message_time': None,
                'category': 'sensor'
            },
            'GV': { # 10~30
                'topic_name': '/md200/out',
                'msg_type': MD200Status,
                'error_code': 3,
                'last_message_time': None,
                'category': 'actuator'
            },
            'LIFT': { # GV
                'topic_name': '/lift/out',
                'msg_type': LiftStatus,
                'error_code': 4,
                'last_message_time': None,
                'category': 'actuator'
            },
            'MANIPULATOR': { # 200hz
                'topic_name': '/manipulator/out',
                'msg_type': ManipulatorStatus,
                'error_code': 5,
                'last_message_time': None,
                'category': 'actuator'
            },
            'EEF': { # 10 ~ 30
                'topic_name': '/mf_eef_raw/out',
                'msg_type': Endeffector,
                'error_code': 6,
                'last_message_time': None,
                'category': 'actuator'
            },
        }

        if self.param['unit_device'] != '':
            topics_to_remove = [key for key in self.topics.keys() if key not in self.param['unit_device']]
            for key in topics_to_remove:
                self.topics.pop(key)

        self.mf_logger.info(f"subscribed topics: {self.topics}")
        for topic_key, topic_info in self.topics.items():
            self.add_async_subscriber(
                topic_info['msg_type'],
                topic_info['topic_name'],
                lambda msg, key=topic_key: self.health_check_callback(msg, key),
                1
            )

        # self.add_async_timer(1.0, self.check_health)
        self.timeout_start = time.time()


        #----------------------------------------#
        self.health_msg = HealthManager()

        self.health_status = {
            'sensor':
                {'status': 'UNKNOWN',
                 'error_topic': set() },
            'actuator':
                {'status': 'UNKNOWN',
                 'error_topic': set()},
            'etc':
                {'status': 'UNKNOWN',
                 'error_topic': set()}
        }

        self.health_pub = self.create_publisher(
            HealthManager,          # 또는 String
            '/device_health_check_node/health_status',      # 1개의 토픽으로 합치기
            1
        )

        #----------------------------------------#


        self.mf_logger.info('Health check node has been started')


    @BehaviorTreeServerNodeV2.available_action()
    def health_check_action(self,
                   input_dict: Optional[Dict],
                   target_sector: Sector,
                   target_work: ALLOWED_WORKS):
        return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'status': 'good'}

    @BehaviorTreeServerNodeV2.available_action()
    def is_cpu_usage_ok(self,
                        input_dict: Optional[Dict],
                        target_sector: Sector,
                        target_work: ALLOWED_WORKS):
        cpu_timeout = self.param['cpu_timeout']
        start_time = time.time()

        for p in psutil.process_iter():
            p.cpu_percent(None)

        while time.time() - start_time < cpu_timeout:
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage < self.param['cpu_threshold']:
                self.mf_logger.info(f"cpu_usage: {cpu_usage}")
                return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {'status': 'cpu_usage_ok'}
            else:
                self.mf_logger.error(f"cpu_usage: {cpu_usage}")
        return BehaviorTreeStatus.TASK_STATUS_FAILURE, {'status': 'cpu_usage_too_high'}

    def health_check_callback(self, msg, topic_key: str):
        self.topics[topic_key]['last_message_time'] = self.get_clock().now()
        self.unmark_heartbeat(self.topics[topic_key]['error_code'])

    def main_check_health(self):
        #----- initialize health status -----#
        while rclpy.ok():
            # time.sleep(1)
            cpu_percent = psutil.cpu_percent(interval=1)
            self.mf_logger.info(f'current cpu usage: {cpu_percent}')

            for cat in self.health_status.values():
                cat['status'] = 'OK'
                cat['error_topic'].clear()

            #------- check health status --------#
            status_log = {}

            for _ in range(100):
                rclpy.spin_once(self)

            for key, info in self.topics.items():
                #----- initialize unhealthy flag -----#
                unhealthy = False

                name = info['topic_name']
                cat  = info['category']
                last_msg_time = info['last_message_time']

                #----- check if there is no message -----#
                if last_msg_time is None:
                    unhealthy = True
                    reason = 'No message'
                #----- check if there is timeout -----#
                else:
                    dt = (self.get_clock().now() - last_msg_time).nanoseconds / 1e9
                    unhealthy = (dt > self.param['heartbeat_timeout'])
                    reason = f'Timeout {dt:.1f}s' if unhealthy else 'Healthy'


                status_log[name] = reason

                #----- update health status -----#
                if unhealthy:
                    self.health_status[cat]['status'] = 'ERROR'
                    self.health_status[cat]['error_topic'].add(key)
                    self.mark_heartbeat(info['error_code'])
                else:
                    self.unmark_heartbeat(info['error_code'])

            #----- log health status -----#
            for t, s in status_log.items():
                self.mf_logger.info(f'{t}: {s}')

            #----- update health status -----#
            if all(state == "Healthy" for state in status_log.values()):
                self.mark_heartbeat(0)
                self.mf_logger.info("All devices are healthy")
                self.health_status['sensor']['status'] = 'OK'
                self.health_status['actuator']['status'] = 'OK'
                self.health_status['sensor']['error_topic'].clear()
                self.health_status['actuator']['error_topic'].clear()
            else:
                self.mf_logger.warning("Some devices are unhealthy")

            #----- publish health status -----#
            hm = HealthManager()
            hm.sensor_status   = self.health_status['sensor']['status']
            hm.actuator_status = self.health_status['actuator']['status']

            details = []
            details += list(self.health_status['sensor'  ]['error_topic'])
            details += list(self.health_status['actuator']['error_topic'])
            hm.detail = ', '.join(details)

            self.health_pub.publish(hm)

def main(args=None):
    rclpy.init(args=args)
    run_mode = 'server'
    node = DeviceHealthCheckNode()
    try:
        if run_mode == 'server':
            # node.start_ros_thread(async_spin=False)
            node.main_check_health()
        elif run_mode == 'standalone':
            context = BehaviorTreeContext()
            lookup_table = LeafActionLookup.build_leaf_action_lookup_instance()
            print('lookup_table: ', lookup_table)

            node.start_ros_thread(async_spin=True)

            la_health_check = lookup_table.find(repo='health_manager', node_name='device_health_check_node', action_name='health_check_action')[0]
            la_is_cpu_usage_ok = lookup_table.find(repo='health_manager', node_name='device_health_check_node', action_name='is_cpu_usage_ok')[0]


            context.set_target_leaf_action(la_health_check)
            context = node.run_leaf_action(context)
            context.set_target_leaf_action(la_is_cpu_usage_ok)
            context = node.run_leaf_action(context)
            node.mf_logger.info(f'health_check done: {context}')
        else:
            raise NotImplementedError(f'run_mode: {run_mode} is not implemented')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
