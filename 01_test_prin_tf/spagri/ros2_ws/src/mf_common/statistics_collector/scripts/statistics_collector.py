#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict
from std_msgs.msg import String
import rclpy
import queue
import logging
import os
import atexit

from mf_msgs.msg import BehaviorTreeCommand, BehaviorTreeStatus
from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext
from mflib.common.mf_base import raise_with_log
from mflib.common.mf_base import (
    yaml_file_to_instance,
    yaml_instance_to_file,
    yaml_instance_to_json_string,
    yaml_instance_to_string,
    yaml_json_string_to_instance,
    yaml_string_to_instance,
    yaml_print
)
from typing import Optional, Dict
from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS, Pollination, Harvesting

from config.common.statistics_collector.commander_status import CommanderStatusFrozen, CommanderStatusWorking
from config.common.statistics_collector.statistics_manager import StatisticsManager, create_stats_model
import time

def get_statistics_logger(work_type: str, base_name: str = "statistics_result"):
    log_queue = queue.Queue(-1)  # Unlimited-size queue for asynchronous logging

    logger = logging.getLogger(f"{base_name}_{work_type}")
    logger.setLevel(logging.DEBUG)

    # Create a date-based directory: /opt/logs/YYYYMMDD/statistics_result/
    start_time = time.localtime()
    date_folder = time.strftime("%Y%m%d", start_time)
    log_dir = os.path.join("/opt/logs", date_folder, "statistics_result", work_type)
    os.makedirs(log_dir, exist_ok=True)

    # Find the next available log number
    existing_logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    existing_numbers = []
    for filename in existing_logs:
        try:
            num = int(filename.replace(".log", ""))
            existing_numbers.append(num)
        except ValueError:
            continue
    next_number = max(existing_numbers, default=0) + 1

    # Define the log file path: (increment number).log
    log_path = os.path.join(log_dir, f"{next_number}.log")

    # Add a QueueHandler to push logs into the memory queue
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # Create a RotatingFileHandler to write logs to file
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        mode='a',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_formatter = logging.Formatter("[%(levelname)s, %(asctime)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    # Start a QueueListener to asynchronously write logs from the queue to the file
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()

    # Register listener.stop() to be called when the program exits
    atexit.register(listener.stop)

    return logger

class StatisticsCollectorNode(BehaviorTreeServerNodeV2):
    repo = 'mf_common'
    node_name = 'statistics_collector_node'

    def __init__(self, run_mode):
        super().__init__(run_mode)
        self.mark_heartbeat(0)

        self.statistics_manager = StatisticsManager()
        self.commander_status = CommanderStatusWorking()

        # 작업별 로거 초기화
        self.statistics_loggers = {}
        # 작업별 토픽 퍼블리셔 초기화
        self.statistics_publishers = {}

        self.add_async_subscriber(
            BehaviorTreeCommand,
            '/behavior_tree_commander/btc_status',
            self.btc_status_callback
        )

        self.add_async_timer(1.0, self.publish_statistics)

    def _ensure_work_type_setup(self, work_type: str):
        """작업 유형에 대한 로거와 퍼블리셔가 설정되어 있는지 확인하고, 없다면 초기화합니다."""
        if work_type not in self.statistics_loggers:
            self.statistics_loggers[work_type] = get_statistics_logger(work_type)
        if work_type not in self.statistics_publishers:
            self.statistics_publishers[work_type] = self.create_publisher(
                String,
                f'~/statistics/{work_type}',
                10
            )

    @BehaviorTreeServerNodeV2.available_action()
    def increment_success(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
        if target_work is None:
            return BehaviorTreeStatus.TASK_STATUS_FAILURE, {}

        work_type = target_work.work_type
        self._ensure_work_type_setup(work_type)
        self.statistics_manager.increment_success(work_type)
        self.log_new(work_type)
        return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

    @BehaviorTreeServerNodeV2.available_action()
    def increment_skipped(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
        if target_work is None:
            return BehaviorTreeStatus.TASK_STATUS_FAILURE, {}

        work_type = target_work.work_type
        self._ensure_work_type_setup(work_type)
        self.statistics_manager.increment_skipped(work_type)
        self.log_new(work_type)
        return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

    @BehaviorTreeServerNodeV2.available_action()
    def mark_start(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
        if target_work is None:
            return BehaviorTreeStatus.TASK_STATUS_FAILURE, {}

        work_type = target_work.work_type
        self._ensure_work_type_setup(work_type)
        self.statistics_manager.mark_start(work_type)
        self.log_new(work_type)
        return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}

    @BehaviorTreeServerNodeV2.available_action()
    def mark_end(self, input_dict: Optional[Dict], target_sector: Sector, target_work: ALLOWED_WORKS):
        if target_work is None:
            return BehaviorTreeStatus.TASK_STATUS_FAILURE, {}

        work_type = target_work.work_type
        self._ensure_work_type_setup(work_type)
        self.statistics_manager.mark_end(work_type)
        self.log_new(work_type)
        return BehaviorTreeStatus.TASK_STATUS_SUCCESS, {}


    def btc_status_callback(self, msg: BehaviorTreeCommand):
        frozen_status = yaml_json_string_to_instance(msg.context_json, CommanderStatusFrozen)
        self.mf_logger.info(f"btc_status_callback: {frozen_status.target_leaf_action.action_name}")
        self.commander_status = CommanderStatusWorking()
        self.commander_status.is_on_boot_succeeded = frozen_status.is_on_boot_succeeded
        self.commander_status.last_command = frozen_status.last_command
        self.commander_status.state = frozen_status.state
        self.commander_status.target_leaf_action = frozen_status.target_leaf_action
        self.commander_status.target_sector = frozen_status.target_sector
        self.commander_status.target_work = frozen_status.target_work

    def log_new(self, work_type: str):
        stats_model = create_stats_model(
            commander_status=self.commander_status.to_frozen(),
            statistics=self.statistics_manager.export_frozen(work_type)
        )
        self.statistics_loggers[work_type].info(f"{yaml_instance_to_string(stats_model)}")

    def publish_statistics(self):
        frozen_commander_status = self.commander_status.to_frozen()

        # StatisticsManager에서 관리하는 모든 작업 유형에 대해 통계 발행
        for work_type in self.statistics_manager.working_stats.keys():
            try:
                frozen_statistics = self.statistics_manager.export_frozen(work_type)

                stats_model = create_stats_model(
                    commander_status=frozen_commander_status,
                    statistics=frozen_statistics
                )

                msg = String()
                msg.data = yaml_instance_to_string(stats_model)
                self._ensure_work_type_setup(work_type)
                self.statistics_publishers[work_type].publish(msg)
                print(f"Published statistics for {work_type}")

            except ValueError as e:
                self.mf_logger.error(f"Error publishing statistics for {work_type}: {e}")

def main(args=None):
    rclpy.init(args=args)

    run_mode = 'server'
    node = StatisticsCollectorNode(run_mode)
    if run_mode == 'standalone':
        try:
            node.start_ros_thread(async_spin=True)
            # 테스트용 작업 정의
            pollination_task = Pollination()  # 생성자 인자 필요시 채워넣기
            harvest_task = Harvesting()  # 생성자 인자 필요시 채워넣기
            # 테스트 실행
            node.mark_start(None, None, pollination_task)
            node.mark_start(None, None, harvest_task)
            i = 0
            while i < 100:
              time.sleep(1)
              node.increment_success(None, None, pollination_task)
              node.increment_success(None, None, harvest_task)
              node.increment_skipped(None, None, pollination_task)
              node.increment_skipped(None, None, harvest_task)
              i += 1
            node.mark_end(None, None, harvest_task)
            node.mark_end(None, None, pollination_task)


        except Exception as e:
            node.mf_logger.error(f"Error during test: {e}")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        node.start_ros_thread(async_spin=False)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
