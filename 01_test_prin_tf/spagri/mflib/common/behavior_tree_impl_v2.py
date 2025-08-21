#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
import rclpy.executors
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from mf_msgs.msg import BehaviorTreeCommand, BehaviorTreeStatus, Heartbeat
from mf_msgs.srv import MfService
from typing import Optional, Dict
from ruamel.yaml import YAML
import threading
import time
import json
import inspect
import signal
from typing import get_type_hints

from mflib.common.mf_base import get_logger_for_node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_json_string,
  yaml_instance_to_string,
  yaml_json_string_to_instance,
  yaml_string_to_instance,
  yaml_print
)

import config.common.leaves.leaf_actions as la
import config.common.services.mf_services as ls
from config.common.heartbeat.heartbeat_lib import HeartbeatHandler
from config.common.specs.job_manager import Specifications, DefaultJob
from mflib.common.behavior_tree_context import BehaviorTreeContext

from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
from config.common.specs.job_manager import JobManager

JOB_MANAGER_NODE_NAME = 'job_manager_node'

class BehaviorTreeServerNodeV2(Node):
  available_actions = [] # used before initialization
  available_actions_to_func = {}
  leaf_action_lookup_table = la.LeafActionLookup.build_leaf_action_lookup_instance()
  available_services = [] # used before initialization
  available_services_to_func_client = {}
  service_running_status = {}
  mf_service_lookup_table = ls.LeafServiceLookup.build_mf_service_lookup_instance()
  heartbeat_handler = HeartbeatHandler()
  lookup = la.LeafActionLookup.build_leaf_action_lookup_instance()

  def __init__(self, run_mode: str, **kwargs):
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

    cls = self.__class__

    if not hasattr(cls, 'repo') or not hasattr(cls, 'node_name'):
      raise ValueError(f"{cls.__name__} must define class variables 'repo' and 'node_name'")

    super().__init__(self.node_name)
    self.is_job_manager_node = False
    if self.node_name == JOB_MANAGER_NODE_NAME:
      self.is_job_manager_node = True

    self.mf_logger = get_logger_for_node(self.node_name, log_dir='/opt/logs')

    ## if functions can be run in parallel, use this group
    self._parallel_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
    ## if functions must be run in sequence, use this group
    self._sequential_callback_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

    # Set default remote node name to None
    self.remote_node = None

    self.map_la_to_func()
    self.map_ls_to_func()

    # load config
    self.yaml_instance = YAML()
    config_path_file = '/opt/config/common/behavior_tree_server_config.yaml'
    self.bt_server_config = self.yaml_instance.load(open(config_path_file, 'r'))

    # select run mode and load specs
    # BT server related
    self.job_manager = None
    self.run_mode = None
    self.run_mode_metadata = {'server': {"initialized": False}, 'standalone': {}}
    self.select_run_mode(run_mode, **kwargs)

    # check error action
    self.on_failure_action = self.bt_server_config['on_failure_action'].lower()
    failure_actions = ['infinite_wait', 'raise_value_error']
    if self.on_failure_action not in failure_actions:
      raise ValueError('on_failure_action must be in ', failure_actions)
    self.mf_logger.info(f'BehaviorTreeServerNodeV2 on_failure_action: {self.on_failure_action}')

    # prepare members
    self.last_msg_id = 0

    self._current_status = BehaviorTreeStatus()
    self._status_lock = threading.Lock()

    self._marked_heartbeats = []
    self._heartbeat_lock = threading.Lock()
    self._timers = []

    # for derived class
    self.subs = {}
    self.param = {}
    self.add_on_set_parameters_callback(self._on_params_changed)

    # publishers
    self._status_publisher = self.create_publisher(BehaviorTreeStatus, f'{self.node_name}/status', 1)
    self._heartbeat_publisher = self.create_publisher(Heartbeat, f'{self.node_name}/heartbeat', 1)

    # subscribers
    self._command_subscriber = self.add_async_subscriber(BehaviorTreeCommand,
                                                        f'{self.node_name}/command',
                                                        self._command_callback)

    # status publisher timer
    self.add_async_timer(
      self.bt_server_config['status_publish_latency_seconds'],
      self._publish_status)
    self.add_async_timer(
      self.bt_server_config['heartbeat_publish_latency_seconds'],
      self._publish_heartbeat)


  def _signal_handler(self, signum, frame):
    self.mf_logger.info(f"Signal received: {signum}")
    self.cleanup()
    self.destroy_node()
    rclpy.shutdown()

  def cleanup(self):
    self.mf_logger.warn("Please implement cleanup() in your node!")
    pass

  def select_run_mode(self, run_mode: str, **kwargs):
    if run_mode not in ['server', 'standalone']:
      raise ValueError('run_mode must be in [server, standalone]')
    self.run_mode = run_mode

    self.load_default_job()

    if self.run_mode not in ['server', 'standalone']:
      raise ValueError(f'{run_mode} not in [server, standalone]')
    
  def load_default_job(self):
    self.job_manager = JobManager()
    self.job_manager.setup(specs_file=self.bt_server_config['default_specs'])

  def get_specs(self):
    return self.job_spec

  def _on_params_changed(self, params):
    for param in params:
      try:
        if isinstance(param.value, str):
          try:    data = json.loads(param.value)
          except: data = param.value
        else:
          data = param.value
        self.param[param.name] = data
      #   self._logger.info(f"[Parameter set] {param.name} = {self.param[param.name]}  (type: {type(self.param[param.name])})")
      except Exception as e:
        self._handle_bt_error(f'Failed to set parameter: {param.name}', str(e))
    self.on_params_changed()
    return SetParametersResult(successful=True, reason='Parameter set')

  # TODO: override this function in derived class
  def on_params_changed(self):
    pass

  def declare_parameter(self, name, value):
    p_value, _ = self._check_param(value)
    super().declare_parameter(name, p_value)

  def _check_param(self, value):
    type_mapping = {
      bool: rclpy.Parameter.Type.BOOL,
      int: rclpy.Parameter.Type.INTEGER,
      float: rclpy.Parameter.Type.DOUBLE,
      str: rclpy.Parameter.Type.STRING,
    }
    python_type = type(value)

    if python_type in [list, dict, tuple]:
      return json.dumps(value), rclpy.Parameter.Type.STRING
    elif python_type in type_mapping:
      return value, type_mapping[python_type]
    else:
      self._handle_bt_error(f'Invalid parameter type of {python_type}')

  def set_param(self, name, value):
    try:
      has_param = self.has_parameter(name)
      ros_param_value, ros_param_type = self._check_param(value)

      if not has_param:
        self.declare_parameter(name, ros_param_value)
      else:
        current_value = self.get_parameter(name).value
        self.mf_logger.info(f"Parameter '{name}' updated: {current_value} -> {value}")
        self.set_parameters([rclpy.parameter.Parameter(name, ros_param_type, ros_param_value)])

      self.param[name] = value

    except Exception as e:
      self.mf_logger.error(f"Parameter '{name}' processing error: {str(e)}")

      if self.run_mode_metadata['server']['initialized']:
        self._handle_bt_error(f'Error setting parameter {name}', str(e))
      else:
        ros_param_value, ros_param_type = self._check_param(value)

        if name not in self.param:
          self.declare_parameter(name, ros_param_value)
        else:
          self.set_parameters([rclpy.parameter.Parameter(name, ros_param_type, ros_param_value)])

        self.param[name] = value

    return self.param[name]

  def _handle_bt_error(self, msg: str='', msg_verbose: str=''):
    msg = f'BtServerNode caught an error: {msg}'
    if self.bt_server_config['verbose']:
      msg += f'\ndetail: {msg_verbose}'
    if self.on_failure_action == 'raise_value_error':
      self.mf_logger.error(msg)
      raise ValueError(msg)
    elif self.on_failure_action == 'infinite_wait':
      self.mf_logger.error(msg)
      while True:
        self.mf_logger.info('infinite wait in bt server...')
        time.sleep(1)
    else:
      raise NotImplementedError('Impossible control flow reached. Something wrong with on_failure_action check')

  def _publish_heartbeat(self):
    msg = None
    with self._heartbeat_lock:
      msg = self.heartbeat_handler.to_msg(self.node_name, self._marked_heartbeats, self.get_clock().now().to_msg())
    self._heartbeat_publisher.publish(msg)

  def get_heartbeat(self):
    with self._heartbeat_lock:
      return self._marked_heartbeats

  def mark_heartbeat(self, heartbeat: int):
    with self._heartbeat_lock:
      if heartbeat == 0:
        self._marked_heartbeats = []
      else:
        self._marked_heartbeats.append(heartbeat)
        self._marked_heartbeats = sorted(set(self._marked_heartbeats))

  def unmark_heartbeat(self, heartbeat: int):
    if heartbeat == 0:
      return # ignore
    with self._heartbeat_lock:
      if heartbeat in self._marked_heartbeats:
        self._marked_heartbeats.remove(heartbeat)

  def start_ros_thread(self, async_spin=True):
    # multi-threaded executor
    self.my_executor = rclpy.executors.MultiThreadedExecutor(
      self.bt_server_config['num_executor_threads'])
    self.my_executor.add_node(self)
    if self.run_mode == 'server':
      self.run_mode_metadata['server']['initialized'] = True
      # check heartbeat
    self.mf_logger.info(f'core of bt server is initialized. Start spin: {self.node_name}')
    if async_spin:
      self._start_spin_async()
    else:
      self._start_spin()

  def _start_spin_async(self):
    def run(executor):
      executor.spin()
    thr = threading.Thread(target=run, args=(self.my_executor,), daemon=True)
    thr.start()

  def _start_spin(self):
    self.my_executor.spin()

  @classmethod
  def available_action(cls):
    def decorator(func):
      cls.available_actions.append(func)
      return func
    return decorator

  @classmethod
  def available_service(cls):
    def decorator(func):
      cls.available_services.append(func)
      return func
    return decorator

  def map_la_to_func(self):
    cls = self.__class__
    for func in self.available_actions:
      sig = inspect.signature(func)
      params = list(sig.parameters.values())

      # Check number of parameters
      if len(params) != 4:
        raise ValueError(f"{func.__name__} must have exactly 4 parameters (self, input_dict, target_sector, target_work)")

      # Check second parameter name
      if params[1].name != 'input_dict':
        raise ValueError(f"{func.__name__} second parameter must be 'input_dict'")

      if params[2].name != 'target_sector':
        raise ValueError(f"{func.__name__} second parameter must be 'target_sector'")

      if params[3].name != 'target_work':
        raise ValueError(f"{func.__name__} second parameter must be 'target_work'")

      # Check second parameter annotation
      type_hints = get_type_hints(func)
      expected_type = Optional[Dict]

      name_type_dict = {
        'input_dict': expected_type,
        'target_sector': Sector,
        'target_work': ALLOWED_WORKS
      }

      for name, expected_type in name_type_dict.items():
        actual_type = type_hints.get(name, None)
        if actual_type != expected_type:
          raise ValueError(f"{func.__name__} {name} type must be {expected_type.__name__}, but got {actual_type.__name__ if actual_type else 'missing type hint'}")

      # check if name is listed in registered_actions
      action_name = func.__name__
      la_list = cls.leaf_action_lookup_table.find(repo=cls.repo, node_name=cls.node_name, action_name=action_name)
      if len(la_list) == 0:
        raise ValueError(f"Leaf action not found for {cls.repo}, {cls.node_name}, {action_name} in the lookup table")
      leaf_action = la_list[0]

      cls.available_actions_to_func[leaf_action] = func
      self.mf_logger.info(f"Action registered: {action_name} -> {func.__name__}")


  def map_ls_to_func(self):
    cls = self.__class__
    for func in self.available_services:
      service_name = func.__name__
      sig = inspect.signature(func)
      params = list(sig.parameters.values())

      filtered_params = params[1:]
      if len(filtered_params) != 1 or filtered_params[0].name != 'args_json':
        raise TypeError(
          f"{func.__name__} must have exactly one parameter named 'args_json'"
        )

      type_hints = get_type_hints(func)
      expected_type = Optional[dict]
      actual_type = type_hints.get('args_json', None)
      if actual_type != expected_type:
        raise TypeError(
          f"Parameter 'args_json' in leaf_service [{func.__name__}] must be of type {expected_type.__name__}, "
          f"got {actual_type.__name__ if actual_type else 'missing type hint'}"
        )

      # check if name is listed in registered_services
      target_service = cls.mf_service_lookup_table.find(service_name)
      if target_service is None:
        raise ValueError(f"MF service not found for {service_name} in the lookup table")

      # create ROS2 service with reliable QoS
      service_callback = self._create_service_callback(service_name)
      self.create_service(
        MfService,
        service_name,
        service_callback,
        qos_profile=QoSProfile(
          reliability=ReliabilityPolicy.RELIABLE,
          durability=DurabilityPolicy.TRANSIENT_LOCAL,
          depth=10
        ),
        callback_group=self._parallel_callback_group)

      # Use service_name as the key instead of target_service object
      cls.available_services_to_func_client[service_name] = [func, None]
      print(f"Service registered: {service_name} -> {func.__name__}")


  def _create_service_callback(self, service_name):
    """Service callback function"""
    self.mf_logger.info(f"[create_service_callback] start: {service_name}")
    self.service_running_status[service_name] = False

    def service_callback(request, response):
      if self.service_running_status[service_name] == True:
        response.success = False
        response.message = json.dumps({'why': 'Service is already running'})
        return response

      self.service_running_status[service_name] = True

      cls = self.__class__
      # Find the leaf service
      target_service = cls.mf_service_lookup_table.find(service_name)
      if target_service is None:
        raise ValueError(f"Leaf service not found: {service_name}")

      if service_name not in cls.available_services_to_func_client:
        raise ValueError(f"Service function not registered: {service_name}")

      # Parse args_json
      args_json = json.loads(request.args_json) if request.args_json else {}

      # Call service function using service_name as key
      service_func = cls.available_services_to_func_client[service_name][0]
      self.mf_logger.info("SERVICE start!!!")
      result = service_func(self, args_json)
      self.mf_logger.info("SERVICE end!!!")

      # Check if all required output keys are present in the result
      if target_service.output_keys:
        missing_keys = [key for key in target_service.output_keys if key not in result]
        if missing_keys:
          error_msg = f"Missing required output keys in service response from '{service_name}': {', '.join(missing_keys)}"
          self.get_logger().error(error_msg)
          raise ValueError(error_msg)
      self.mf_logger.info(f"SERVICE result: {result}")

      response.success = True
      response.message = json.dumps(result)
      self.service_running_status[service_name] = False
      return response
    return service_callback


  def call_service(self, service_name: str, args_json: dict, timeout_sec: float = 30.0) -> dict:
    """call service function and return result"""
    cls = self.__class__

    # Find services from lookup table
    target_service = cls.mf_service_lookup_table.find(service_name)
    if target_service is None:
      raise ValueError(f"MF service not found: {service_name}")

    # Check if args_json has required keys before making the service call
    if target_service.input_keys:
      missing_keys = [key for key in target_service.input_keys if key not in args_json]
      if missing_keys:
        error_msg = f"Missing required keys in service call to '{service_name}': {', '.join(missing_keys)}"
        self.get_logger().error(error_msg)
        raise ValueError(error_msg)

    # check if client is already created - use service_name as key
    target_client = cls.available_services_to_func_client.get(service_name, [None, None])[1]
    if target_client is None:
      target_client = self.create_client(
        MfService,
        service_name,
        qos_profile=QoSProfile(
          reliability=ReliabilityPolicy.RELIABLE,
          durability=DurabilityPolicy.TRANSIENT_LOCAL,
          depth=10
        ),
        callback_group=self._parallel_callback_group)
      if service_name in cls.available_services_to_func_client:
        cls.available_services_to_func_client[service_name][1] = target_client
      else:
        cls.available_services_to_func_client[service_name] = [None, target_client]

    # Wait for service to be available
    if not target_client.service_is_ready():
      self.get_logger().warn(f"Service '{service_name}' is not immediately available. Waiting for timeout...")

    # Wait for service to be available
    if not target_client.wait_for_service(timeout_sec=timeout_sec):
      error_msg = f"Service '{service_name}' not available after {timeout_sec} seconds timeout."
      self.get_logger().error(error_msg)
      raise ValueError(error_msg)

    # create request
    request = MfService.Request()
    request.args_json = json.dumps(args_json)

    # call service asynchronously
    try:
      future = target_client.call_async(request)

      # Wait for the future to complete
      start_time = time.time()
      while not future.done():
        time.sleep(0.01)
        if time.time() - start_time > timeout_sec:
          success = False
          result = {'why': 'Timeout'}
          return success, result

      response = future.result()
      success = response.success
      result = json.loads(response.message)
      self.mf_logger.info(f'Service completed: {service_name}')
      return success, result

    except Exception as e:
      error_msg = f"Service call to '{service_name}' failed with exception: {str(e)}"
      self.mf_logger.error(error_msg)
      raise ValueError(error_msg)


  def print_helper_functions(self):
    # self.mf_logger.info functions that are not start with _ or __ and defined in the current class, not in the parent class
    method_names = []
    for func in dir(self):
      if not callable(getattr(self, func)):
        continue
      if func not in dir(BehaviorTreeServerNodeV2):
        continue
      if func in dir(Node):
        continue
      if func.startswith('_'):
        continue
      method_names.append(func)
    self.mf_logger.info('implement your own class with using below functions - ')
    for f in method_names:
      self.mf_logger.info(f'  {f}')


  def _add_subscriber(self, msg_type, topic, callback, callback_group, qos=1):
    if self.run_mode_metadata['server']['initialized'] == True:
      self._handle_bt_error('You must add subscriber before start_ros_thread()')
    sub = self.create_subscription(
      msg_type,
      topic,
      callback,
      qos,
      callback_group=callback_group)
    self.subs[topic] = sub

  def get_subscriber(self, sub_topic_name):
    return self.subs.get(sub_topic_name)

  def add_async_subscriber(self, msg_type, topic, callback, qos=1):
    self._add_subscriber(msg_type, topic, callback, self._parallel_callback_group, qos)

  def add_sequential_subscriber(self, msg_type, topic, callback, qos=1):
    self._add_subscriber(msg_type, topic, callback, self._sequential_callback_group, qos)

  def add_async_timer(self, seconds, func):
    return self.create_timer(seconds, func, callback_group=self._parallel_callback_group)

  def add_sequential_timer(self, seconds, func):
    return self.create_timer(seconds, func, callback_group=self._sequential_callback_group)

  def run_leaf_action(self, context: BehaviorTreeContext): # for standalone test
    leaf_action = context.target_leaf_action
    # check if set_status is called in the leaf action function
    self._run_action(leaf_action, context)
    return context

  @staticmethod
  def validate_leaf_action_result(result) -> tuple:
    if not isinstance(result, tuple) or len(result) != 2:
      raise ValueError("Return value must be a tuple of (task_status, dict_to_add)")

    task_status, dict_to_add = result

    allowed_states = (BehaviorTreeStatus.TASK_STATUS_SUCCESS, BehaviorTreeStatus.TASK_STATUS_FAILURE, BehaviorTreeStatus.TASK_STATUS_RUNNING)
    if task_status not in allowed_states:
      raise ValueError(f"task_status must be in {allowed_states} Got {task_status}")

    # if not isinstance(dict_to_add, dict):
    #   raise ValueError(f"Second return value must be a dict. Got {type(dict_to_add).__name__}")

    return task_status, dict_to_add

  def _run_action(self, leaf_action:la.LeafAction, context: BehaviorTreeContext):
    input_dict = context.query_input_of_target_work()
    if self.run_mode == 'standalone':
      target_sector = self.job_manager.default_job.sector_default
      target_work = self.job_manager.default_job.work_default
    elif self.run_mode == 'server':
      target_sector = context.target_sector
      target_work = context.target_work
    else:
      raise ValueError(f'{self.run_mode} not in [standalone, server]')

    if self.is_job_manager_node:
      self.bt_context = context # enable get_context() in leaf action
    leaf_action_ret = BehaviorTreeServerNodeV2.available_actions_to_func[leaf_action](self, input_dict, target_sector, target_work)
    task_status, dict_to_add = self.validate_leaf_action_result(leaf_action_ret)
    if self.is_job_manager_node:
      self.bt_context = BehaviorTreeContext()

    self.set_status(task_status, context, dict_to_add)
    self.mf_logger.info(f'execute action done: {leaf_action.shortly()}')

  def _command_callback(self, msg: BehaviorTreeCommand): # this function has dependency on msg_id management
    # handle msg id
    if self.last_msg_id is not None:
      current_msg_id = msg.msg_id

      # if already completed action, ignore it
      if current_msg_id == self.last_msg_id:
        return

    self.last_msg_id = msg.msg_id

    bt_context = yaml_json_string_to_instance(msg.context_json, BehaviorTreeContext)
    target_leaf_action = bt_context.target_leaf_action

    # execute action
    self.mf_logger.info(f'execute action: {target_leaf_action.shortly()}')

    self._run_action(target_leaf_action, bt_context)


  def _publish_status(self):
    status = self.get_status()
    if status is None:
      return
    self._status_publisher.publish(status)

  def set_status( self,
                  task_status: int,
                  bt_context,
                  memory: dict = None):
    allowed_task_status = [
      BehaviorTreeStatus.TASK_STATUS_SUCCESS,
      BehaviorTreeStatus.TASK_STATUS_FAILURE,
      BehaviorTreeStatus.TASK_STATUS_RUNNING]
    if not task_status in allowed_task_status:
      raise ValueError('task_status must be belong to BehaviorTreeStatus.TASK_STATUS_XXX')

    with self._status_lock:
      if memory is not None:
        target_leaf_action = bt_context.target_leaf_action
        for k, v in memory.items():
          bt_context.set_memory_dict(target_leaf_action, k, v)

      self._current_status.context_json = yaml_instance_to_json_string(bt_context, self.mf_logger)

      # update status for server mode
      self._current_status.header.stamp = self.get_clock().now().to_msg()
      self._current_status.task_status = task_status
      self._current_status.msg_id = self.last_msg_id

      # mark as read
      self._is_current_status_updated = True

  def get_status(self):
    with self._status_lock:
      return self._current_status


  ##################### job manager related #####################
  def get_context(self):
    if self.is_job_manager_node:
      return self.bt_context
    else:
      raise ValueError('get_context() is only available in job manager node')
