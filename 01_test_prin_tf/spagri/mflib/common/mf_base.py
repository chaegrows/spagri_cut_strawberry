import logging
import logging.handlers
import queue
import os
import time
import atexit

from ruamel.yaml import YAML
from pydantic import BaseModel
import sys
import json
import io
from typing import Type

# Never import any other custom moduls in this file

# ANSI 색상 및 컬러 로그 포맷터 클래스
class ColoredFormatter(logging.Formatter):
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    RESET = '\033[0m'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            color = self.WHITE
        elif record.levelno == logging.INFO:
            color = self.WHITE
        elif record.levelno == logging.WARNING:
            color = self.YELLOW
        elif record.levelno == logging.ERROR:
            color = self.RED
        else:
            color = self.WHITE

        current_time = time.strftime("%H:%M:%S", time.localtime())

        # WARNING, ERROR는 prefix도 색상 적용
        if record.levelno in (logging.WARNING, logging.ERROR):
            prefix = f"{color}[{record.levelname}, {current_time}]{self.RESET}"
        else:
            prefix = f"{self.WHITE}[{record.levelname}, {current_time}]{self.RESET}"
        message = f"{color}{record.getMessage()}{self.RESET}"

        return f"{prefix} {message}"

############ yaml related
_yaml = YAML()
_yaml.default_flow_style = False

def _dump_yaml(data, stream):
    _yaml.dump(data, stream)

def _load_yaml(stream):
    return _yaml.load(stream)

def yaml_instance_to_string(instance: BaseModel) -> str:
    try:
        stream = io.StringIO()
        _dump_yaml(instance.model_dump(mode="json"), stream)
    except Exception:
        print(instance)
        return ''
    return stream.getvalue()

def yaml_string_to_instance(yaml_string: str, model_class: Type[BaseModel]) -> BaseModel:
    data = _load_yaml(io.StringIO(yaml_string))
    return model_class.model_validate(data)

def yaml_file_to_instance(file_path: str, model_class: Type[BaseModel]) -> BaseModel:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = _load_yaml(f)
    return model_class.model_validate(data)

def yaml_instance_to_file(instance: BaseModel, file_path: str):
    if instance is None:
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        _dump_yaml(instance.model_dump(mode="json"), f)

def yaml_print(instance: BaseModel):
    if instance is None:
        print("None")
        return
    _dump_yaml(instance.model_dump(mode="json"), sys.stdout)

def yaml_instance_to_json_string(instance: BaseModel, logger = None) -> str:
    if instance is None:
        return
    ret = instance.model_dump_json()
    if logger is not None:
        logger.info(f'instance in function yaml_instance_to_json_string: {instance.model_dump_json()}')
    return instance.model_dump_json()

def yaml_json_string_to_instance(json_string: str, model_class: Type[BaseModel], logger = None) -> BaseModel:
    if logger is not None:
        logger.info(f'json_string in function yaml_json_string_to_instance: {json_string}')
    data = json.loads(json_string)
    return model_class.model_validate(data)

############# logger related

def get_logger_for_node(node_name, log_dir='/opt/logs'):
  log_queue = queue.Queue(-1)  # Create an unlimited-size queue for asynchronous logging

  logger = logging.getLogger(node_name)
  logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG

  # 콘솔 핸들러
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_formatter = ColoredFormatter(
    "%(message)s",  # 포맷 문자열을 단순화
    datefmt="%H:%M:%S"
  )
  console_handler.setFormatter(console_formatter)
  logger.addHandler(console_handler)

  # Add a QueueHandler to push logs into the memory queue
  queue_handler = logging.handlers.QueueHandler(log_queue)
  logger.addHandler(queue_handler)

  # Create a date-based directory: log_dir/YYYYMMDD/
  start_time = time.localtime()
  date_folder = time.strftime("%Y%m%d", start_time)
  log_dir = os.path.join(log_dir, date_folder)
  os.makedirs(log_dir, exist_ok=True)

  # Define the log file path: node_name.log
  log_path = os.path.join(log_dir, f"{node_name}.log")

  if os.path.exists(log_path):
    # If the log file already exists, insert a marker line with timestamp
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_path, mode='a') as f:
      f.write(f"\n\n===== New Process Started at {now} =====\n")

  # Create a RotatingFileHandler for writing logs to file
  file_handler = logging.handlers.RotatingFileHandler(
    log_path,
    mode='a',                  # Append mode
    maxBytes=10 * 1024 * 1024,  # 10 MB before rotating
    backupCount=5               # Keep up to 5 backup files
  )
  file_formatter = logging.Formatter("[%(levelname)s, %(asctime)s] %(message)s")
  file_handler.setFormatter(file_formatter)
  file_handler.setLevel(logging.DEBUG)

  # Start a QueueListener to asynchronously write logs from the queue to the file
  listener = logging.handlers.QueueListener(log_queue, file_handler)
  listener.start()

  # Register listener.stop() to be called when the program exits
  atexit.register(listener.stop)

  return logger

##############

########### exception related

def raise_with_log(logger, exc_class, message):
  logger.error(message)
  raise exc_class(message)

############
