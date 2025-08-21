from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import List, Optional, Union
import os
import sys

from config.common.specs.farm_def import FarmSpec
from config.common.specs.robot_def import RobotSpec
from config.common.specs.work_def import WorkSpec

FARM_BASE_PATH = '/opt/config/common/specs/farm_configs'
ROBOT_BASE_PATH = '/opt/config/common/specs/robot_configs'
WORK_BASE_PATH = '/opt/config/common/specs/work_configs'

def load_farm_spec(yaml_path: str) -> FarmSpec:
  yaml_file = os.path.join(FARM_BASE_PATH, yaml_path, 'def.yaml')
  if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"Could not find YAML at: {yaml_file}")

  yaml = YAML()
  with open(yaml_file, 'r') as f:
    data = yaml.load(f)

  try:
    farm_spec = FarmSpec(**data)
  except ValidationError as e:
    print("Validation failed when parsing the farm spec YAML:")
    print(e)
    raise

  return farm_spec

def load_robot_spec(yaml_path: str) -> BaseModel:
  yaml_file = os.path.join(ROBOT_BASE_PATH, yaml_path, 'def.yaml')
  if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"Could not find YAML at: {yaml_file}")

  yaml = YAML()
  with open(yaml_file, 'r') as f:
    data = yaml.load(f)

  try:
    robot_spec = RobotSpec(**data)
  except ValidationError as e:
    print("Validation failed when parsing the robot spec YAML:")
    print(e)
    raise

  return robot_spec

def load_work_spec(yaml_path: str) -> WorkSpec:
  yaml_file = os.path.join(WORK_BASE_PATH, yaml_path, 'def.yaml')
  if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"Could not find YAML at: {yaml_file}")

  yaml = YAML()
  with open(yaml_file, 'r') as f:
    data = yaml.load(f)

  try:
    work_spec = WorkSpec(**data)
  except ValidationError as e:
    print("Validation failed when parsing the work spec YAML:")
    print(e)
    raise

  return work_spec


def _test_farm_spec():
  # create example farm spec
  example_farm_spec = FarmSpec()

  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # dump to stdout
  with open(os.path.join(FARM_BASE_PATH, 'example_farm', 'def.yaml'), 'w') as f:
    yaml.dump(example_farm_spec.model_dump(), f)

  loaded_farm_spec = load_farm_spec('example_farm')
  yaml.dump(loaded_farm_spec.model_dump(), sys.stdout)

def _test_robot_spec():
   # create example farm spec
  example_robot_spec = RobotSpec()

  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # dump to stdout
  with open(os.path.join(ROBOT_BASE_PATH, 'example_robot', 'def.yaml'), 'w') as f:
    yaml.dump(example_robot_spec.model_dump(), f)

  loaded_robot_spec = load_robot_spec('example_robot')
  yaml.dump(loaded_robot_spec.model_dump(), sys.stdout)

def _test_work_spec():
  # create example work spec
  example_work_spec = WorkSpec()

  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # dump to stdout
  with open(os.path.join(WORK_BASE_PATH, 'example_work', 'def.yaml'), 'w') as f:
    yaml.dump(example_work_spec.model_dump(), f)

  loaded_work_spec = load_work_spec('example_work')
  yaml.dump(loaded_work_spec.model_dump(), sys.stdout)


if __name__ == "__main__":
  print('#### farm spec example ####')
  _test_farm_spec()
  print('############################\n\n')

  print('#### robot spec example ####')
  _test_robot_spec()
  print('############################\n\n')

  print('#### work spec example ####')
  _test_work_spec()
  print('############################\n\n')

  