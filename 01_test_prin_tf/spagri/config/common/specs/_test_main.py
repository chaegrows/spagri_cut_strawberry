from config.common.specs.job_manager import JobManager
from ruamel.yaml import YAML
import os
import sys

from config.common.specs.farm_def import FarmSpec, FARM_BASE_PATH
from config.common.specs.robot_def import RobotSpec, ROBOT_BASE_PATH
from config.common.specs.work_def import WorkSpec, WORK_BASE_PATH, Pollination
from config.common.specs.job_manager import JobSchedule, JobManager, _load_yaml

def _test_farm_spec():
  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # create example farm spec
  example_farm_spec = FarmSpec()

  # dump to stdout
  with open(os.path.join(FARM_BASE_PATH, 'example_farm', 'def.yaml'), 'w') as f:
    yaml.dump(example_farm_spec.model_dump(), f)

  loaded_farm_spec = FarmSpec.create_spec_by_name('example_farm')
  yaml.dump(loaded_farm_spec.model_dump(), sys.stdout)

def _test_robot_spec():
   # create example farm spec
  example_robot_spec = RobotSpec()

  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # dump to stdout
  with open(os.path.join(ROBOT_BASE_PATH, 'example_robot', 'def.yaml'), 'w') as f:
    yaml.dump(example_robot_spec.model_dump(), f)

  loaded_robot_spec = RobotSpec.create_spec_by_name('example_robot')
  yaml.dump(loaded_robot_spec.model_dump(), sys.stdout)

def _test_work_spec():
  # create example work spec
  example_work_spec = WorkSpec()

  yaml = YAML()
  yaml.default_flow_style = False  # 줄바꿈 있는 형태로 출력

  # dump to stdout
  with open(os.path.join(WORK_BASE_PATH, 'example_work', 'def.yaml'), 'w') as f:
    yaml.dump(example_work_spec.model_dump(), f)

  loaded_work_spec = WorkSpec.create_spec_by_name('example_work')
  yaml.dump(loaded_work_spec.model_dump(), sys.stdout)

def _test_job_manager():
  # create manager
  job_manager = JobManager()

  # load specs
  # job_manager.setup(specs_file='example_job')
  job_manager.setup(specs_file='dongtan_pollinate_job')

  # schedule sector and work
  target_work = Pollination(
    work_type='pollination',
    name='example_pollination',
    perception_algorithm='strawberry_harvesting_pollination',
    vibration_seconds=1
  )
  # job_manager.create_schedule(target_work, 'simple')
  job_manager.create_schedule(target_work, 'only_left')
  # job_manager.create_schedule(target_work, 'only_right')
  
  # print
  job_manager.print_summary()


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

  print('#### job manager example ####')
  _test_job_manager()
  print('############################\n\n')

  