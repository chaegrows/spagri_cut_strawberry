from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import List, Optional, Union, Dict, Any
import os
import sys
import re

from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_json_string,
  yaml_instance_to_string,
  yaml_json_string_to_instance,
  yaml_string_to_instance,
  yaml_print
)

from config.common.specs.farm_def import FarmSpec, Sector, FARM_BASE_PATH
from config.common.specs.robot_def import RobotSpec, ROBOT_BASE_PATH
from config.common.specs.work_def import WorkSpec, ALLOWED_WORKS, WORK_BASE_PATH, Pollination

JOB_BASE_PATH = '/opt/config/common/specs/job_configs'

def parse_sec_pointer(sec_pointer: str):
    match = re.match(r"l(\d+)h(\d+)h(\d+)r([lr])f(\d)", sec_pointer)
    if not match:
        raise ValueError(f"Invalid format for sec_pointer: {sec_pointer}")
    
    rack_id = int(match.group(1))
    height_id = int(match.group(2))
    horizon_id = int(match.group(3))
    required_viewpoint = 'left' if match.group(4) == 'l' else 'right'
    required_flipped = bool(int(match.group(5)))
    
    return rack_id, height_id, horizon_id, required_viewpoint, required_flipped

def _load_yaml(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"Spec file not found: {path}")
  with open(path, "r") as f:
    return YAML().load(f)

class Specifications(BaseModel):
  farm_name: str = Field(description="Farm specification path", default="example_farm")
  robot_name: str = Field(description="Robot specification path", default="example_robot")
  work_name: str = Field(description="Work specification path", default="example_work")

  farm_spec: Optional[FarmSpec] = Field(description="Farm specification", default=None)
  robot_spec: Optional[RobotSpec] = Field(description="Robot specification", default=None)
  work_spec: Optional[WorkSpec] = Field(description="Work specification", default=None)

  @staticmethod
  def load_specs(job_name):
    # create specification instance
    specs_path = os.path.join(JOB_BASE_PATH, job_name, 'specs.yaml')
    specification = Specifications.model_validate(_load_yaml(specs_path))
    # initiate specs
    path = os.path.join(FARM_BASE_PATH, specification.farm_name, 'def.yaml')
    specification.farm_spec = FarmSpec.model_validate(_load_yaml(path))
    path = os.path.join(ROBOT_BASE_PATH, specification.robot_name, 'def.yaml')
    specification.robot_spec = RobotSpec.model_validate(_load_yaml(path))
    path = os.path.join(WORK_BASE_PATH, specification.work_name, 'def.yaml')
    specification.work_spec = WorkSpec.model_validate(_load_yaml(path))

    return specification

class SectorRef(BaseModel):
  rack_id: int = Field(description="Identifier for the lane. Left to right: 0, 1, 2, ...", default=1)
  required_viewpoint: str = Field(description="Identifier for the required viewpoint. only 'left' or 'right' is allowed", default='left')
  height_id: int = Field(description="Identifier for the height. Bottom to top: 0, 1, 2, ...", default=1)
  horizon_id: int = Field(description="Identifier for the horizon. Close to far: 0, 1, 2, ...", default=1)
  required_flipped: bool = Field(description="Whether the sector is flipped", default=False)

  class Config:
    frozen = True

class WorkRef(BaseModel):
  name: str = Field(description="Name of the work spec", default="example_harvesting")

  class Config:
    frozen = True

class DefaultJob(BaseModel):
  sector_ref: SectorRef = Field(description="Reference to the sector", default_factory=SectorRef)
  work_ref: WorkRef = Field(description="Reference to the work spec", default_factory=WorkRef)
  sector_default: Optional[Sector] = Field(description="Default sector", default=None)
  work_default: Optional[ALLOWED_WORKS] = Field(description="Default work", default=None)

  @staticmethod
  def load_default_job(job_name):
    path = os.path.join(JOB_BASE_PATH, job_name, 'default_job.yaml')
    if not os.path.exists(path):
      raise FileNotFoundError(f"Default job spec not found: {path}")
    return DefaultJob.model_validate(_load_yaml(path))


class ScheduleMapping(BaseModel):
  schedule_keys: List[str]
  keys_to_sector_pointers: Dict[str, List[str]] = {}

  @model_validator(mode='before')
  @classmethod
  def move_dynamic_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    schedule_keys = data.get("schedule_keys", [])
    extracted = {}

    for key in schedule_keys:
      if key in data:
        extracted[key] = data.pop(key)

    data["keys_to_sector_pointers"] = extracted
    return data

  def get_sector_pointers(self, key: str) -> List[SectorRef]:
    if key not in self.keys_to_sector_pointers:
      raise ValueError(f"Key {key} not defined in the schedule mapping")
    return self.keys_to_sector_pointers[key]

  @staticmethod
  def load_schedule_mapping(job_name):
    path = os.path.join(JOB_BASE_PATH, job_name, 'schedule_mapping.yaml')
    if not os.path.exists(path):
      return None
    return ScheduleMapping.model_validate(_load_yaml(path))


class JobSchedule(BaseModel): # mutable
  target_work: ALLOWED_WORKS = Field(description="Target work to be done", default=None)
  sectors: List[Sector] = Field(description="List of sectors to work on", default_factory=lambda: [])
  current_sector_idx: Optional[int] = Field(description="Index of the current sector", default=0)

  def switch_to_next_job(self):
    if self.target_work is None:
      raise ValueError("target_work can not be None when calling this function")
    if len(self.sectors) == 0:
      raise ValueError("len(sectors) == 0")
    if self.current_sector_idx < len(self.sectors) - 1:
      self.current_sector_idx += 1
    else:
      return None, None
    return self.sectors[self.current_sector_idx], self.target_work

  def set_current_sector(self, sector: Sector):
    if sector in self.sectors:
      self.current_sector_idx = self.sectors.index(sector)
    else:
      raise ValueError(f"Sector {sector} not found in the schedule")

  @staticmethod
  def create_schedule(
      sectors: List[Sector],
      target_work: ALLOWED_WORKS,
      rule: str,
      schedule_mapping: Optional[ScheduleMapping] = None) -> JobSchedule:
    if not sectors:
      raise ValueError("No sectors provided")

    scheduled_sectors = []
    if rule == 'simple':
      scheduled_sectors = sectors  # defined in farm yaml
    else:
      if schedule_mapping is None:
        raise ValueError("schedule_mapping is required for non-simple rules")
      if rule not in schedule_mapping.schedule_keys:
        raise ValueError(f"Rule {rule} not found in schedule mapping")
      sector_pointers = schedule_mapping.get_sector_pointers(rule)
      for sec_pointer in sector_pointers:
          rack_id, height_id, horizon_id, required_viewpoint, required_flipped = parse_sec_pointer(sec_pointer)

          found = False
          for sector in sectors:
              if (
                  int(sector.rack_id) == rack_id and
                  int(sector.height_id) == height_id and
                  int(sector.horizon_id) == horizon_id and
                  sector.required_viewpoint == required_viewpoint and
                  sector.required_flipped == required_flipped
              ):
                  scheduled_sectors.append(sector)
                  found = True
                  break

          if not found:
              raise ValueError(f"Sector {sec_pointer} not found in the farm spec")


    return JobSchedule(
      sectors=scheduled_sectors,
      current_sector_idx=-1,
      target_work=target_work
    )



class JobManager:
  specifications: Optional[Specifications] = Field(description="All specs required for job", default_factory=Specifications)

  schedule_mapping: Optional[ScheduleMapping] = Field(description="Schedule mapping for the job", default=None)
  job_scheduler: Optional[JobSchedule] = Field(description="Work schedule to execute", default=None)

  def __init__(self):
    self.specifications = None
    self.job_scheduler = None
    self.default_job = None
    self.schedule_mapping = None

  def setup(self, **kwargs):
    specs_file = kwargs.get('specs_file')
    if specs_file is None:
      raise ValueError("specs_file is required")
    # specifications
    self.specifications = Specifications.load_specs(specs_file)
    # default job
    self.default_job = DefaultJob.load_default_job(specs_file)
    self._validate_default_job()
    # schedule_mapping
    self.schedule_mapping = ScheduleMapping.load_schedule_mapping(specs_file)
    # job_scheduler
    self.job_scheduler = JobSchedule()

  def _validate_default_job(self):
    # check if default sector is in the farm spec
    is_sector_ok = False
    for sectors in self.specifications.farm_spec.sectors:
      if  sectors.rack_id == self.default_job.sector_ref.rack_id and \
          sectors.height_id == self.default_job.sector_ref.height_id and \
          sectors.horizon_id == self.default_job.sector_ref.horizon_id and \
          sectors.required_viewpoint == self.default_job.sector_ref.required_viewpoint and \
          sectors.required_flipped == self.default_job.sector_ref.required_flipped:
        self.default_job.sector_default = sectors
        is_sector_ok = True
        break
    if not is_sector_ok:
      raise ValueError(f"Default job sector not found in farm spec")

    is_work_ok = False
    for work_defined in self.specifications.work_spec:
      if self.default_job.work_ref.name == work_defined[1].name:
        self.default_job.work_default = work_defined[1]
        is_work_ok = True
        break
    if not is_work_ok:
      raise ValueError(f"Default job work not found in work spec")

  def create_schedule(self, target_work, rule):
    sectors = self.specifications.farm_spec.sectors
    self.job_scheduler = JobSchedule.create_schedule(sectors, target_work, rule, self.schedule_mapping)

  def print_summary(self):
    print('specifications: ')
    yaml_print(self.specifications)
    print('\ndefault job: ')
    yaml_print(self.default_job)
    print('\nschedule mapping: ')
    yaml_print(self.schedule_mapping)
    print('\ncurrent_schedule: ')
    for sector in self.job_scheduler.sectors:
      print(sector)

if __name__ == "__main__":
  sample_specs = Specifications()
  filepath = os.path.join(JOB_BASE_PATH, 'example_job', 'specs.yaml')
  yaml_instance_to_file(sample_specs, filepath)
  yaml_print(sample_specs)

  default_job = DefaultJob()
  filepath = os.path.join(JOB_BASE_PATH, 'example_job', 'default_job.yaml')
  yaml_instance_to_file(default_job, filepath)
  yaml_print(default_job)
