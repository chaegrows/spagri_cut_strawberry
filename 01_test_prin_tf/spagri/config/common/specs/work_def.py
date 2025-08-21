from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import List, Optional, Union
import os
import sys

WORK_BASE_PATH = '/opt/config/common/specs/work_configs'

class BaseWorkSpec(BaseModel):
  work_type: str = Field(description="Type of work to be done", default="pollination")
  perception_algorithm: str = Field(description="Perception algorithm to be used", default="strawberry_harvesting_pollination")

  class Config:
    frozen = True

  @model_validator(mode="after")
  def check_required_fields(self) -> 'BaseWorkSpec':
    missing = []
    if not self.work_type:
      missing.append("work_type")
    if not self.name:
      missing.append("name")
    if not self.perception_algorithm:
      missing.append("perception_algorithm")

    if missing:
      raise ValueError(f"Missing required fields: {missing}")

    if self.work_type not in ["harvesting", "pollination"]:
      raise ValueError(f"Invalid work_type: {self.work_type}. Must be 'harvesting' or 'pollination'")
    if self.perception_algorithm not in ['strawberry_harvesting_pollination']:
      raise ValueError(f"Invalid perception_algorithm: {self.perception_algorithm}. Must be 'strawberry_harvesting_pollination'")
    
    return self


class Harvesting(BaseWorkSpec):
  n_slots: Optional[int] = Field(description="Number of slots for the harvesting carrier", default=1)
  name: str = Field(description="Name of the harvesting work spec", default="example_harvesting")  

  class Config:
    frozen = True

  @model_validator(mode="after")
  def check_n_slots(self) -> 'Harvesting':
    if self.n_slots is not None and self.n_slots < 1:
      raise ValueError("n_slots must be >= 1")
    return self


class Pollination(BaseWorkSpec):
  vibration_seconds: Optional[float] = Field(description="Vibration time in seconds", default=1)
  name: str = Field(description="Name of the pollination work spec", default="example_pollination")  

  class Config:
    frozen = True

  @model_validator(mode="after")
  def check_vibration(self) -> 'Pollination':
    if self.vibration_seconds is not None and self.vibration_seconds <= 0:
      raise ValueError("vibration_seconds must be positive")
    return self

ALLOWED_WORKS = Union[Harvesting, Pollination]
class WorkSpec(BaseModel):
  harvesting: Optional[Harvesting] = Field(description="Harvesting work spec", default_factory=Harvesting)
  pollination: Optional[Pollination] = Field(description="Pollination work spec", default_factory=Pollination)

  @staticmethod
  def create_spec_by_name(work_name):
    yaml_path = os.path.join(WORK_BASE_PATH, work_name, 'def.yaml')
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Work spec not found: {yaml_path}")
    with open(yaml_path, "r") as f:
      return WorkSpec.model_validate(YAML().load(f))

  class Config:
    frozen = True

  @model_validator(mode="after")
  def validate_children(self) -> 'WorkSpec':
    # check if names are duplicated
    if self.harvesting is None:
      raise ValueError("harvesting is None")
    if self.pollination is None:
      raise ValueError("pollination is None")
    
    names = [work.name for work in [self.harvesting, self.pollination]]
    if len(names) != len(set(names)):
      raise ValueError(f"Duplicate work names found: {names}")
    return self

