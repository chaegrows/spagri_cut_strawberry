from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import List, Optional, Union
import os
import sys

FARM_BASE_PATH =  '/opt/config/common/specs/farm_configs'


class Rack(BaseModel):
  rack_id: int = Field(description="Identifier for the rack. 1, 2, ...", default=1)
  rack_start_xyz: List[float] = Field(description="Start position of the rack", default_factory=lambda: [1, 2, 3])
  rack_end_xyz: List[float] = Field(description="End position of the rack", default_factory=lambda: [1, 2, 3])
  n_horizon: int = Field(description="Number of horizon", default=1)
  n_vertical: int = Field(description="Number of vertical", default=1)
  horizon_poses: List[float] = Field(description="Horizon poses", default_factory=lambda: [1, 2, 3])
  vertical_poses: List[float] = Field(description="Vertical poses", default_factory=lambda: [1, 2, 3])
  plant_offsets: List[float] = Field(description="Plant offsets", default_factory=lambda: [0.0])
  ws_list_xyzwlh: List[List[float]] = Field(description="Global ground vehicle pose=> is available workspace param", default_factory=lambda: [1, 1, 1, 0.1, 0.1, 0.1])


  class Config:
    frozen = True

  @model_validator(mode='after')
  def check_validity(self) -> 'Rack':
    if self.rack_id <= 0:
      raise ValueError(f"Invalid rack ID: {self.rack_id}. Rack ID must be non-negative")
    return self


class Sector(BaseModel):
  rack_id: int = Field(description="Identifier for the rack. Left to right: 1, 2, ...", default=1)
  height_id: int = Field(description="Identifier for the height. Bottom to top: 1, 2, ...", default=1)
  horizon_id: int = Field(description="Identifier for the horizon. Close to far: 1, 2, ...", default=1)
  required_viewpoint: str = Field('left', description="Identifier for the required viewpoint. only 'left' or 'right' is allowed")
  required_flipped: bool = Field(description="Whether the sector is flipped", default=False)
  global_height: float = Field(description="Global height of the sector", default=1.2)
  global_gv_pose: List[float] = Field(description="Global ground vehicle pose", default_factory=lambda: [1, 2, 3])
  depth_to_plants: float = Field(description="Depth to plants from center of the rack", default=0.6)

  class Config:
    frozen = True

  @model_validator(mode='after')
  def check_validity(self) -> 'Sector':
    if self.rack_id <= 0 or self.height_id <= 0 or self.horizon_id <= 0:
      raise ValueError(f"Invalid lane, height, or horizon ID: {self.rack_id}, {self.height_id}, {self.horizon_id}. Must be non-negative")
    if self.required_viewpoint not in ['left', 'right']:
      raise ValueError(f"Invalid required viewpoint: {self.required_viewpoint}. Only 'left' or 'right' is allowed")
    if len(self.global_gv_pose) != 3:
      raise ValueError(f"Invalid global ground vehicle pose: {self.global_gv_pose}. Must be a list of 3 floats")
    if self.depth_to_plants < 0:
      raise ValueError(f"Invalid depth to plants: {self.depth_to_plants}. Must be non-negative")
    return self


  def get_sector_id(self) -> str:
    required_flipped = 1 if self.required_flipped else 0
    return f"l{self.rack_id}h{self.height_id}h{self.horizon_id}r{self.required_viewpoint}f{required_flipped}"




class Lane(BaseModel):
  rack_id: int = Field(description="Identifier for the lane. Left to right: 0, 1, 2, ...", default=1)
  dir_from_rack_to_plants: str = Field(description="Direction from rack to plants. only 'left' or 'right' is allowed", default='left')
  lane_home: List[float] = Field(description="Home position of the lane", default_factory=lambda: [-10, 2, 3])
  lane_end: List[float] = Field(description="End position of the lane", default_factory=lambda: [20, 2, 3])

  class Config:
    frozen = True

  @model_validator(mode='after')
  def check_validity(self) -> 'Lane':
    if len(self.lane_home) != 3 or len(self.lane_end) != 3:
      raise ValueError(f"Invalid lane home or end position: {self.lane_home}, {self.lane_end}. Must be a list of 3 floats")
    if self.rack_id <= 0:
      raise ValueError(f"Invalid lane ID: {self.rack_id}. Lane ID must be non-negative")
    return self


class FarmSpec(BaseModel):
  name: str = Field(description="Name of the farm", default="example_farm")
  pcd_map_file: str = Field(description="Path to the pcd map file for localizing", default="example_map.pcd")
  pcd_viz_map_file: str = Field(description="Path to the pcd viz map file for visualizing", default="example_viz_map.pcd")
  racks: List[Rack] = Field(description="List of racks in the farm", default_factory=lambda: [Rack()])
  sectors: List[Sector] = Field(description="List of sectors in the farm", default_factory=lambda: [Sector()])
  lanes: List[Lane] = Field(description="List of lanes in the farm", default_factory=lambda: [Lane()])

  class Config:
    frozen = True

  @staticmethod
  def create_spec_by_name(farm_name):
    yaml_path = os.path.join(FARM_BASE_PATH, farm_name, 'def.yaml')
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Farm spec not found: {yaml_path}")
    with open(yaml_path, "r") as f:
      return FarmSpec.model_validate(YAML().load(f))

  @model_validator(mode='after')
  def check_rack_id_consistency(self) -> 'FarmSpec':
    rack_ids_in_sectors = set(sector.rack_id for sector in self.sectors)
    rack_ids_in_lanes = set(lane.rack_id for lane in self.lanes)
    if rack_ids_in_sectors != rack_ids_in_lanes:
      raise ValueError(f"Lane IDs in sectors and lanes do not match: {rack_ids_in_sectors} vs {rack_ids_in_lanes}")
    return self

  def get_lane_by_id(self, rack_id: int, required_viewpoint: str) -> Lane:
    if required_viewpoint not in ['left', 'right']:
      raise ValueError(f"Invalid dir_from_rack_to_plants: {dir_from_rack_to_plants}. Only 'left' or 'right' is allowed")
    dir_from_rack_to_plants = 'left' if required_viewpoint == 'right' else 'right'
    for lane in self.lanes:
      if lane.rack_id == rack_id and lane.dir_from_rack_to_plants == dir_from_rack_to_plants:
        return lane
    raise ValueError(f"Lane with ID {rack_id} not found")
