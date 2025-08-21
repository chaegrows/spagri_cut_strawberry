from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import List, Optional, Union
import os
import sys

ROBOT_BASE_PATH = '/opt/config/common/specs/robot_configs'

class GroundVehicle(BaseModel):
  name: str = Field(description="Name of the gv", default="gv")

  # communication
  communication_interface: str = Field(description="Communication interface for the gv", default="USB_serial")
  usb_serial_vendor_id: Optional[str] = Field(description="USB vendor ID for the gv", default='0x1234')
  usb_serial_product_id: Optional[str] = Field(description="USB product ID for the gv", default='0x5678')
  usb_serial_baudrate: Optional[int] = Field(description="USB baudrate for the gv", default=115200)
  usb_serial_reconnect: Optional[bool] = Field(description="Reconnect to the gv if disconnected", default=True)
  usb_serial_reconnect_delay_ms: Optional[int] = Field(description="Delay in ms before reconnecting to the gv", default=1000)
  usb_serial_verbose: bool = Field(description="Verbose mode for the gv", default=False)

  # lidar
  mid360_lidar_ip: str = Field(description="IP address of the mid360 lidar", default='192.168.5.100')

  # velocities
  standard_linear_velocity: float = Field(description="Standard linear velocity of the gv in m/s", default=0.5)
  standard_angular_velocity: float = Field(description="Standard angular velocity of the gv in rad/s", default=0.5)
  max_linear_velocity: float = Field(description="Maximum linear velocity of the gv in m/s", default=1.0)
  max_angular_velocity: float = Field(description="Maximum angular velocity of the gv in rad/s", default=0.5)

  class Config:
    frozen = True

  @model_validator(mode='after')
  def validate(self) -> 'GroundVehicle':
    allowed = ['USB_serial']
    if self.communication_interface not in allowed:
      raise ValueError(f"Invalid communication interface. Must be one of: {allowed}")

    if self.communication_interface == 'USB_serial':
      if not all([self.usb_serial_vendor_id, self.usb_serial_product_id, self.usb_serial_baudrate,
                  self.usb_serial_reconnect, self.usb_serial_reconnect_delay_ms]):
        raise ValueError("All USB serial communication fields must be specified")

    if self.standard_linear_velocity > self.max_linear_velocity:
      raise ValueError("Standard linear velocity cannot exceed maximum linear velocity.")
    if self.standard_angular_velocity > self.max_angular_velocity:
      raise ValueError("Standard angular velocity cannot exceed maximum angular velocity.")
    if self.standard_linear_velocity < 0 or self.standard_angular_velocity < 0:
      raise ValueError("Standard velocities must be non-negative.")
    if self.max_linear_velocity < 0 or self.max_angular_velocity < 0:
      raise ValueError("Max velocities must be non-negative.")

    return self


class Lift(BaseModel):
  name: str = Field(description="Name of the lift", default="md200lift")
  max_height: float = Field(description="Maximum height of the lift in meters", default=1.0)
  min_height: float = Field(description="Minimum height of the lift in meters", default=0.0)

  class Config:
    frozen = True

  @model_validator(mode='after')
  def validate(self) -> 'Lift':
    if self.min_height < 0:
      raise ValueError("Minimum height cannot be negative.")
    if self.max_height < self.min_height:
      raise ValueError("Maximum height cannot be less than minimum height.")
    return self

class EndEffector(BaseModel):
  name: str = Field(description="Name of the endeffector", default='endeffector')
  device: str = Field(description="Device of the endeffector", default='/dev/ttyTHS1')
  baudrate: int = Field(description="Baudrate of the endeffector", default=115200)
  verbose: Optional[bool] = Field(description="Verbose mode of the endeffector", default=True)
  reconnect: Optional[bool] = Field(description="Reconnect mode of the endeffector", default=True)
  reconnect_delay_ms: Optional[int] = Field(description="Reconnect delay of the endeffector in ms", default=1000)

  class Config:
    frozen = True

  @model_validator(mode='after')
  def validate(self) -> 'EndEffector':
    if self.reconnect_delay_ms < 0:
      raise ValueError("Reconnect delay cannot be negative.")
    return self


class StaticTransforms(BaseModel):
  frame_id: str = Field(description="Frame ID of the static transform", default="your_base_link")
  child_frame_id: str = Field(description="Child frame ID of the static transform", default="my_camera")
  translation: List[float] = Field(description="Translation of the static transform in meters", default=[0.0, 0.0, 0.0])
  rotation: List[float] = Field(description="Rotation of the static transform in radians", default=[0.0, 0.0, 0.0])

  class Config:
    frozen = True

  @model_validator(mode='after')
  def validate(self) -> 'StaticTransforms':
    if len(self.translation) != 3:
      raise ValueError("Translation must be a list of 3 floats.")
    if len(self.rotation) != 3:
      raise ValueError("Rotation must be a list of 3 floats.")
    return self

class RobotSpec(BaseModel):
  uuid: str = Field(description="UUID of the robot", default="robot_uuid")
  ground_vehicle: GroundVehicle = Field(description="Ground vehicle of the robot", default_factory=GroundVehicle)
  lift: Lift = Field(description="Lift of the robot", default_factory=Lift)
  end_effector: EndEffector = Field(description="Endeffector of the robot", default_factory=EndEffector)
  static_transforms: List[StaticTransforms] = Field(description="Static transforms of the robot", default_factory=lambda: [StaticTransforms()])

  @staticmethod
  def create_spec_by_name(robot_name):
    yaml_path = os.path.join(ROBOT_BASE_PATH, robot_name, 'def.yaml')
    print(f"@@@@@ yaml_path: {yaml_path}")
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Robot spec not found: {yaml_path}")
    with open(yaml_path, "r") as f:
      return RobotSpec.model_validate(YAML().load(f))

  class Config:
    frozen = True

  @model_validator(mode='after')
  def validate(self) -> 'RobotSpec':
    # each instances are already validated
    return self
