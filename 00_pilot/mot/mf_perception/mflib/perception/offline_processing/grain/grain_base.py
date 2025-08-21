from __future__ import annotations # late type check

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML
from typing import ClassVar, Tuple, Dict, Type
from abc import abstractmethod, ABC
import mflib.perception.offline_processing.grain as grain_pkg
import pkgutil
import importlib


from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_string,
  yaml_string_to_instance,
  yaml_instance_to_json_string,
  yaml_json_string_to_instance,
  yaml_print,
) 

BASE_GRAIN_PATH = '/opt/mflib/perception/offline_processing/grain'

class GrainBase(BaseModel, ABC):
  grain_class_names: ClassVar[Tuple[str]] = ("default1", "default2")
  # example: ['crop_type', 'strawberry', '...]
  # example2: ['raw_data', 'rosbag2', ...]

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if not isinstance(cls.grain_class_names, Tuple) or not all(isinstance(x, str) for x in cls.grain_class_names):
      raise TypeError(f"grain_class_names must be List[str], got {cls.grain_class_names}")

  @classmethod
  @abstractmethod
  def get_instance(cls) -> 'GrainBase':
    pass

  @abstractmethod
  @model_validator(mode='after')
  def check_validity(self) -> GrainBase:
    """
    Check if the class names are valid.
    """
    pass

class GrainRegistry:
  _registry: Dict[str, Type[GrainBase]] = {}

  @classmethod
  def register(cls, grain_cls: Type[GrainBase]):
    cls._registry[grain_cls.__name__] = grain_cls
    return grain_cls

  @classmethod
  def all(cls) -> Dict[str, Type[GrainBase]]:
    return cls._registry
  
def import_all_grains():
  for _, module_name, _ in pkgutil.iter_modules(grain_pkg.__path__):
    importlib.import_module(f"{grain_pkg.__name__}.{module_name}")
  # check for duplicate grain_class_names
  seen = {}
  for cls in GrainRegistry.all().values():
    if cls is GrainBase:
      continue  # Skip base class
    key = tuple(cls.grain_class_names)
    if key in seen:
      raise ValueError(f"Duplicate grain_class_names found:\n  {seen[key].__name__}\n  {cls.__name__}")
    seen[key] = cls

def find_grain_by_class_names(name_tuple: Tuple[str]) -> Type[GrainBase]:
  for cls in GrainRegistry.all().values():
    if tuple(cls.grain_class_names) == name_tuple:
      return cls
  return None