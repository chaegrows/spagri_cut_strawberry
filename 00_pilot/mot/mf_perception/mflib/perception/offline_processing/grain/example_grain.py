from __future__ import annotations
from pydantic import model_validator, Field
from mflib.common.mf_base import yaml_print, raise_with_log
from typing import ClassVar, Tuple

from mflib.perception.offline_processing.grain.grain_base import GrainBase, GrainRegistry

@GrainRegistry.register
class ExampleGrain(GrainBase):
  grain_class_names: ClassVar[Tuple[str]] = ("crop_type", "strawberry", "fruit")

  some_attribute1: str = Field(..., description="attr1")
  some_attribute2: int = Field(..., description="attr2")
  
  @model_validator(mode='after')
  def check_validity(self) -> 'ExampleGrain':
    if self.some_attribute1 != 'hello':
      ValueError(
        f"some_attribute should be 'hello', but got {self.some_attribute1}"
      )
    if self.some_attribute2 != 0:
      ValueError(
        f"some_attribute2 should be 0, but got {self.some_attribute2}"
      )
    return self
  
  def get_instance() -> ExampleGrain:
    """
    Get an instance of ExampleGrain.
    """
    return ExampleGrain(
      some_attribute1='hello',
      some_attribute2=0
    )
  
def main():
  example_grain = ExampleGrain(some_attribute1='hello', some_attribute2=0)
  yaml_print(example_grain)


if __name__ == "__main__":
  main()