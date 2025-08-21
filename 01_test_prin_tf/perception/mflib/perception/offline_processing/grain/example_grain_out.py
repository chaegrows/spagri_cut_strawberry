from __future__ import annotations
from pydantic import model_validator, Field
from mflib.common.mf_base import yaml_print, raise_with_log
from typing import ClassVar, Tuple

from mflib.perception.offline_processing.grain.grain_base import GrainBase, GrainRegistry

@GrainRegistry.register
class ExampleGrainOut(GrainBase):
  grain_class_names: ClassVar[Tuple[str]] = ("crop_type", "strawberry", "fruit_images_preprocessed")

  some_attribute3: str = Field(..., description="attr1")
  
  @model_validator(mode='after')
  def check_validity(self) -> 'ExampleGrainOut':
    return self
  
  def get_instance() -> ExampleGrainOut:
    """
    Get an instance of ExampleGrainOut.
    """
    return ExampleGrainOut(
      some_attribute3='default_value'
    )
  
def main():
  example_grain = ExampleGrainOut(some_attribute3='hi')
  yaml_print(example_grain)


if __name__ == "__main__":
  main()