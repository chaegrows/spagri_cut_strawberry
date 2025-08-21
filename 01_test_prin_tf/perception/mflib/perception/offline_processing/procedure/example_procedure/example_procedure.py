from mflib.perception.offline_processing.procedure.procedure_base import ProcedureBase, ProcedureRegistry
from mflib.perception.offline_processing.grain import example_grain, example_grain_out

from typing import Tuple, ClassVar

@ProcedureRegistry.register
class ExampleProcedure(ProcedureBase):
  procedure_class_names: ClassVar[Tuple[str]] = ('deep_learning', 'data_preprocessing', 'example_procedure')
  from_grain_names_tuple: ClassVar[Tuple[Tuple[str]]] = (("crop_type", "strawberry", "fruit"),)
  output_grain_names: ClassVar[Tuple[str]] = ("crop_type", "strawberry", "fruit_images_preprocessed")

  def _run(self, *args, **kwargs) -> None:
    if len(args) != 1:
      raise ValueError("ExampleProcedure requires exactly one argument: an instance of ExampleGrain")
    example_grain_instance = args[0]
    if not isinstance(example_grain_instance, example_grain.ExampleGrain):
      raise TypeError("Expected an instance of ExampleGrain")

    attr3 = f'{example_grain_instance.some_attribute1}_{example_grain_instance.some_attribute2}'

    out = example_grain_out.ExampleGrainOut(
      some_attribute3=attr3
    )
    return out
  
  @classmethod
  def get_instance(self):
    """
    Get an instance of ExampleProcedure.
    """
    return ExampleProcedure()
  
def main():
  example_grain_instance = example_grain.ExampleGrain(
    some_attribute1='hello',
    some_attribute2=0
  )
  
  procedure = ExampleProcedure()
  output_grain = procedure._run(example_grain_instance)
  
  print(output_grain)

if __name__ == "__main__":
  main()