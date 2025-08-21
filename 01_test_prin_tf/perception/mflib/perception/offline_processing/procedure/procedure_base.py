from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type, ClassVar
from pydantic import BaseModel, Field, model_validator
import pkgutil
import importlib
from pathlib import Path


class ProcedureBase(BaseModel, ABC):
  procedure_class_names: ClassVar[Tuple[str]] = ('class1', 'class2')
  from_grain_names_tuple: ClassVar[Tuple[Tuple[str]]] = (('default1', 'default2'),)
  output_grain_names: ClassVar[Tuple[str]] = ('default11', 'default22')

  def run(self, *args, **kwargs) -> None:
    """
    Run the procedure.
    """
    self._run(*args, **kwargs)
    pass

  @abstractmethod
  def _run(self, *args, **kwargs) -> None:
    pass

  @classmethod
  @abstractmethod
  def get_instance(cls) -> 'ProcedureBase':
    pass


class ProcedureRegistry:
  _registry: Dict[str, Type[ProcedureBase]] = {}

  @classmethod
  def register(cls, procedure_cls: Type[ProcedureBase]):
    cls._registry[procedure_cls.__name__] = procedure_cls
    return procedure_cls

  @classmethod
  def all(cls) -> Dict[str, Type[ProcedureBase]]:
    return cls._registry
  
def import_all_procedures():
  import mflib.perception.offline_processing.procedure as procedure_pkg
  base_path = Path(procedure_pkg.__path__[0])
  
  for item in base_path.iterdir():
    if item.is_dir():
      py_files = list(item.glob("*.py"))
      if len(py_files) == 2:
        module_name = py_files[0].stem  # 'example_procedure.py' → 'example_procedure'
        if module_name == "__init__":
          continue
        full_import_path = f"{procedure_pkg.__name__}.{item.name}.{module_name}"
        print(f"Importing procedure module: {full_import_path}")
        importlib.import_module(full_import_path)
      else:
        print(f"Skipping directory {item.name} as it does not contain exactly two .py file.")

  # check for duplicate grain_class_names
  seen = {}
  for cls in ProcedureRegistry.all().values():
    if cls is ProcedureBase:
      continue  # Skip base class
    key = tuple(cls.procedure_class_names)
    if key in seen:
      raise ValueError(f"Duplicate procedure_class_names found:\n  {seen[key].__name__}\n  {cls.__name__}")
    seen[key] = cls

def find_procedure_by_class_names(name_tuple: Tuple[str]) -> Type[ProcedureBase]:
  """
  Find a procedure class by its class names.
  """
  for cls in ProcedureRegistry.all().values():
    if cls.procedure_class_names == name_tuple:
      return cls
  raise ValueError(f"Procedure with class names {name_tuple} not found.")