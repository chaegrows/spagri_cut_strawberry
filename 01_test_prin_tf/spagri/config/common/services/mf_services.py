from typing import Dict, List, Optional, TypedDict, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
import sys
import os



# MfService model definition
class MfService(BaseModel):
  service_name: str = Field(description="Service name", default='default_service_name')
  input_keys: List[str] = Field(description="List of required input keys for the service", default=['test_key'])
  output_keys: List[str] = Field(description="List of required output keys for the service", default=['test_output'])
  description: Optional[str] = Field(description="Description of the service", default='default_description')


class MfServiceList(BaseModel):
  node_name: str = Field(description="Node name", default='default_node_name')
  services: List[MfService] = Field(description="List of services", default=[MfService()])

  @staticmethod
  def create_by_filename(filename):
    yaml_path = os.path.join(filename)
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Farm spec not found: {yaml_path}")
    with open(yaml_path, "r") as f:
      return MfServiceList.model_validate(YAML().load(f))


# Class that stores and retrieves MfService objects
class LeafServiceLookup:
  def __init__(self):
    # Nested dictionary structure for organized storage
    self.store: Dict[str, MfService] = {}

  @classmethod
  def build_mf_service_lookup_instance(cls):
    """Create and initialize a LeafServiceLookup instance with all available services"""
    instance = cls()
    instance.load_all_mf_services()
    return instance

  def find(self, service_name) -> MfService:
    return self.store.get(service_name)


  def load_all_mf_services(self):
    BASE_PATH = '/opt/config/common/services'
    files = os.listdir(BASE_PATH)
    all_services = []

    for filename in files:
      if filename.endswith('.yaml'):
        mfs_in_node = MfServiceList.create_by_filename(os.path.join(BASE_PATH, filename))
        for service in mfs_in_node.services:
          if service.service_name in all_services:
            raise ValueError(f"Service name '{service.service_name}' is duplicated in the {filename}")
          all_services.append(service.service_name)
          self.store[service.service_name] = service


if __name__ == '__main__':
  # Example usage
  # leaf_service_lookup = LeafServiceLookup.build_leaf_service_lookup_instance()
  # services = leaf_service_lookup.find(service_name='get_manipulator_state', repo='mf_manipulator')
  # # services = leaf_service_lookup.find(service_name='get_manipulator_state', repo='error_repo') # this will not return anything
  # for service in services:
  #   print(service)


  mf_service_list = MfServiceList(
    node_name='default_node_name',
    services=[MfService(
      service_name='get_server_info',
      input_keys=[],
      output_keys=['server_name', 'repo', 'node_name', 'uptime', 'version'],
      description='default_description'
    )]
  )

  yaml = YAML()
  yaml.default_flow_style = False
  yaml.dump(mf_service_list.model_dump(), sys.stdout)
