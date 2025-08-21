from typing import Dict, List, Optional, TypedDict, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field
from collections import Counter
import os

from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_json_string,
  yaml_instance_to_string,
  yaml_json_string_to_instance,
  yaml_string_to_instance,
  yaml_print
)



# Explicit reference to another action's output
class InputRef(BaseModel):
  repo: str = Field(description="Repository name", default='maybe_other_repo')
  node_name: str = Field(description="Node name", default='maybe_other_node')
  action_name: str = Field(description="Action name", default='an_action_name')
  output_name: str = Field(description="The name of the output required from the source action", default='frame_id')

  class Config:
    frozen = True

# LeafAction model definition
class LeafAction(BaseModel):
  action_name: str = Field(description="Action name", default='action_name')
  input_refs: List[InputRef] = Field(description="List of required input references", default=[InputRef()])
  description: Optional[str] = Field(description="Description of the action", default='')

  def shortly(self) -> str:
    # Return a short string representation of the action
    return f"{self.action_name}"
  
  def __hash__(self):
    return hash(self.action_name)

  class Config:
    frozen = True

class LeafActionList(BaseModel):
  repo: str = Field(description="Repository name", default='some_fancy_repo')
  node_name: str = Field(description="Node name", default='some_fancy_node')
  actions: List[LeafAction] = Field(description="List of LeafAction objects", default=[LeafAction()])

  
  class Config:
    frozen = True

  @staticmethod
  def create_by_filename(filename):
    yaml_path = os.path.join(filename)
    if not os.path.exists(yaml_path):
      raise FileNotFoundError(f"Leaf action list not found: {yaml_path}")
    return yaml_file_to_instance(yaml_path, LeafActionList)

# Class that stores and retrieves LeafAction objects
class LeafActionQueryParams(TypedDict, total=False):
  repo: str
  node_name: str
  action_name: str

class LeafActionLookup:
  def __init__(self):
    # Nested dictionary structure for organized storage
    self.store: Dict[str, Dict[str, Dict[str, LeafAction]]] = defaultdict(lambda: defaultdict(dict))
    self.node_name_to_actions_dict = {}

  def print_all_actions(self):
    for repo, nodes in self.store.items():
      print(f"Repository: {repo}") 
      for node_name, actions in nodes.items():
        print(f"  Node: {node_name}")
        for action_name, action in actions.items():
          print(f"    Action: {action_name}, Description: {action.description}")
    
  def add(self, repo, node_name, action: LeafAction):
    # Add a LeafAction to the registry
    self.store[repo][node_name][action.action_name] = action    
    
    # self.node_name_to_actions_dict
    if node_name not in self.node_name_to_actions_dict:
      self.node_name_to_actions_dict[node_name] = []
    self.node_name_to_actions_dict[node_name].append(action.action_name)
    
  def action_name_to_node_name(self, action_name: str) -> str:
    for node_name, actions in self.node_name_to_actions_dict.items():
      if action_name in actions:
        return node_name
    raise ValueError(f"Action name {action_name} not found in any node")

  def find(self, **kwargs) -> List[LeafAction]:
    query: LeafActionQueryParams = kwargs  # 타입 힌트로 강제
    result = []

    repo_q = query.get("repo")
    node_q = query.get("node_name")
    action_q = query.get("action_name")
    if action_q is None:
      raise ValueError("action_name is required")

    # Iterate through the registry to find matching actions
    for repo, nodes in self.store.items():
      if repo_q and repo != repo_q:
        continue
      for node_name, actions in nodes.items():
        if node_q and node_name != node_q:
          continue
        for action_name, leaf_action in actions.items():
          if action_name == action_q:
            result.append(leaf_action)

    return result
  
  @staticmethod
  def build_leaf_action_lookup_instance():
    BASE_PATH = '/opt/config/common/leaves'
    files = os.listdir(BASE_PATH)
    
    node_name_list = []
    action_name_list = []

    leaf_action_lookup = LeafActionLookup()
    
    for filename in files:
      if filename.endswith('.yaml'):
        filepath = os.path.join(BASE_PATH, filename)
        leaf_action_list = LeafActionList.create_by_filename(filepath)
        if leaf_action_list.repo == 'some_fancy_repo': # ignore example
          continue
        node_name = leaf_action_list.node_name
        if node_name is None:
          raise ValueError(f"[{filename}] does not have a valid node_name.")
        for leaf_action in leaf_action_list.actions:
          leaf_action_lookup.add(leaf_action_list.repo, node_name, leaf_action)

        # for validation
        node_name_list.append(node_name)
        action_name_list.extend([action.action_name for action in leaf_action_list.actions])
    
    # check if node names are unique
    counts = Counter(node_name_list)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    if duplicates:
      for name, count in duplicates.items():
        raise ValueError(f"  - node_name '{name}' appears {count} times in {BASE_PATH}")
    # check if action names are unique
    counts = Counter(action_name_list)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    if duplicates:
      for name, count in duplicates.items():
        raise ValueError(f"  - action_name '{name}' appears {count} times in {BASE_PATH}")
  
    return leaf_action_lookup



if __name__ == '__main__':
  # Example usage
  leaf_action_lookup = LeafActionLookup.build_leaf_action_lookup_instance()
  leaf_action_lookup.print_all_actions()
  actions = leaf_action_lookup.find(action_name='approach_to_fruit', repo='mf_manipulator')
  # actions = leaf_action_lookup.find(action_name='approach_to_fruit', repo='error_repo') # this will not return anything
  for action in actions:
    print(action)

  # leaf_action_list = LeafActionList()
  # example_file = open('./example_node.yaml', 'w')
  # flush_pydantic_yaml(leaf_action_list, example_file)
  # print_pydantic_yaml(leaf_action_list)
  # example_file.close()
