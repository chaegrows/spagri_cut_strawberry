from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
from pydantic import BaseModel, Field
from typeguard import typechecked


import config.common.leaves.leaf_actions as la
from typing import Optional, Dict, Mapping
import json

class BehaviorTreeContext(BaseModel):
  target_leaf_action: Optional[la.LeafAction] = Field(description="Target leaf action to be executed", default=None)
  target_sector:   Optional[Sector] = Field(description="Target sector to be worked on", default=None)
  target_work:     Optional[ALLOWED_WORKS] = Field(description="Target work to be done", default=None)
  memory: Dict = Field(default_factory=dict)

  @typechecked
  def set_sector(self, sector: Sector):
    self.target_sector = sector

  @typechecked
  def set_work(self, work: ALLOWED_WORKS):
    self.target_work = work

  @typechecked
  def set_target_leaf_action(self, leaf_action: la.LeafAction):
    if not isinstance(leaf_action, la.LeafAction):
      raise ValueError("leaf_action must be an instance of LeafAction")
    self.target_leaf_action = leaf_action

  # memory related
  def query_input_of_target_work(self):
    # need to search (leaf action, key)
    leaf_action_lookup_table = la.LeafActionLookup.build_leaf_action_lookup_instance()
    dict_to_return = {}
    for input_ref in self.target_leaf_action.input_refs:
      repo = input_ref.repo
      node_name = input_ref.node_name
      action_name = input_ref.action_name
      target_key = input_ref.output_name
      
      leaf_action_list = leaf_action_lookup_table.find(repo=repo, node_name=node_name, action_name=action_name) # must be list of length 0
      if len(leaf_action_list) == 0:
        raise ValueError(f"Leaf action not found for {repo}, {node_name}, {action_name} in the lookup table")
      leaf_action = leaf_action_list[0]

      if leaf_action.shortly() not in self.memory:
        raise ValueError(f"Leaf action {leaf_action} not found in memory")
      if target_key not in self.memory[leaf_action.shortly()]:
        raise ValueError(f"Key {target_key} not found in memory for leaf action {leaf_action}")

      value = self.memory[leaf_action.shortly()][target_key]
      dict_to_return[target_key] = value
    
    return dict_to_return

  def set_memory_dict(self, leaf_action, key, value):
    try: hash(key)
    except TypeError: raise ValueError(f"Key {key} is not hashable")
    try: hash(value)
    except TypeError: raise ValueError(f"Value {value} is not hashable")
    if leaf_action.shortly() not in self.memory:
      self.memory[leaf_action.shortly()] = {}
    self.memory[leaf_action.shortly()][key] = value
  
  class Config:
    frozen = False
  
  
  def print_all(self, logger):
    logger.info(self.to_string())