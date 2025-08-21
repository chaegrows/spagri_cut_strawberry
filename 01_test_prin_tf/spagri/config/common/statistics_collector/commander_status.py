from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from typeguard import typechecked
from datetime import datetime, timezone
from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_json_string,
  yaml_instance_to_string,
  yaml_json_string_to_instance,
  yaml_string_to_instance,
  yaml_print
)

from config.common.specs.farm_def import Sector
from config.common.specs.work_def import ALLOWED_WORKS
import config.common.leaves.leaf_actions as la

class CommanderState(str, Enum):
  IDLE = "IDLE"
  EXECUTING = "EXECUTING"
  ERROR = "ERROR"

class CommanderStatusFrozen(BaseModel):
  is_on_boot_succeeded: bool = Field(default=False, description="Whether the boot sequence succeeded.")
  last_command: Optional[str] = Field(default=None, description="Last command that was executed.")
  state: CommanderState = Field(default=CommanderState.IDLE, description="Current operation state.")
  updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the status was last updated (UTC).")
  target_leaf_action: Optional[la.LeafAction] = Field(description="Target leaf action to be executed", default=None)
  target_sector:   Optional[Sector] = Field(description="Target sector to be worked on", default=None)
  target_work:     Optional[ALLOWED_WORKS] = Field(description="Target work to be done", default=None)
  class Config:
    frozen = True

class CommanderStatusWorking:
  def __init__(self):
    self.is_on_boot_succeeded: bool = False
    self.last_command: Optional[str] = None
    self.state: CommanderState = CommanderState.IDLE
    self.updated_at: datetime = datetime.now(timezone.utc)
    self.target_leaf_action: Optional[la.LeafAction] = None
    self.target_sector: Optional[Sector] = None
    self.target_work: Optional[ALLOWED_WORKS] = None

  @typechecked
  def set_boot_state(self, is_on_boot_succeeded: bool):
    self.is_on_boot_succeeded = is_on_boot_succeeded
    self._update_timestamp()

  @typechecked
  def set_last_command(self, command_name: str):
    self.last_command = command_name
    self._update_timestamp()

  @typechecked
  def set_state(self, new_state: CommanderState):
    self.state = new_state
    self._update_timestamp()

  def _update_timestamp(self):
    self.updated_at = datetime.now(timezone.utc)

  def to_frozen(self) -> CommanderStatusFrozen:
    return CommanderStatusFrozen(
      is_on_boot_succeeded=self.is_on_boot_succeeded,
      last_command=self.last_command,
      state=self.state,
      updated_at=self.updated_at,
      target_leaf_action=self.target_leaf_action,
      target_sector=self.target_sector,
      target_work=self.target_work
    )

  def reset(self):
    """Reset commander status to default."""
    self.is_on_boot_succeeded = False
    self.last_command = None
    self.state = CommanderState.IDLE
    self._update_timestamp()


if __name__ == "__main__":
  status = CommanderStatusWorking()

  # from btc
  status.set_boot_state(True)
  status.set_last_command("move_to_target")
  status.set_state(CommanderState.EXECUTING)

  frozen = status.to_frozen()
  print(yaml_instance_to_json_string(frozen))

  status.reset()

  frozen = status.to_frozen()
  print(yaml_instance_to_json_string(frozen))
