from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import copy

from config.common.statistics_collector.commander_status import CommanderStatusFrozen

# --- Pollination Statistics ---

class PollinationStatisticsWorking:
  def __init__(self):
    self.n_flowers_pollinated: int = 0
    self.n_flowers_skipped: int = 0
    self.start_time: datetime = datetime.now(timezone.utc)
    self.end_time: datetime = datetime.now(timezone.utc)

  def increment_pollinated(self):
    self.n_flowers_pollinated += 1

  def increment_skipped(self):
    self.n_flowers_skipped += 1

  def mark_start(self):
    self.start_time = datetime.now(timezone.utc)

  def mark_end(self):
    self.end_time = datetime.now(timezone.utc)

  def reset(self):
    """Reset all pollination statistics."""
    self.n_flowers_pollinated = 0
    self.n_flowers_skipped = 0
    self.start_time = datetime.now(timezone.utc)
    self.end_time = None

  def to_frozen(self) -> 'PollinationStatisticsFrozen':
    if self.end_time is None:
      raise ValueError("End time not set.")

    total_attempts = self.n_flowers_pollinated + self.n_flowers_skipped
    success_rate = (self.n_flowers_pollinated / total_attempts) * 100 if total_attempts > 0 else 0.0
    total_work_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
    if total_work_time is not None and total_work_time < 0: total_work_time = 0.0
    avg_time_per_attempt = total_work_time / total_attempts if total_attempts > 0 else -1
    if avg_time_per_attempt is not None and avg_time_per_attempt < 0: avg_time_per_attempt = 0.0
    efficiency = self.n_flowers_pollinated / (total_work_time / 60) if total_work_time > 0 else -1

    return PollinationStatisticsFrozen(
      n_flowers_pollinated=self.n_flowers_pollinated,
      n_flowers_skipped=self.n_flowers_skipped,
      success_rate=success_rate,
      start_time=self.start_time,
      end_time=self.end_time,
      total_work_time=total_work_time,
      avg_time_per_attempt=avg_time_per_attempt,
      efficiency=efficiency
    )

class PollinationStatisticsFrozen(BaseModel):
  n_flowers_pollinated: int
  n_flowers_skipped: int
  success_rate: float
  start_time: datetime
  end_time: datetime
  total_work_time: float
  avg_time_per_attempt: Optional[float]
  efficiency: Optional[float]

  class Config:
    frozen = True

class PollinationStatisticsManager:
  def __init__(self):
    self.working = PollinationStatisticsWorking()

  def increment_pollinated(self):
    self.working.increment_pollinated()

  def increment_skipped(self):
    self.working.increment_skipped()

  def mark_start(self):
    self.working.mark_start()

  def mark_end(self):
    self.working.mark_end()

  def reset(self):
    self.working.reset()

  def export_frozen(self) -> PollinationStatisticsFrozen:
    return self.working.to_frozen()


class Isu2025(BaseModel):
  commander_status: CommanderStatusFrozen = Field(..., description="Commander current status.")
  pollination_statistics: PollinationStatisticsFrozen = Field(..., description="Pollination statistics result.")

  class Config:
    frozen = True

def create_isu2025(
    commander_status: CommanderStatusFrozen,
    pollination_statistics: PollinationStatisticsFrozen
    ) -> Isu2025:
  ret = Isu2025(
    commander_status=commander_status,
    pollination_statistics=pollination_statistics
  )
  return ret