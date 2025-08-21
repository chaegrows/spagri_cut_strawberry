from pydantic import BaseModel, Field
from typing import List

class RosbagRecorderConfig(BaseModel):
  topics: List[str] = Field(description="List of topic configurations")
  queue_len: int = Field(description="max queue length of subscribed topics")
  force_1to1_connection: bool = Field(description="force 1to1 pub and sub connection")
  class Config:
    frozen = True