from config.common.statistics_collector.isu2025 import Isu2025
from mflib.common.mf_base import (
  yaml_file_to_instance,
  yaml_instance_to_file,
  yaml_instance_to_string,
  yaml_string_to_instance,
  yaml_instance_to_json_string,
  yaml_json_string_to_instance,
  yaml_print,
)

from config.common.statistics_collector.commander_status import CommanderStatusFrozen, CommanderStatusWorking
from config.common.statistics_collector.isu2025 import PollinationStatisticsManager, create_isu2025

commander_status = CommanderStatusWorking()
commander_status.is_on_boot_succeeded = frozen_status.is_on_boot_succeeded
commander_status.last_command = frozen_status.last_command
commander_status.state = frozen_status.state



status = yaml_json_string_to_instance(, Isu2025)
