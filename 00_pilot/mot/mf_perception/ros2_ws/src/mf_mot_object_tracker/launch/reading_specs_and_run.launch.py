from launch import LaunchDescription
from launch_ros.actions import Node
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2

# class BTserver(BehaviorTreeServerNodeV2):
#   def __init__(self):
#     super().__init__('gonna_die_soon', [])
#     self.load_specs('metafarmers250402_harvesting.yaml')
#     self.perception_algorithm = self.get_job_spec().query_spec([])


def generate_launch_description():
  return LaunchDescription([
    Node(
        package='mf_mot_object_tracker',
        # executable='mot_by_bbox_server.py',
        # executable='mot_by_bbox.py',
        executable='mot_by_bbox_pre_a.py',
        output='screen',
        emulate_tty=True,
        arguments=['strawberry_harvesting_pollination']
    ),
    Node(
        package='mf_mot_object_tracker',
        executable='yolo.py',
        output='screen',
        emulate_tty=True,
        arguments=['strawberry_harvesting_pollination']
    ),
  ])