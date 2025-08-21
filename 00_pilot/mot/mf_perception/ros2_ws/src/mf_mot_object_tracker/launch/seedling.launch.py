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
        executable='mot_seedling_arrow2d.py',
        output='screen',
        emulate_tty=True,
        arguments=['seedling_arrow2d'],
        # parameters=[{'use_sim_time': True}],
    ),
    Node(
        package='mf_mot_object_tracker',
        executable='yolo_seedling.py',
        output='screen',
        emulate_tty=True,
        arguments=['seedling_arrow2d']
    ),
  ])
