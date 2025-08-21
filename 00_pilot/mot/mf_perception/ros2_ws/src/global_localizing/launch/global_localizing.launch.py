from launch import LaunchDescription
from launch_ros.actions import Node

from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2

import rclpy

class SuddenServer(BehaviorTreeServerNodeV2):
  repo = 'mf_common'
  node_name = 'sudden_server'
  def __init__(self):
    super().__init__('standalone')
    self.mark_heartbeat(0)

def generate_launch_description():
  rclpy.init()

  server = SuddenServer()
  farm_spec = server.job_manager.specifications.farm_spec
  farm_name = farm_spec.name
  pcd_name = farm_spec.pcd_map_file
  PCD_FULL_PATH = f'/opt/config/common/specs/farm_configs/{farm_name}/{pcd_name}'

  print('publish pcd map:', PCD_FULL_PATH)

  pcl_ros = Node(
    package='pcl_ros',
    executable='pcd_to_pointcloud',
    name='pcd_to_pointcloud_node',
    output='screen',
    parameters=[
      {'file_name': PCD_FULL_PATH},
      {'tf_frame': 'farm_pcd'},
    ],
  )

  reloc = Node(
    package='global_localizing',
    executable='reloc_bt_server.py',
    output='screen',
    parameters=[
      {'pcd_map_path': PCD_FULL_PATH},
      {'init_xyz': [2.2, 1.5, 1.0]},
      {'init_yaw_deg': 0.0},
      {'lidar_topic': '/livox/lidar'},
      {'odom_topic': '/livox/imu'},
      {'do_reloc_pose_diff': 0.1},
      {'do_reloc_scan_accumulate': 20},
      {'min_dist_to_lidar': 0.1},
      {'max_dist_to_lidar': 30.0},
      {'voxel_size_map': 0.4},
      {'voxel_size_scan': 0.1},
      {'fov_degree': 300.0},
    ],
  )

  farm_configuration_viz = Node(
    package='global_localizing',
    executable='farm_configuration_visualizer.py',
    output='screen'
  )

  farm_pcd_tf = Node(
      package='tf2_ros',
      executable='static_transform_publisher',
      arguments=['0.0', '0.0', '0.1', '0.0', '0', '0.0', 'farm', 'farm_pcd'],
      output='screen'
  )

  glim = Node(
    package='glim_ros',
    executable='glim_rosnode',
    output='screen',
    parameters=[
      {'config_path': '/workspace/third_party/glim/glim_ws/src/glim/config'}
    ],
  )

  return LaunchDescription([
    pcl_ros,
    reloc,
    farm_configuration_viz,
    farm_pcd_tf,
    # glim,
  ])
