#! /usr/bin/env python3
import rclpy
from scipy.spatial.transform import Rotation as R

from config.common.leaves.leaf_actions import LeafActionLookup
from mflib.common.behavior_tree_impl_v2 import BehaviorTreeServerNodeV2
from mflib.common.behavior_tree_context import BehaviorTreeContext

from visualization_msgs.msg import Marker, MarkerArray

from copy import copy

class FarmConfigurationVisualizer(BehaviorTreeServerNodeV2):
    repo = 'mf_perception'
    node_name = 'farm_configuration_visualizer'
  
    def __init__(self, run_mode):
      super().__init__(run_mode)
      self.mark_heartbeat(0)

      self.publisher = self.create_publisher(MarkerArray, '/farm_configuration_visualizer/configurations', 10)
      self.timer = self.create_timer(1, self.viz_timer)  

    def viz_timer(self):
      farm_spec = self.job_manager.specifications.farm_spec
      farm_configs = [sector.configuration for sector in farm_spec.sectors]


      marker_array = MarkerArray()

      for idx, (cx, cy, cz, w, l, h) in enumerate(farm_configs):
        marker = Marker()
        marker.header.frame_id = "farm"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "farm_cubes"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = cx + w / 2
        marker.pose.position.y = cy + l / 2
        marker.pose.position.z = cz + h / 2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = w
        marker.scale.y = l
        marker.scale.z = h

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3  

        marker_array.markers.append(marker)

      self.publisher.publish(marker_array)

if __name__=='__main__':
  rclpy.init()

  # run_mode = 'standalone'
  run_mode = 'server'
  server = FarmConfigurationVisualizer(run_mode)
  server.start_ros_thread(async_spin=False)
  rclpy.shutdown()