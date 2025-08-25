#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs

class ClickedPointDistance(Node):
    """
    Subscribe to clicked points and calculate distance between last two points
    """

    def __init__(self):
        super().__init__('clicked_point_distance')

        # Parameters
        self.declare_parameter('clicked_topic', '/clicked_point')
        self.declare_parameter('target_frame', 'camera_hand_link') 
        self.declare_parameter('tf_timeout_sec', 0.5)

        self.clicked_topic = self.get_parameter('clicked_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.tf_timeout = self.get_parameter('tf_timeout_sec').get_parameter_value().double_value

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to clicked points
        self.sub = self.create_subscription(PointStamped, self.clicked_topic, self.on_clicked, 10)
        
        # Store last two clicked points
        self.last_points = []

        self.get_logger().info(f'Listening to clicked points on: {self.clicked_topic}')

    def transform_point(self, point_stamped: PointStamped, to_frame: str) -> Optional[PointStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                point_stamped.header.frame_id,
                Time(),
                timeout=Duration(seconds=self.tf_timeout)
            )
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed_point
        except TransformException as e:
            self.get_logger().warn(f'Point transform failed to "{to_frame}": {e}')
            return None

    def on_clicked(self, msg: PointStamped):
        # Transform point to target frame
        pt_in_target = self.transform_point(msg, self.target_frame)
        if pt_in_target is None:
            return

        # Add point to list
        self.last_points.append(pt_in_target)
        
        # Keep only last 2 points
        if len(self.last_points) > 2:
            self.last_points.pop(0)

        # Calculate distance if we have 2 points
        if len(self.last_points) == 2:
            p1 = self.last_points[0].point
            p2 = self.last_points[1].point
            
            dist = math.sqrt(
                (p2.x - p1.x) ** 2 + 
                (p2.y - p1.y) ** 2 + 
                (p2.z - p1.z) ** 2
            )
            
            self.get_logger().info(
                f'Distance between points:\n'
                f'Point 1: ({p1.x:.3f}, {p1.y:.3f}, {p1.z:.3f})\n'
                f'Point 2: ({p2.x:.3f}, {p2.y:.3f}, {p2.z:.3f})\n'
                f'Distance: {dist:.3f} meters'
            )

def main():
    rclpy.init()
    node = ClickedPointDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
