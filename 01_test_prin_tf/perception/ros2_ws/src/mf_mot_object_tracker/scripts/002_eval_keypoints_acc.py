#!/usr/bin/env python3
# filename: clicked_point_distance_to_fruit.py
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
import tf2_geometry_msgs  # Add this import


class ClickedPointToFruit(Node):
    """
    - /clicked_points (geometry_msgs/PointStamped) 구독
    - 포인트를 target_frame(=camera_hand_link)로 변환
    - fruit_frame(=fruit_)의 위치(원점)를 같은 frame으로 가져와 거리 계산
    - 결과를 log + /fruit_clicked_distance(Float32) 발행
    - 시각화를 위해 /fruit_clicked_markers(Marker) 발행 (line + points)
    """

    def __init__(self):
        super().__init__('clicked_point_to_fruit_distance')

        # 파라미터
        self.declare_parameter('clicked_topic', '/clicked_point')   # RViz 기본은 "/clicked_point"
        self.declare_parameter('target_frame', 'camera_hand_link')
        self.declare_parameter('fruit_frame',  'fruit_0_oriented')
        self.declare_parameter('tf_timeout_sec', 0.5)

        self.clicked_topic: str = self.get_parameter('clicked_topic').get_parameter_value().string_value
        self.target_frame:  str = self.get_parameter('target_frame').get_parameter_value().string_value
        self.fruit_frame:   str = self.get_parameter('fruit_frame').get_parameter_value().string_value
        self.tf_timeout     = self.get_parameter('tf_timeout_sec').get_parameter_value().double_value

        # TF 버퍼/리스너
        self.tf_buffer   = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 구독/발행
        self.sub = self.create_subscription(PointStamped, self.clicked_topic, self.on_clicked, 10)
        self.pub_dist = self.create_publisher(Float32, '/fruit_clicked_distance', 10)
        self.pub_marker = self.create_publisher(Marker, '/fruit_clicked_markers', 10)

        self.get_logger().info(
            f'Listening: {self.clicked_topic}\n'
            f'Comparing distance between CLICKED point and TF frame "{self.fruit_frame}"\n'
            f'All in target frame: "{self.target_frame}"'
        )

        # 마커 ID 관리
        self.marker_seq = 0

    def lookup_transform_safely(self, target: str, source: str) -> Optional[tf2_ros.TransformStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, Time(),
                timeout=Duration(seconds=self.tf_timeout)
            )
            return tf
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed ({target} <- {source}): {e}')
            return None

    def transform_point(self, point_stamped: PointStamped, to_frame: str) -> Optional[PointStamped]:
        try:
            # tf2_geometry_msgs를 사용한 수동 변환
            transform = self.tf_buffer.lookup_transform(
                to_frame, point_stamped.header.frame_id, Time(),
                timeout=Duration(seconds=self.tf_timeout)
            )
            
            # tf2_geometry_msgs를 사용하여 포인트 변환
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed_point
            
        except TransformException as e:
            self.get_logger().warn(f'Point transform failed to "{to_frame}": {e}')
            return None

    def on_clicked(self, msg: PointStamped):
        # 1) 클릭 포인트를 target_frame으로 변환
        pt_in_target = self.transform_point(msg, self.target_frame)
        if pt_in_target is None:
            return

        # 2) fruit_frame의 원점을 target_frame 기준으로 얻기 (translation)
        tf_fruit = self.lookup_transform_safely(self.target_frame, self.fruit_frame)
        if tf_fruit is None:
            return

        fx = tf_fruit.transform.translation.x
        fy = tf_fruit.transform.translation.y
        fz = tf_fruit.transform.translation.z

        px = pt_in_target.point.x
        py = pt_in_target.point.y
        pz = pt_in_target.point.z

        # 3) 거리 계산
        dist = math.sqrt((px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2)

        # 4) 로그 + 퍼블리시
        self.get_logger().info(
            f'[frame={self.target_frame}] | fruit_@({fx:.3f},{fy:.3f},{fz:.3f}) '
            f'<-> clicked@({px:.3f},{py:.3f},{pz:.3f})  => distance = {dist:.3f} m'
        )
        self.pub_dist.publish(Float32(data=float(dist)))

        # 5) RViz 마커(라인 + 포인트) 표시
        self.publish_markers(pt_in_target, fx, fy, fz)

    def publish_markers(self, pt_in_target: PointStamped, fx: float, fy: float, fz: float):
        # Line marker (fruit_ ↔ clicked)
        line = Marker()
        line.header.frame_id = self.target_frame
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'fruit_clicked_link'
        line.id = self.marker_seq
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.scale.x = 0.005  # line width (m)
        # 색상 (RGBA)
        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 1.0
        line.color.a = 1.0
        # 두 점 추가
        from geometry_msgs.msg import Point
        p1 = Point(x=fx, y=fy, z=fz)
        p2 = Point(x=pt_in_target.point.x, y=pt_in_target.point.y, z=pt_in_target.point.z)
        line.points = [p1, p2]
        self.pub_marker.publish(line)

        # Points marker (click, fruit positions)
        pts = Marker()
        pts.header.frame_id = self.target_frame
        pts.header.stamp = line.header.stamp
        pts.ns = 'fruit_clicked_points'
        pts.id = self.marker_seq
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.scale.x = 0.01  # point size
        pts.scale.y = 0.01
        # fruit_ = green, clicked = red
        # Marker.POINTS는 단일 색만 줄 수 있어 색 분리가 필요하면 별도 마커 2개로 나눠도 됨.
        pts.color.r = 0.2
        pts.color.g = 0.8
        pts.color.b = 0.2
        pts.color.a = 1.0
        pts.points = [p1, p2]
        self.pub_marker.publish(pts)

        self.marker_seq += 1


def main():
    rclpy.init()
    node = ClickedPointToFruit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
