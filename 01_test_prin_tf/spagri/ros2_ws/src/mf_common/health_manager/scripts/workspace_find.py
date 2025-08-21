#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
꽃 TF(flower_*)를 (x,z) 복셀 히스토그램으로 분석해
가장 밀집된 복셀 중심(cx, cy_fixed, cz)을 RViz Marker(CUBE)와 Point로 발행.

ros2 run mf_tf_tools flower_voxelizer_xz
"""
import math, re, collections, numpy as np, rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker
import tf2_ros


# ────────────────────────────────────────────────────────────────
def voxel_key_xz(p, vx, vz):
    """X-Z 평면에서 복셀 인덱스 반환"""
    return (math.floor(p[0] / vx),   # ix
            math.floor(p[2] / vz))   # iz


# ────────────────────────────────────────────────────────────────
class FindLiftVoxelizerXZ(BehaviorTreeServerNodeV2):
    def __init__(self):
        super().__init__('find_lift_voxelizer_xz')

        self.declare_parameter('reference_frame', 'manipulator_base_link',
            ParameterDescriptor(description='기준 TF 프레임'))
        self.declare_parameter('voxel_size', [0.3, 0.2, 0.4],
            ParameterDescriptor(description='[lx, ly, lz] [m]'))
        self.declare_parameter('center_y', -0.4,
            ParameterDescriptor(description='고정 y 좌표 [m]'))

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub_center = self.create_publisher(
            PointStamped, 'voxel_center', 10)
        self.pub_marker = self.create_publisher(
            Marker, 'voxel_marker', 10)

        self.timer = self.create_timer(0.5, self.timer_cb)

    # ────────────────────────────────────────────────────────
    def timer_cb(self):
        ref_frame = self.get_parameter('reference_frame').value
        vx, vy, vz = self.get_parameter('voxel_size').value
        cy_fixed   = float(self.get_parameter('center_y').value)

        positions = []
        for line in self.tf_buffer.all_frames_as_string().split('\n'):
            m = re.match(r'^\s*Frame\s+(\w+)', line)
            if not m or not m.group(1).startswith('flower_'):
                continue
            child = m.group(1)
            try:
                t: TransformStamped = self.tf_buffer.lookup_transform(
                    ref_frame, child, rclpy.time.Time())
                tr = t.transform.translation
                positions.append([tr.x, tr.y, tr.z])
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass

        if not positions:
            self.get_logger().warn('flower_* TF를 찾지 못했습니다.')
            return

        P = np.asarray(positions)
        counts = collections.Counter(voxel_key_xz(p, vx, vz) for p in P)

        if not counts:
            self.get_logger().warn('Counter가 비었습니다.')
            return

        (ix_best, iz_best), best_cnt = counts.most_common(1)[0]

        cx = (ix_best + 0.5) * vx
        cz = (iz_best + 0.5) * vz
        cy = cy_fixed

        self.publish_center(cx, cy, cz, ref_frame, best_cnt)
        self.publish_marker(cx, cy, cz, vx, vy, vz, ref_frame)

    # ────────────────────────────────────────────────────────
    def publish_center(self, x, y, z, frame, n):
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = x, y, z
        self.pub_center.publish(msg)
        self.get_logger().info(
            f'[voxel-xz] center=({x:.3f},{y:.3f},{z:.3f}) m, count={n}')

    # ────────────────────────────────────────────────────────
    def publish_marker(self, x, y, z, lx, ly, lz, frame):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns  = 'flower_voxel'
        m.id  = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = lx, ly, lz
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.4, 1.0, 0.4
        m.lifetime.sec = 1
        self.pub_marker.publish(m)


# ────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = FlowerVoxelizerXZ()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
