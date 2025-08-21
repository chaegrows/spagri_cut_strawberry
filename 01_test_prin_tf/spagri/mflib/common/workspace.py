import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, Point
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple, Union
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from rclpy.time import Time

class WorkspaceManager:
    def __init__(self, ros_node):
        self.ros_node = ros_node
        self.workspaces = []
        self.marker_pub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.ros_node)

        # Half-space 표현을 위한 데이터 구조
        self.cube_constraints = {
            'A': [],  # 부등식 계수 행렬 (Ax <= b)
            'b': [],  # 부등식 상수 벡터
            'orientations': [],  # 방향 벡터
            'thresholds': []  # 방향 허용 범위
        }
        self.sphere_constraints = {
            'centers': [],  # 구의 중심점
            'radii': [],  # 구의 반경
            'orientations': [],  # 방향 벡터
            'thresholds': []  # 방향 허용 범위
        }

    def clear_ws(self):
        self.workspaces = []

    def add_ws_cube(self,
                   ref_frame: str,
                   ws_center_xyz: List[float],
                   ws_size_wlh: List[float],
                   ws_orientation: Optional[List[float]] = None,
                   rot_threshold: float = np.pi/4) -> None:
        """
        큐브 형태의 작업 공간을 half-space 부등식으로 추가합니다.
        Ax <= b 형태의 선형 부등식으로 변환합니다.

        Args:
            ref_frame: 기준 좌표계
            ws_center_xyz: 큐브의 중심 좌표 [cx, cy, cz]
            ws_size_wlh: 큐브의 크기 [width, length, height]
            ws_orientation: 큐브의 방향 (오일러 각, [rx, ry, rz])
            rot_threshold: 방향 허용 임계값 (라디안)
        """
        workspace = {
            'type': 'cube',
            'ref_frame': ref_frame,
            'ws_center_xyz': np.array(ws_center_xyz),
            'ws_size_wlh': np.array(ws_size_wlh),
            'ws_orientation': np.array([0.0, 0.0, 0.0]) if ws_orientation is None else np.array(ws_orientation),
            'rot_threshold': rot_threshold
        }
        self.workspaces.append(workspace)

        # Half-space 부등식 계수 행렬 생성 (6면에 대한 부등식)
        # x >= x_min, x <= x_max, y >= y_min, y <= y_max, z >= z_min, z <= z_max
        A = np.array([
            [-1, 0, 0],  # x >= x_min
            [1, 0, 0],   # x <= x_max
            [0, -1, 0],  # y >= y_min
            [0, 1, 0],   # y <= y_max
            [0, 0, -1],  # z >= z_min
            [0, 0, 1]    # z <= z_max
        ])

        # 부등식 상수 벡터 생성
        half_size = np.array(ws_size_wlh) / 2
        center = np.array(ws_center_xyz)

        b = np.array([
            -(center[0] - half_size[0]),  # -x + x_min <= 0
            center[0] + half_size[0],     # x - x_max <= 0
            -(center[1] - half_size[1]),  # -y + y_min <= 0
            center[1] + half_size[1],     # y - y_max <= 0
            -(center[2] - half_size[2]),  # -z + z_min <= 0
            center[2] + half_size[2]      # z - z_max <= 0
        ])

        # 부등식 저장
        self.cube_constraints['A'].append(A)
        self.cube_constraints['b'].append(b)
        self.cube_constraints['orientations'].append(workspace['ws_orientation'])
        self.cube_constraints['thresholds'].append(rot_threshold)

    def add_ws_sphere(self,
                     ref_frame: str,
                     ws_center_xyz: List[float],
                     ws_radius: float,
                     ws_orientation: Optional[List[float]] = None,
                     rot_threshold: float = np.pi/4) -> None:
        """구형 작업 공간을 추가합니다.

        Args:
            ref_frame: 기준 좌표계
            ws_center_xyz: 구의 중심 좌표 [cx, cy, cz]
            ws_radius: 구의 반경
            ws_orientation: 구의 방향 (오일러 각, [rx, ry, rz])
            rot_threshold: 방향 허용 임계값 (라디안)
        """
        workspace = {
            'type': 'sphere',
            'ref_frame': ref_frame,
            'ws_center_xyz': np.array(ws_center_xyz),
            'ws_radius': ws_radius,
            'ws_orientation': np.array([0.0, 0.0, 0.0]) if ws_orientation is None else np.array(ws_orientation),
            'rot_threshold': rot_threshold
        }
        self.workspaces.append(workspace)

        # 구 제약조건 저장
        self.sphere_constraints['centers'].append(workspace['ws_center_xyz'])
        self.sphere_constraints['radii'].append(ws_radius)
        self.sphere_constraints['orientations'].append(workspace['ws_orientation'])
        self.sphere_constraints['thresholds'].append(rot_threshold)

    def _angle_between_quaternions(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """두 쿼터니언 사이의 각도를 계산합니다."""
        dot_product = np.dot(q1, q2)
        return 2 * np.arccos(min(abs(dot_product), 1.0))

    def _angle_between_quat_and_vector(self, quat: np.ndarray, vector: np.ndarray) -> float:
        """쿼터니언과 방향 벡터 사이의 각도를 계산합니다."""
        # 쿼터니언을 회전 행렬로 변환
        rot = Rotation.from_quat(quat)
        rot_matrix = rot.as_matrix()

        # 회전 행렬의 z축 방향 벡터 추출 (쿼터니언의 방향)
        quat_direction = rot_matrix[:, 2]

        # 두 벡터 사이의 각도 계산 (0 ~ pi 범위)
        dot_product = np.dot(quat_direction, vector)
        return np.arccos(min(max(dot_product, -1.0), 1.0))


    def check_vector_inside_ws(self,
                             pos_from_ref: List[float],
                             rot_from_ref: Optional[List[float]] = None) -> bool:
        """
        주어진 위치와 방향이 작업 공간 내에 있는지 확인합니다.
        Half-space 부등식을 사용하여 효율적으로 체크합니다.

        Args:
            pos_from_ref: 기준 프레임으로부터의 위치 [x, y, z]
            rot_from_ref: 기준 프레임으로부터의 회전 [rx, ry, rz] 또는 [qx, qy, qz, qw]

        Returns:
            bool: 작업 공간 내에 있으면 True, 아니면 False
        """
        pos = np.array(pos_from_ref)

        # 1. 큐브 workspace 체크 (Half-space 부등식 사용)
        if self.cube_constraints['A']:
            # 모든 큐브의 부등식을 한 번에 계산
            A_stack = np.vstack(self.cube_constraints['A'])  # (6N, 3)
            b_stack = np.hstack(self.cube_constraints['b'])  # (6N,)

            # Ax <= b 체크
            inequalities = A_stack @ pos <= b_stack  # (6N,)

            # 각 큐브별로 6개의 부등식이 모두 만족하는지 체크
            in_cubes = np.all(inequalities.reshape(-1, 6), axis=1)  # (N,)

            if rot_from_ref is not None and np.any(in_cubes):
                # 위치 체크를 통과한 workspace의 방향만 체크
                valid_orientations = np.array(self.cube_constraints['orientations'])[in_cubes]
                valid_thresholds = np.array(self.cube_constraints['thresholds'])[in_cubes]

                # 입력 회전을 방향 벡터로 변환
                if len(rot_from_ref) == 3:
                    input_rot = Rotation.from_euler('zyx', rot_from_ref)
                else:
                    input_rot = Rotation.from_quat(rot_from_ref)
                input_direction = input_rot.apply([0, 0, 1])

                # 모든 방향 체크를 한 번에 수행
                ws_rots = Rotation.from_euler('zyx', valid_orientations)
                ws_directions = ws_rots.apply([0, 0, 1])  # (M, 3)

                # 각도 계산
                dot_products = np.sum(ws_directions * input_direction, axis=1)  # (M,)
                angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # (M,)

                if np.any(angles <= valid_thresholds):
                    return True
            elif np.any(in_cubes):
                return True

        # 2. 구 workspace 체크
        if self.sphere_constraints['centers']:
            centers = np.array(self.sphere_constraints['centers'])  # (N, 3)
            radii = np.array(self.sphere_constraints['radii'])  # (N,)

            # 모든 구와의 거리를 한 번에 계산
            distances = np.linalg.norm(pos - centers, axis=1)  # (N,)
            in_spheres = distances <= radii  # (N,)

            if rot_from_ref is not None and np.any(in_spheres):
                # 위치 체크를 통과한 workspace의 방향만 체크
                valid_orientations = np.array(self.sphere_constraints['orientations'])[in_spheres]
                valid_thresholds = np.array(self.sphere_constraints['thresholds'])[in_spheres]

                # 입력 회전을 방향 벡터로 변환
                if len(rot_from_ref) == 3:
                    input_rot = Rotation.from_euler('zyx', rot_from_ref)
                else:
                    input_rot = Rotation.from_quat(rot_from_ref)
                input_direction = input_rot.apply([0, 0, 1])

                # 모든 방향 체크를 한 번에 수행
                ws_rots = Rotation.from_euler('zyx', valid_orientations)
                ws_directions = ws_rots.apply([0, 0, 1])  # (M, 3)

                # 각도 계산
                dot_products = np.sum(ws_directions * input_direction, axis=1)  # (M,)
                angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # (M,)

                if np.any(angles <= valid_thresholds):
                    return True
            elif np.any(in_spheres):
                return True

        return False


    def check_vector_outside_ws(self,
                               pos_from_ref: List[float],
                               rot_from_ref: Optional[List[float]] = None) -> bool:
        """
        주어진 위치와 방향이 작업 공간 외부에 있는지 확인합니다.
        """
        return not self.check_vector_inside_ws(pos_from_ref, rot_from_ref)


    def check_tf_inside_ws(self,
                         frame_name: str,
                         check_rot: bool = False) -> bool:
        """
        주어진 TF 프레임이 작업 공간 내에 있는지 확인합니다.

        Args:
            frame_name: 확인할 TF 프레임 이름
            check_rot: 방향도 검사할지 여부

        Returns:
            bool: 작업 공간 내에 있으면 True, 아니면 False
        """
        try:
          # TF 변환 가져오기
          transform = self.tf_buffer.lookup_transform(
              self.workspaces[0]['ref_frame'],
              frame_name,
              Time()
          )
          # 위치와 방향 추출
          trans = [
              transform.transform.translation.x,
              transform.transform.translation.y,
              transform.transform.translation.z
          ]

          if check_rot:
              rot = [
                  transform.transform.rotation.x,
                  transform.transform.rotation.y,
                  transform.transform.rotation.z,
                  transform.transform.rotation.w
              ]
              return self.check_vector_inside_ws(trans, rot)
          else:
              return self.check_vector_inside_ws(trans)
        except Exception as e:
          self.ros_node.get_logger().warn(f"TF 변환 실패: {str(e)}")
          return False




    def check_tf_outside_ws(self,
                           frame_name: str,
                           check_rot: bool = False) -> bool:
        """
        주어진 TF 프레임이 작업 공간 외부에 있는지 확인합니다.
        """
        return not self.check_tf_inside_ws(frame_name, check_rot)

    def publish_workspace_markers(self, topic_name: str, rgba_color: List[float] = [1.0, 1.0, 0.0, 0.2]) -> None:
        """
        작업 공간을 시각화하는 마커를 발행합니다.

        Args:
            topic_name: 마커를 발행할 토픽 이름
        """
        if self.marker_pub is None:
            self.marker_pub = self.ros_node.create_publisher(MarkerArray, topic_name, 1)

        m_array = MarkerArray()
        marker_id = 0

        for ws in self.workspaces:
            # 작업 공간 마커
            marker = Marker()
            marker.header.frame_id = ws['ref_frame']
            marker.header.stamp = self.ros_node.get_clock().now().to_msg()
            marker.ns = "workspace"
            marker.id = marker_id
            marker_id += 1

            if ws['type'] == 'cube':
                marker.type = Marker.CUBE
                marker.pose.position.x = ws['ws_center_xyz'][0]
                marker.pose.position.y = ws['ws_center_xyz'][1]
                marker.pose.position.z = ws['ws_center_xyz'][2]
                marker.scale.x = ws['ws_size_wlh'][0]
                marker.scale.y = ws['ws_size_wlh'][1]
                marker.scale.z = ws['ws_size_wlh'][2]

            elif ws['type'] == 'sphere':
                marker.type = Marker.SPHERE
                marker.pose.position.x = ws['ws_center_xyz'][0]
                marker.pose.position.y = ws['ws_center_xyz'][1]
                marker.pose.position.z = ws['ws_center_xyz'][2]
                marker.scale.x = ws['ws_radius'] * 2
                marker.scale.y = ws['ws_radius'] * 2
                marker.scale.z = ws['ws_radius'] * 2

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = rgba_color[0]
            marker.color.g = rgba_color[1]
            marker.color.b = rgba_color[2]
            marker.color.a = rgba_color[3]

            m_array.markers.append(marker)

            # Orientation 마커 추가
            orientation_marker = Marker()
            orientation_marker.header.frame_id = ws['ref_frame']
            orientation_marker.header.stamp = self.ros_node.get_clock().now().to_msg()
            orientation_marker.ns = "workspace_orientation"
            orientation_marker.id = marker_id
            marker_id += 1
            orientation_marker.type = Marker.ARROW
            orientation_marker.action = Marker.ADD

            # ws_orientation을 이용해 방향 벡터 계산
            rotation = Rotation.from_euler('zyx', ws['ws_orientation'])
            z_axis = np.array([0.0, 0.0, 1.0])
            rotated_vector = rotation.apply(z_axis)

            # 화살표 시작점과 끝점 설정
            start_point = ws['ws_center_xyz']
            end_point = start_point + rotated_vector * 0.05  # 0.5m 길이의 화살표

            orientation_marker.points = [
                Point(x=start_point[0], y=start_point[1], z=start_point[2]),
                Point(x=end_point[0], y=end_point[1], z=end_point[2])
            ]

            # 화살표 스타일 설정
            orientation_marker.scale.x = 0.03  # 화살표 몸통 두께
            orientation_marker.scale.y = 0.05  # 화살표 머리 두께
            orientation_marker.scale.z = 0.02

            # 노란색으로 표시
            orientation_marker.color.r = 1.0
            orientation_marker.color.g = 1.0
            orientation_marker.color.b = 0.0
            orientation_marker.color.a = 1.0

            m_array.markers.append(orientation_marker)

        self.marker_pub.publish(m_array)



# if __name__ == "__main__":
#     import rclpy
#     from rclpy.node import Node
#     import time
#     import numpy as np

#     class TestNode(Node):
#         def __init__(self):
#             super().__init__('test_workspace_node')

#             # WorkspaceManager 초기화
#             self.ws_manager = WorkspaceManager(self)

#             # Static TF Broadcaster 초기화
#             self.tf_broadcaster = StaticTransformBroadcaster(self)

#             # 테스트 TF 프레임 발행
#             self.publish_test_frames()

#             # 테스트용 작업 공간 추가
#             # 1. 큐브 작업 공간
#             self.ws_manager.add_ws_cube(
#                 ref_frame='base_link',
#                 ws_center_xyz=[0.0, 0.0, 0.0],  # 중심점 좌표
#                 ws_size_wlh=[1.0, 1.0, 1.0],    # 크기
#                 ws_orientation=[0.0, 0.0, 0.0],
#             )

#             # 2. 구형 작업 공간
#             self.ws_manager.add_ws_sphere(
#                 ref_frame='base_link',
#                 ws_center_xyz=[0.5, 0.5, 0.0],
#                 ws_radius=0.3,
#                 ws_orientation=[0.0, 0.0, 0.0],
#             )

#             # 마커 발행을 위한 타이머 생성
#             self.create_timer(1.0, self.publish_markers)

#             # 테스트 포인트들
#             self.test_points = [
#                 [0.0, 0.0, 0.0],    # 큐브 내부
#                 [0.5, 0.5, 0.0],    # 구 내부
#                 [1.0, 1.0, 1.0],    # 외부
#                 [-0.2, -0.2, 0.0],  # 큐브 내부
#                 [0.6, 0.6, 0.0]     # 구 외부
#             ]

#             # 테스트 회전값들 (오일러 각)
#             self.test_rotations = [
#                 [0.0, 0.0, 0.0],    # 기준 방향
#                 [0.0, 3.14, 0.0],    # y축으로 0.1 라디안 회전
#                 [0.0, 0.0, 3.14],    # y축으로 0.2 라디안 회전
#                 [3.14, 3.14, 0.0],    # z축으로 0.1 라디안 회전
#                 [1.57, 0.0, 0.0],     # x축으로 0.1 라디안 회전
#                 [np.pi/6, 0.0, 0.0],     # x축으로 0.1 라디안 회전
#                 [0.0, 0.0, -np.pi/6],     # x축으로 0.1 라디안 회전
#             ]


#             # 테스트 벡터 시각화를 위한 퍼블리셔 생성
#             self.vector_marker_pub = self.create_publisher(MarkerArray, 'test_vectors', 1)

#             # 테스트 실행
#             self.create_timer(2.0, self.run_tests)

#         def publish_test_frames(self):
#             """테스트용 TF 프레임들을 발행"""
#             test_frames = [
#                 # frame_id, x, y, z, roll, pitch, yaw
#                 ('test_frame1', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),      # 원점에 있는 프레임
#                 ('test_frame2', 0.3, 0.3, 0.0, 0.0, 0.0, np.pi/6),  # 45도 회전한 프레임
#                 ('test_frame3', -0.2, -0.2, 0.0, 0.0, np.pi/6, 0.0), # pitch가 30도인 프레임
#                 ('test_frame4', 0.8, 0.8, 0.0, 0.0, 0.0, 0.0),      # workspace 밖의 프레임
#             ]

#             transforms = []
#             for frame_id, x, y, z, roll, pitch, yaw in test_frames:
#                 t = TransformStamped()
#                 t.header.stamp = self.get_clock().now().to_msg()
#                 t.header.frame_id = 'base_link'
#                 t.child_frame_id = frame_id

#                 # 위치 설정
#                 t.transform.translation.x = x
#                 t.transform.translation.y = y
#                 t.transform.translation.z = z

#                 # 회전 설정
#                 rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
#                 quat = rotation.as_quat()
#                 t.transform.rotation.x = quat[0]
#                 t.transform.rotation.y = quat[1]
#                 t.transform.rotation.z = quat[2]
#                 t.transform.rotation.w = quat[3]

#                 transforms.append(t)

#             # TF 발행
#             self.tf_broadcaster.sendTransform(transforms)

#         def publish_markers(self):
#             """작업 공간 마커 발행"""
#             self.ws_manager.publish_workspace_markers('workspace_markers')
#             self.publish_test_vectors()

#         def publish_test_vectors(self):
#             """테스트 벡터들을 시각화하는 마커 발행"""
#             m_array = MarkerArray()
#             marker_id = 0

#             # 각 테스트 포인트에 대해
#             for point_idx, point in enumerate(self.test_points):
#                 # 각 회전값에 대해
#                 for rot_idx, rot in enumerate(self.test_rotations):
#                     # 회전 행렬 생성
#                     rotation = Rotation.from_euler('zyx', rot)
#                     # z축 방향 벡터를 회전
#                     z_axis = np.array([0.0, 0.0, 1.0])
#                     rotated_vector = rotation.apply(z_axis)

#                     # 화살표 마커 생성
#                     marker = Marker()
#                     marker.header.frame_id = 'base_link'
#                     marker.header.stamp = self.get_clock().now().to_msg()
#                     marker.ns = f"test_vector_{point_idx}_{rot_idx}"
#                     marker.id = marker_id
#                     marker_id += 1
#                     marker.type = Marker.ARROW
#                     marker.action = Marker.ADD

#                     # 시작점과 끝점 설정
#                     start_point = point
#                     end_point = np.array(point) + rotated_vector * 0.3  # 0.3m 길이의 화살표

#                     marker.points = [
#                         Point(x=start_point[0], y=start_point[1], z=start_point[2]),
#                         Point(x=end_point[0], y=end_point[1], z=end_point[2])
#                     ]

#                     # 화살표 스타일 설정
#                     marker.scale.x = 0.02  # 화살표 몸통 두께
#                     marker.scale.y = 0.04  # 화살표 머리 두께
#                     marker.scale.z = 0.0   # 사용하지 않음

#                     # workspace 내부 여부 확인
#                     is_inside = self.ws_manager.check_vector_inside_ws(point, rot)

#                     # 색상 설정 (내부면 초록색, 외부면 빨간색)
#                     marker.color.r = 0.0 if is_inside else 1.0
#                     marker.color.g = 1.0 if is_inside else 0.0
#                     marker.color.b = 0.0
#                     marker.color.a = 1.0

#                     m_array.markers.append(marker)

#             self.vector_marker_pub.publish(m_array)

#         def run_tests(self):
#             """작업 공간 테스트 실행"""
#             self.get_logger().info("=== Workspace 테스트 시작 ===")

#             # 위치 테스트
#             self.get_logger().info("\n1. 위치 테스트:")
#             for point in self.test_points:
#                 is_inside = self.ws_manager.check_vector_inside_ws(point)
#                 self.get_logger().info(f"포인트 {point}: 작업 공간 내부 {'O' if is_inside else 'X'}")

#                 is_outside = self.ws_manager.check_vector_outside_ws(point)
#                 self.get_logger().info(f"포인트 {point}: 작업 공간 외부 {'O' if is_outside else 'X'}")

#             # 위치 + 방향 테스트
#             self.get_logger().info("\n2. 위치 + 방향 테스트:")
#             for point in self.test_points:
#                 for rot in self.test_rotations:
#                     is_inside = self.ws_manager.check_vector_inside_ws(point, rot)
#                     self.get_logger().info(f"포인트 {point}, 회전 {rot}: 작업 공간 내부 {'O' if is_inside else 'X'}")

#                     is_outside = self.ws_manager.check_vector_outside_ws(point, rot)
#                     self.get_logger().info(f"포인트 {point}, 회전 {rot}: 작업 공간 외부 {'O' if is_outside else 'X'}")

#             # TF 테스트
#             self.get_logger().info("\n3. TF 테스트:")
#             test_frames = ['test_frame1', 'test_frame2', 'test_frame3', 'test_frame4']
#             for frame in test_frames:
#                 is_inside = self.ws_manager.check_tf_inside_ws(frame, check_rot=True)
#                 self.get_logger().info(f"프레임 {frame}: 작업 공간 내부 {'O' if is_inside else 'X'}")

#             self.get_logger().info("\n=== Workspace 테스트 완료 ===")

#     def main():
#         rclpy.init()
#         node = TestNode()
#         try:
#             rclpy.spin(node)
#         except KeyboardInterrupt:
#             pass
#         finally:
#             node.destroy_node()
#             rclpy.shutdown()

#     if __name__ == "__main__":
#         main()

