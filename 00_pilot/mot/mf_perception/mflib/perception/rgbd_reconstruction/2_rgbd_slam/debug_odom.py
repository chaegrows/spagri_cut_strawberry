import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
from visualization_msgs.msg import Marker, MarkerArray

tf_csv = '/shared_dir/mf_recon/data/step1_out/do_flower_far2_modified/tf_synced.txt'

class TransformPublisher(Node):
    def __init__(self, csv_file):
        super().__init__('transform_publisher')

        self.marker_pub = self.create_publisher(MarkerArray, 'marker', 10)

        # TransformBroadcaster 객체 생성
        self.br = TransformBroadcaster(self)

        # CSV 파일에서 변환 행렬을 파싱
        if csv_file.endswith('.txt'):
          self.transforms = self.parse_csv(csv_file)
        elif csv_file.endswith('.log'):
          self.transforms = self.parse_csv2(csv_file)
        else:
            raise NotImplementedError(f'Unsupported file format: {csv_file}')

        # 타이머로 주기적으로 tf 발행
        self.current_transform_index = 0
        self.timer = self.create_timer(0.02, self.broadcast_transform)


    def parse_csv(self, csv_file):
        """CSV 파일을 파싱하여 변환 행렬 리스트를 반환"""
        transforms = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # 16개의 요소로 구성된 행렬 파싱
                matrix = np.array(row, dtype=float)[1:].reshape(4, 4)
                transforms.append(matrix)
        
        
        # first as identify
        # init_tf = transforms[0]
        # for i in range(len(transforms)):
        #     transforms[i] = np.linalg.inv(init_tf) @ transforms[i]

        # to camera axis
        # to_camera = np.array([[0, 1, 0], 
        #                       [0, 0, -1], 
        #                       [-1,0, 0]])
        # for i in range(len(transforms)):
        #     transforms[i][:3, :3] = to_camera @ transforms[i][:3, :3]

        # to_optical_axis = np.array([[0, 0, 1, 0], 
        #                             [1, 0, 0, 0], 
        #                             [0, 1, 0, 0],
        #                             [0, 0, 0, 1]])

        # to_optical_axis = np.array([[0, 0, 1, 0], 
        #                             [1, 0, 0, 0], 
        #                             [0, 1, 0, 0],
        #                             [0, 0, 0, 1]])
        # to_R = np.array([[[0, 0, -1],
        #                   [0, -1, 0],
        #                   [-1, 0, 0]
        #                   ]])
        # for i in range(len(transforms)):
        #     transforms[i] = to_optical_axis @ transforms[i]
        #     # transforms[i][:3, :3] =  transforms[i][:3, :3] @ to_R
        return transforms
  

    def parse_csv2(self, csv_file):
        """CSV 파일을 파싱하여 변환 행렬 리스트를 반환"""
        transforms = []
        i = 0
        one_tf = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                i += 1
                if i == 1:
                    continue
                floats = [float(f) for f in row[0].split(' ')]
                one_tf += floats
                if i % 5 == 0:
                    tf = np.array(one_tf).reshape(4, 4)
                    # transforms.append(np.linalg.inv(tf))
                    transforms.append(tf)
                    one_tf = []
                    i = 0
                
        return transforms

    def broadcast_transform(self):
        


        # 현재 발행할 변환 행렬 가져오기
        matrix = self.transforms[self.current_transform_index]

        # 평행 이동 (translation) 추출
        translation = matrix[:3, 3]

        # 회전 (rotation) 추출
        rotation_matrix = matrix[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w] 형식의 쿼터니언

        # TransformStamped 메시지 생성
        t = TransformStamped()

        # tf 프레임 정보 설정
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'  # 부모 프레임 이름 설정
        t.child_frame_id = f'femto_bolt'  # 자식 프레임 이름

        # 변환 설정 (translation)
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        # 변환 설정 (rotation)
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        # tf 발행
        self.br.sendTransform(t)

        # 다음 변환 행렬로 이동 (순환)
        self.current_transform_index = (self.current_transform_index + 1) % len(self.transforms)

        # pub marker array to visualize self.current_transform_index
        m_array = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.text = str(self.current_transform_index)
        m.pose.position.x = translation[0]
        m.pose.position.y = translation[1]
        m.pose.position.z = translation[2]
        m.pose.orientation.x = float(0)
        m.pose.orientation.y = float(0)
        m.pose.orientation.z = float(0)
        m.pose.orientation.w = float(1)
        m.scale.x = float(1)
        m.scale.y = float(1)
        m.scale.z = float(1)
        m.color.a = float(1)
        m.color.r = float(0)
        m.color.g = float(1)
        m.color.b = float(0)
        m_array.markers.append(m)
        # self.marker_pub.publish(m_array)


def main(args=None):
    rclpy.init(args=args)

    node = TransformPublisher(tf_csv)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
