import rclpy
from rclpy.node import Node
import numpy as np
import csv
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import signal
import sys
from scipy.spatial.transform import Rotation as R

def handle_sigint(signum, frame):
    print("got sigint")
    global node
    node.save_matrix_to_csv('odometry.csv')

    rclpy.shutdown()
    
def pose_to_T_mat4x4(pose):

    x, y, z = pose.position.x, pose.position.y, pose.position.z

    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w

    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [x, y, z]

    return T

class OdomListener(Node):
    def __init__(self):
        super().__init__('odom_listener')
        
        self.subscription = self.create_subscription(
            Odometry,
            '/mavros/odometry/out',
            self.odometry_callback,
            10)
        
        self.odom_data = []

    def odometry_callback(self, msg):
        try:
            pose: Pose = msg.pose.pose

            # Pose를 4x4 변환 행렬로 변환
            T = pose_to_T_mat4x4(pose)

            # 타임스탬프 저장 (초 단위)
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            if len(self.odom_data) == 0 or sec != self.odom_data[-1][0]:
                self.odom_data.append((sec, T))

        except Exception as e:
            self.get_logger().error(f'Could not process odometry: {e}')

    def save_matrix_to_csv(self, filename):
        """ 4x4 변환 행렬을 CSV 파일로 저장 """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sec, tf in self.odom_data:
                writer.writerow([f'{sec:09}'] + tf.flatten().tolist())

node = None

def main(args=None):
    signal.signal(signal.SIGINT, handle_sigint)
    rclpy.init(args=args)

    global node
    node = OdomListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.save_matrix_to_csv('odometry.csv')
    node.destroy_node()
    rclpy.shutdown()
    print('Odometry transform is successfully created')

if __name__ == '__main__':
    main()

