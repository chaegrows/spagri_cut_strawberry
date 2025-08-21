import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
import numpy as np
import csv
from geometry_msgs.msg import TransformStamped

import signal
import rclpy
import mflib.brain.mot.utils.utils as utils

def handle_sigint(signum, frame):
    print("got sigint")
    global node
    node.save_matrix_to_csv('transform.csv')

    rclpy.shutdown()



class TFListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(1./1000., self.timer_callback)
        self.tfs = []

    def timer_callback(self):
        try:
            # map -> lidar 변환을 조회
            trans = self.tf_buffer.lookup_transform('map', 'femto_bolt', rclpy.time.Time())
            
            # 변환을 4x4 행렬로 변환
            if len(self.tfs) == 0 or trans.header.stamp != self.tfs[-1][0]:
              T = utils.tf_msg_to_T_mat4x4(trans)
              self.tfs.append((trans.header.stamp, T))

            # 행렬을 CSV 파일로 저장
            # self.save_matrix_to_csv(transform_matrix, 'transform.csv')
        
        except Exception as e:
            self.get_logger().error(f'Could not get transform: {e}')


    def save_matrix_to_csv(self, filename):
        """ 4x4 행렬을 CSV 파일로 저장 """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for ts, tf in self.tfs:
                sec = ts.sec + ts.nanosec/1e9
                writer.writerow([f'{sec:09}'] + tf.flatten().tolist())

node = None

def main(args=None):
    signal.signal(signal.SIGINT, handle_sigint)
    rclpy.init(args=args)

    global node
    node = TFListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.save_matrix_to_csv('transform.csv')
    node.destroy_node()
    rclpy.shutdown()
    print('transform is succesfully created')


if __name__ == '__main__':
    main()
