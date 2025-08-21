from mf_msgs.msg import MD200Control
import rclpy

if __name__ == '__main__':
  rclpy.init()
  node = rclpy.create_node('pub_md200_cmd')
  pub = node.create_publisher(MD200Control, '/md200/in', 1)

  msg = MD200Control()
  msg.enable_estop = True
  msg.do_free_wheel = False
  msg.twist.linear.x = 0.00
  msg.twist.angular.z = 0.00

  while rclpy.ok():
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=1)
