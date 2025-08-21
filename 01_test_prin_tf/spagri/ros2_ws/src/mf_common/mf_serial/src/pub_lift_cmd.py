from mf_msgs.msg import LiftControl
import rclpy
import time
from std_msgs.msg import Float32



def pub_new(cmd_type, height_mm=None):
  rclpy.init()
  node = rclpy.create_node('pub_lift_cmd')
  pub = node.create_publisher(LiftControl, '/lift/in', 1)

  msg = LiftControl()
  msg.cmd_type = cmd_type
  if height_mm is not None:
    msg.height_mm = height_mm

  while rclpy.ok():
    pub.publish(msg) 
    rclpy.spin_once(node, timeout_sec=1)
    time.sleep(1)
    

if __name__ == '__main__':
  pub_new(cmd_type=LiftControl.CMD_MOVE_ABS, height_mm=0)
  # # pub_new(cmd_type=LiftControl.CMD_MOVE_REL, height_mm=200)
  # time.sleep(1)
  # pub_new(cmd_type=LiftControl.CMD_MOVE_HOME)
  # time.sleep(1)
  # pub_new(cmd_type=LiftControl.CMD_ESTOP)
  # pub_cmd_set_stop()