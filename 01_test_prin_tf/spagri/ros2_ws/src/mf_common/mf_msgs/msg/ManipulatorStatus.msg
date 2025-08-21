std_msgs/Header header

string base_frame # Manipulator vendor specific parameter for ROS2
string tool_frame # Manipulator vendor specific parameter for ROS2
string move_group # Manipulator vendor specific parameter for ROS2
string[] joint_names # Manipulator vendor specific parameter for ROS2

uint64 seq
uint32 current_status

bool is_left # Current status of the robot
bool is_flipped # Current status of the robot

float64[] current_joints # [rad]
geometry_msgs/Pose current_pose # [m, rad]


uint32 ROBOT_STATUS_IDLE=0
uint32 ROBOT_STATUS_RUNNING=1
uint32 ROBOT_STATUS_ESTOP=2
uint32 ROBOT_STATUS_NOT_AVAILABLE=3

