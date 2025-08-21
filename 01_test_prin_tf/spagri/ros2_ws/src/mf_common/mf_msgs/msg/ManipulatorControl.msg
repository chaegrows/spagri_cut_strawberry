std_msgs/Header header

uint64 seq
uint32 cmd_type

float64[] target_joints
geometry_msgs/Pose target_pose
geometry_msgs/Twist target_twist

string digital_io

float64 speed_factor  # 0.0 - 1.0 [%]

# [0~99] Basic manipulator motion commands
uint32 CMD_MOVEJ=0
uint32 CMD_MOVEL_FROM_BASE=1 # ref_frame = manipulator_base_link
uint32 CMD_MOVEL_FROM_EEF=2 # ref_frame = endeffector
uint32 CMD_MOVE_SERVO=3
uint32 CMD_MOVE_RECOVERY=4
uint32 CMD_SET_DIGITAL_IO=10

#################################################################################
uint32 CMD_MOVEJ_ID_START=1000
uint32 CMD_MOVEJ_ID_END=3999

# [1000~1999] MOVEJ to predefined configuration (left-side)
uint32 CMD_MOVEJ_LEFT_NORMAL_ZERO=1000
uint32 CMD_MOVEJ_LEFT_NORMAL_SAFE=1001
uint32 CMD_MOVEJ_LEFT_NORMAL_HOME=1002
uint32 CMD_MOVEJ_LEFT_NORMAL_FLIPPED_1=1003
uint32 CMD_MOVEJ_LEFT_NORMAL_FLIPPED_2=1004
uint32 CMD_MOVEJ_LEFT_NORMAL_FLIPPED_3=1005
uint32 CMD_MOVEJ_LEFT_FLIPPED_SAFE=1006
uint32 CMD_MOVEJ_LEFT_FLIPPED_HOME=1007

# [2000~2999] MOVEJ to predefined configuration (right-side)
uint32 CMD_MOVEJ_RIGHT_NORMAL_ZERO=2000
uint32 CMD_MOVEJ_RIGHT_NORMAL_SAFE=2001
uint32 CMD_MOVEJ_RIGHT_NORMAL_HOME=2002
uint32 CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_1=2003
uint32 CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_2=2004
uint32 CMD_MOVEJ_RIGHT_NORMAL_FLIPPED_3=2005
uint32 CMD_MOVEJ_RIGHT_FLIPPED_SAFE=2006
uint32 CMD_MOVEJ_RIGHT_FLIPPED_HOME=2007

# [3000~3999] others
uint32 CMD_GRIPPER=3000
uint32 CMD_MOVEJ_COLLECT_1=3001
uint32 CMD_MOVEJ_COLLECT_2=3002
uint32 CMD_MOVEJ_COLLECT_3=3003
uint32 CMD_MOVEJ_COLLECT_4=3004
uint32 CMD_MOVEJ_COLLECT_5=3005
uint32 CMD_MOVEJ_COLLECT_6=3006
uint32 CMD_MOVEJ_COLLECT_7=3007
uint32 CMD_MOVEJ_TEST=3008

#################################################################################
