lidar_topic = '/livox/lidar'
# odom_topic = '/mavros/odometry/out'
odom_topic = '/livox/imu'
do_reloc_pose_diff = 0.1 # meter
do_reloc_scan_accumulate = 20
pcd_map_path = "/workspace/pcd/mf250401.pcd"
# pcd_map_path = "/workspace/pcd/mf250406_dongtan.pcd"
# pcd_map_path = "/workspace/pcd/mf250401.pcd"
# @TODO : 동적으로 받아서 map 쓰기 

# @TODO : 대충 rviz 에서 보고 넣어라 
# @TODO , future future : 아무대나 둬도 될 수 있도록 
init_xyz = (3.37525, 4.71832, 0)
init_yaw_deg = -0.1

#
min_dist_to_lidar = 0.1 # specification 상 0.3 부터 잘 된다고.. 원래
max_dist_to_lidar = 30


# registration params
voxel_size_map = 0.4 
voxel_size_scan = 0.1
fov_degree = 300 # front is 0 degree. FOV spread from -fov_degree/2 to fov_degree/2


#pose: Frame:map, Position(3.37525, 4.71832, 0), Orientation(0, 0, -0.0294966, 0.999565) = Angle: -0.0590018
#3.6201; 4.5221; -1.3123