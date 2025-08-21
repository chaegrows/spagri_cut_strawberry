ROS_DOMAIN_ID=7
BAG_PATH=/home/tw/docker/shared_dir/mf_recon/data/bag/do_flower_far2_modified
docker run \
  -it \
  --rm \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e=DISPLAY \
  -e=ROS_DOMAIN_ID \
  -v /home/tw/docker/shared_dir/glim/config:/glim/config \
  -v $BAG_PATH:/bag \
  koide3/glim_ros2:humble_cuda12.2 \
  ros2 run glim_ros glim_rosbag /bag --ros-args -p config_path:=/glim/config
  #ros2 run glim_ros glim_rosnode --ros-args -p config_path:=/glim/config
