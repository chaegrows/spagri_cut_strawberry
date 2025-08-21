ROS_DOMAIN_ID=7

docker run \
  -it \
  --rm \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e=DISPLAY \
  -e=ROS_DOMAIN_ID \
  -v ./dump:/tmp/dump \
  -v ./glim_config_offline:/glim/config \
  koide3/glim_ros2:humble_cuda12.2 \
  ros2 run glim_ros offline_viewer
  #/bin/bash \
  # -v ./config.orig:/glim/config \
