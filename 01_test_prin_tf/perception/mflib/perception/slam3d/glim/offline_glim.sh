ROS_DOMAIN_ID=21

docker run \
  -it \
  --rm \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e=DISPLAY \
  -e=ROS_DOMAIN_ID \
  -v ./glim_config_offline:/glim/config \
  -v $PATH_BAG:/bag \
  -v ./dump:/tmp/dump \
  koide3/glim_ros2:humble_cuda12.2 \
  ros2 run glim_ros offline_viewer
  #/bin/bash \
