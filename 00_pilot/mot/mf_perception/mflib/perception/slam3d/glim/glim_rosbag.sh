#!/bin/bash

if [ -z "$1" ]; then
  echo "usage: ./glim_rosbag.sh [path/to/rosbag2]"
  exit 1
fi

xhost +
ROS_DOMAIN_ID=7

docker run \
  -it --rm\
  --net=host \
  --ipc=host \
  --pid=host \
  --privileged --gpus all  \
  --hostname=$USER \
  --name glim \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw \
  -v /dev:/dev \
  -v /tmp:/tmp \
  -e ROS_DOMAIN_ID=7 \
  -v /dev:/dev \
  -v ./config_offline_mapping:/glim/config \
  -v ./dump:/tmp/dump \
  -v $1:/rosbag \
  koide3/glim_ros2:humble_cuda12.2 \
  ros2 run glim_ros glim_rosbag /rosbag --ros-args -p config_path:=/glim/config 
  # -v ./config.orig:/glim/config \
  #/bin/bash \
