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
  -v ./config.orig:/glim/config \
  -v ./dump:/tmp/dump \
  -v $1:/rosbag \
  -v ./check_topic_frequency.py:/glim/check_topic_frequency.sh \
  koide3/glim_ros2:humble_cuda12.2 \
  python3 /glim/check_topic_frequency.sh /rosbag
