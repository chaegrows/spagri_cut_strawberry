#!/bin/bash

ARCH=$(uname -m)
VERSION=latest

CURRENT_DIR=$(basename $(pwd))
PROJECT_NAME=${CURRENT_DIR#mf_}  # mf_ 

CONTAINER_NAME=${1:-"${PROJECT_NAME}"}

echo "System architecture: $ARCH"
echo "Project name: $PROJECT_NAME"
echo "Container name: $CONTAINER_NAME"

# X11 authentication setup
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# GPU setup
if [ "$2" = "cpu" ]; then
    gpu=""
else
    if [ "$ARCH" = "aarch64" ]; then
        gpu="--runtime=nvidia"
    else
        gpu="--gpus all"
    fi
fi

# Get the absolute path of the current directory
PROJECT_PATH=$(pwd)
echo "PROJECT_PATH: $PROJECT_PATH"

COMMON_PATH=$(cd "$PROJECT_PATH/../common" && pwd)
if [ ! -d "$COMMON_PATH" ]; then
    echo "Error: COMMON_PATH '$COMMON_PATH' does not exist."
    echo "Please download the 'mf_common' repository and place it in the correct directory."
    echo "git clone -b dev git@github.com:Metafarmers/mf_common.git"
    exit 1
fi
echo "COMMON_PATH: $COMMON_PATH"

if [ "$ARCH" = "aarch64" ]; then
    echo "Detect : Jetson system"
    DOCKER_IMAGE="perception_localization:${VERSION}"
    EXTRA_ARGS="bash"
else
    echo "Detect : AMD/Intel system"
    # DOCKER_IMAGE="metafarmers/ubuntu2204_cuda12_ros2humble:${VERSION}"
    # DOCKER_IMAGE="metafarmers/ubuntu2204_cuda12_ros2humble:${VERSION}"
    DOCKER_IMAGE="mf_perception:250405"
    EXTRA_ARGS=""
fi

docker run -it --rm \
    --shm-size=10G \
    --ipc=host \
    --privileged $gpu \
    --hostname=$USER \
    --name mf_perception_localization \
    --network="host" \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw \
    -v /dev:/dev \
    -v /tmp:/tmp \
    -v ${COMMON_PATH}/ros2_ws/src/mf_common:/root/ros2_ws/src/mf_common\
    -v ${COMMON_PATH}/mflib/common:/opt/mflib/common \
    -v ${COMMON_PATH}/config/common:/opt/config/common \
    -v ${PROJECT_PATH}/config/${PROJECT_NAME}:/opt/config/${PROJECT_NAME} \
    -v ${PROJECT_PATH}/ros2_ws/build:/root/ros2_ws/build \
    -v ${PROJECT_PATH}/ros2_ws/log:/root/ros2_ws/log \
    -v ${PROJECT_PATH}/ros2_ws/install:/root/ros2_ws/install \
    -v ${PROJECT_PATH}/ros2_ws/src:/root/ros2_ws/src/mf_${PROJECT_NAME} \
    -v ${PROJECT_PATH}/mflib/${PROJECT_NAME}:/opt/mflib/${PROJECT_NAME} \
    -v ${PROJECT_PATH}/ai_models:/opt/ai_models \
    -v ${PROJECT_PATH}/third_party:/opt/third_party \
    -v ${PROJECT_PATH}/rviz:/opt/rviz \
    -v ${PROJECT_PATH}:/workspace \
    ${DOCKER_IMAGE}


# if you want to use rviz and pcd, you can use the following command
#       -v ${PROJECT_PATH}/pcd:/opt/pcd \
#    -v ${PROJECT_PATH}:/workspace


# if you want to using liDAR 
# source /workspace/third_party/livox_ws/install/setup.bash