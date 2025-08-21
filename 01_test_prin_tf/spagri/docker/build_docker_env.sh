#!/bin/bash

ARCH=$(uname -m)
VERSION=v1.0.0

echo "System architecture: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "Detect : Jetson system"
    docker build -t metafarmers/jetson-jetpack6:${VERSION} -f ./docker/jetson-jetpack6/Dockerfile .
elif [ "$ARCH" = "x86_64" ]; then
    echo "Detect : AMD/Intel system"
    docker build -t metafarmers/ubuntu2204_cuda12_ros2humble:${VERSION} -f ./docker/ubuntu2204_cuda12_ros2humble/Dockerfile .
else
    echo "Unsupported architecture: $ARCH"

fi
