#!/bin/bash

ARCH=$(uname -m)
VERSION=v1.0.0_isu_glim

echo "System architecture: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "Detect : Jetson system"
    sudo docker build -t metafarmers/jetson-jetpack6:${VERSION} -f docker/jetson-jetpack6/isu/Dockerfile .
fi
