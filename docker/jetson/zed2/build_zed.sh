#!/bin/sh

# Script to build docker image for the ZED ROS2 wrapper, since the official build was broken at the time of writing.

# Could be that the official build at https://github.com/stereolabs/zed-ros2-wrapper/tree/master/docker
# is working again, so might be worth checking that out first.

# Clone https://github.com/stereolabs/zed-ros2-wrapper somewhere and provide the path to the directory here
ZED_ROS2_WRAPPER_DIR=''

DOCKER_BUILDKIT=1 docker build \
    --build-arg IMAGE_NAME=dustynv/ros:humble-ros-base-l4t-r35.4.1 \
    --build-arg JETPACK_MAJOR=5 \
    --build-arg JETPACK_MINOR=1 \
    --build-arg L4T_MAJOR=35 \
    --build-arg L4T_MINOR=4 \
    -t zed -f Dockerfile.zed $ZED_ROS2_WRAPPER_DIR/docker