#!/bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

# If the jetson-containers has not been cloned, clone and install it
JETSON_CONTAINERS_DIR=$(realpath $SCRIPT_DIR/../../jetson-containers)
if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
    git clone https://github.com/dusty-nv/jetson-containers $JETSON_CONTAINERS_DIR
    cd $JETSON_CONTAINERS_DIR
    git checkout f38a23025619afd95ae79ca7c3e1c609964c3117
    bash $JETSON_CONTAINERS_DIR/install.sh
    cd $SCRIPT_DIR
fi

# Get the L4T version
L4T_VERSION=$($JETSON_CONTAINERS_DIR/jetson_containers/l4t_version.sh | awk '/L4T_VERSION:/ {print $2}')

DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE=dustynv/ros:humble-ros-base-l4t-r$L4T_VERSION \
    -t usb_cam -f Dockerfile.usb_cam .