#!/bin/sh

IMAGE_NAME="faint"

# Dependency versions
ROS_DISTRO="humble-ros-base"
PYTORCH_VERSION="2.2"
TORCHVISION_VERSION="0.17.2"
OPENCV_VERSION="4.8.1"

while getopts r:b:t: flag
do
    case "${flag}" in
        t) IMAGE_NAME=${OPTARG};;
        *) error "Unexpected option ${flag}" ;;
    esac
done

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

# If the jetson-containers has not been cloned, clone and install it
JETSON_CONTAINERS_DIR=$(realpath $SCRIPT_DIR/../jetson-containers)
if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
    git clone https://github.com/dusty-nv/jetson-containers $JETSON_CONTAINERS_DIR
    cd $JETSON_CONTAINERS_DIR
    git checkout f38a23025619afd95ae79ca7c3e1c609964c3117
    bash $JETSON_CONTAINERS_DIR/install.sh
    cd $SCRIPT_DIR
fi

# Get the L4T version
L4T_VERSION=$($JETSON_CONTAINERS_DIR/jetson_containers/l4t_version.sh | awk '/L4T_VERSION:/ {print $2}')

# Build a jetson-container with the base dependencies: pytorch, torchvision, ros, opencv
BASE_IMAGE_NAME="${IMAGE_NAME}_base"
BASE_IMAGE_TAG="r$L4T_VERSION"
jetson-containers build --skip-tests=all --name=$BASE_IMAGE_NAME \
    opencv:$OPENCV_VERSION \
    ros:$ROS_DISTRO \
    pytorch:$PYTORCH_VERSION \
    torchvision:$TORCHVISION_VERSION \

echo "Building $IMAGE_NAME with base image $BASE_IMAGE_NAME:$BASE_IMAGE_TAG"
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE=$BASE_IMAGE_NAME:$BASE_IMAGE_TAG \
    -t $IMAGE_NAME:$BASE_IMAGE_TAG \
    -f Dockerfile ../..