#!/bin/sh

IMAGE_NAME="faint"

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

# Parse command line arguments
ARCH="x86"
while getopts r:b:t: flag
do
    case "${flag}" in
        t) ARCH=${OPTARG};;
        *) echo "Unexpected option ${flag}" && exit 1 ;;
    esac
done

# Assert that the architecture is either x86 or aarch64
if [ "$ARCH" != "x86" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Error: Architecture must be either 'x86' or 'aarch64'."
    exit 1
fi

# The Isaac-ROS images get updated frequently.
# See https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/containers/ros/tags
# for up-to-date tags
if [ "$ARCH" = "x86" ]; then
    BASE_IMAGE="nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble_6f2a6bddf70fcd928f08e874635efe43"
    TAG="x86-0.0.1"
else
    BASE_IMAGE="nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_77e6a678c2058abf96bedcb8f7dd4330"
    TAG="aarch64-0.0.1"
fi

# GPU capabilites require Nvidia container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg ARCH=$ARCH \
    -t $IMAGE_NAME:$TAG \
    -f Dockerfile ../..
