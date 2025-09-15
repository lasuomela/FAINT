#!/bin/bash

# Set default camera type
CAMERA_TYPE='zed2'

# Set default ROS discovery server IP and port here or in CLI args
# if you want to use discovery server. Blank will use default
# ROS2 multicast discovery.
ROS_DISCOVERY_SERVER_IP='192.168.185.3'
ROS_DISCOVERY_SERVER_PORT='11811'
export ROS_DOMAIN_ID='0'


while getopts c:i:p: flag
do
    case "${flag}" in
        c) CAMERA_TYPE=${OPTARG};;
        i) ROS_DISCOVERY_SERVER_IP=${OPTARG};;
        p) ROS_DISCOVERY_SERVER_PORT=${OPTARG};;
        *) error "Unexpected option ${flag}" ;;
    esac
done

if [ -n "$ROS_DISCOVERY_SERVER_IP" ]; then
    export ROS_DISCOVERY_SERVER="$ROS_DISCOVERY_SERVER_IP:$ROS_DISCOVERY_SERVER_PORT"
    export IS_ROS_SUPER_CLIENT='True'
fi

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
export WORKDIR=$(dirname $(dirname "$SCRIPT_DIR")) # Top level directory

# Get the L4T version
# Assume jetson-containers was cloned during docker build
JETSON_CONTAINERS_DIR=$(realpath $SCRIPT_DIR/../jetson-containers)
export L4T_VERSION=$($JETSON_CONTAINERS_DIR/jetson_containers/l4t_version.sh | awk '/L4T_VERSION:/ {print $2}')

xhost +local:docker
docker-compose -f docker-compose.yml up --remove-orphans --force-recreate -d $CAMERA_TYPE faint
docker attach faint_container

# Kill running containers
docker-compose -f docker-compose.yml down
xhost -local:docker