#!/bin/bash

# Set default camera type
CAMERA_TYPE=''

# Set default ROS discovery server IP and port here or in CLI args
# if you want to use discovery server. Blank will use default
# ROS2 multicast discovery.
ROS_DISCOVERY_SERVER_IP=''
ROS_DISCOVERY_SERVER_PORT=''
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

FILE=$(readlink -f "$0")
export WORKDIR=$(dirname $(dirname $(dirname "$FILE")))

xhost +local:docker
docker-compose -f docker-compose.yml up --remove-orphans --force-recreate -d $CAMERA_TYPE faint
docker attach faint_container

# Kill running containers
docker-compose -f docker-compose.yml down
xhost -local:docker