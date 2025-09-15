#!/bin/bash

echo "Configuring:"

if [ -z "$ROS_DISCOVERY_SERVER" ]; then
    echo "ROS_DISCOVERY_SERVER is not set, using default multicast"
else
    echo "Using ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
fi
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

# If discovery server ip is localhost, start the discovery server
if [ "$ROS_DISCOVERY_SERVER" == "127.0.0.1" ];
then
    echo "Starting FastDDS discovery server"
    source ${ROS_ROOT}/install/setup.bash
    fastdds discovery --server-id $domain_id &
fi

# Add ros2 setup.bash to the bashrc
echo "source ${ROS_ROOT}/install/setup.bash" >> /root/.bashrc

# Check if the faint ros package is installed, and build it
if [ ! -d "/opt/faint/faint/deployment/src/install" ];
then
    echo "Building faint deployment package"
    cd /opt/faint/faint/deployment/src
    colcon build
    cd /opt/faint/
fi
# Add faint setup.bash to the bashrc
echo "source /opt/faint/faint/deployment/src/install/setup.bash" >> /root/.bashrc

# Launch foxglove bridge in the background
ros2 launch foxglove_bridge foxglove_bridge_launch.xml > foxglove_output.log 2>&1 &
exec "$@"