#!/bin/bash

# Add ros2 setup.bash to the bashrc
echo "source ${ROS_ROOT}/setup.bash" >> /root/.bashrc

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

# Add faint setup.bash to the bashrc
echo "source /opt/faint/faint/deployment/src/install/setup.bash" >> /root/.bashrc
exec "$@"