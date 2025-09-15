ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install tools and messages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    mesa-utils \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install ROS packages using the jetson-containers script
# For some reason the packages have to be installed one by one or the script fails
RUN /ros2_install.sh turtlebot4_msgs
RUN /ros2_install.sh irobot_create_msgs
RUN /ros2_install.sh foxglove_bridge
RUN /ros2_install.sh theora_image_transport
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install deployment python dependencies
COPY faint/deployment/requirements.txt /tmp/requirements.txt

# Update pip
RUN pip3 install --upgrade pip

# Fix pyopenssl problem from base container
RUN pip3 install pyopenssl==24.0.0 && pip3 install --upgrade requests

# Install dependencies and provide C_INCLUDE_PATH for h5py
RUN C_INCLUDE_PATH=/usr/lib/aarch64-linux-gnu/openmpi/include/ pip install -r /tmp/requirements.txt
RUN pip install -e "git+https://github.com/real-stanford/diffusion_policy.git@main#egg=diffusion_policy" --config-settings editable_mode=compat

# Install the python package
COPY . /opt/faint
WORKDIR /opt/faint
RUN pip3 install -e .

RUN pip3 install setuptools==58.2.0

COPY ./docker/release/ros_entrypoint.sh /opt/ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh", "/opt/ros_entrypoint.sh", "bash"]