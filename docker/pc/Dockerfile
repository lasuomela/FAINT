ARG BASE_IMAGE
FROM $BASE_IMAGE
ARG ARCH

# Install deployment python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip3 install --upgrade pip

# Install pip dependencies (But assume that torch and torchvision are provided by the base image)
COPY faint/deployment/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-binary torch,torchvision -r /tmp/requirements.txt

# Install torchvision from source
RUN git clone https://github.com/pytorch/vision.git /opt/vision \
    && cd /opt/vision
RUN cd /opt/vision && git checkout v0.17.0 && FORCE_CUDA=0 python3 setup.py install

# Install the python package
COPY . /opt/faint
WORKDIR /opt/faint
RUN pip3 install -e .

COPY docker/x86/ros_entrypoint.sh /opt/ros_entrypoint.sh
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/ros_entrypoint.sh", "bash"]