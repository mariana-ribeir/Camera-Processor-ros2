FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install base tools
RUN apt update && apt install -y \
        python3-pip \
        python3-opencv \
        git wget curl \
        ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch (CPU)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install YOLO (this pulls numpy >=2)
RUN pip3 install ultralytics

# ⚠️ FIX: YOLO installs numpy 2.x → reinstall numpy < 2
RUN pip3 install --force-reinstall "numpy<2"

# Verify numpy version
RUN python3 -c "import numpy; print('NUMPY VERSION:', numpy.__version__)"

# Copy ROS2 workspace
COPY . /ros2_ws
WORKDIR /ros2_ws

# Build ROS2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && bash"]
