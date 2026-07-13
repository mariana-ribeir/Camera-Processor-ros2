FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# --- 1. FERRAMENTAS E DEPENDÊNCIAS ---
# Install GTK3 for OpenCV display (like Humble used) instead of Qt6
RUN apt update && apt install -y \
    python3-pip \
    python3-setuptools \
    git wget curl \
    ros-jazzy-vision-opencv \
    ros-jazzy-rqt-image-view \
    build-essential \
    libboost-all-dev \
    # GTK3 dependencies (works better in Docker than Qt6)
    libgtk-3-0 \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# --- 1.5. ROS WEB VIDEO SERVER ---
RUN apt update && apt install -y \
    ros-jazzy-web-video-server \
    && rm -rf /var/lib/apt/lists/*

# Upgrade setuptools to support --editable
RUN pip3 install --break-system-packages --upgrade setuptools

# --- 1b. INSTALL OPENCV AND NUMPY STABLE ---
# We pin numpy to 1.26.x and opencv to 4.10.x to maintain compatibility with cv_bridge
RUN pip3 install --break-system-packages --ignore-installed \
    "numpy<2.0.0" \
    "opencv-python==4.10.0.84"

# Environment variable to help with display
ENV QT_X11_NO_MITSHM=1
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

# --- 2. DEPENDÊNCIAS PYTHON ---
# We must re-pin them here because ultralytics/torch might try to upgrade them
RUN pip3 install --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --break-system-packages \
    onnx onnxruntime ultralytics torchreid gdown tensorboard scipy scikit-learn lap \
    "numpy<2.0.0" \
    "opencv-python==4.10.0.84" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# --- 3. FIX CV_BRIDGE ---
WORKDIR /opt/cv_bridge_build
RUN git clone https://github.com/ros-perception/vision_opencv.git -b rolling src/vision_opencv --depth 1

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --packages-select cv_bridge --symlink-install"

RUN mkdir -p /opt/ros/jazzy/lib/python3.12/site-packages/
RUN cp -r install/cv_bridge/lib/python3.12/site-packages/cv_bridge /opt/ros/jazzy/lib/python3.12/site-packages/

# --- 3.5. ROS-TCP-ENDPOINT PARA UNITY ---
RUN apt update && apt install -y \
    python3-rospkg \
    ros-jazzy-ros-base \
    && rm -rf /var/lib/apt/lists/*

# --- 4. WORKSPACE DO UTILIZADOR ---
ARG WORKSPACE=/workspaces/ros2_ws
WORKDIR $WORKSPACE
COPY . $WORKSPACE

# Removidas as variáveis SETUPTOOLS_USE_DISTUTILS que causaram erro no Python 3.12
# Compilar o teu código
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build"

# --- 5. CONFIGURAÇÃO DO TERMINAL ---
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source $WORKSPACE/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]