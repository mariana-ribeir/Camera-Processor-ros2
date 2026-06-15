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
RUN pip3 install --upgrade setuptools --break-system-packages

# --- 1b. INSTALL OPENCV WITH GTK SUPPORT ---
# Use --ignore-installed to bypass system numpy that can't be uninstalled
RUN pip3 install --break-system-packages --ignore-installed opencv-python

# Environment variable to help with display
ENV QT_X11_NO_MITSHM=1
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

# --- 2. DEPENDÊNCIAS PYTHON ---
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
ARG ONNXRUNTIME=cpu
ENV ONNXRUNTIME=${ONNXRUNTIME}

# Install Python dependencies. Choose ONNXRUNTIME at build time via --build-arg.
# Valid values: 'cpu' (default) or 'gpu'.
RUN if [ "$ONNXRUNTIME" = "gpu" ] ; then \
        pip3 install --break-system-packages --ignore-installed numpy onnxruntime-gpu ultralytics torchreid gdown tensorboard scipy scikit-learn lap ; \
    else \
        pip3 install --break-system-packages --ignore-installed numpy onnxruntime ultralytics torchreid gdown tensorboard scipy scikit-learn lap ; \
    fi

# NOTE: For CUDA (x86_64) use `--build-arg ONNXRUNTIME=gpu` and ensure the
# host/container CUDA toolkit and drivers match the `onnxruntime-gpu` wheel.
# For NVIDIA Jetson (ARM) you must use a Jetson-compatible ONNX Runtime
# wheel (or build from source). Installing both CPU and GPU wheels together
# may cause package conflicts, so the Dockerfile installs only the selected
# variant at build time.

# --- 3. FIX CV_BRIDGE ---
WORKDIR /opt/cv_bridge_build
RUN git clone https://github.com/ros-perception/vision_opencv.git -b rolling src/vision_opencv --depth 1

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --packages-select cv_bridge --symlink-install"

RUN mkdir -p /opt/ros/jazzy/lib/python3.12/site-packages/
RUN cp -r install/cv_bridge/lib/python3.12/site-packages/cv_bridge /opt/ros/jazzy/lib/python3.12/site-packages/

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