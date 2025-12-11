FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# --- 1. INSTALAÇÃO DE FERRAMENTAS BASE E DEPENDÊNCIAS DE BUILD ---
RUN apt update && apt install -y \
    python3-pip \
    python3-opencv \
    git wget curl \
    ros-humble-vision-opencv \
    build-essential \
    libboost-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# --- 2. INSTALAÇÃO DE TODAS AS DEPENDÊNCIAS PYTHON (NumPy 2.x) ---
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install ultralytics
RUN pip3 install torchreid gdown tensorboard scipy scikit-learn

# ⚠️ NOVIDADE: Limpeza de qualquer cv_bridge instalado via APT
RUN apt remove -y ros-humble-cv-bridge || true

# --- 3. FIX CRÍTICO: BUILD DO CV_BRIDGE CONTRA NUMPY 2.x ---
WORKDIR /
RUN git clone https://github.com/ros-perception/vision_opencv.git
WORKDIR /vision_opencv
RUN git checkout humble

RUN mkdir -p /cv_bridge_ws/src
RUN mv cv_bridge /cv_bridge_ws/src/
WORKDIR /cv_bridge_ws

# Compila o cv_bridge usando o NumPy 2.x
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
                  rosdep install --from-paths src --ignore-src -r -y && \
                  colcon build --packages-select cv_bridge --symlink-install"

# --- 4. SUBSTITUIÇÃO FORÇADA ---
# 1. Limpa o cache de otimização do Python.
RUN find /usr/lib/python3.10 -name "__pycache__" -exec rm -rf {} +

# 2. Elimina a versão antiga do cv_bridge que era incompatível.
RUN rm -rf /opt/ros/humble/local/lib/python3.10/dist-packages/cv_bridge

# 3. ⚠️ CORREÇÃO DEFINITIVA: Copia o conteúdo C++ e headers.
# Copia o conteúdo principal de "share" e "include" para o destino.
RUN cp -r /cv_bridge_ws/install/cv_bridge/share/* /opt/ros/humble/share/
RUN cp -r /cv_bridge_ws/install/cv_bridge/include/* /opt/ros/humble/include/

# 4. 🔑 CÓPIA DO MÓDULO PYTHON: Copia o módulo compilado para o caminho Python.
RUN cp -r /cv_bridge_ws/install/cv_bridge/local/lib/python3.10/dist-packages/cv_bridge /opt/ros/humble/local/lib/python3.10/dist-packages/
RUN cp -r /cv_bridge_ws/install/cv_bridge/local/lib/python3.10/dist-packages/cv_bridge-3.2.1-py3.10.egg-info /opt/ros/humble/local/lib/python3.10/dist-packages/
# --- 5. BUILD DO WORKSPACE ROS2 (Focando no registro do seu pacote) ---

COPY . /ros2_ws
WORKDIR /ros2_ws

# Compila o pacote 'camera'
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
                  colcon build --symlink-install --packages-select camera"

# Instalação editável do pacote 'camera' a partir do diretório 'src'.
# Isto força a ligação Python e deve resolver o ModuleNotFoundError.
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
                  pip3 install -e src/camera"

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && bash"]
