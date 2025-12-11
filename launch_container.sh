#!/bin/bash

# --- Configuration Variables ---
CONTAINER_NAME="ros2_dev"
IMAGE_TAG="ros2_ws_humble_image"

# --- Step 1: Build the Image (Reconstrói apenas se o Dockerfile mudar) ---
echo "--- 1. Building Base Image: $IMAGE_TAG ---"
docker build -t $IMAGE_TAG .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker image build failed. Exiting."
    exit 1
fi

# --- Step 2: Clean Up Previous Container (Garantir que o nome está livre) ---
echo "--- 2. Stopping and Forcing Removal of old container: $CONTAINER_NAME ---"
# O '-f' (force) garante que o contentor é removido mesmo se estiver a correr.
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# --- Step 3: Run the New Container (Montar o Código Git Local) ---
echo "--- 3. Starting new container and mounting local Git repo to /ros2_ws ---"

LOCAL_PATH=$(pwd -W)

# Monta o volume, executa em detached mode (-d) para que possamos usar 'exec'
docker run -it -d \
    --name $CONTAINER_NAME \
    -v "$LOCAL_PATH:/ros2_ws" \
    --rm \
    $IMAGE_TAG

# Esperar 1 segundo para garantir que o bash iniciou antes de executar comandos
sleep 1

# ...
# --- Step 4: First-Time Build and Setup (Compilação do Código Git Atualizado) ---
echo "--- 4. Building and registering the local code (ROS setup)... ---"

# 1. Colcon build do seu código (COMPILOU COM SUCESSO!)
docker exec $CONTAINER_NAME bash -c 'source /opt/ros/humble/setup.bash && colcon build --symlink-install'

# --- Step 5: Attach to Container ---
echo "--- 5. Setup complete. Attaching to container shell. Use 'exit' to stop the container. ---"

# Anexa o seu terminal ao shell do contentor
docker attach $CONTAINER_NAME

echo "Container stopped"