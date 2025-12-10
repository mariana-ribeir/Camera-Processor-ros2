#!/bin/bash

# --- Configuration Variables ---
CONTAINER_NAME="ros2_dev"
IMAGE_TAG="ros2_ws_humble_image"

# --- Step 1: Build the Image (Using your Dockerfile) ---
echo "--- Step 1: Building Docker Image: $IMAGE_TAG ---"
# The '.' at the end means "use the Dockerfile in the current directory"
docker build -t $IMAGE_TAG .

# Check if the build failed
if [ $? -ne 0 ]; then
    echo "ERROR: Docker image build failed. Exiting."
    exit 1
fi

# --- Step 2: Clean Up Previous Container (Stop and Remove) ---
echo "--- Step 2: Stopping and Removing old container: $CONTAINER_NAME ---"
# The '|| true' ensures the script doesn't crash if the container doesn't exist
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# --- Step 3: Run the New Container (Assigning the Name) ---
echo "--- Step 3: Starting new container: $CONTAINER_NAME ---"
echo "--- You will be inside the container with your ROS workspace mounted at /ros2_ws ---"

# The command below starts the container:
# -it: Interactive and TTY (allows you to use the shell)
# --name: Assigns the name
# -v "$(pwd):/ros2_ws": Mounts your current host directory to /ros2_ws in the container
docker run -it \
    --name $CONTAINER_NAME \
    -v "$(pwd):/ros2_ws" \
    $IMAGE_TAG