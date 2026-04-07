#!/bin/bash

# Script to update the camera_processor package after code changes

echo "Building camera_processor package..."
colcon build --packages-select camera_processor

echo "Sourcing ROS2 setup..."
source install/setup.bash

echo "Setting PYTHONPATH..."
export PYTHONPATH="/workspaces/Camera-Processor-ros2/install/camera_processor/lib/python3.12/site-packages:$PYTHONPATH"

echo "Update complete! Now you can run: ros2 run camera_processor ia_pose"