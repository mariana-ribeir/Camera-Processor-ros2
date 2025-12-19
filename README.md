# Camera Processor ROS2 Package.

This ROS2 package simulates a camera using a video file and publishes frames as ROS2 image messages.  
It also includes a placeholder for vision processing (`processor.py`).

## Features

- Simulates a camera from a video file (`walk_people.mp4` or any `.mp4` video)
- Publishes frames to the `/camera/image_raw` topic
- Ready for adding custom image processing

## Docker Workflow

This project uses Docker to ensure a consistent ROS2 environment across machines.

### 1. Prerequisites
* Docker Desktop (WSL Option).
* Visual Studio Code (VS Code).
* VS Code Extension: Dev Containers (formerly "Remote - Containers").

### 2.Launching the Environment

You do not need to run manual Docker commands. VS Code will handle the container setup automatically:

1.  **Clone the Repository::** Clone this repository to your local system.
2.  **Open in VS Code:** Open the project's root folder (`/ros2_ws`) in VS Code .
3.  **Launch the Container:** When VS Code opened notification should appear asking **"Reopen in Container?"**, click 
    * (If the notification doesn't appear, press $\text{Ctrl} + \text{Shift} + \text{P}$ (or $\text{Cmd} + \text{Shift} + \text{P}$) and execute the command Dev Containers: Reopen in Container).
    
VS Code will build the image and launch a Linux container where your source code is mounted, with all ROS 2 dependencies pre-configured in `Dockerfile`

### 3. ROS 2 Development Workflow

All commands must be executed within the VS Code Integrated Terminal (which is already inside the Linux container).

To test if the ROS2 is currently installed:

```bash
source /opt/ros/humble/setup.bash
ros2 --help
ros2 doctor
```

To build the workspace:
(Whenever you make changes to Python or C++ files, recompile using the terminal)

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Usage

Run the camera simulator:
```bash
ros2 run camera camera_simulator
```

The node will publish frames from the video to `/camera/image_raw` topic, its possible check the topic: 

```bash
ros2 topic list
ros2 topic info /camera/image_raw
```

To visualize the image view, or in this case the video view: 
```bash
ros2 run rqt_image_view rqt_image_view
```

## Architecture Overview

```
                                       +--------------------+
                                       |   Camera Node      |
                                       | (camera_simulator) |
                                       +--------------------+
                                                  | 
              -----------------+------------------+----------------------------------+
              |                                   |                                  |
      /camera/image_raw                 /camera/image_raw                    /camera/image_raw
              |                                   |                                  |
  +-------------------------+         +-------------------------+        +-------------------------+
  |   Color Processor Node  |         |  Person Processor Node  |        |   Pose Processor Node   |
  |   (color_processor)     |         |   (person_processor)    |        |   (pose_processor)      |
  +-------------------------+         +-------------------------+        +-------------------------+
  | /color/frame_processed  |         |  person/detected        |        |  pose/detected          |
  | /color/red_detected     |         |  person/count           |        |                         |
  +-------------------------+         +-------------------------+        +-------------------------+


```

| Package                     | Node name          | Purpose                                                     |
| --------------------------- | ------------------ | ----------------------------------------------------------  |
| `camera`                    | `camera_publisher` | Publishes raw video frames                                  |
| `camera-processor`          | `color_processor`  | Processes frames to detect colors and publish results       |
|                             | `person_processor` | Processes frames to detect persons and publish results      |
|                             | `pose_processor`   | Processes frames to detect pose persons and publish results |



## Current Stage: Early

Current this project is in **Early Stage** its like the initial phase, undertstand the big problem by split in some little problems.


- ✔️ Camera Node
    -  ✔️ Publishes raw video frames on `/camera/image_raw` 
- ✔️ Camera Processor 
    - ✔️ Color Processor Node
        -  ✔️ Subscribes to `/camera/image_raw` 
        -  ✔️ Publishes real frames
        -  ✔️ Processes frames to detect red objects
        -  ✔️ Publishes processed frames 
        -  ✔️ Publishes boolean red detection on `color/red_detected`
    - ✔️ Person Processor Node
       - ✔️ Subscribes to `/camera/image_raw` 
       - ✔️ Processes frames to detect persons 
       - ✔️ Publishes boolean person detected `/person_detected`
       - ✔️ Publishes count person `/count_person`
    - ✔️ Pose Person Processor Node
       - ✔️ Subscribes to `/camera/image_raw` 
       - ✔️ Processes frames to pose persons 
       - ✔️ Publishes string pose person detected `/pose_detected`
<!--
## Final Stage

By the end of the project should be possible, identify a human in the video near the robo that its present in the video too and tell the robo to stop moving to keep the human safe.
-->
