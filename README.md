# Camera Processor ROS2 Package

This ROS2 package simulates a camera using a video file and publishes frames as ROS2 image messages.  
It also includes a placeholder for vision processing (`processor.py`).

## Features

- Simulates a camera from a video file (`walk_people.mp4` or any `.mp4` video)
- Publishes frames to the `/camera/image_raw` topic
- Ready for adding custom image processing

## Docker Workflow

This project uses Docker to ensure a consistent ROS2 environment across machines.


Inside the project root lets create the Docker image: 

Pull the ROS 2 Humble Desktop image: 

```bash
docker build -t ros2-humble-yolo .
```

Run a container mapping a Windows folder


```bash
docker run -it --name ros2_yolo_container -v C:\path\to\windows\folder:/ros2_ws ros:ros2-humble-yolo 
```

Inside the container, your Windows files are accessible at /ros2_ws. You can now run ROS 2 commands or launch nodes directly.

To test if the ROS2 is currently installed:

```bash
source /opt/ros/humble/setup.bash
ros2 --help
ros2 doctor
```

To stop the running container:

```bash
docker stop ros2_yolo_container
```

To restart it later:

```bash
docker start -ai ros2_yolo_container
```
-a ->  attacth to the container
-i ->  keep the container interactive

## Installation

Make sure you made the Docker steps, or have ROS2 installed by another away (directly on WSL or Linux, Virtual Machines...)

Clone the package into your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/mariana-ribeiro/camera-processor-ros2.git
```

Build the workspace:

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
              -----------------+-----------------
              |                                   |
  /camera/image_raw                        /camera/image_raw
              |                                   |
  +-------------------------+         +-------------------------+
  |   Color Processor Node  |         |  Person Processor Node  |
  |   (color_processor)     |         |   (person_processor)    |
  +-------------------------+         +-------------------------+
  | /color/frame_processed  |         |  person/detected        |
  | /color/red_detected     |         |  person/count           |
  |                         |         |  person/frame_processed |
  +-------------------------+         +-------------------------+

```

| Package                     | Node name          | Purpose                                                |
| --------------------------- | ------------------ | ------------------------------------------------------ |
| `camera`                    | `camera_publisher` | Publishes raw video frames                             |
| `camera-processor`          | `color_processor`  | Processes frames to detect colors and publish results  |
|                             | `person_processor` | Processes frames to detect persons and publish results |



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
<!--
## Final Stage

By the end of the project should be possible, identify a human in the video near the robo that its present in the video too and tell the robo to stop moving to keep the human safe.
-->
