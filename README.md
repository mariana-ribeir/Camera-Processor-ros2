# ROS2 Camera Processor Package

This ROS2 package simulates a camera using a video file and publishes frames as ROS2 image messages.  
It also includes a placeholder for vision processing (`processor.py`).

## Features

- Simulates a camera from a video file (`walk_people.mp4` or any `.mp4` video)
- Publishes frames to the `/camera/image_raw` topic
- Ready to add custom image processing

## Workflow with Docker

This project uses Docker to ensure a consistent ROS2 Jazzy environment across all machines.

### 1. Prerequisites
* Docker Desktop (WSL option).
* Visual Studio Code (VS Code).
* VS Code Extension: Dev Containers (formerly "Remote - Containers").
* **VcXsrv** (X Server for Windows) - Required to display OpenCV windows.

### 2. VcXsrv Setup (Required for Display)

Since the application uses `cv2.imshow()` to display video frames, you need an X server on Windows:

1. **Download VcXsrv:** https://sourceforge.net/projects/vcxsrv/
2. **Install** with default options
3. **Run XLaunch** from the Start menu with these settings:
   - Select "Multiple windows" → Next
   - Select "Start no client" → Next
   - ✅ Check **"Disable access control"** (important!)
   - Click Finish
4. The VcXsrv icon should appear in the system tray

> **Note:** You must start VcXsrv **before** opening the Development Container.

### 3. Launching the Environment

You can use Docker Compose for easier execution, or proceed with Dev Containers in VS Code.

#### Option A: Using Docker Compose (Recommended for manual execution)

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Access the container:**
   ```bash
   docker exec -it camera_processor_ws bash
   ```

3. **Inside the container, build the workspace if necessary:**
   ```bash
   cd /workspaces/ros2_ws
   source /opt/ros/jazzy/setup.bash
   colcon build
   source install/setup.bash
   ```

#### Option B: Using Dev Containers in VS Code

You don't need to run Docker commands manually. VS Code will handle the container setup automatically:

1.  **Clone the Repository:** Clone this repository to your local system.
2.  **Start VcXsrv:** Make sure XLaunch is running (see step 2 above).
3.  **Open in VS Code:** Open the root project folder (`/ros2_ws`) in VS Code.
4.  **Launch the Container:** When the VS Code notification appears asking **"Reopen in Container?"**, click it.
    * (If the notification doesn't appear, press Ctrl+Shift+P (or Cmd+Shift+P) and run the Dev Containers: Reopen in Container command).
    
VS Code will build the image and launch a Linux container where your source code is mounted, with all ROS 2 dependencies preconfigured in `Dockerfile`.

### 4. ROS 2 Development Workflow

All commands must be run inside the VS Code Integrated Terminal (which is already inside the Linux container).

To test if ROS2 is currently installed:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
ros2 doctor
```

To build the workspace:
(When you make changes to Python or C++ files, recompile using the terminal)

```bash
cd /workspaces/ros2_ws
colcon build
source install/setup.bash
```

## Usage

Run the camera simulator:
```bash
ros2 run camera camera_simulator
```

The node will publish frames from the video to the `/camera/image_raw` topic, you can check the topic:

```bash
ros2 topic list
ros2 topic info /camera/image_raw
```

To visualize the image view, or in this case the video view:
```bash
ros2 run rqt_image_view rqt_image_view
```

## Architecture Summary

```
                                        +--------------------+
                                        |   Camera Node      |
                                        | (camera_simulator) |
                                        +--------------------+
                                                |
                                                |  /camera/image_raw
                                                |
          +---------------------------+------------------+-----------------------+
          |                           |                  |                       |
          v                           v                  v                       v
+---------------------------+ +------------------+  +------------------+ +-------------------------+
| Color Processor           | | Person Processor  | | IA Pose          | | Heuristic Pose          |
| (color_processor)         | | (person_processor)| | (ai_pose)        | | (heuristic_pose)        |
+---------------------------+ +-------------------+ +------------------+ +-------------------------+
| /color/frame_processed    | | person/detected   | | pose/ia/detected | | pose/heuristic/detected |
| processed                 | | person/count      | |                  | | detected                |
| /color/red_detected       | |                   | |                  | |                         |
|                           | |                   | |                  | |                         |
+---------------------------+ +-------------------+ +------------------+ +-------------------------+      
                                                            |                        |
                                                            +------------------------+
                                                                        |
                                                                        v
                                                                +------------------+
                                                                | Pose Processor   |
                                                                | (pose_processor) |
                                                                +------------------+
                                                                | pose/detected    |
                                                                |                  |
                                                                +------------------+
```

**Data Flow:**
- `camera_simulator` publishes images to `/camera/image_raw`.
- Four nodes subscribe to `/camera/image_raw`: `color_processor`, `person_processor`, `ai_pose`, `heuristic_pose`.
- `ai_pose` and `heuristic_pose` publish results to `pose/ia/detected` and `pose/heuristic/detected`, which converge in `pose_processor`.
- `pose_processor` combines the results and publishes to `pose/detected`.
- `color_processor` and `person_processor` publish to their own topics.

| Package                     | Node Name         | Purpose                                                  |
| --------------------------- | ----------------- | ------------------------------------------------------- |
| `camera`                    | `camera_publisher`| Publishes raw video frames                              |
| `camera-processor`          | `color_processor` | Processes frames to detect colors and publish results   |
|                             | `person_processor`| Processes frames to detect people and publish results   |
|                             | `heuristic_pose`  | Processes frames to detect poses using heuristics and publish results |
|                             | `ai_pose`         | Processes frames to detect poses using AI and publish results |
|                             | `pose_processor`  | Combines AI and heuristic detections to publish final results |

### Relationship between Pose Nodes

The `heuristic_pose` and `ai_pose` nodes subscribe directly to the `/camera/image_raw` topic and process frames independently:

- `heuristic_pose`: Uses heuristic algorithms to detect human poses.
- `ai_pose`: Uses an AI model (YOLO) to detect poses.

Each publishes their results to their respective topics (`pose/heuristic/detected` and `pose/ia/detected`).

The `pose_processor` node acts as an aggregator: it subscribes to the topics from `heuristic_pose` and `ai_pose`, combines or compares the detections, and publishes the final result to `pose/detected`. It does not receive images directly; it only processes the results already calculated by the other nodes.



## Current Stage: Early

Currently this project is in **Early Stage**, it's like the initial phase, understanding the big problem by dividing it into some small problems.


- ✔️ Camera Node
    -  ✔️ Publishes raw video frames to `/camera/image_raw` 
- ✔️ Camera Processor 
    - ✔️ Color Processor Node
        -  ✔️ Subscribes to `/camera/image_raw` 
        -  ✔️ Publishes real frames
        -  ✔️ Processes frames to detect red objects
        -  ✔️ Publishes processed frames 
        -  ✔️ Publishes boolean red detection to `color/red_detected`
    - ✔️ Person Processor Node
       - ✔️ Subscribes to `/camera/image_raw` 
       - ✔️ Processes frames to detect people 
       - ✔️ Publishes boolean person detected `/person_detected`
       - ✔️ Publishes person count `/count_person`
    - ✔️ Person Pose Processor Node
       - ✔️ Subscribes to `/camera/image_raw` 
       - ✔️ Processes frames for person poses 
       - ✔️ Publishes string person pose detected `/pose_detected`
    - ✔️ Heuristic Pose Node
       - ✔️ Subscribes to `/camera/image_raw`
       - ✔️ Processes frames to detect poses using heuristics
       - ✔️ Publishes pose detections to `pose/heuristic/detected`
    - ✔️ IA Pose Node
       - ✔️ Subscribes to `/camera/image_raw`
       - ✔️ Processes frames to detect poses using AI (YOLO)
       - ✔️ Publishes pose detections to `pose/ia/detected`
<!--
## Final Stage

At the end of the project it should be possible to identify a human in the video near the robot that is also present in the video and tell the robot to stop moving to keep the human safe.
-->
