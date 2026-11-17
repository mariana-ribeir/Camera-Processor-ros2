# ROS2 Camera Processor Package

This ROS2 package simulates a camera using a video file and publishes frames as ROS2 image messages.
It includes multiple processing nodes for perception tasks such as person detection, pose estimation, and color detection.

## Features

- Simulates a camera from a video file (`walk_people.mp4` or any `.mp4` video)
- Publishes frames to the `/camera/image_raw` topic
- Modular perception system:
  - Person detection
  - Pose estimation (AI + heuristic)
  - Color detection
- Web-based visualization using `web_video_server`

## Workflow with Docker

This project uses Docker to ensure a consistent ROS2 Jazzy environment across all machines.

### 1. Prerequisites
- Docker Desktop (WSL option)
- Visual Studio Code (VS Code)
- VS Code Extension: Dev Containers
- Web browser for visualizing topics


### 2. Launching the Environment

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

## Launch File

The system can be started using a ROS2 launch file that initializes all nodes:
- camera_simulator
- person_processor
- ai_pose
- heuristic_pose
- pose_processor
- empatica_gmm_monitor
- polarh10_monitor
- web_video_server

```bash
ros2 launch camera_processor your_launch_file.py
```


## Web Visualization 

The `web_video_server` allows you to view image topics in your browser.

Open in browser:

```
http://localhost:8080/stream?topic=/camera/image_raw
```

You can also visualize processed topics such as:
- /color/frame_processed
- /person/detected
- /pose/ia/detected
- /pose/heuristic/detected
- /pose/detected

## Empatica GMM Monitor

The independent `empatica_gmm_monitor` node checks the configured S3 prefix every 60 seconds, loads the newest day of digital biomarkers, applies the notebook's preprocessing and trained scaler/PCA/GMM bundle, and publishes only timestamps not previously published. Its first run records the existing valid rows as a baseline; later rows are published in batches on `/empatica/gmm/predictions`. The state file stores timestamps only for the active S3 date and drops the previous day's timestamps when a new date is detected.

Before launching, configure AWS and the participant in a root `.env` file (start from `.env.example`) and place the trusted `gmm_model.pkl` bundle at `src/camera_processor/models/gmm_model.pkl`, or set `GMM_MODEL_PATH`. The bundle must contain `model`, `scaler`, `pca`, `features`, and `use_pca`. Do not commit AWS credentials or an untrusted pickle file.

Each `camera_interfaces/msg/EmpaticaPrediction` contains the Unix timestamp in milliseconds, ISO timestamp, cluster ID (0-8), notebook class label, and maximum cluster probability. The array message groups all newly available predictions from one poll:

```bash
ros2 topic echo /empatica/gmm/predictions
```

Every S3 poll also publishes a `camera_interfaces/msg/EmpaticaQueryStatus` on `/empatica/gmm/status`. Its `status` is `error`, `no_data`, `no_new_data`, `new_data`, or `baseline`; `row_count` and `detail` report the outcome. Monitor it with `ros2 topic echo /empatica/gmm/status`.

The node is included in both `launch.py` and `launch_nogui.py` and does not subscribe to the camera pipeline.

## Polar H10 Fatigue Monitor

For concise PC and Jetson setup steps, see [POLAR_QUICKSTART.md](./docs/POLAR_QUICKSTART.md).

The `polarh10_monitor` node connects to a Polar H10 chest strap over Bluetooth (BLE), receives HR/R-R and ECG (130 Hz via the Polar PMD service), reproduces the training preprocessing in 60-second windows, and estimates fatigue with 1-4 ONNX models. It is included in `launch.py`, `launch_nogui.py` and `launch_gmm.py`.

The node supports simultaneous Android sessions and up to two concurrent Polar BLE bridge connections. Each identified connection sends the worker name and Polar BLE ID; buffers, inference history, logs and baselines are separated by that pair. The app asks the user to select a sensor if it finds multiple H10s nearby.

**Baseline:** all input modes use the same persistent `/workspaces/ros2_ws/.cache/polar_baselines` directory by default (override with `POLAR_BASELINE_DIR`). Identified bridge and Android sessions load their file by participant name and readable Polar ID. Older bridge clients without identity fields continue using the configured legacy subject. If a baseline file is missing, the first **15 valid 60 s windows** build it; no predictions are published during calibration. Set `recalibrate:=true` to replace a baseline; it is only overwritten after 15 new windows complete.

**Raw data CSV:** each accepted connection writes a separate timestamped file under `/workspaces/ros2_ws/.cache/polar_logs` by default. Identified bridge and Android file names and rows include the participant name and Polar ID; the CSV columns are `UNIX Timestamp,ECG,HR,RR,Marker,Participant Name,Polar Device ID`. Older bridge clients without identity fields use the configured legacy subject. ECG is in mV, HR in bpm and RR in ms. HR and R-R values are associated with the nearest ECG sample; values without a nearby ECG sample are retained in their own timestamped row. The file is flushed during the session and closed on disconnect. On Windows and Jetson, the bind-mounted files appear under `<project folder>\.cache\polar_logs` and `~/camera-processor-ros2/.cache/polar_logs`, respectively. Override the container directory with `POLAR_LOG_DIR`.

**ROS topic recording:** set `POLAR_ROSBAG_ENABLED=true` in the root `.env` before starting the system to automatically record all `/polar/fatigue/*` inference topics and `/empatica/gmm/predictions` plus `/empatica/gmm/status` together in one timestamped rosbag2 under `.cache/rosbags`. Set it to `false` (the default) to disable recording. The rosbag starts and stops with the selected ROS launch; raw HR, RR and ECG continue to be saved separately as session CSV files.

**How to run on Windows with Docker Desktop:** the ROS node, calibration, ONNX models and ROS publishers run inside Docker. Since Docker Desktop does not directly expose Windows' built-in Bluetooth adapter to Linux containers, the BLE bridge runs on Windows and opens one TCP connection per Polar (up to two simultaneously). It forwards HR/R-R, ECG, participant name and BLE ID over the localhost TCP port; inference and ROS publishing remain in Docker.

1. Rebuild and start the container, then launch the GMM pipeline (which also starts `polarh10_monitor`):

   ```powershell
   docker-compose up -d --build
   docker exec camera_processor_ws bash -lc "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 launch camera_processor launch_gmm.py"
   ```

2. In a separate Windows PowerShell terminal, create a Python environment for this project, install its BLE dependency from the project requirements file, and start the bridge from the repository root:

   ```powershell
   cd "C:\Users\Carlo\Desktop\Camera-Processor-ros2"
   py -3.10 -m venv .venv-polar
   .\.venv-polar\Scripts\python.exe -m pip install -r .\requirements-ble.txt
   .\.venv-polar\Scripts\python.exe .\scripts\polar_h10_tcp_bridge.py --device both
   ```

   Leave both the Docker launch and bridge running. The bridge reconnects if the ROS node restarts. Stop it with Ctrl+C. The TCP port is published only on `127.0.0.1:8765`; it is not exposed to the LAN.

The bridge needs Python 3.9 or newer and Bleak 1.x for Windows' pre-connection pairing support. The Python launcher command above uses Python 3.10; if that version is not installed, install Python 3.10 or newer first.

**Input mode and bridge selection:** set `POLAR_INPUT_MODE=android` to receive phone streams, or `POLAR_INPUT_MODE=bridge` to receive the Python BLE bridge. The phone mode supports simultaneous app connections. In bridge mode, `POLAR_DEVICE=polar1`, `polar2` or `both` selects one or both configured sensors; `POLAR1_ADDRESS` and `POLAR2_ADDRESS` hold their BLE addresses, with optional `POLAR1_PAIR`/`POLAR2_PAIR` pairing flags. Bridge identities are fixed as `POLAR1` and `POLAR2`; the same labels are used for their baseline and log filenames. `POLAR_DEVICE` is ignored in phone mode. You can override the selection for one run with `--device polar1`, `--device polar2` or `--device both`.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `subject` | `POLAR1` | Fallback identity for older bridge clients without identity fields |
| `recalibrate` | `false` | Discard the existing baseline and record 15 new windows |
| `models` | `[logistic_regression, random_forest, svm, xgboost]` | Subset of 1-4 models to run |
| `models_root` | `<package_share>/models/polar` | Folder with the `LOGISTIC_REGRESSION`, `RANDOM_FOREST`, `SVM`, `XGBOOST` model dirs |
| `baseline_dir` | `/workspaces/ros2_ws/.cache/polar_baselines` in Docker | Single persistent folder where all bridge and Android baseline CSVs are read/written |
| `polar_log_dir` | `/workspaces/ros2_ws/.cache/polar_logs` in Docker | Folder for raw ECG/HR/R-R CSV files, one per connection |
| `tcp_host` | `0.0.0.0` | Address on which the ROS node accepts bridge and Android TCP sessions |
| `tcp_port` | `8765` | TCP port for BLE bridge and Android sessions |

**Model selection:** set `POLAR_MODELS` in the root `.env` file as a comma-separated list. For example, `POLAR_MODELS=svm,xgboost` runs only those two models; `POLAR_MODELS=random_forest` runs only Random Forest. Valid names are `logistic_regression`, `random_forest`, `svm` and `xgboost`. Restart the system (`.\run_system.ps1 -GMM`) for the node to load the selection. Models cannot be changed while the node is running.

**Topics:** each model publishes `camera_interfaces/msg/FatiguePrediction` on its own topic: `/polar/fatigue/logistic_regression`, `/polar/fatigue/random_forest`, `/polar/fatigue/svm` and `/polar/fatigue/xgboost`. The message carries the session subject, participant name, Polar ID, model name, predicted label (`Low`, `Medium`, `High`), the three class probabilities, the number of imputed features and the temporal history size. Rebuild the ROS workspace and Docker image after changing the message definition.

```bash
ros2 topic echo /polar/fatigue/xgboost
```

**Requirements:** the H10 must be worn (skin contact) and not connected to another app. Install `bleak` in the Windows Python environment and run the bridge as described above. Docker runs Polar inference from ONNX models and imputer medians stored in their metadata; the `.joblib` training artifacts are not loaded at runtime. These outputs are model estimates, not a medical evaluation.

## Architecture Summary
```text
                                        +--------------------+        +---------------------------+   +---------------------------+
                                        |   Camera Node      |        | Empatica GMM Monitor      |   | Polar H10 Monitor         |
                                        | (camera_simulator) |        | (empatica_gmm_monitor)    |   | (polarh10_monitor)        |
                                        +--------------------+        +---------------------------+   +---------------------------+
                                                |                     | /empatica/gmm/predictions |   | /polar/fatigue/<model>    |
                                                |  /camera/image_raw  | /empatica/gmm/status      |   |                           |
                                                |                     +---------------------------+   +---------------------------+
          +---------------------------+---------------------+
          |                           |                     |
          v                           v                     |
+---------------------------+ +------------------+          |
| Color Processor           | | Person Processor |          |
| (color_processor)         | | (person_processor)|         |
+---------------------------+ +------------------+          |
| /color/frame_processed    | | person/detected  |          |
| /color/red_detected       | | person/detections|          |
+---------------------------+ +------------------+          |
                                    |                       |
                                    |                       |
                                    v                       v
                          +---------------------------------+-------------------+
                          |                                                     |
                          v                                                     v
               +------------------+                               +-------------------------+
               | IA Pose          |                               | Heuristic Pose          |
               | (ai_pose)        |                               | (heuristic_pose)        |
               +------------------+                               +-------------------------+
               | pose/ia/detected |                               | pose/heuristic/detected |
               +------------------+                               +-------------------------+
                          ^                                                    ^
                          |                                                    |
                          +-------------------- person/detected ---------------+
                                                     |
                                                     v
                                            +------------------+
                                            | Pose Processor   |
                                            | (pose_processor) |
                                            +------------------+
                                            | pose/detected    |
                                            +------------------+
```

**Data Flow:**
- `camera_simulator` publishes images to `/camera/image_raw`.

- The following nodes subscribe independently to the camera stream:
  - `color_processor`
  - `person_processor`
  - `ai_pose`
  - `heuristic_pose`

- Each node processes the incoming data and publishes its own results:
  - `color_processor` → `/color/frame_processed`, `/color/red_detected`
  - `person_processor` → `person/detected`, `person/count`
  - `ai_pose` → `pose/ia/detected`
  - `heuristic_pose` → `pose/heuristic/detected`

- The `ai_pose` and `heuristic_pose` nodes use two inputs:
  - `/camera/image_raw` (raw frames)
  - `person/detected` (person detections)

  This allows pose estimation to be guided by detected persons while still using the full image.

- The `pose_processor` node:
  - Subscribes to both `pose/ia/detected` and `pose/heuristic/detected`
  - Combines the results
  - Publishes the final output to `pose/detected`

- `empatica_gmm_monitor` polls the configured S3 bucket for the participant's
  biomarker CSVs, preprocesses new rows, runs the scaler/PCA/GMM pipeline, and
  publishes predictions and query status. It does not connect directly to the
  watch.
- `polarh10_monitor` receives HR/R-R and ECG from the BLE bridge over TCP,
  records each connection to a CSV, processes signal windows, and publishes
  fatigue predictions. It accepts concurrent identified connections from
  Android phones and up to two Polar BLE bridge profiles.

| Package                     | Node Name         | Purpose                                                               |
| --------------------------- | ----------------- | --------------------------------------------------------------------- |
| `camera`                    | `camera_simulator`| Publishes raw video frames                                            | 
| `camera-processor`          | `color_processor` | Processes frames to detect colors and publish results                 |
|                             | `person_processor`| Processes frames to detect people and publish results                 |
|                             | `heuristic_pose`  | Processes frames to detect poses using heuristics and publish results |
|                             | `ai_pose`         | Processes frames to detect poses using AI and publish results         |
|                             | `pose_processor`  | Combines AI and heuristic detections to publish final results         |
|                             | `empatica_gmm_monitor` | Reads Empatica biomarker CSVs from S3 and publishes GMM predictions |
|                             | `polarh10_monitor` | Receives Polar data over TCP, logs CSV, and publishes fatigue predictions |

### Relationship between Pose Nodes

The `heuristic_pose` and `ai_pose` nodes subscribe to multiple inputs:

- `/camera/image_raw`: raw image frames
- `person/detected`: detected persons

- `heuristic_pose`: Uses heuristic algorithms to detect human poses.
- `ai_pose`: Uses an AI model (YOLO) to detect poses.

Each node publishes its results to:
- `pose/heuristic/detected`
- `pose/ia/detected`

The `pose_processor` node acts as an aggregator:
- Subscribes to both pose topics
- Combines or compares the detections
- Publishes the final result to `pose/detected`

The `pose_processor` does not receive image data directly.


<!--
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

## Final Stage

At the end of the project it should be possible to identify a human in the video near the robot that is also present in the video and tell the robot to stop moving to keep the human safe.
-->
