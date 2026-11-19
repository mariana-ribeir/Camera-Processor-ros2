#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/ros/jazzy/setup.bash
source "$HOME/.venvs/camera-processor-native/bin/activate"
source install-native/setup.bash

set -a
source .env
set +a

export GMM_MODEL_PATH="${GMM_MODEL_PATH:-$PWD/src/camera_processor/models/gmm_model.pkl}"
export EMPATICA_STATE_FILE="${EMPATICA_STATE_FILE:-$PWD/.cache/empatica_gmm_state.json}"
export POLAR_BASELINE_DIR="${POLAR_BASELINE_DIR:-$PWD/.cache/polar_baselines}"
export POLAR_LOG_DIR="${POLAR_LOG_DIR:-$PWD/.cache/polar_logs}"
export POLAR_ROSBAG_DIR="${POLAR_ROSBAG_DIR:-$PWD/.cache/rosbags}"

for name in GMM_MODEL_PATH EMPATICA_STATE_FILE POLAR_BASELINE_DIR POLAR_LOG_DIR POLAR_ROSBAG_DIR; do
    value="${!name}"
    case "$value" in
        /workspaces/ros2_ws/*)
            export "$name=$PWD/${value#/workspaces/ros2_ws/}"
            ;;
    esac
done

exec python /opt/ros/jazzy/bin/ros2 launch camera_processor "${1:-launch.py}"
