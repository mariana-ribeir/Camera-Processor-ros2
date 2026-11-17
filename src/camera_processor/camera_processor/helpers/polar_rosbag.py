import os
from datetime import datetime


POLAR_ROSBAG_DIR = "/workspaces/ros2_ws/.cache/rosbags"
POLAR_TOPIC_REGEX = r"^/(?:polar/fatigue/.*|empatica/gmm/(?:predictions|status))$"


def polar_rosbag_command(enabled=None):
    value = os.getenv("POLAR_ROSBAG_ENABLED", "false") if enabled is None else enabled
    if str(value).strip().lower() not in {"1", "true", "yes", "on"}:
        return None

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = f"{POLAR_ROSBAG_DIR}/polar_{timestamp}"
    return [
        "ros2",
        "bag",
        "record",
        "--output",
        output,
        "--regex",
        POLAR_TOPIC_REGEX,
    ]
