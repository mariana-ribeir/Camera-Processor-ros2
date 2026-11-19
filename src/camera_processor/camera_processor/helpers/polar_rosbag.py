import os
from datetime import datetime


POLAR_ROSBAG_DIR = "/workspaces/ros2_ws/.cache/rosbags"
ROSBAG_TOPIC_PATTERNS = (
    r"/polar/fatigue/.*",
    r"/empatica/gmm/(?:predictions|status)",
    r"/person/(?:detected|detections)",
    r"/pose/(?:ia|heuristic)/detected",
    r"/pose/detected",
)


def _is_enabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _topic_regex(include_camera_image):
    topic_patterns = list(ROSBAG_TOPIC_PATTERNS)
    if include_camera_image:
        topic_patterns.append(r"/camera/image_raw")
    return r"^(?:" + "|".join(topic_patterns) + r")$"


ROSBAG_TOPIC_REGEX = _topic_regex(include_camera_image=False)


def polar_rosbag_command(enabled=None):
    value = os.getenv("POLAR_ROSBAG_ENABLED", "false") if enabled is None else enabled
    if not _is_enabled(value):
        return None

    camera_image_enabled = _is_enabled(
        os.getenv("POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED", "false")
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = os.getenv("POLAR_ROSBAG_DIR", POLAR_ROSBAG_DIR)
    if not output_dir.strip():
        raise ValueError("POLAR_ROSBAG_DIR no puede estar vacio.")
    output = os.path.join(output_dir, f"polar_{timestamp}")
    return [
        "ros2",
        "bag",
        "record",
        "--output",
        output,
        "--regex",
        _topic_regex(include_camera_image=camera_image_enabled),
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
    ]
