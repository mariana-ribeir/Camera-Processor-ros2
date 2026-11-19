import os
import re

import pytest

from camera_processor.helpers.polar_rosbag import (
    POLAR_ROSBAG_DIR,
    ROSBAG_TOPIC_REGEX,
    polar_rosbag_command,
)


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_polar_rosbag_is_disabled_by_default_or_false_values(monkeypatch, value):
    if value == "":
        monkeypatch.delenv("POLAR_ROSBAG_ENABLED", raising=False)
    else:
        monkeypatch.setenv("POLAR_ROSBAG_ENABLED", value)

    assert polar_rosbag_command() is None


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
def test_polar_rosbag_command_records_polar_topics(monkeypatch, value):
    monkeypatch.setenv("POLAR_ROSBAG_ENABLED", value)
    monkeypatch.delenv("POLAR_ROSBAG_DIR", raising=False)
    monkeypatch.delenv("POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED", raising=False)

    command = polar_rosbag_command()

    assert command[:4] == ["ros2", "bag", "record", "--output"]
    assert command[5:7] == ["--regex", ROSBAG_TOPIC_REGEX]
    assert command[7:] == [
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
    ]
    assert os.path.dirname(command[4]) == POLAR_ROSBAG_DIR
    assert re.fullmatch(r"polar_\d{8}_\d{6}", os.path.basename(command[4]))


def test_polar_rosbag_uses_native_output_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("POLAR_ROSBAG_ENABLED", "true")
    monkeypatch.setenv("POLAR_ROSBAG_DIR", str(tmp_path))

    command = polar_rosbag_command()

    assert command[4].startswith(str(tmp_path / "polar_"))


def test_polar_rosbag_rejects_empty_output_directory(monkeypatch):
    monkeypatch.setenv("POLAR_ROSBAG_ENABLED", "true")
    monkeypatch.setenv("POLAR_ROSBAG_DIR", " ")

    with pytest.raises(ValueError, match="POLAR_ROSBAG_DIR"):
        polar_rosbag_command()


@pytest.mark.parametrize(
    "topic",
    [
        "/polar/fatigue/xgboost",
        "/polar/fatigue/logistic_regression",
        "/empatica/gmm/predictions",
        "/empatica/gmm/status",
        "/person/detected",
        "/person/detections",
        "/pose/ia/detected",
        "/pose/heuristic/detected",
        "/pose/detected",
    ],
)
def test_polar_rosbag_regex_includes_requested_topics(topic):
    assert re.fullmatch(ROSBAG_TOPIC_REGEX, topic)


@pytest.mark.parametrize(
    "topic",
    [
        "/camera/image_raw",
        "/person/processed_image",
        "/pose/ia/processed_image",
        "/pose_heuristic/processed_image",
        "/empatica/raw",
    ],
)
def test_polar_rosbag_regex_excludes_unrequested_topics(topic):
    assert not re.fullmatch(ROSBAG_TOPIC_REGEX, topic)


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
def test_camera_raw_topic_is_added_when_enabled(monkeypatch, value):
    monkeypatch.setenv("POLAR_ROSBAG_ENABLED", "true")
    monkeypatch.setenv("POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED", value)

    command = polar_rosbag_command()
    topic_regex = command[command.index("--regex") + 1]

    assert re.fullmatch(topic_regex, "/camera/image_raw")
    assert re.fullmatch(topic_regex, "/person/detected")
    assert not re.fullmatch(topic_regex, "/person/processed_image")


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_camera_raw_topic_is_not_added_when_disabled(monkeypatch, value):
    monkeypatch.setenv("POLAR_ROSBAG_ENABLED", "true")
    if value == "":
        monkeypatch.delenv("POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED", raising=False)
    else:
        monkeypatch.setenv("POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED", value)

    command = polar_rosbag_command()
    topic_regex = command[command.index("--regex") + 1]

    assert not re.fullmatch(topic_regex, "/camera/image_raw")
    assert re.fullmatch(topic_regex, "/person/detected")
