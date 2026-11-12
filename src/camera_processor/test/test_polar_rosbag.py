import re

import pytest

from camera_processor.helpers.polar_rosbag import (
    POLAR_ROSBAG_DIR,
    POLAR_TOPIC_REGEX,
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

    command = polar_rosbag_command()

    assert command[:4] == ["ros2", "bag", "record", "--output"]
    assert command[5:] == ["--regex", POLAR_TOPIC_REGEX]
    assert command[4].startswith(f"{POLAR_ROSBAG_DIR}/polar_")
    assert re.fullmatch(r".*/polar_\d{8}_\d{6}", command[4])
