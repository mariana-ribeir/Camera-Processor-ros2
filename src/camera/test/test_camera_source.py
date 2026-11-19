import pytest

from camera.camera_simulator import resolve_capture_source


def test_video_source_uses_configured_video_path():
    assert resolve_capture_source(
        "video", "/videos/sample.mp4", "/dev/video0"
    ) == ("video", "/videos/sample.mp4")


def test_camera_source_uses_configured_device():
    assert resolve_capture_source(
        "camera", "/videos/sample.mp4", "/dev/video0"
    ) == ("camera", "/dev/video0")


def test_camera_source_is_case_and_whitespace_insensitive():
    assert resolve_capture_source(
        " Camera ", "/videos/sample.mp4", "/dev/video0"
    ) == ("camera", "/dev/video0")


def test_unknown_camera_source_is_rejected():
    with pytest.raises(ValueError, match="CAMERA_SOURCE"):
        resolve_capture_source(
            "webcam", "/videos/sample.mp4", "/dev/video0"
        )
