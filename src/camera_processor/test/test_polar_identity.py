import pytest

from camera_processor.helpers.polar_identity import make_polar_subject


def test_subject_is_stable_for_the_same_worker_and_sensor():
    first = make_polar_subject("Maria Lopez", "AA:BB:CC:DD:EE:01")
    repeated = make_polar_subject("Maria Lopez", "AA:BB:CC:DD:EE:01")

    assert first == repeated
    assert first[0] == "maria-lopez_AA-BB-CC-DD-EE-01"
    assert first[1:] == ("Maria Lopez", "AA:BB:CC:DD:EE:01")


@pytest.mark.parametrize(
    ("name", "device_id", "expected"),
    [
        (
            "POLAR1",
            "24:AC:AC:01:8E:82",
            "polar1_24-AC-AC-01-8E-82",
        ),
        (
            "POLAR2",
            "A0:9E:1A:EA:3A:E4",
            "polar2_A0-9E-1A-EA-3A-E4",
        ),
    ],
)
def test_bridge_profile_baseline_keys(name, device_id, expected):
    subject, _, _ = make_polar_subject(name, device_id)

    assert subject == expected


@pytest.mark.parametrize(
    ("name", "device_id"),
    [
        ("Juan Perez", "AA:BB:CC:DD:EE:01"),
        ("Maria Lopez", "AA:BB:CC:DD:EE:02"),
    ],
)
def test_subject_separates_worker_and_sensor_pairs(name, device_id):
    expected = make_polar_subject("Maria Lopez", "AA:BB:CC:DD:EE:01")[0]
    actual = make_polar_subject(name, device_id)[0]

    assert actual != expected


@pytest.mark.parametrize(
    ("name", "device_id"),
    [
        ("", "AA:BB:CC:DD:EE:01"),
        ("Worker", ""),
        ("Worker", ":::"),
        ("x" * 81, "AA:BB:CC:DD:EE:01"),
        ("Worker", "x" * 129),
        ("Worker\nName", "AA:BB:CC:DD:EE:01"),
    ],
)
def test_subject_rejects_invalid_identity(name, device_id):
    with pytest.raises(ValueError):
        make_polar_subject(name, device_id)
