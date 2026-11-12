"""Tests for timestamped raw Polar CSV recording."""

import csv

from camera_processor.helpers.polar_csv_recorder import (
    POLAR_EPOCH_UNIX_NS,
    PolarCsvRecorder,
)


def test_records_aligned_samples_and_unmatched_hr(tmp_path):
    recorder = PolarCsvRecorder(tmp_path, "test subject")
    sensor_end_ns = 2_000_000_000
    expected_end_ns = POLAR_EPOCH_UNIX_NS + sensor_end_ns + 10_000_000_000

    recorder.record_ecg(
        expected_end_ns, sensor_end_ns, [-553] * 260
    )
    recorder.record_hr(expected_end_ns - 100_000_000, 84, [800])
    recorder.record_hr(expected_end_ns + 5_000_000_000, 70, [])
    recorder.close()

    with recorder.path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        "UNIX Timestamp",
        "ECG",
        "HR",
        "RR",
        "Marker",
        "Participant Name",
        "Polar Device ID",
    ]
    ecg_rows = [row for row in rows if row["ECG"]]
    assert len(ecg_rows) == 260
    assert all(float(row["ECG"]) == -0.553 for row in ecg_rows)
    timestamps = [int(row["UNIX Timestamp"]) for row in ecg_rows]
    assert timestamps[-1] == expected_end_ns
    assert timestamps[0] == expected_end_ns - round(259 * 1_000_000_000 / 130)

    hr_rows = [row for row in rows if row["HR"]]
    assert len(hr_rows) == 2
    assert any(row["HR"] == "84" for row in ecg_rows)
    assert any(row["RR"] == "800.0" for row in ecg_rows)
    hr_timestamp = min(
        (timestamp for row, timestamp in zip(ecg_rows, timestamps) if row["HR"]),
        key=lambda timestamp: abs(timestamp - (expected_end_ns - 100_000_000)),
    )
    rr_timestamp = min(
        (timestamp for row, timestamp in zip(ecg_rows, timestamps) if row["RR"]),
        key=lambda timestamp: abs(timestamp - (expected_end_ns - 900_000_000)),
    )
    assert hr_timestamp == min(
        timestamps, key=lambda timestamp: abs(timestamp - (expected_end_ns - 100_000_000))
    )
    assert rr_timestamp == min(
        timestamps, key=lambda timestamp: abs(timestamp - (expected_end_ns - 900_000_000))
    )
    assert any(row["HR"] == "70" and not row["ECG"] for row in rows)
    assert all(not row["Marker"] for row in rows)
    assert all(row["Participant Name"] == "test subject" for row in rows)
    assert all(not row["Polar Device ID"] for row in rows)


def test_logs_include_worker_and_polar_identity_in_file_and_rows(tmp_path):
    recorder = PolarCsvRecorder(
        tmp_path,
        "maria-lopez_0123456789abcdef",
        participant_name="Maria Lopez",
        polar_device_id="AA:BB:CC:DD:EE:01",
    )
    recorder.record_ecg(1_700_000_000_000_000_000, None, [125, -50])
    recorder.close()

    assert "MariaLopez_AABBCCDDEE01" in recorder.path.name
    with recorder.path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert all(row["Participant Name"] == "Maria Lopez" for row in rows)
    assert all(row["Polar Device ID"] == "AA:BB:CC:DD:EE:01" for row in rows)


def test_bridge_logs_use_generic_polar_identity(tmp_path):
    recorder = PolarCsvRecorder(
        tmp_path,
        "polar1_24-AC-AC-01-8E-82",
        participant_name="POLAR1",
        polar_device_id="24:AC:AC:01:8E:82",
    )
    recorder.record_ecg(1_700_000_000_000_000_000, None, [125])
    recorder.close()

    assert recorder.path.name.startswith("POLAR1_24ACAC018E82_")
    with recorder.path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["Participant Name"] == "POLAR1"
    assert rows[0]["Polar Device ID"] == "24:AC:AC:01:8E:82"
