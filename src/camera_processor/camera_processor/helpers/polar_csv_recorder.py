"""Write timestamped raw Polar measurements to one CSV per BLE session."""

import csv
from datetime import datetime, timezone
from pathlib import Path

NANOSECONDS_PER_SECOND = 1_000_000_000
POLAR_EPOCH_UNIX_NS = 946_684_800 * NANOSECONDS_PER_SECOND
SAMPLE_RATE_HZ = 130
FLUSH_DELAY_NS = 5 * NANOSECONDS_PER_SECOND
ASSOCIATION_LOOKAHEAD_NS = NANOSECONDS_PER_SECOND // (2 * SAMPLE_RATE_HZ)
MAX_ASSOCIATION_DELTA_NS = 500_000_000
CSV_COLUMNS = (
    "UNIX Timestamp",
    "ECG",
    "HR",
    "RR",
    "Marker",
    "Participant Name",
    "Polar Device ID",
)


class PolarCsvRecorderError(RuntimeError):
    """Raised when a Polar session CSV cannot be written."""


class PolarCsvRecorder:
    """Buffer and align Polar samples before writing timestamp-ordered CSV rows."""

    def __init__(
        self, directory, subject, participant_name=None, polar_device_id=None
    ):
        safe_subject = "".join(
            character
            for character in str(subject)
            if character.isalnum() or character in "-_"
        ) or "polar"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.participant_name = participant_name or str(subject)
        self.polar_device_id = polar_device_id or ""
        filename_identity = safe_subject
        if polar_device_id:
            safe_name = self._safe_component(self.participant_name)
            safe_device = self._safe_component(polar_device_id)
            filename_identity = f"{safe_name}_{safe_device}"
        self.path = Path(directory) / f"{filename_identity}_{timestamp}.csv"
        self._rows = {}
        self._pending_metrics = []
        self._latest_ecg_timestamp_ns = None
        self._sensor_clock_offset_ns = None
        self._closed = False

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
            self._writer.writeheader()
            self._file.flush()
        except OSError as error:
            raise PolarCsvRecorderError(
                f"No se pudo crear el CSV Polar en {self.path}: {error}"
            ) from error

    def record_ecg(self, received_at_ns, sensor_end_ns, samples_uv):
        """Add ECG samples, mapping the Polar clock to the host Unix clock."""
        self._ensure_open()
        if sensor_end_ns is None:
            end_timestamp_ns = received_at_ns
        else:
            sensor_timestamp_ns = sensor_end_ns + POLAR_EPOCH_UNIX_NS
            if self._sensor_clock_offset_ns is None:
                self._sensor_clock_offset_ns = (
                    received_at_ns - sensor_timestamp_ns
                )
            end_timestamp_ns = sensor_timestamp_ns + self._sensor_clock_offset_ns

        last_index = len(samples_uv) - 1
        for index, sample_uv in enumerate(samples_uv):
            timestamp_ns = end_timestamp_ns - round(
                (last_index - index) * NANOSECONDS_PER_SECOND / SAMPLE_RATE_HZ
            )
            row = self._row(timestamp_ns)
            row["ECG"] = float(sample_uv) / 1000.0
            if (
                self._latest_ecg_timestamp_ns is None
                or timestamp_ns > self._latest_ecg_timestamp_ns
            ):
                self._latest_ecg_timestamp_ns = timestamp_ns

        self._match_pending_metrics()
        self._flush_ready_rows()

    def record_hr(self, timestamp_ns, heart_rate, rr_ms):
        """Associate HR and beat-timed R-R readings with nearby ECG samples."""
        self._ensure_open()
        heart_rate = float(heart_rate)
        if heart_rate.is_integer():
            heart_rate = int(heart_rate)
        self._pending_metrics.append((timestamp_ns, "HR", heart_rate))

        beat_timestamp_ns = timestamp_ns
        for interval_ms in reversed(rr_ms):
            beat_timestamp_ns -= round(float(interval_ms) * 1_000_000)
            self._pending_metrics.append(
                (beat_timestamp_ns, "RR", float(interval_ms))
            )

        self._match_pending_metrics()
        self._flush_ready_rows()

    def _row(self, timestamp_ns):
        return self._rows.setdefault(
            timestamp_ns,
            {
                "UNIX Timestamp": timestamp_ns,
                "ECG": "",
                "HR": "",
                "RR": "",
                "Marker": "",
                "Participant Name": self.participant_name,
                "Polar Device ID": self.polar_device_id,
            },
        )

    @staticmethod
    def _safe_component(value):
        return "".join(
            character
            for character in str(value)
            if character.isalnum() or character in "-_"
        )[:48] or "polar"

    def _match_pending_metrics(self, force=False):
        remaining = []
        latest = self._latest_ecg_timestamp_ns
        flush_before = latest - FLUSH_DELAY_NS if latest is not None else None

        for timestamp_ns, column, value in self._pending_metrics:
            has_lookahead = (
                latest is not None
                and latest >= timestamp_ns + ASSOCIATION_LOOKAHEAD_NS
            )
            expired = flush_before is not None and timestamp_ns < flush_before
            if not (force or has_lookahead or expired):
                remaining.append((timestamp_ns, column, value))
                continue

            candidates = [
                candidate
                for candidate, row in self._rows.items()
                if row["ECG"] != "" and row[column] == ""
            ]
            nearest = min(
                candidates,
                key=lambda candidate: abs(candidate - timestamp_ns),
                default=None,
            )
            if (
                nearest is not None
                and abs(nearest - timestamp_ns) <= MAX_ASSOCIATION_DELTA_NS
            ):
                self._rows[nearest][column] = value
            elif force or expired:
                self._row(timestamp_ns)[column] = value
            else:
                remaining.append((timestamp_ns, column, value))

        self._pending_metrics = remaining

    def _flush_ready_rows(self):
        latest = self._latest_ecg_timestamp_ns
        if latest is None:
            return
        cutoff = latest - FLUSH_DELAY_NS
        timestamps = sorted(timestamp for timestamp in self._rows if timestamp < cutoff)
        self._write_rows(timestamps)

    def _write_rows(self, timestamps):
        try:
            for timestamp_ns in timestamps:
                self._writer.writerow(self._rows.pop(timestamp_ns))
            self._file.flush()
        except (OSError, csv.Error) as error:
            raise PolarCsvRecorderError(
                f"No se pudo escribir el CSV Polar {self.path}: {error}"
            ) from error

    def _ensure_open(self):
        if self._closed:
            raise PolarCsvRecorderError("El CSV de la sesion Polar ya esta cerrado.")

    def close(self):
        """Write all remaining rows and close this session's CSV."""
        if self._closed:
            return
        try:
            self._match_pending_metrics(force=True)
            self._write_rows(sorted(self._rows))
        finally:
            try:
                self._file.close()
            except OSError as error:
                raise PolarCsvRecorderError(
                    f"No se pudo cerrar el CSV Polar {self.path}: {error}"
                ) from error
            finally:
                self._closed = True
