"""Training-compatible features, personal calibration and ONNX inference.

Ported from the standalone Polar H10 scripts (fatigue_inference.py) with
support for several models sharing a single personal baseline.
"""

import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from antropy import sample_entropy
from scipy.signal import find_peaks

WINDOW_SECONDS = 60
SAMPLE_RATE_HZ = 130
BASELINE_WINDOWS = 15
BASE_FEATURES = (
    "ECG_mean", "ECG_std", "ECG_range", "ECG_energy",
    "ECG_samp_ent", "ECG_missing_peaks", "SDNN", "RMSSD", "pNN50",
    "HR_mean", "HR_max", "HR_min",
)
TEMPORAL_FEATURES = (
    "HR_mean_ultimos_5min", "cambio_HR_vs_baseline",
    "tendencia_HR", "cambio_RMSSD",
)
FEATURES = BASE_FEATURES + TEMPORAL_FEATURES
LABELS = ("Low", "Medium", "High")


def compute_features(ecg, hr, rr, logger=print):
    """Match process_all_workers_tempvar.py (ECG in mV, R-R in ms)."""
    result = dict.fromkeys(BASE_FEATURES, math.nan)
    if not ecg:
        return result
    count = len(ecg)
    mean = sum(ecg) / count
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in ecg) / (count - 1))
        if count > 1 else 0
    )
    result.update(
        ECG_mean=mean, ECG_std=std, ECG_range=max(ecg) - min(ecg),
        ECG_energy=sum(value ** 2 for value in ecg) / count,
    )
    try:
        entropy = float(sample_entropy(ecg))
        if not math.isfinite(entropy):
            logger("Entropia ECG no finita; se usa 0, como en entrenamiento.")
            entropy = 0.0
    except (ValueError, ZeroDivisionError, FloatingPointError) as error:
        logger(
            "No se pudo calcular entropia ECG; se usa 0 como en "
            f"entrenamiento: {error}"
        )
        entropy = 0.0
    result["ECG_samp_ent"] = entropy
    if hr and sum(hr) > 0:
        input_mean = sum(hr) / len(hr)
        peaks, _ = find_peaks(
            ecg, height=mean + 0.5 * std if std > 0 else 0,
            distance=max(1, int(SAMPLE_RATE_HZ / (input_mean / 60))),
        )
        expected_beats = (input_mean / 60) * WINDOW_SECONDS
        result["ECG_missing_peaks"] = float(max(0, int(expected_beats) - len(peaks)))
    rr = [value for value in rr if value > 0 and math.isfinite(value)]
    if rr:
        rr_mean = sum(rr) / len(rr)
        differences = [right - left for left, right in zip(rr, rr[1:])]
        result.update(
            SDNN=math.sqrt(sum((value - rr_mean) ** 2 for value in rr) / len(rr)),
            RMSSD=(
                math.sqrt(sum(value ** 2 for value in differences) / len(differences))
                if differences else 0
            ),
            pNN50=(
                100 * sum(abs(value) > 50 for value in differences) / len(differences)
                if differences else 0
            ),
            HR_mean=60000 / rr_mean,
            HR_max=max(60000 / value for value in rr),
            HR_min=min(60000 / value for value in rr),
        )
    return result


@dataclass(frozen=True)
class Statistic:
    """Baseline statistics of one feature."""

    count: int
    mean: float
    std: float


@dataclass(frozen=True)
class Prediction:
    """Inference result of one model for one 60 s window."""

    model: str
    label: str
    probabilities: tuple
    imputed: int
    history_minutes: int


class Baseline:
    """Personal z-score baseline built from 15 valid 60 s windows."""

    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def calculate(cls, rows):
        """Build baseline statistics from exactly 15 valid windows."""
        if len(rows) != BASELINE_WINDOWS:
            raise ValueError("La baseline requiere exactamente 15 ventanas validas.")
        statistics = {}
        for name in FEATURES:
            values = [row[name] for row in rows if math.isfinite(row[name])]
            mean = sum(values) / len(values) if values else math.nan
            std = (
                math.sqrt(
                    sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                )
                if len(values) > 1 else math.nan
            )
            statistics[name] = Statistic(len(values), mean, std)
        return cls(statistics)

    @classmethod
    def load(cls, path, subject):
        """Load and validate a baseline CSV for the given subject."""
        statistics = {}
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["Trabajador"] != subject:
                    raise ValueError(f"Baseline de otra persona: {path}")
                name = row["Feature"]
                if name in statistics:
                    raise ValueError(f"Caracteristica duplicada en baseline: {name}")
                statistic = Statistic(
                    int(row["Baseline_n"]),
                    float(row["Baseline_mean"]),
                    float(row["Baseline_std"]),
                )
                if not 0 <= statistic.count <= BASELINE_WINDOWS:
                    raise ValueError(f"Recuento invalido en baseline: {name}")
                if name in BASE_FEATURES and (
                    statistic.count != BASELINE_WINDOWS
                    or not math.isfinite(statistic.mean)
                ):
                    raise ValueError(f"Baseline incompleta: {name}")
                if statistic.count > 1 and (
                    not math.isfinite(statistic.std) or statistic.std < 0
                ):
                    raise ValueError(f"Desviacion invalida: {name}")
                if statistic.count and not math.isfinite(statistic.mean):
                    raise ValueError(f"Media invalida: {name}")
                statistics[name] = statistic
        if set(statistics) != set(FEATURES):
            raise ValueError(
                f"Las caracteristicas de la baseline no coinciden: {path}"
            )
        return cls(statistics)

    def save(self, path, subject):
        """Persist the baseline CSV atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Trabajador", "Feature", "Baseline_n",
                     "Baseline_mean", "Baseline_std"]
                )
                for name, statistic in self.statistics.items():
                    writer.writerow(
                        [subject, name, statistic.count,
                         statistic.mean, statistic.std]
                    )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def normalize(self, row, columns):
        """Z-score the row against the baseline in the given column order."""
        values = []
        for column in columns:
            name = column.removesuffix("_z")
            statistic = self.statistics[name]
            value = row[name]
            values.append(
                (value - statistic.mean) / statistic.std
                if math.isfinite(value)
                and math.isfinite(statistic.std)
                and statistic.std > 0
                else math.nan
            )
        return np.asarray([values], dtype=np.float64)


class Predictor:
    """ONNX classifier with training imputer medians from its metadata."""

    def __init__(self, model_dir, name=None):
        self.name = name or model_dir.name
        metadata_files = sorted(model_dir.glob("*_metadata.json"))
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if len(metadata_files) != 1 or len(onnx_files) != 1:
            raise ValueError(
                "Se esperaba exactamente un *_metadata.json y un .onnx en "
                f"{model_dir}"
            )
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        self.columns = metadata["feature_columns"]
        self.labels = metadata["labels"]
        imputer_medians = metadata.get("imputer_medians")
        if (
            set(self.columns) != {f"{name}_z" for name in FEATURES}
            or len(self.columns) != len(FEATURES)
            or metadata["feature_count"] != len(FEATURES)
            or self.labels != list(LABELS)
        ):
            raise ValueError("Metadatos incompatibles con el preprocesamiento.")
        if not isinstance(imputer_medians, list):
            raise ValueError("Los metadatos no contienen las medianas del imputador.")
        self.medians = np.asarray(imputer_medians, dtype=np.float64)
        if self.medians.shape != (len(self.columns),) or not np.isfinite(
            self.medians
        ).all():
            raise ValueError("Medianas de imputacion invalidas.")
        self.session = ort.InferenceSession(
            str(onnx_files[0]), providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != [None, len(self.columns)]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError("Entrada ONNX incompatible.")
        self.input_name = inputs[0].name
        outputs = self.session.get_outputs()
        label_outputs = [
            output.name for output in outputs if "label" in output.name.lower()
        ]
        probability_outputs = [
            output.name for output in outputs if "probab" in output.name.lower()
        ]
        if len(label_outputs) != 1 or len(probability_outputs) != 1:
            raise ValueError("Salida ONNX incompatible.")
        self.label_output = label_outputs[0]
        self.probability_output = probability_outputs[0]

    def _label_code(self, label):
        if isinstance(label, (str, np.str_)):
            if label not in self.labels:
                raise ValueError("Salida ONNX invalida.")
            return self.labels.index(label)
        code = int(label)
        if code not in range(len(self.labels)):
            raise ValueError("Salida ONNX invalida.")
        return code

    def _probability_row(self, probabilities):
        if isinstance(probabilities, np.ndarray):
            return np.asarray(probabilities, dtype=np.float64)
        # ZipMap: lista de diccionarios {clase: probabilidad} (skl2onnx).
        row = np.full((1, len(self.labels)), np.nan)
        for key, value in probabilities[0].items():
            row[0, self._label_code(key)] = value
        return row

    def predict(self, baseline, row):
        """Return (label, probabilities, imputed_count) for one window."""
        normalized = baseline.normalize(row, self.columns)
        missing = ~np.isfinite(normalized)
        prepared = np.where(missing, self.medians, normalized).astype(np.float32)
        label, probabilities = self.session.run(
            [self.label_output, self.probability_output],
            {self.input_name: prepared},
        )
        code = self._label_code(np.asarray(label)[0])
        probabilities = self._probability_row(probabilities)
        if (
            probabilities.shape != (1, len(self.labels))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-5)
        ):
            raise ValueError("Salida ONNX invalida.")
        return self.labels[code], probabilities[0], int(missing.sum())


class FatigueMonitor:
    """Calibration and multi-model inference sharing one baseline."""

    def __init__(
        self,
        model_dirs,
        baseline_dir,
        subject,
        recalibrate=False,
        logger=print,
        predictors=None,
    ):
        if not model_dirs:
            raise ValueError("Se necesita al menos un modelo para la inferencia.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", subject):
            raise ValueError(
                "La persona debe identificarse con letras, numeros, guion o "
                "guion bajo."
            )
        self.log = logger
        self.subject = subject
        self.path = Path(baseline_dir) / f"{subject}_baseline.csv"
        self.predictors = predictors or {
            name: Predictor(Path(model_dir), name=name)
            for name, model_dir in model_dirs.items()
        }
        self.baseline = (
            Baseline.load(self.path, subject)
            if self.path.exists() and not recalibrate
            else None
        )
        self.history = deque(maxlen=5)
        self.previous_rmssd = math.nan
        self.calibration = []
        if self.baseline:
            self.log(f"Baseline cargada: {self.path}")
        else:
            self.log(
                "Calibracion: se requieren 15 ventanas validas de 60 s; aun no "
                "hay predicciones."
            )

    def process(self, ecg, hr, rr, quality=True):
        """Process one window; return a Prediction per model (empty if not)."""
        row = compute_features(ecg, hr, rr, logger=self.log)
        self.history.append(row)
        recent = [
            item["HR_mean"] for item in self.history
            if math.isfinite(item["HR_mean"])
        ]
        mean = sum(recent) / len(recent) if recent else math.nan
        x_mean = (len(recent) - 1) / 2
        denominator = sum((index - x_mean) ** 2 for index in range(len(recent)))
        row["HR_mean_ultimos_5min"] = mean
        row["tendencia_HR"] = (
            sum(
                (index - x_mean) * (value - mean)
                for index, value in enumerate(recent)
            ) / denominator
            if denominator else math.nan
        )
        row["cambio_RMSSD"] = row["RMSSD"] - self.previous_rmssd
        if math.isfinite(row["RMSSD"]):
            self.previous_rmssd = row["RMSSD"]
        valid = quality and all(
            math.isfinite(row[name]) for name in BASE_FEATURES
        )
        if not valid:
            self.log(
                "Ventana descartada: ECG/HR/R-R insuficientes o interrupcion "
                "de la senal."
            )
            return []
        if self.baseline is None:
            self.calibration.append(row)
            self.log(
                f"Calibracion: {len(self.calibration)}/{BASELINE_WINDOWS} "
                "ventanas validas"
            )
            if len(self.calibration) == BASELINE_WINDOWS:
                baseline_hr = (
                    sum(item["HR_mean"] for item in self.calibration)
                    / BASELINE_WINDOWS
                )
                for item in self.calibration:
                    item["cambio_HR_vs_baseline"] = item["HR_mean"] - baseline_hr
                self.baseline = Baseline.calculate(self.calibration)
                self.baseline.save(self.path, self.subject)
                self.log(
                    f"Baseline guardada: {self.path}. Predicciones desde la "
                    "siguiente ventana."
                )
                self.calibration.clear()
            return []
        row["cambio_HR_vs_baseline"] = (
            row["HR_mean"] - self.baseline.statistics["HR_mean"].mean
        )
        predictions = []
        for name, predictor in self.predictors.items():
            label, probabilities, imputed = predictor.predict(self.baseline, row)
            detail = " | ".join(
                f"{label_name}={probability:.1%}"
                for label_name, probability in zip(predictor.labels, probabilities)
            )
            self.log(
                f"[{name}] Fatiga estimada: {label} | {detail} | "
                f"imputadas={imputed}/16 | historial={len(self.history)}/5 min"
            )
            predictions.append(
                Prediction(
                    model=name,
                    label=label,
                    probabilities=tuple(float(p) for p in probabilities),
                    imputed=imputed,
                    history_minutes=len(self.history),
                )
            )
        return predictions


class WindowBuffer:
    """Keep each real sample once; ECG time comes from the Polar PMD clock."""

    def __init__(self):
        self.start = None
        self.origin = None
        self.ecg = []
        self.hr = []
        self.rr = []

    def add_hr(self, now, hr, rr):
        """Append an HR notification; R-R beats are spread backwards in time."""
        if self.start is None:
            self.hr[:] = [
                (time, value) for time, value in self.hr
                if time >= now - WINDOW_SECONDS
            ]
            self.rr[:] = [
                (time, value) for time, value in self.rr
                if time >= now - WINDOW_SECONDS
            ]
        self.hr.append((now, hr))
        beat_time = now
        beats = []
        for value in reversed(rr):
            beats.append((beat_time, value))
            beat_time -= value / 1000
        self.rr.extend(reversed(beats))

    def add_ecg(self, now, sensor_end, samples_uv):
        """Append ECG samples (uV) converted to mV on the sensor clock."""
        if not samples_uv:
            raise ValueError("Trama ECG sin muestras.")
        if self.origin is None:
            self.origin = now - sensor_end
            self.start = now - (len(samples_uv) - 1) / SAMPLE_RATE_HZ
        end = self.origin + sensor_end
        # CSVs used in training contain ECG in mV; BLE PMD supplies integer uV.
        self.ecg.extend(
            (end - (len(samples_uv) - 1 - index) / SAMPLE_RATE_HZ, value / 1000)
            for index, value in enumerate(samples_uv)
        )

    def pop_ready(self, now):
        """Return completed 60 s windows as (ecg, hr, rr, quality) tuples."""
        windows = []
        # Leave time for notifications arriving on the other BLE characteristic.
        while self.start is not None and now >= self.start + WINDOW_SECONDS + 2:
            end = self.start + WINDOW_SECONDS
            selected = []
            ecg_times = []
            for stream in (self.ecg, self.hr, self.rr):
                samples = sorted(
                    (time, value) for time, value in stream
                    if self.start <= time < end
                )
                selected.append([value for _, value in samples])
                if stream is self.ecg:
                    ecg_times = [time for time, _ in samples]
                stream[:] = [
                    (time, value) for time, value in stream if time >= end
                ]
            gaps = [right - left for left, right in zip(
                [self.start] + ecg_times, ecg_times + [end],
            )]
            quality = (
                len(selected[0]) >= 0.9 * WINDOW_SECONDS * SAMPLE_RATE_HZ
                and max(gaps) <= 2
            )
            windows.append((selected[0], selected[1], selected[2], quality))
            # The offline processor anchors each window on its first ECG row.
            self.start = (
                min(time for time, _ in self.ecg) if self.ecg else end
            )
        return windows
