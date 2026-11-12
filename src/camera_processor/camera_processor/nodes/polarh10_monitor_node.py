"""ROS 2 Polar H10 inference node receiving samples from the Windows BLE bridge."""

import asyncio
import json
import math
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from camera_interfaces.msg import FatiguePrediction
from rclpy.node import Node

from camera_processor.helpers.polar_fatigue import (
    FatigueMonitor,
    Predictor,
    WindowBuffer,
)
from camera_processor.helpers.polar_csv_recorder import (
    PolarCsvRecorder,
    PolarCsvRecorderError,
)
from camera_processor.helpers.polar_discovery import (
    DEFAULT_DISCOVERY_PORT,
    make_discovery_response,
)
from camera_processor.helpers.polar_identity import make_polar_subject

AVAILABLE_MODELS = {
    "logistic_regression": "LOGISTIC_REGRESSION",
    "random_forest": "RANDOM_FOREST",
    "svm": "SVM",
    "xgboost": "XGBOOST",
}


@dataclass
class PolarClientSession:
    participant_name: str
    polar_device_id: str
    subject: str
    buffer: WindowBuffer
    monitor: FatigueMonitor
    recorder: PolarCsvRecorder | None
    started: float
    last_hr: float | None = None
    last_ecg: float | None = None


class PolarH10MonitorNode(Node):
    """Receive Polar samples over TCP, infer fatigue and publish ROS topics."""

    def __init__(self):
        super().__init__("polarh10_monitor")

        package_share = Path(get_package_share_directory("camera_processor"))
        default_models_root = package_share / "models" / "polar"
        default_baseline_dir = Path(
            os.getenv(
                "POLAR_BASELINE_DIR",
                "/workspaces/ros2_ws/.cache/polar_baselines",
            )
        )
        self.declare_parameter("subject", "POLAR1")
        self.declare_parameter("recalibrate", False)
        default_models = [
            model.strip().lower()
            for model in os.getenv(
                "POLAR_MODELS", ",".join(AVAILABLE_MODELS)
            ).split(",")
            if model.strip()
        ]
        self.declare_parameter("models", default_models)
        self.declare_parameter("models_root", str(default_models_root))
        self.declare_parameter("baseline_dir", str(default_baseline_dir))
        self.declare_parameter(
            "polar_log_dir",
            os.getenv(
                "POLAR_LOG_DIR", "/workspaces/ros2_ws/.cache/polar_logs"
            ),
        )
        self.declare_parameter("tcp_host", "0.0.0.0")
        self.declare_parameter("tcp_port", 8765)
        self.declare_parameter(
            "input_mode", os.getenv("POLAR_INPUT_MODE", "bridge")
        )
        self.declare_parameter("discovery_port", DEFAULT_DISCOVERY_PORT)

        self.subject = self.get_parameter("subject").value
        recalibrate = bool(self.get_parameter("recalibrate").value)
        models = list(self.get_parameter("models").value)
        models_root = Path(self.get_parameter("models_root").value)
        baseline_dir = Path(self.get_parameter("baseline_dir").value)
        self.baseline_dir = baseline_dir
        self.polar_log_dir = Path(self.get_parameter("polar_log_dir").value)
        tcp_host = str(self.get_parameter("tcp_host").value)
        tcp_port = int(self.get_parameter("tcp_port").value)
        input_mode = str(self.get_parameter("input_mode").value).strip().lower()
        self.input_mode = input_mode
        discovery_port = int(self.get_parameter("discovery_port").value)

        unknown = sorted(set(models) - set(AVAILABLE_MODELS))
        if unknown:
            raise ValueError(
                f"Modelos desconocidos: {', '.join(unknown)}. "
                f"Opciones: {', '.join(sorted(AVAILABLE_MODELS))}"
            )
        if not models:
            raise ValueError("El parametro models necesita al menos un modelo.")
        if len(set(models)) != len(models):
            raise ValueError("El parametro models no debe repetir modelos.")
        if not 1 <= tcp_port <= 65535:
            raise ValueError("tcp_port debe estar entre 1 y 65535.")
        if input_mode not in {"bridge", "android"}:
            raise ValueError(
                "POLAR_INPUT_MODE debe ser 'bridge' o 'android'."
            )
        if not 1 <= discovery_port <= 65535:
            raise ValueError("discovery_port debe estar entre 1 y 65535.")

        baseline_dir.mkdir(parents=True, exist_ok=True)
        model_dirs = {
            name: models_root / AVAILABLE_MODELS[name] for name in models
        }
        self._model_dirs = model_dirs
        self._recalibrate = recalibrate
        self._predictors = {
            name: Predictor(model_dir, name=name)
            for name, model_dir in model_dirs.items()
        }
        self._legacy_monitor = None
        if input_mode == "bridge":
            self._legacy_monitor = FatigueMonitor(
                model_dirs,
                baseline_dir,
                self.subject,
                recalibrate=recalibrate,
                logger=lambda message: self.get_logger().info(message),
                predictors=self._predictors,
            )
        self._model_publishers = {
            name: self.create_publisher(
                FatiguePrediction, f"/polar/fatigue/{name}", 10
            )
            for name in models
        }
        self._predictions = queue.Queue()
        self._active_sessions = {}
        self._publisher_timer = self.create_timer(0.1, self._publish_pending)
        self._stop_event = threading.Event()
        self._tcp_server_ready = threading.Event()
        self._server_thread = threading.Thread(
            target=self._run_tcp_server,
            args=(tcp_host, tcp_port),
            name="polarh10_tcp_server",
            daemon=True,
        )
        self._server_thread.start()
        self._discovery_thread = None
        if input_mode == "android":
            self._discovery_thread = threading.Thread(
                target=self._run_discovery_server,
                args=(discovery_port, tcp_port),
                name="polarh10_discovery_server",
                daemon=True,
            )
            self._discovery_thread.start()
        self.get_logger().info(
            f"polarh10_monitor activo con modelos: {', '.join(models)}. "
            f"Modo de entrada: {input_mode}. Receptor TCP: {tcp_host}:{tcp_port}."
        )
        if self._legacy_monitor is not None:
            self.get_logger().info(
                f"Baseline del puente existente: {self._legacy_monitor.path}."
            )
        if input_mode == "android":
            self.get_logger().info(
                f"Descubrimiento UDP habilitado en el puerto {discovery_port}."
            )

    def _run_tcp_server(self, host, port):
        try:
            asyncio.run(self._tcp_server(host, port))
        except Exception as error:
            if not self._stop_event.is_set():
                self.get_logger().error(f"Servidor TCP Polar detenido: {error}")

    async def _tcp_server(self, host, port):
        server = await asyncio.start_server(
            self._handle_client, host, port, limit=65536
        )
        sockets = server.sockets or []
        addresses = ", ".join(str(sock.getsockname()) for sock in sockets)
        self.get_logger().info(f"Servidor TCP Polar escuchando en {addresses}.")
        self._tcp_server_ready.set()
        async with server:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.25)
        server.close()
        await server.wait_closed()

    def _run_discovery_server(self, discovery_port, tcp_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as discovery_socket:
                discovery_socket.bind(("0.0.0.0", discovery_port))
                discovery_socket.settimeout(0.5)
                self.get_logger().info(
                    f"Descubrimiento Polar escuchando en UDP 0.0.0.0:{discovery_port}."
                )
                while not self._stop_event.is_set():
                    try:
                        payload, peer = discovery_socket.recvfrom(1024)
                    except socket.timeout:
                        continue
                    except OSError:
                        if not self._stop_event.is_set():
                            raise
                        break

                    response = make_discovery_response(payload, tcp_port)
                    if response is None or not self._tcp_server_ready.is_set():
                        continue

                    discovery_socket.sendto(response, peer)
        except OSError as error:
            if not self._stop_event.is_set():
                self.get_logger().error(
                    f"Servidor de descubrimiento Polar detenido: {error}"
                )

    async def _handle_client(self, reader, writer):
        peer = writer.get_extra_info("peername")
        session = None
        loop = asyncio.get_running_loop()
        self.get_logger().info(f"Cliente Polar conectado desde {peer}.")
        try:
            while not self._stop_event.is_set():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    line = None
                if line == b"":
                    break
                if line:
                    event = self._parse_event(line)
                    identity = (
                        event["participant_name"],
                        event["polar_device_id"],
                    )
                    if session is None:
                        session = self._create_client_session(
                            identity[0], identity[1], writer, loop.time()
                        )
                    elif identity != (None, None) and identity != (
                        session.participant_name,
                        session.polar_device_id,
                    ):
                        raise ValueError(
                            "La identidad del cliente cambio durante la sesion."
                        )
                    now = loop.time()
                    if event["type"] == "hr":
                        session.buffer.add_hr(now, event["hr"], event["rr_ms"])
                        if session.recorder is not None:
                            session.recorder.record_hr(
                                time.time_ns(), event["hr"], event["rr_ms"]
                            )
                        session.last_hr = now
                    else:
                        session.buffer.add_ecg(
                            now, event["sensor_end"], event["samples_uv"]
                        )
                        if session.recorder is not None:
                            session.recorder.record_ecg(
                                time.time_ns(),
                                event["sensor_end_ns"],
                                event["samples_uv"],
                            )
                        session.last_ecg = now

                if session is not None:
                    now = loop.time()
                    self._check_streams(
                        now, session.started, session.last_hr, session.last_ecg
                    )
                    for ecg, hr, rr, quality in session.buffer.pop_ready(now):
                        predictions = await asyncio.to_thread(
                            session.monitor.process, ecg, hr, rr, quality
                        )
                        for prediction in predictions:
                            self._predictions.put(
                                (
                                    session.participant_name,
                                    session.polar_device_id,
                                    session.subject,
                                    prediction,
                                )
                            )
        except (ValueError, TypeError, KeyError) as error:
            self.get_logger().error(f"Trama invalida del cliente Polar {peer}: {error}")
        except (ConnectionError, OSError) as error:
            self.get_logger().warning(
                f"Conexion Polar perdida desde {peer}: {error}"
            )
        except PolarCsvRecorderError as error:
            self.get_logger().error(str(error))
        finally:
            if session is not None and session.recorder is not None:
                try:
                    session.recorder.close()
                    self.get_logger().info(
                        f"CSV de {session.participant_name} "
                        f"({session.polar_device_id}) cerrado: "
                        f"{session.recorder.path}."
                    )
                except PolarCsvRecorderError as error:
                    self.get_logger().error(str(error))
            if (
                session is not None
                and self._active_sessions.get(session.subject) is writer
            ):
                del self._active_sessions[session.subject]
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self.get_logger().warning(
                f"Cliente Polar desconectado ({peer}); "
                "el nodo sigue esperando conexiones."
            )

    def _create_client_session(
        self, participant_name, polar_device_id, writer, started
    ):
        legacy_session = participant_name is None and polar_device_id is None
        if legacy_session:
            if self.input_mode == "android":
                raise ValueError(
                    "La app debe enviar el nombre y el ID del Polar H10."
                )
            participant_name = self.subject
            polar_device_id = "bridge"
            subject = self.subject
        else:
            if participant_name is None or polar_device_id is None:
                raise ValueError(
                    "Cada trama debe incluir nombre e ID del Polar."
                )
            subject, participant_name, polar_device_id = make_polar_subject(
                participant_name, polar_device_id
            )

        if subject in self._active_sessions:
            raise ValueError(
                f"Ya hay una conexión activa para {participant_name} "
                f"({polar_device_id})."
            )

        if legacy_session:
            monitor = self._legacy_monitor
            if monitor is None:
                raise RuntimeError("El monitor del puente BLE no está disponible.")
        else:
            monitor = FatigueMonitor(
                self._model_dirs,
                self.baseline_dir,
                subject,
                recalibrate=self._recalibrate,
                logger=lambda message: self.get_logger().info(
                    f"[{participant_name} | {polar_device_id}] {message}"
                ),
                predictors=self._predictors,
            )

        recorder = None
        try:
            recorder = PolarCsvRecorder(
                self.polar_log_dir,
                subject,
                participant_name=participant_name,
                polar_device_id=polar_device_id,
            )
            self.get_logger().info(
                f"Guardando datos de {participant_name} ({polar_device_id}) "
                f"en {recorder.path}. Baseline: {monitor.path}."
            )
        except PolarCsvRecorderError as error:
            self.get_logger().error(str(error))

        session = PolarClientSession(
            participant_name=participant_name,
            polar_device_id=polar_device_id,
            subject=subject,
            buffer=WindowBuffer(),
            monitor=monitor,
            recorder=recorder,
            started=started,
        )
        self._active_sessions[subject] = writer
        return session

    @staticmethod
    def _parse_event(line):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("El evento debe ser un objeto JSON.")
        participant_name = event.get("participant_name")
        polar_device_id = event.get("polar_device_id")
        if participant_name is not None or polar_device_id is not None:
            _, participant_name, polar_device_id = make_polar_subject(
                participant_name, polar_device_id
            )
        event_type = event.get("type")
        if event_type == "hr":
            hr = float(event["hr"])
            rr_ms = [float(value) for value in event["rr_ms"]]
            if not math.isfinite(hr) or hr <= 0:
                raise ValueError("Valor HR invalido.")
            if any(not math.isfinite(value) or value <= 0 for value in rr_ms):
                raise ValueError("Valor R-R invalido.")
            return {
                "type": "hr",
                "hr": hr,
                "rr_ms": rr_ms,
                "participant_name": participant_name,
                "polar_device_id": polar_device_id,
            }
        if event_type == "ecg":
            sensor_end = float(event["sensor_end"])
            sensor_end_ns = event.get("sensor_end_ns")
            samples = event["samples_uv"]
            if not math.isfinite(sensor_end) or not isinstance(samples, list):
                raise ValueError("Evento ECG invalido.")
            if sensor_end_ns is not None and (
                not isinstance(sensor_end_ns, int)
                or isinstance(sensor_end_ns, bool)
                or sensor_end_ns < 0
            ):
                raise ValueError("Timestamp ECG invalido.")
            if not samples or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in samples
            ):
                raise ValueError("Muestras ECG invalidas.")
            return {
                "type": "ecg",
                "sensor_end": sensor_end,
                "sensor_end_ns": sensor_end_ns,
                "samples_uv": samples,
                "participant_name": participant_name,
                "polar_device_id": polar_device_id,
            }
        raise ValueError(f"Tipo de evento desconocido: {event_type!r}.")

    def _check_streams(self, now, started, last_hr, last_ecg):
        missing = []
        for name, last in (("HR", last_hr), ("ECG", last_ecg)):
            if last is None and now - started > 60:
                missing.append(f"{name}: no se recibio ninguna muestra en 60 s")
            elif last is not None and now - last > 10:
                missing.append(
                    f"{name}: ultima notificacion hace {now - last:.1f} s"
                )
        if missing:
            raise ConnectionError(
                "; ".join(missing)
                + ". Comprueba el Polar H10 y su conexion Bluetooth."
            )

    def _publish_pending(self):
        while True:
            try:
                participant_name, polar_device_id, subject, prediction = (
                    self._predictions.get_nowait()
                )
            except queue.Empty:
                return
            message = FatiguePrediction()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = subject
            message.subject = subject
            message.participant_name = participant_name
            message.polar_device_id = polar_device_id
            message.model = prediction.model
            message.label = prediction.label
            message.probability_low = prediction.probabilities[0]
            message.probability_medium = prediction.probabilities[1]
            message.probability_high = prediction.probabilities[2]
            message.imputed_features = prediction.imputed
            message.history_minutes = prediction.history_minutes
            self._model_publishers[prediction.model].publish(message)

    def destroy_node(self):
        self._stop_event.set()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        if (
            self._discovery_thread is not None
            and self._discovery_thread.is_alive()
        ):
            self._discovery_thread.join(timeout=2)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PolarH10MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
