"""Forward Polar H10 BLE measurements from Windows to the ROS 2 Docker node."""

import argparse
import asyncio
import json
import os
import struct
from pathlib import Path

from bleak import BleakClient, BleakScanner

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
PMD_CONTROL_UUID = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA_UUID = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
ECG_START_CMD = bytearray(
    [0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00]
)
ECG_STOP_CMD = bytearray([0x03, 0x00])


def decode_hr(data):
    """Decode HR and R-R intervals from the standard 0x2A37 characteristic."""
    if len(data) < 2:
        raise ValueError("Trama HR incompleta.")
    flags = data[0]
    if flags & 0x01:
        hr = struct.unpack_from("<H", data, 1)[0]
        index = 3
    else:
        hr = data[1]
        index = 2
    if flags & 0x08:
        index += 2

    rr_ms = []
    if flags & 0x10:
        if index >= len(data) or (len(data) - index) % 2:
            raise ValueError("Trama R-R incompleta.")
        while index + 1 < len(data):
            rr_1024 = struct.unpack_from("<H", data, index)[0]
            rr_ms.append(rr_1024 / 1024.0 * 1000.0)
            index += 2
    return hr, rr_ms


def decode_ecg(data):
    """Decode a Polar PMD ECG frame into sensor time and signed uV samples."""
    if not data:
        raise ValueError("Trama PMD vacia.")
    if data[0] != 0x00:
        return None
    if len(data) < 13 or (len(data) - 10) % 3 or data[9] != 0:
        raise ValueError("Trama ECG incompleta o formato PMD no soportado.")
    sensor_end_ns = int.from_bytes(data[1:9], "little")
    sensor_end = sensor_end_ns / 1e9
    samples = [
        int.from_bytes(data[index:index + 3], "little", signed=True)
        for index in range(10, len(data) - 2, 3)
    ]
    return sensor_end, samples, sensor_end_ns


def make_stream_event(
    event_type, fields, participant_name=None, polar_device_id=None
):
    if (participant_name is None) != (polar_device_id is None):
        raise ValueError("El nombre y el ID del Polar deben enviarse juntos.")
    event = {"type": event_type, **fields}
    if participant_name is not None:
        event["participant_name"] = participant_name
        event["polar_device_id"] = polar_device_id
    return event


def load_env_file(path):
    """Carga pares KEY=VALUE del .env sin reemplazar variables existentes."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def parse_bool_env(name, default=False):
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} debe ser true o false.")


def load_device_profiles():
    profiles = {}
    for participant_name in ("POLAR1", "POLAR2"):
        key = participant_name.lower()
        profiles[key] = {
            "address": os.getenv(f"{participant_name}_ADDRESS", "").strip(),
            "participant_name": participant_name,
            "pair": parse_bool_env(f"{participant_name}_PAIR"),
        }
    return profiles


async def select_device(address, scan_timeout, scan_lock=None):
    """Find one configured Polar by its BLE address."""
    if scan_lock is None:
        discovered = await BleakScanner.discover(timeout=scan_timeout)
    else:
        async with scan_lock:
            discovered = await BleakScanner.discover(timeout=scan_timeout)
    return next(
        (
            device
            for device in discovered
            if device.address.lower() == address.lower()
        ),
        None,
    )


async def forward_session(args, device, profile, polar_device_id):
    _, writer = await asyncio.open_connection(args.host, args.port)
    queue = asyncio.Queue()
    errors = asyncio.Queue()
    loop = asyncio.get_running_loop()
    start_response = loop.create_future()
    participant_name = profile["participant_name"] or None

    async def send_events():
        while True:
            event = await queue.get()
            writer.write(json.dumps(event, separators=(",", ":")).encode() + b"\n")
            await writer.drain()

    send_task = asyncio.create_task(send_events())

    def receive_control(_, data):
        if len(data) >= 3 and data[:3] == bytearray([0xF0, 0x02, 0x00]):
            if len(data) < 5:
                if not start_response.done():
                    start_response.set_exception(
                        ValueError("Respuesta PMD de inicio ECG incompleta.")
                    )
            elif not start_response.done():
                start_response.set_result(data[3])

    def receive_hr(_, data):
        try:
            hr, rr_ms = decode_hr(data)
            queue.put_nowait(
                make_stream_event(
                    "hr",
                    {"hr": hr, "rr_ms": rr_ms},
                    participant_name,
                    polar_device_id if participant_name else None,
                )
            )
        except (ValueError, IndexError, struct.error) as error:
            errors.put_nowait(error)

    def receive_ecg(_, data):
        try:
            decoded = decode_ecg(data)
            if decoded is not None:
                sensor_end, samples, sensor_end_ns = decoded
                queue.put_nowait(
                    make_stream_event(
                        "ecg",
                        {
                            "sensor_end": sensor_end,
                            "sensor_end_ns": sensor_end_ns,
                            "samples_uv": samples,
                        },
                        participant_name,
                        polar_device_id if participant_name else None,
                    )
                )
        except (ValueError, IndexError, struct.error) as error:
            errors.put_nowait(error)

    print(
        f"Conectando {participant_name or profile['participant_name']} "
        f"a {device.name or device.address}..."
    )
    if profile["pair"]:
        print("Emparejando el Polar antes de conectar (BLE authentication)...")
    try:
        async with BleakClient(
            device, pair=profile["pair"], timeout=30.0
        ) as client:
            try:
                print(
                    f"Conectado al Polar H10. Enviando datos a "
                    f"{args.host}:{args.port}."
                )
                print("Activando notificaciones de ritmo cardiaco...")
                await client.start_notify(HR_MEASUREMENT_UUID, receive_hr)
                print("Activando notificaciones del servicio ECG...")
                await client.start_notify(PMD_CONTROL_UUID, receive_control)
                await client.start_notify(PMD_DATA_UUID, receive_ecg)
                print("Solicitando inicio del stream ECG...")
                await client.write_gatt_char(
                    PMD_CONTROL_UUID, ECG_START_CMD, response=True
                )
                try:
                    status = await asyncio.wait_for(start_response, timeout=15)
                except asyncio.TimeoutError as error:
                    raise TimeoutError(
                        "El Polar no confirmo el inicio del stream ECG en 15 s."
                    ) from error
                if status != 0:
                    raise RuntimeError(
                        f"El Polar rechazo iniciar ECG: codigo PMD {status}."
                    )
                print("ECG confirmado. Ctrl+C para detener.")

                while client.is_connected:
                    await asyncio.sleep(1)
                    if send_task.done():
                        send_task.result()
                    if not errors.empty():
                        raise errors.get_nowait()
                raise ConnectionError("El Polar H10 se desconecto.")
            finally:
                if client.is_connected:
                    await client.write_gatt_char(PMD_CONTROL_UUID, ECG_STOP_CMD)
    finally:
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


async def run_profile(args, profile, address, scan_lock, initial_device=None):
    device = initial_device
    while True:
        try:
            if device is None:
                device = await select_device(address, args.scan_timeout, scan_lock)
            if device is None:
                print(
                    f"No se encontro {profile['participant_name']} ({address}). "
                    "Revisa que este encendido y no conectado a otra app."
                )
            else:
                await forward_session(args, device, profile, address)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Puente Polar: {error}")
        device = None
        print(f"Reintentando en {args.reconnect_seconds:g} s...")
        await asyncio.sleep(args.reconnect_seconds)


async def run_both_profiles(args):
    profiles = args.device_profiles
    addresses = [profiles[key]["address"] for key in ("polar1", "polar2")]
    if addresses[0].lower() == addresses[1].lower():
        raise ValueError("Los dos perfiles deben tener direcciones BLE distintas.")

    print("Buscando los dos Polar H10 configurados...")
    while True:
        discovered = await BleakScanner.discover(timeout=args.scan_timeout)
        by_address = {
            device.address.lower(): device
            for device in discovered
        }
        initial_devices = {
            key: by_address.get(profile["address"].lower())
            for key, profile in profiles.items()
        }
        if any(initial_devices.values()):
            break
        print("No se encontro ninguno de los dos Polar H10.")
        print(f"Reintentando en {args.reconnect_seconds:g} s...")
        await asyncio.sleep(args.reconnect_seconds)

    scan_lock = asyncio.Lock()
    tasks = []
    for key in ("polar1", "polar2"):
        profile = profiles[key]
        device = initial_devices[key]
        if device is None:
            print(
                f"No se encontro {profile['participant_name']} "
                f"({profile['address']})."
            )
        tasks.append(
            asyncio.create_task(
                run_profile(
                    args,
                    profile,
                    profile["address"],
                    scan_lock,
                    initial_device=device,
                )
            )
        )
    await asyncio.gather(*tasks)


async def run(args):
    if args.device == "both":
        await run_both_profiles(args)
        return

    profile = args.device_profiles[args.device]
    await run_profile(args, profile, profile["address"], asyncio.Lock())


def main():
    load_env_file(ENV_PATH)
    parser = argparse.ArgumentParser(
        description="Puente BLE Polar H10 (Windows) -> nodo ROS 2 en Docker."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host TCP del contenedor.")
    parser.add_argument("--port", type=int, default=8765, help="Puerto TCP del contenedor.")
    parser.add_argument(
        "--device",
        choices=("polar1", "polar2", "both"),
        default=os.getenv("POLAR_DEVICE", "polar1").strip().lower(),
        help="Polar a conectar: polar1, polar2 o ambos.",
    )
    parser.add_argument("--scan-timeout", type=float, default=20.0)
    parser.add_argument("--reconnect-seconds", type=float, default=3.0)
    args = parser.parse_args()
    try:
        args.device_profiles = load_device_profiles()
    except ValueError as error:
        parser.error(str(error))
    if args.device == "both":
        for key in ("polar1", "polar2"):
            profile = args.device_profiles[key]
            if not profile["address"]:
                parser.error(f"Configura {profile['participant_name']}_ADDRESS.")
        addresses = [
            args.device_profiles[key]["address"]
            for key in ("polar1", "polar2")
        ]
        if addresses[0].lower() == addresses[1].lower():
            parser.error("POLAR1_ADDRESS y POLAR2_ADDRESS deben ser distintas.")
    else:
        selected_profile = args.device_profiles[args.device]
        if not selected_profile["address"]:
            parser.error(f"Configura {selected_profile['participant_name']}_ADDRESS.")
    if not 1 <= args.port <= 65535:
        parser.error("--port debe estar entre 1 y 65535.")
    if args.scan_timeout <= 0 or args.reconnect_seconds <= 0:
        parser.error("--scan-timeout y --reconnect-seconds deben ser positivos.")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nPuente Polar detenido.")


if __name__ == "__main__":
    main()
