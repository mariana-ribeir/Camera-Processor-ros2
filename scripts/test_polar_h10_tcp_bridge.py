import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.polar_h10_tcp_bridge import (
    load_device_profiles,
    make_stream_event,
    run_both_profiles,
)


class PolarBridgeEventTests(unittest.TestCase):
    def test_hr_event_includes_participant_and_polar_identity(self):
        event = make_stream_event(
            "hr",
            {"hr": 84, "rr_ms": [800.0]},
            "POLAR1",
            "24:AC:AC:01:8E:82",
        )

        self.assertEqual(
            event,
            {
                "type": "hr",
                "hr": 84,
                "rr_ms": [800.0],
                "participant_name": "POLAR1",
                "polar_device_id": "24:AC:AC:01:8E:82",
            },
        )

    def test_ecg_event_includes_participant_and_polar_identity(self):
        event = make_stream_event(
            "ecg",
            {
                "sensor_end": 2.0,
                "sensor_end_ns": 2_000_000_000,
                "samples_uv": [-553, 120],
            },
            "POLAR2",
            "A0:9E:1A:EA:3A:E4",
        )

        self.assertEqual(event["participant_name"], "POLAR2")
        self.assertEqual(event["polar_device_id"], "A0:9E:1A:EA:3A:E4")
        self.assertEqual(event["samples_uv"], [-553, 120])

    def test_legacy_event_omits_identity_and_rejects_partial_identity(self):
        event = make_stream_event("hr", {"hr": 70, "rr_ms": []})

        self.assertNotIn("participant_name", event)
        self.assertNotIn("polar_device_id", event)
        with self.assertRaises(ValueError):
            make_stream_event("hr", {}, "POLAR1")

    @patch.dict(
        os.environ,
        {
            "POLAR1_ADDRESS": "24:AC:AC:01:8E:82",
            "POLAR1_PAIR": "false",
            "POLAR2_ADDRESS": "A0:9E:1A:EA:3A:E4",
            "POLAR2_PAIR": "true",
        },
        clear=True,
    )
    def test_profiles_load_separate_participant_names(self):
        profiles = load_device_profiles()

        self.assertEqual(profiles["polar1"]["participant_name"], "POLAR1")
        self.assertEqual(profiles["polar2"]["participant_name"], "POLAR2")
        self.assertEqual(profiles["polar1"]["pair"], False)
        self.assertEqual(profiles["polar2"]["pair"], True)

    def test_both_profiles_start_with_their_configured_devices(self):
        addresses = (
            "24:AC:AC:01:8E:82",
            "A0:9E:1A:EA:3A:E4",
        )
        devices = [
            SimpleNamespace(name="Polar H10 018E823A", address=addresses[0]),
            SimpleNamespace(name="Polar H10 EA3AE42B", address=addresses[1]),
        ]
        profiles = {
            "polar1": {
                "address": addresses[0],
                "name": devices[0].name,
                "participant_name": "POLAR1",
                "pair": False,
            },
            "polar2": {
                "address": addresses[1],
                "name": devices[1].name,
                "participant_name": "POLAR2",
                "pair": True,
            },
        }
        args = SimpleNamespace(
            device_profiles=profiles,
            scan_timeout=1,
            reconnect_seconds=0.1,
        )
        started = {}

        async def discover(timeout):
            return devices

        async def run_profile(*args, **kwargs):
            started[args[1]["participant_name"]] = (
                kwargs["initial_device"].address
            )

        with patch(
            "scripts.polar_h10_tcp_bridge.BleakScanner.discover",
            new=discover,
        ), patch(
            "scripts.polar_h10_tcp_bridge.run_profile",
            new=run_profile,
        ):
            asyncio.run(run_both_profiles(args))

        self.assertEqual(
            started,
            {
                "POLAR1": addresses[0],
                "POLAR2": addresses[1],
            },
        )


if __name__ == "__main__":
    unittest.main()
