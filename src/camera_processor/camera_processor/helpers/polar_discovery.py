"""Local UDP discovery protocol for the Polar H10 ROS 2 receiver."""

import json

DISCOVERY_REQUEST_TYPE = "polar_h10_discovery"
DISCOVERY_RESPONSE_TYPE = "polar_h10_receiver"
DISCOVERY_PROTOCOL_VERSION = 1
DEFAULT_DISCOVERY_PORT = 8766


def make_discovery_response(payload, tcp_port):
    try:
        request = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(request, dict):
        return None
    version = request.get("version")
    if (
        request.get("type") != DISCOVERY_REQUEST_TYPE
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version != DISCOVERY_PROTOCOL_VERSION
    ):
        return None

    return json.dumps(
        {
            "type": DISCOVERY_RESPONSE_TYPE,
            "version": DISCOVERY_PROTOCOL_VERSION,
            "tcp_port": tcp_port,
        },
        separators=(",", ":"),
    ).encode("utf-8")
