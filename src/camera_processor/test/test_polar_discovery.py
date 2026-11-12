import json

import pytest

from camera_processor.helpers.polar_discovery import (
    DISCOVERY_PROTOCOL_VERSION,
    DISCOVERY_REQUEST_TYPE,
    DISCOVERY_RESPONSE_TYPE,
    make_discovery_response,
)


def test_make_discovery_response_returns_tcp_endpoint():
    request = json.dumps(
        {"type": DISCOVERY_REQUEST_TYPE, "version": DISCOVERY_PROTOCOL_VERSION}
    ).encode("utf-8")

    response = make_discovery_response(request, 8765)

    assert json.loads(response) == {
        "type": DISCOVERY_RESPONSE_TYPE,
        "version": DISCOVERY_PROTOCOL_VERSION,
        "tcp_port": 8765,
    }


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"[]",
        b'{"type":"other","version":1}',
        b'{"type":"polar_h10_discovery","version":true}',
        b'{"type":"polar_h10_discovery","version":2}',
    ],
)
def test_make_discovery_response_ignores_invalid_requests(payload):
    assert make_discovery_response(payload, 8765) is None
