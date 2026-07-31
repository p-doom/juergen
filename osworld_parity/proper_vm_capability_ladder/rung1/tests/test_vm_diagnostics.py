from __future__ import annotations

import base64
import hashlib
import json
import struct

from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1 import vm
from osworld_parity.proper_vm_capability_ladder.rung1.vm import (
    CHROME_LOG_PREFIX,
    POINTER_STATE_PREFIX,
    KvmFixtureSession,
)


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit):
        return self.payload[:limit]


def _server_frame(payload: bytes, *, opcode: int = 0x1, final: bool = True) -> bytes:
    first = (0x80 if final else 0) | opcode
    if len(payload) < 126:
        return bytes((first, len(payload))) + payload
    if len(payload) <= 0xFFFF:
        return bytes((first, 126)) + struct.pack("!H", len(payload)) + payload
    return bytes((first, 127)) + struct.pack("!Q", len(payload)) + payload


def _decode_client_frame(frame: bytes) -> tuple[int, bool, bytes]:
    opcode = frame[0] & 0x0F
    masked = bool(frame[1] & 0x80)
    length = frame[1] & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack("!H", frame[offset : offset + 2])[0]
        offset += 2
    elif length == 127:
        length = struct.unpack("!Q", frame[offset : offset + 8])[0]
        offset += 8
    mask = frame[offset : offset + 4] if masked else b""
    offset += 4 if masked else 0
    payload = frame[offset : offset + length]
    if masked:
        payload = bytes(
            value ^ mask[index % 4] for index, value in enumerate(payload)
        )
    return opcode, masked, payload


def test_websocket_client_frames_are_masked(monkeypatch) -> None:
    class Socket:
        sent = b""

        def sendall(self, value):
            self.sent += value

    monkeypatch.setattr(vm.os, "urandom", lambda length: b"M" * length)
    sock = Socket()
    vm._send_websocket_frame(sock, b'{"id":1}', opcode=0x1)
    opcode, masked, payload = _decode_client_frame(sock.sent)
    assert opcode == 0x1
    assert masked is True
    assert payload == b'{"id":1}'


def test_websocket_receive_handles_fragmentation_and_ping(monkeypatch) -> None:
    incoming = b"".join(
        (
            _server_frame(b'{"id":', final=False),
            _server_frame(b"probe", opcode=0x9),
            _server_frame(b"1}", opcode=0x0),
        )
    )

    class Socket:
        def __init__(self):
            self.incoming = bytearray(incoming)
            self.sent = []

        def recv(self, length):
            size = min(length, 2, len(self.incoming))
            value = bytes(self.incoming[:size])
            del self.incoming[:size]
            return value

        def sendall(self, value):
            self.sent.append(value)

    monkeypatch.setattr(vm.os, "urandom", lambda length: b"P" * length)
    sock = Socket()
    assert vm._recv_websocket_message(sock) == b'{"id":1}'
    assert len(sock.sent) == 1
    opcode, masked, payload = _decode_client_frame(sock.sent[0])
    assert opcode == 0xA
    assert masked is True
    assert payload == b"probe"


def test_cdp_evaluate_upgrades_sends_request_and_extracts_result(monkeypatch) -> None:
    observed = {}

    class Socket:
        def __init__(self):
            self.incoming = bytearray()
            self.requests = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def settimeout(self, timeout):
            observed["socket_timeout"] = timeout

        def recv(self, length):
            size = min(length, 3, len(self.incoming))
            value = bytes(self.incoming[:size])
            del self.incoming[:size]
            return value

        def sendall(self, value):
            if value.startswith(b"GET "):
                header = value.decode("ascii")
                key = next(
                    line.split(":", 1)[1].strip()
                    for line in header.split("\r\n")
                    if line.lower().startswith("sec-websocket-key:")
                )
                accept = base64.b64encode(
                    hashlib.sha1((key + vm._WEBSOCKET_GUID).encode()).digest()
                ).decode()
                self.incoming.extend(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode()
                )
                return
            opcode, masked, payload = _decode_client_frame(value)
            assert opcode == 0x1 and masked is True
            request = json.loads(payload)
            self.requests.append(request)
            event = {"method": "Runtime.executionContextCreated", "params": {}}
            response = {
                "id": request["id"],
                "result": {
                    "result": {
                        "type": "object",
                        "value": {"queue": "captured", "pending": 2},
                    }
                },
            }
            self.incoming.extend(_server_frame(json.dumps(event).encode()))
            self.incoming.extend(_server_frame(json.dumps(response).encode()))

    sock = Socket()

    def connect(address, *, timeout):
        observed["address"] = address
        observed["connect_timeout"] = timeout
        return sock

    monkeypatch.setattr(vm.socket, "create_connection", connect)
    monkeypatch.setattr(vm.os, "urandom", lambda length: b"K" * length)
    value = vm._cdp_evaluate(
        "ws://127.0.0.1:43123/devtools/page/target?token=one",
        "window.__RUNG1A_DIAGNOSTICS__",
        timeout_s=1.25,
    )
    assert value == {"queue": "captured", "pending": 2}
    assert observed == {
        "address": ("127.0.0.1", 43123),
        "connect_timeout": 1.25,
        "socket_timeout": 1.25,
    }
    assert sock.requests == [
        {
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {
                "expression": "window.__RUNG1A_DIAGNOSTICS__",
                "returnByValue": True,
                "awaitPromise": False,
            },
        }
    ]


def test_browser_diagnostic_rewrites_advertised_cdp_endpoint(monkeypatch) -> None:
    fixture = load_manifest().by_id("r1a-click-dev-1101")
    advertised = "ws://127.0.0.1:9222/devtools/page/fixture-target"
    targets = [
        {
            "id": "fixture-target",
            "type": "page",
            "url": f"http://10.0.2.2/fixture/{fixture.id}",
            "title": "Capability fixture",
            "webSocketDebuggerUrl": advertised,
        }
    ]
    monkeypatch.setattr(
        vm.urllib.request, "urlopen", lambda *a, **k: _Response(targets)
    )
    observed = {}

    def evaluate(url, expression, *, timeout_s):
        observed.update(url=url, expression=expression, timeout_s=timeout_s)
        return {
            "diagnostics": {"page_events": [], "report_queue": {"pending": 0}},
            "dom": {"outer_html": "<html></html>"},
        }

    monkeypatch.setattr(vm, "_cdp_evaluate", evaluate)
    session = KvmFixtureSession()
    session._chromium_port = 43123
    evidence = session.capture_browser_diagnostics(fixture)
    assert observed["url"] == "ws://127.0.0.1:43123/devtools/page/fixture-target"
    assert "window.__RUNG1A_DIAGNOSTICS__" in observed["expression"]
    assert "document.documentElement.outerHTML" in observed["expression"]
    assert "captured_browser_wall_time_ms: Date.now()" in observed["expression"]
    assert "performance_time_origin_ms: performance.timeOrigin" in observed["expression"]
    assert evidence["target"]["advertised_websocket_url"] == advertised
    assert evidence["target"]["local_websocket_url"] == observed["url"]


def test_guest_pointer_diagnostic_separates_raw_and_pointer_masks() -> None:
    class Transport:
        program = ""

        def execute_argv(self, argv, *, check):
            self.program = argv[2]
            payload = {
                "schema_version": 1,
                "cursor": [300, 400],
                "raw_x_mask": 258,
                "pointer_button_mask": 256,
                "guest_wall_before_ns": 10,
                "guest_wall_after_ns": 20,
                "guest_monotonic_before_ns": 30,
                "guest_monotonic_after_ns": 40,
            }
            return {
                "status": "success",
                "returncode": 0,
                "output": POINTER_STATE_PREFIX + json.dumps(payload) + "\n",
                "error": "",
            }

    session = KvmFixtureSession()
    transport = Transport()
    session.transport = transport
    evidence = session.capture_guest_pointer_state()
    compile(transport.program, "<pointer-state-diagnostic>", "exec")
    assert "int(q.mask)&1792" in transport.program
    assert evidence["raw_x_mask"] == 258
    assert evidence["pointer_button_mask"] == 256
    assert evidence["guest_returncode"] == 0
    assert evidence["guest_monotonic_after_ns"] == 40
    assert evidence["raw_result_marker"].startswith(POINTER_STATE_PREFIX)


def test_chrome_log_diagnostic_is_bounded_and_hashes_full_file() -> None:
    class Transport:
        program = ""

        def execute_argv(self, argv, *, check):
            self.program = argv[2]
            payload = {
                "schema_version": 1,
                "total_bytes": 2_000_000,
                "captured_bytes": 1_048_576,
                "sha256": "b" * 64,
                "truncated": True,
                "tail": "bounded chrome tail",
            }
            return {
                "status": "success",
                "returncode": 0,
                "output": CHROME_LOG_PREFIX + json.dumps(payload) + "\n",
                "error": "",
            }

    session = KvmFixtureSession()
    transport = Transport()
    session.transport = transport
    evidence = session.capture_chrome_log()
    compile(transport.program, "<chrome-log-diagnostic>", "exec")
    assert "source.read(65536)" in transport.program
    assert "if len(tail)>1048576: del tail[:-1048576]" in transport.program
    assert "digest.update(chunk)" in transport.program
    assert evidence["total_bytes"] == 2_000_000
    assert evidence["captured_bytes"] == 1_048_576
    assert evidence["sha256"] == "b" * 64
    assert evidence["truncated"] is True
    assert evidence["content_tail"] == "bounded chrome tail"
    assert "output" not in evidence["raw_guest_result"]
