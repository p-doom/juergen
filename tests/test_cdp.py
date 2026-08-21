"""`evals/fixtures/cdp.py` — the hand-rolled websocket client.

The client is driven against a real socket server that speaks RFC6455 back, not a
mock of its own helpers: a masking or continuation-frame error does not crash, it
returns wrong evidence.

The bounds are bounds on evidence size — a page returning 40 MB of `outerHTML` must
fail loudly rather than silently fill a trace — so they are asserted too.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading

import pytest

from evals.fixtures.cdp import (
    MAX_CDP_MESSAGE_BYTES,
    MAX_CDP_TARGET_LIST_BYTES,
    WEBSOCKET_GUID,
    CdpError,
    cdp_evaluate,
    find_page_target,
    local_websocket_url,
)


def _accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")


def _read_headers(conn: socket.socket) -> dict[str, str]:
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        chunk = conn.recv(1)
        if not chunk:
            break
        raw.extend(chunk)
    lines = bytes(raw).decode("iso-8859-1").split("\r\n")
    return {
        name.strip().lower(): value.strip()
        for line in lines[1:]
        if ":" in line
        for name, value in [line.split(":", 1)]
    }


def _server_frame(payload: bytes, *, opcode: int, final: bool = True) -> bytes:
    """A server->client frame: unmasked, per RFC6455."""
    first = (0x80 if final else 0x00) | opcode
    length = len(payload)
    if length < 126:
        return bytes((first, length)) + payload
    if length <= 0xFFFF:
        return bytes((first, 126)) + struct.pack("!H", length) + payload
    return bytes((first, 127)) + struct.pack("!Q", length) + payload


def _read_client_message(conn: socket.socket) -> bytes:
    """Read one masked client frame (the client must always mask)."""
    header = conn.recv(2)
    first, second = header[0], header[1]
    assert second & 0x80, "an RFC6455 client MUST mask its frames"
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", conn.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", conn.recv(8))[0]
    mask = conn.recv(4)
    payload = bytearray()
    while len(payload) < length:
        payload.extend(conn.recv(length - len(payload)))
    return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


class FakeCdpServer:
    """Serves one connection, then replies with `frames_for(request)`."""

    def __init__(self, frames_for, *, handshake: bytes | None = None, bad_accept: bool = False):
        self.frames_for = frames_for
        self.handshake = handshake
        self.bad_accept = bad_accept
        self.requests: list[dict] = []
        self.pongs: list[bytes] = []
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        with conn:
            try:
                headers = _read_headers(conn)
                if self.handshake is not None:
                    conn.sendall(self.handshake)
                    return
                key = headers.get("sec-websocket-key", "")
                accept = "wrong-accept-key" if self.bad_accept else _accept_key(key)
                conn.sendall(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("ascii")
                )
                if self.bad_accept:
                    return
                request = json.loads(_read_client_message(conn))
                self.requests.append(request)
                if getattr(self, "ping_first", False):
                    # Deterministic ordering: ping, block on the pong, then reply.
                    conn.sendall(_server_frame(b"hb", opcode=0x9))
                    self.pongs.append(_read_client_message(conn))
                for frame in self.frames_for(request):
                    conn.sendall(frame)
            except (OSError, AssertionError, IndexError, json.JSONDecodeError):
                return

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/devtools/page/ABC"

    def close(self) -> None:
        self._listener.close()


def _ok(request, value):
    return [
        _server_frame(
            json.dumps({"id": request["id"], "result": {"result": {"value": value}}}).encode(),
            opcode=0x1,
        )
    ]


@pytest.fixture
def server_factory():
    servers: list[FakeCdpServer] = []

    def build(frames_for, **kwargs) -> FakeCdpServer:
        server = FakeCdpServer(frames_for, **kwargs)
        servers.append(server)
        return server

    yield build
    for server in servers:
        server.close()


def test_a_round_trip_returns_the_evaluated_value(server_factory) -> None:
    server = server_factory(lambda r: _ok(r, {"title": "Capability fixture", "n": 3}))
    value = cdp_evaluate(server.url, "document.title", timeout_s=5)
    assert value == {"title": "Capability fixture", "n": 3}
    request = server.requests[0]
    assert request["method"] == "Runtime.evaluate"
    assert request["params"]["expression"] == "document.title"
    assert request["params"]["returnByValue"] is True
    assert request["params"]["awaitPromise"] is False, (
        "awaiting would let the page's async work change what is being measured"
    )


def test_the_client_masks_its_frames(server_factory) -> None:
    """Asserted by the server: an unmasked client frame fails its assertion."""
    server = server_factory(lambda r: _ok(r, 1))
    assert cdp_evaluate(server.url, "1", timeout_s=5) == 1


@pytest.mark.parametrize("size", [10, 200, 70_000])
def test_every_payload_length_encoding_round_trips(server_factory, size: int) -> None:
    """<126 inline, <=0xFFFF 16-bit, else 64-bit — all three headers."""
    expression = "x" * size
    server = server_factory(lambda r: _ok(r, len(r["params"]["expression"])))
    assert cdp_evaluate(server.url, expression, timeout_s=10) == size


def test_a_large_response_is_reassembled(server_factory) -> None:
    blob = "y" * 200_000
    server = server_factory(lambda r: _ok(r, blob))
    assert cdp_evaluate(server.url, "big", timeout_s=10) == blob


def test_continuation_frames_are_reassembled(server_factory) -> None:
    def split(request):
        body = json.dumps(
            {"id": request["id"], "result": {"result": {"value": "abcdefghij"}}}
        ).encode()
        half = len(body) // 2
        return [
            _server_frame(body[:half], opcode=0x1, final=False),
            _server_frame(body[half:], opcode=0x0, final=True),
        ]

    server = server_factory(split)
    assert cdp_evaluate(server.url, "x", timeout_s=5) == "abcdefghij"


def test_a_ping_is_answered_with_a_pong(server_factory) -> None:
    """A Chromium that gets no pong drops the connection mid-evaluate.

    The server blocks on the pong before replying, so the reply can only arrive if
    the client really answered the ping.
    """
    server = server_factory(lambda r: _ok(r, "after-ping"))
    server.ping_first = True
    assert cdp_evaluate(server.url, "x", timeout_s=10) == "after-ping"
    assert server.pongs == [b"hb"], "the pong must carry the ping's payload back"


def test_a_pong_is_ignored(server_factory) -> None:
    def with_pong(request):
        return [_server_frame(b"", opcode=0xA), *_ok(request, "after-pong")]

    assert cdp_evaluate(server_factory(with_pong).url, "x", timeout_s=5) == "after-pong"


def test_a_binary_frame_is_accepted(server_factory) -> None:
    def binary(request):
        body = json.dumps({"id": request["id"], "result": {"result": {"value": 7}}}).encode()
        return [_server_frame(body, opcode=0x2)]

    assert cdp_evaluate(server_factory(binary).url, "x", timeout_s=5) == 7


def test_an_unrelated_event_before_the_reply_is_skipped(server_factory) -> None:
    """CDP interleaves events with replies; only the matching id is the answer."""

    def noisy(request):
        event = _server_frame(
            json.dumps({"method": "Runtime.consoleAPICalled", "params": {}}).encode(), opcode=0x1
        )
        other = _server_frame(
            json.dumps({"id": 999, "result": {"result": {"value": "wrong"}}}).encode(), opcode=0x1
        )
        return [event, other, *_ok(request, "right")]

    assert cdp_evaluate(server_factory(noisy).url, "x", timeout_s=5) == "right"


def test_a_falsy_value_is_returned_not_treated_as_missing(server_factory) -> None:
    for value in (False, 0, "", None):
        server = server_factory(lambda r, v=value: _ok(r, v))
        assert cdp_evaluate(server.url, "x", timeout_s=5) == value


@pytest.mark.parametrize("url", ["http://127.0.0.1:9/x", "wss://host/x", "ws:///x", "nonsense"])
def test_an_unsupported_websocket_url_is_refused(url: str) -> None:
    with pytest.raises(CdpError, match="unsupported CDP websocket URL"):
        cdp_evaluate(url, "1", timeout_s=1)


def test_a_failed_upgrade_is_reported(server_factory) -> None:
    server = server_factory(
        lambda r: [], handshake=b"HTTP/1.1 500 Server Error\r\nContent-Length: 0\r\n\r\n"
    )
    with pytest.raises(CdpError, match="upgrade failed"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_an_invalid_accept_key_is_refused(server_factory) -> None:
    """Otherwise any HTTP server answering 101 would be treated as a websocket."""
    server = server_factory(lambda r: [], bad_accept=True)
    with pytest.raises(CdpError, match="invalid accept key"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_a_close_frame_before_the_reply_is_an_error(server_factory) -> None:
    server = server_factory(lambda r: [_server_frame(b"", opcode=0x8)])
    with pytest.raises(CdpError, match="closed before the requested response"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_a_truncated_frame_is_an_error(server_factory) -> None:
    server = server_factory(lambda r: [_server_frame(b"partial", opcode=0x1)[:3]])
    with pytest.raises(CdpError, match="closed before its response completed"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_an_unsupported_opcode_is_an_error(server_factory) -> None:
    server = server_factory(lambda r: [_server_frame(b"x", opcode=0x3)])
    with pytest.raises(CdpError, match="unsupported CDP websocket opcode"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_a_continuation_with_no_start_frame_is_an_error(server_factory) -> None:
    server = server_factory(lambda r: [_server_frame(b"orphan", opcode=0x0)])
    with pytest.raises(CdpError, match="unsupported CDP websocket opcode"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_a_frame_over_the_evidence_bound_is_refused(server_factory) -> None:
    """A page returning 40 MB of outerHTML must fail loudly, not fill a trace."""
    oversized = struct.pack("!BBQ", 0x81, 127, MAX_CDP_MESSAGE_BYTES + 1)
    server = server_factory(lambda r: [oversized])
    with pytest.raises(CdpError, match="exceeded"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_invalid_json_is_reported_as_a_protocol_failure(server_factory) -> None:
    server = server_factory(lambda r: [_server_frame(b"{not json", opcode=0x1)])
    with pytest.raises(CdpError, match="invalid JSON"):
        cdp_evaluate(server.url, "1", timeout_s=5)


def test_a_protocol_error_reply_is_reported(server_factory) -> None:
    def failing(request):
        return [
            _server_frame(
                json.dumps({"id": request["id"], "error": {"code": -32000, "message": "nope"}}).encode(),
                opcode=0x1,
            )
        ]

    with pytest.raises(CdpError, match="Runtime.evaluate failed"):
        cdp_evaluate(server_factory(failing).url, "1", timeout_s=5)


def test_a_page_side_exception_is_reported(server_factory) -> None:
    def raising(request):
        return [
            _server_frame(
                json.dumps(
                    {
                        "id": request["id"],
                        "result": {
                            "result": {"type": "object"},
                            "exceptionDetails": {"text": "ReferenceError"},
                        },
                    }
                ).encode(),
                opcode=0x1,
            )
        ]

    with pytest.raises(CdpError, match="raised"):
        cdp_evaluate(server_factory(raising).url, "1", timeout_s=5)


def test_a_reply_with_no_value_is_reported(server_factory) -> None:
    def valueless(request):
        return [
            _server_frame(
                json.dumps({"id": request["id"], "result": {"result": {"type": "undefined"}}}).encode(),
                opcode=0x1,
            )
        ]

    with pytest.raises(CdpError, match="returned no value"):
        cdp_evaluate(server_factory(valueless).url, "1", timeout_s=5)


def test_a_non_object_result_is_reported(server_factory) -> None:
    def scalar(request):
        return [
            _server_frame(
                json.dumps({"id": request["id"], "result": {"result": "oops"}}).encode(), opcode=0x1
            )
        ]

    with pytest.raises(CdpError, match="no result object"):
        cdp_evaluate(server_factory(scalar).url, "1", timeout_s=5)


class _TargetList:
    """Serves `/json/list` like Chromium's debugging endpoint."""

    def __init__(self, payload: bytes) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def target_list():
    servers: list[_TargetList] = []

    def build(targets) -> _TargetList:
        payload = targets if isinstance(targets, bytes) else json.dumps(targets).encode()
        server = _TargetList(payload)
        servers.append(server)
        return server

    yield build
    for server in servers:
        server.close()


def _page(id_: str, url: str, *, type_: str = "page", ws: bool = True) -> dict:
    target = {"id": id_, "type": type_, "url": url, "title": id_}
    if ws:
        target["webSocketDebuggerUrl"] = f"ws://10.0.2.15:9222/devtools/page/{id_}"
    return target


def test_exactly_one_matching_page_target_is_returned(target_list) -> None:
    server = target_list([_page("A", "http://10.0.2.2:1/fixture/fx_click")])
    target = find_page_target(server.port, "fx_click")
    assert target["id"] == "A"


def test_a_non_page_target_is_not_matched(target_list) -> None:
    server = target_list(
        [
            _page("SW", "http://10.0.2.2:1/fixture/fx_click", type_="service_worker"),
            _page("A", "http://10.0.2.2:1/fixture/fx_click"),
        ]
    )
    assert find_page_target(server.port, "fx_click")["id"] == "A"


def test_a_target_without_a_websocket_url_is_not_matched(target_list) -> None:
    server = target_list(
        [
            _page("NoWs", "http://10.0.2.2:1/fixture/fx_click", ws=False),
            _page("A", "http://10.0.2.2:1/fixture/fx_click"),
        ]
    )
    assert find_page_target(server.port, "fx_click")["id"] == "A"


def test_two_matching_targets_is_an_error_not_the_first_one(target_list) -> None:
    """Picking the first silently reads whichever tab Chromium ordered first."""
    server = target_list(
        [
            _page("A", "http://10.0.2.2:1/fixture/fx_click"),
            _page("B", "http://10.0.2.2:1/fixture/fx_click"),
        ]
    )
    with pytest.raises(CdpError, match="found 2"):
        find_page_target(server.port, "fx_click")


def test_no_matching_target_is_an_error_that_lists_what_was_found(target_list) -> None:
    server = target_list([_page("A", "chrome://newtab")])
    with pytest.raises(CdpError, match="found 0") as excinfo:
        find_page_target(server.port, "fx_click")
    assert "chrome://newtab" in str(excinfo.value), "the error must be diagnosable"


def test_a_non_array_target_list_is_refused(target_list) -> None:
    server = target_list({"not": "an array"})
    with pytest.raises(CdpError, match="was not an array"):
        find_page_target(server.port, "fx_click")


def test_an_oversized_target_list_is_refused(target_list) -> None:
    server = target_list(b"[" + b" " * (MAX_CDP_TARGET_LIST_BYTES + 8) + b"]")
    with pytest.raises(CdpError, match="exceeded its evidence bound"):
        find_page_target(server.port, "fx_click")


def test_the_guest_websocket_url_is_rewritten_onto_the_forwarded_port() -> None:
    target = {"webSocketDebuggerUrl": "ws://10.0.2.15:9222/devtools/page/ABC?x=1"}
    assert local_websocket_url(target, 45678) == (
        "ws://127.0.0.1:45678/devtools/page/ABC?x=1"
    )


def test_the_rewritten_url_is_what_cdp_evaluate_accepts(server_factory) -> None:
    server = server_factory(lambda r: _ok(r, "ok"))
    target = {"webSocketDebuggerUrl": "ws://10.0.2.15:9222/devtools/page/ABC"}
    url = local_websocket_url(target, server.port)
    assert cdp_evaluate(url, "1", timeout_s=5) == "ok"
