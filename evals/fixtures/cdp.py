"""A dependency-free Chrome DevTools Protocol client.

One `Runtime.evaluate` over a raw websocket, hand-rolled on `socket` and `struct`.
That is deliberate, and it belongs to *tasks* rather than to the VM layer: the only
reason a fixture needs CDP is to read the DOM state a browser task's oracle scores.
The VM layer forwards a port; it does not know what a page is.

Why not playwright: `SetupController`'s playwright-over-CDP path is the thing that
drags in a browser-automation stack, a matching browser build and a proxy config the
cluster does not have, and it only ever gets used to run one expression. 140 lines of
websocket framing has no install story at all.

Every bound here is a bound on *evidence size*, not on correctness: a page that
returns 40 MB of `outerHTML` should fail loudly rather than silently fill a trace.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import urllib.parse
import urllib.request
from typing import Any

__all__ = [
    "CdpError",
    "MAX_CDP_MESSAGE_BYTES",
    "MAX_CDP_TARGET_LIST_BYTES",
    "cdp_evaluate",
    "find_page_target",
    "local_websocket_url",
]

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_CDP_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_CDP_TARGET_LIST_BYTES = 1024 * 1024


class CdpError(RuntimeError):
    """A CDP transport or protocol failure. Never a task failure."""


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise CdpError("CDP websocket closed before its response completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, payload: bytes, *, opcode: int) -> None:
    first = 0x80 | opcode
    length = len(payload)
    if length < 126:
        header = bytes((first, 0x80 | length))
    elif length <= 0xFFFF:
        header = bytes((first, 0x80 | 126)) + struct.pack("!H", length)
    else:
        header = bytes((first, 0x80 | 127)) + struct.pack("!Q", length)
    mask = os.urandom(4)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(header + mask + masked)


def _recv_message(sock: socket.socket) -> bytes:
    """One complete websocket message, reassembling continuation frames.

    Pings are answered (a Chromium that gets no pong drops the connection
    mid-evaluate), pongs ignored, a close frame is an error — the caller asked a
    question and did not get an answer.
    """
    message = bytearray()
    message_opcode: int | None = None
    while True:
        first, second = _recv_exact(sock, 2)
        final = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
        if length > MAX_CDP_MESSAGE_BYTES:
            raise CdpError(f"CDP websocket frame exceeded {length} bytes")
        mask = _recv_exact(sock, 4) if masked else b""
        payload = _recv_exact(sock, length)
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        if opcode == 0x8:
            raise CdpError("CDP websocket closed before the requested response")
        if opcode == 0x9:
            _send_frame(sock, payload, opcode=0xA)
            continue
        if opcode == 0xA:
            continue
        if opcode in {0x1, 0x2}:
            message_opcode = opcode
        elif opcode != 0x0 or message_opcode is None:
            raise CdpError(f"unsupported CDP websocket opcode {opcode}")
        message.extend(payload)
        if len(message) > MAX_CDP_MESSAGE_BYTES:
            raise CdpError("CDP websocket message exceeded the evidence bound")
        if final:
            return bytes(message)


def cdp_evaluate(websocket_url: str, expression: str, *, timeout_s: float) -> Any:
    """Evaluate `expression` in the page and return its value by value.

    `awaitPromise` is False on purpose: a fixture diagnostic must be a synchronous
    read of current DOM state, and awaiting would let the page's own async work
    change what is being measured.
    """
    parsed = urllib.parse.urlsplit(websocket_url)
    if parsed.scheme != "ws" or parsed.hostname is None:
        raise CdpError(f"unsupported CDP websocket URL: {websocket_url!r}")
    port = parsed.port or 80
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    expected_accept = base64.b64encode(
        hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")
    with socket.create_connection((parsed.hostname, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {parsed.hostname}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii")
        )
        response = bytearray()
        while b"\r\n\r\n" not in response:
            if len(response) > 65536:
                raise CdpError("CDP websocket handshake exceeded its bound")
            response.extend(_recv_exact(sock, 1))
        lines = bytes(response).decode("iso-8859-1").split("\r\n")
        if not lines or " 101 " not in lines[0]:
            raise CdpError(f"CDP websocket upgrade failed: {lines[0]!r}")
        headers = {
            name.strip().lower(): value.strip()
            for line in lines[1:]
            if ":" in line
            for name, value in [line.split(":", 1)]
        }
        if headers.get("sec-websocket-accept") != expected_accept:
            raise CdpError("CDP websocket returned an invalid accept key")
        request_id = 1
        _send_frame(
            sock,
            json.dumps(
                {
                    "id": request_id,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": False,
                    },
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            opcode=0x1,
        )
        while True:
            try:
                payload = json.loads(_recv_message(sock))
            except json.JSONDecodeError as exc:
                raise CdpError("CDP websocket returned invalid JSON") from exc
            if not isinstance(payload, dict) or payload.get("id") != request_id:
                continue
            if "error" in payload:
                raise CdpError(f"CDP Runtime.evaluate failed: {payload['error']}")
            outer = payload.get("result", {})
            result = outer.get("result", {})
            if not isinstance(result, dict):
                raise CdpError("CDP Runtime.evaluate returned no result object")
            if "exceptionDetails" in outer:
                raise CdpError(f"CDP Runtime.evaluate raised: {outer['exceptionDetails']}")
            if "value" not in result:
                raise CdpError(f"CDP Runtime.evaluate returned no value: {result}")
            return result["value"]


def find_page_target(
    chromium_port: int, url_fragment: str, *, timeout_s: float = 2.0
) -> dict[str, Any]:
    """The single live page target whose URL contains `url_fragment`.

    Exactly one, or an error listing what was found. Picking "the first match"
    silently reads whichever tab Chromium happened to order first, which is how a
    diagnostic ends up describing a restored session tab instead of the fixture.
    """
    list_url = f"http://127.0.0.1:{chromium_port}/json/list"
    with urllib.request.urlopen(list_url, timeout=timeout_s) as response:
        payload = response.read(MAX_CDP_TARGET_LIST_BYTES + 1)
    if len(payload) > MAX_CDP_TARGET_LIST_BYTES:
        raise CdpError("Chromium CDP target list exceeded its evidence bound")
    targets = json.loads(payload)
    if not isinstance(targets, list):
        raise CdpError("Chromium CDP target list was not an array")
    matching = [
        target
        for target in targets
        if isinstance(target, dict)
        and target.get("type") == "page"
        and url_fragment in str(target.get("url", ""))
        and isinstance(target.get("webSocketDebuggerUrl"), str)
    ]
    if len(matching) != 1:
        summary = [
            {"id": t.get("id"), "type": t.get("type"), "url": t.get("url")}
            for t in targets
            if isinstance(t, dict)
        ]
        raise CdpError(
            f"expected one live CDP fixture target, found {len(matching)}: {summary}"
        )
    return matching[0]


def local_websocket_url(target: dict[str, Any], chromium_port: int) -> str:
    """Rewrite the guest-advertised websocket URL onto the forwarded host port."""
    advertised = urllib.parse.urlsplit(str(target["webSocketDebuggerUrl"]))
    return urllib.parse.urlunsplit(
        ("ws", f"127.0.0.1:{chromium_port}", advertised.path, advertised.query, "")
    )
