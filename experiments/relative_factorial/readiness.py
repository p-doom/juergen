#!/usr/bin/env python3
"""Wait for a real vision chat completion in the cell's expected action schema."""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import Any


GRAMMAR_ACTIONS = {
    "move_rel": ("tool", "move_rel"),
    "deltatype_raw": ("bare", "delta"),
    "absolute_toolcall": ("tool", "left_click"),
    "absolute_raw": ("bare", "delta"),
}
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _json_request(url: str, *, payload=None, timeout: float = 15.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _png_data_url(size: int = 32) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF))

    pixels = b"".join(b"\x00" + b"\xff\xff\xff" * size for _ in range(size))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _probe_instruction(grammar: str) -> str:
    wrapper, action = GRAMMAR_ACTIONS[grammar]
    if wrapper == "tool":
        return (
            "This is a server readiness probe. Ignore the image. Reply with exactly this "
            "tool call and no other text:\n<tool_call>\n"
            f'{{"name":"computer_use","arguments":{{"action":"{action}",'
            '"coordinate":[0,0]}}\n</tool_call>'
        )
    return (
        "This is a server readiness probe. Ignore the image. Reply with exactly this "
        "single action line and no other text:\n0 0 0 ; +LMB -LMB"
    )


def _arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _probe_schema_ok(message: Any, grammar: str) -> bool:
    if not isinstance(message, dict):
        return False
    wrapper, expected_action = GRAMMAR_ACTIONS[grammar]
    content = message.get("content")
    if wrapper == "bare":
        return isinstance(content, str) and content.strip() == "0 0 0 ; +LMB -LMB"

    payloads = []
    if isinstance(content, str):
        for match in _TOOL_CALL_RE.finditer(content):
            try:
                payloads.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            payloads.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("name") != "computer_use":
            continue
        args = _arguments(payload.get("arguments"))
        if (args is not None and args.get("action") == expected_action
                and args.get("coordinate") == [0, 0]):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--grammar", choices=sorted(GRAMMAR_ACTIONS), required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--pid", type=int, default=None,
                        help="fail immediately if this serving process exits")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + args.timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except OSError:
                print(f"FATAL serving process {args.pid} exited before readiness", file=sys.stderr)
                return 2
        try:
            models = _json_request(base + "/models")
            if not models.get("data"):
                raise RuntimeError("/models returned no models")
            completion = _json_request(
                base + "/chat/completions",
                payload={
                    "model": args.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _probe_instruction(args.grammar)},
                            {"type": "image_url", "image_url": {"url": _png_data_url()}},
                        ],
                    }],
                    "temperature": 0.0,
                    "max_tokens": 96,
                },
                timeout=60.0,
            )
            choices = completion.get("choices")
            message = choices[0].get("message") if choices else None
            if not isinstance(message, dict) or not any(
                key in message for key in ("content", "tool_calls")
            ):
                raise RuntimeError(f"malformed vision chat completion: {completion}")
            # Schema compliance is the scientific outcome, not a readiness
            # condition. Log the diagnostic but let evaluate.py score all 80
            # scenes, including a model that emits zero valid actions.
            schema_ok = _probe_schema_ok(message, args.grammar)
            print(
                f"real vision /v1/chat/completions readiness ({args.grammar}): PASS "
                f"schema_probe={'PASS' if schema_ok else 'FAIL_NONBLOCKING'}"
            )
            return 0
        except (OSError, ValueError, KeyError, RuntimeError, urllib.error.HTTPError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
    print(f"FATAL vision chat-completion readiness failed: {last_error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
