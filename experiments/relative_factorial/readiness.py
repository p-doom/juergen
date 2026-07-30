#!/usr/bin/env python3
"""Wait for and validate an actual OpenAI-compatible chat completion."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
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
                    "messages": [{"role": "user", "content": "Reply with the word ready."}],
                    "temperature": 0.0,
                    "max_tokens": 4,
                },
                timeout=60.0,
            )
            choices = completion.get("choices")
            if not choices or "message" not in choices[0]:
                raise RuntimeError(f"malformed chat completion: {completion}")
            print("real /v1/chat/completions readiness: PASS")
            return 0
        except (OSError, ValueError, KeyError, RuntimeError, urllib.error.HTTPError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
    print(f"FATAL chat-completion readiness failed: {last_error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
