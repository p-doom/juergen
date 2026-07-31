"""Backend-neutral newline-JSON VM protocol used by collection and live replay."""

from __future__ import annotations

import json
import selectors
import shlex
import subprocess
from pathlib import Path
from typing import Any, Self

from experiments.teacher_sft.contracts import ContractError


class JsonlRpcEnvironment:
    """One isolated environment process per rollout.

    The executable receives JSON-RPC-ish requests on stdin. Required methods:
    reset(task, work_dir), step_native(action), step_compact(sequence), reward,
    close. Replies are single JSON objects with matching id, ok, and result/error.
    OSWorld and CUA-Gym wrappers can implement this protocol without leaking
    backend-specific setup or reward logic into the data pipeline.
    """

    def __init__(self, command: str, *, timeout_s: float = 120.0):
        argv = shlex.split(command)
        if not argv:
            raise ContractError("empty environment command")
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.timeout_s = timeout_s
        self.request_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self.process.stdin is None or self.process.stdout is None:
            raise ContractError("environment process pipes are unavailable")
        self.request_id += 1
        request = {"id": self.request_id, "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout_s):
                self.process.kill()
                self.process.wait()
                raise ContractError(
                    f"environment adapter timed out after {self.timeout_s}s in {method}"
                )
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read()[-2000:] if self.process.stderr else ""
            raise ContractError(f"environment adapter exited without reply: {stderr}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"environment adapter emitted invalid JSON: {line[:200]!r}"
            ) from exc
        if reply.get("id") != self.request_id:
            raise ContractError("environment adapter response id mismatch")
        if reply.get("ok") is not True:
            raise ContractError(f"environment adapter error: {reply.get('error')!r}")
        return reply.get("result")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.call("close")
            except ContractError:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def assert_observation(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context}: adapter observation is not an object")
    cursor = value.get("cursor")
    size = value.get("screen_size")
    if (
        not isinstance(cursor, list)
        or len(cursor) != 2
        or any(not isinstance(item, int) for item in cursor)
        or not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(item, int) or item <= 1 for item in size)
    ):
        raise ContractError(f"{context}: invalid cursor/screen telemetry")
    image_path = Path(str(value.get("image_path", ""))).resolve()
    if not image_path.is_file():
        raise ContractError(f"{context}: screenshot is missing: {image_path}")
    return {**value, "image_path": str(image_path)}
