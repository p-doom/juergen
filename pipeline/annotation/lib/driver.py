"""TPM-governed concurrency for Crowd-Cast annotation."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class TpmGovernor:
    def __init__(self, target_tpm: float) -> None:
        if target_tpm <= 0:
            raise ValueError("target_tpm must be positive")
        self.target = target_tpm
        self.inflight: dict[int, float] = {}
        self.completed: deque[tuple[float, int]] = deque()
        self.condition = threading.Condition()
        self.next_handle = 0

    def _prune(self, now: float) -> None:
        while self.completed and now - self.completed[0][0] >= 60:
            self.completed.popleft()

    def acquire(self, estimated_tokens: float) -> int:
        if estimated_tokens <= 0:
            raise ValueError("estimated tokens must be positive")
        with self.condition:
            while True:
                self._prune(time.monotonic())
                reserved = sum(self.inflight.values()) + sum(
                    tokens for _, tokens in self.completed
                )
                if reserved + estimated_tokens <= self.target or reserved == 0:
                    handle = self.next_handle
                    self.next_handle += 1
                    self.inflight[handle] = estimated_tokens
                    return handle
                self.condition.wait(timeout=1)

    def release(self, handle: int, tokens: int) -> None:
        with self.condition:
            self.inflight.pop(handle)
            self.completed.append((time.monotonic(), tokens))
            self.condition.notify_all()


def run_driver(
    items: list[Any],
    *,
    item_id: Callable[[Any], str],
    est_tokens: Callable[[Any], float],
    run_item: Callable[[Any], dict[str, Any]],
    progress_path: Path,
    target_tpm: float,
    max_workers: int,
) -> dict[str, int]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    governor = TpmGovernor(target_tpm)
    output_lock = threading.Lock()

    def run(item: Any) -> dict[str, Any]:
        estimate = int(est_tokens(item))
        handle = governor.acquire(estimate)
        actual = estimate
        try:
            result = run_item(item)
            actual = int(result["actual_tokens"])
            progress = {"id": item_id(item), "status": "ok", **result}
            with output_lock, progress_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(progress) + "\n")
            return result
        finally:
            governor.release(handle, actual)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run, item) for item in items]
        try:
            for future in as_completed(futures):
                future.result()
                completed += 1
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return {"ok": completed}
