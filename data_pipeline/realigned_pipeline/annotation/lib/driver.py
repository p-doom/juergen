"""Threaded annotation driver: TPM-governed, multi-model, resumable.

Ported from the v2 ``run_dataset`` driver, minus the subprocess layer: methods
run in-thread (frames come from the ar:// store, not ffmpeg). Each work item
is routed at dispatch time to whichever model has the most TPM headroom by a
closed-loop governor: it projects each model's tokens/minute from the item's
token estimate and that model's OWN measured call latency, and only admits an
item if the model stays under --target-tpm. A faster model gets fewer
concurrent items that cycle quicker; a slower one gets more — the asymmetric
split emerges live (AIMD on measured TPM).

Resume: every finished item appends to ``progress.jsonl``; already-done ids
are skipped on restart (the per-call response cache additionally makes any
re-run of an unfinished item free).
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from realigned_pipeline.lib.common import read_jsonl


def model_slug(model: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model) if model else "default"


class TpmGovernor:
    """Adaptive-concurrency governor. Each model has an in-flight LIMIT that a
    control loop raises while the model's MEASURED sustained TPM is below
    target and lowers when it exceeds — so each model is driven to ~target_tpm
    regardless of token-estimate error. A loose projection ceiling guards
    against runaway admission during the measurement lag."""

    def __init__(self, models: list[str | None], target_tpm: float, init_call_s: float = 200.0,
                 window_s: float = 180.0, start_limit: int = 24, max_limit: int = 80):
        self.models = list(models)
        self.target = float(target_tpm)
        self.window_s = float(window_s)
        self.max_limit = int(max_limit)
        self.cv = threading.Condition()
        self._next = 0
        self.inflight: dict[Any, dict[int, float]] = {m: {} for m in self.models}  # id -> est
        self.dur: dict[Any, float] = {m: float(init_call_s) for m in self.models}
        self.recent: dict[Any, deque] = {m: deque() for m in self.models}  # (ts, actual_tokens)
        self.limit: dict[Any, int] = {m: int(start_limit) for m in self.models}
        self.tokens: dict[Any, int] = dict.fromkeys(self.models, 0)
        self.done: dict[Any, int] = dict.fromkeys(self.models, 0)
        self.reported: dict[Any, dict[int, int]] = {m: {} for m in self.models}

    def _measured_tpm(self, m: Any, now: float) -> float:
        dq = self.recent[m]
        while dq and now - dq[0][0] > self.window_s:
            dq.popleft()
        return sum(t for _, t in dq) / (self.window_s / 60.0)

    def _proj_tpm(self, m: Any) -> float:
        return sum(self.inflight[m].values()) / max(20.0, self.dur[m]) * 60.0

    def acquire(self, est_tokens: float) -> tuple[Any, int]:
        """Block until a model has a free in-flight slot (under its adaptive
        limit) and isn't wildly over-committed; route to the most-free model."""
        with self.cv:
            while True:
                cands = []
                for m in self.models:
                    free = self.limit[m] - len(self.inflight[m])
                    # loose projection ceiling (1.5x target) as a runaway guard
                    proj_ok = self._proj_tpm(m) + est_tokens / max(20.0, self.dur[m]) * 60.0 <= self.target * 1.5
                    if free > 0 and proj_ok:
                        cands.append((free, m))
                if not cands and all(not self.inflight[m] for m in self.models):
                    cands = [(1, min(self.models, key=lambda m: self.dur[m]))]  # never stall when idle
                if cands:
                    _, m = max(cands, key=lambda x: x[0])
                    h = self._next
                    self._next += 1
                    self.inflight[m][h] = float(est_tokens)
                    return m, h
                self.cv.wait(timeout=0.5)

    def note_tokens(self, m: Any, handle: int, n: int, now: float) -> None:
        """Live token report from a RUNNING item (long chain items report per
        labeler call so the measured TPM window sees a steady stream instead of
        one end-of-item spike). Reported tokens are remembered per handle and
        subtracted from the release-time total, so nothing double-counts."""
        with self.cv:
            self.recent[m].append((now, int(n)))
            self.tokens[m] += int(n)
            self.reported[m][handle] = self.reported[m].get(handle, 0) + int(n)

    def release(self, m: Any, handle: int, actual_tokens: int, dur_s: float, now: float) -> None:
        with self.cv:
            self.inflight[m].pop(handle, None)
            if dur_s > 0:
                self.dur[m] = 0.75 * self.dur[m] + 0.25 * dur_s
            residual = max(0, int(actual_tokens) - self.reported[m].pop(handle, 0))
            if residual:
                self.recent[m].append((now, residual))
                self.tokens[m] += residual
            self.done[m] += 1
            self.cv.notify_all()

    def control_tick(self, now: float) -> None:
        """AIMD on measured TPM: grow the limit when under target and
        saturated, shrink (x0.9) when over."""
        with self.cv:
            for m in self.models:
                mt = self._measured_tpm(m, now)
                if mt > self.target:
                    self.limit[m] = max(2, int(self.limit[m] * 0.9))
                elif mt < 0.9 * self.target and len(self.inflight[m]) >= self.limit[m] - 1:
                    self.limit[m] = min(self.max_limit, self.limit[m] + 2)
            self.cv.notify_all()

    def snapshot(self, now: float) -> str:
        with self.cv:
            return "  ".join(
                f"{model_slug(m)}: meas~{self._measured_tpm(m, now)/1e6:.2f}M tpm, "
                f"inflight={len(self.inflight[m])}/{self.limit[m]}, dur~{self.dur[m]:.0f}s, done={self.done[m]}"
                for m in self.models)


def run_driver(
    items: list[Any],
    *,
    item_id: Callable[[Any], str],
    est_tokens: Callable[[Any], float],
    run_item: Callable[[Any, str | None], dict[str, Any]],
    models: list[str | None],
    progress_path: Path,
    target_tpm: float = 1_800_000,
    max_workers: int = 64,
    init_call_s: float = 200.0,
    tpm_window_s: float = 180.0,
    start_limit: int = 24,
    max_limit: int = 80,
    force: bool = False,
) -> dict[str, int]:
    """Run ``run_item(item, model)`` over all items under the governor.

    ``run_item`` returns a result dict; ``actual_tokens`` in it (when present)
    feeds the governor's TPM measurement. A ``run_item`` that accepts a THIRD
    parameter is handed a ``report_tokens(n)`` callable to stream token counts
    while it runs — required for long chain items (e.g. day-scope annotation),
    whose end-of-item total would otherwise be invisible to the governor for
    hours; release-time accounting subtracts whatever was reported, so the two
    paths never double-count. Each finished item appends a progress row
    {id, status, ...result-lite}; ids already in the progress file are skipped
    unless ``force``."""
    done_ids: set[str] = set()
    if progress_path.exists() and not force:
        done_ids = {str(r.get("id")) for r in read_jsonl(progress_path)
                    if r.get("status") == "ok"}
    todo = [it for it in items if item_id(it) not in done_ids]
    takes_report = len(inspect.signature(run_item).parameters) >= 3

    gov = TpmGovernor(models, target_tpm, init_call_s,
                      window_s=tpm_window_s, start_limit=start_limit, max_limit=max_limit)
    lock = threading.Lock()
    counters = {"done": 0, "ok": 0, "fail": 0}
    total = len(todo)
    stop = threading.Event()
    print(f"[driver] {len(items)} items, {len(items) - total} already done, {total} to do "
          f"| models={[m or 'env' for m in models]} target_tpm={target_tpm:,.0f}/model "
          f"workers={max_workers}", flush=True)
    if not todo:
        return counters

    queue = list(todo)
    q_lock = threading.Lock()

    def next_item() -> Any | None:
        with q_lock:
            return queue.pop(0) if queue else None

    def reporter() -> None:
        i = 0
        while not stop.wait(8.0):
            gov.control_tick(time.time())          # AIMD adjust every 8s
            i += 1
            if i % 2 == 0:                          # report every ~16s
                with lock:
                    d, f = counters["done"], counters["fail"]
                print(f"  [{d}/{total}] fails={f} | {gov.snapshot(time.time())}", flush=True)

    def worker() -> None:
        while True:
            item = next_item()
            if item is None:
                return
            iid = item_id(item)
            est = float(est_tokens(item))
            model, handle = gov.acquire(est)
            t0 = time.time()
            actual: int | None = None
            try:
                if takes_report:
                    report = lambda n, m=model, h=handle: gov.note_tokens(m, h, int(n), time.time())  # noqa: E731
                    rec = run_item(item, model, report)
                else:
                    rec = run_item(item, model)
                actual = int(rec.get("actual_tokens") or 0) or None
                rec = {"id": iid, "status": "ok", "model": model_slug(model), **rec}
            except Exception as exc:
                rec = {"id": iid, "status": "fail", "model": model_slug(model),
                       "error": f"{type(exc).__name__}: {exc}"}
            finally:
                gov.release(model, handle, actual if actual is not None else int(est),
                            time.time() - t0, time.time())
            with lock:
                with progress_path.open("a") as fh:
                    fh.write(json.dumps({k: v for k, v in rec.items()
                                         if k not in ("goals",)}) + "\n")
                counters["done"] += 1
                counters["ok" if rec["status"] == "ok" else "fail"] += 1
                if rec["status"] == "fail":
                    print(f"  FAIL {iid}: {rec.get('error')}", flush=True)

    rep = threading.Thread(target=reporter, daemon=True)
    rep.start()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, max_workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    print(f"[driver] finished: {counters['ok']} ok, {counters['fail']} failed. "
          f"tokens/model: { {model_slug(m): gov.tokens[m] for m in models} }", flush=True)
    return counters
