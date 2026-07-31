"""Stage 1 - Sample: draw k completions per task from a served checkpoint.

Reusable pieces, each of which exists because an ad-hoc version of it failed:

* :class:`ErrorLedger` — explicit per-rollout error accounting with a
  **failure-rate ceiling that aborts** rather than silently degrading.
  **Defect #9**: a probe used ``return_exceptions=True``, filtered the exceptions
  out, got ``success=0/0``, and wrote ``0.0`` as a result. The ledger makes that
  impossible: every attempt is recorded as ok or as a classified failure, an
  empty success set can never be summarised as a number, and crossing the ceiling
  raises :class:`~rft.errors.FailureRateExceeded`.
* :func:`shard_assignments` — deterministic sharding: which ``(task, rollout)``
  units belong to shard *i* of *n*, independent of ordering, resumption state, or
  worker count changes for a fixed *n*.
* :class:`RolloutStore` — append-only JSONL with **resumability**: already-written
  units are skipped on restart, keyed on the collision-proof
  :func:`rft.splits.make_sample_id`.
* :func:`run_sampling` — the orchestration: preflight (a real chat completion, not
  a ``/v1/models`` ping), then the sharded, resumable, ledgered loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.errors import FailureRateExceeded, MissingFieldError, SchemaError
from rft.serving import (
    PreflightResult,
    assert_caches_off_home,
    preflight_chat_completion,
)
from rft.splits import make_sample_id

# ---------------------------------------------------------------------------
# error accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutFailure:
    """One classified rollout failure. Kept, not discarded."""

    sample_id: str
    task_id: str
    rollout_index: int
    kind: str
    detail: str


@dataclass
class ErrorLedger:
    """Per-rollout success/failure accounting with an abort ceiling.

    Args:
        max_failure_rate: abort once the observed failure rate exceeds this and
            at least ``min_attempts_before_abort`` attempts have been made. A run
            that quietly loses a third of its rollouts is not a run with a
            smaller dataset, it is a broken run: prime-rl's session-hashed
            routing turns a per-request 400 into whole-group loss (defect #7),
            so a rising failure rate is the only warning available.
        min_attempts_before_abort: do not judge the rate off two attempts.
    """

    max_failure_rate: float = 0.05
    min_attempts_before_abort: int = 20
    n_ok: int = 0
    failures: list[RolloutFailure] = field(default_factory=list)
    #: Failures already present in a resumed output file, counted separately so a
    #: resumed run does not inherit a stale rate.
    n_resumed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_failure_rate < 1.0:
            raise SchemaError(
                f"max_failure_rate must be in [0,1), got {self.max_failure_rate}"
            )

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def n_attempts(self) -> int:
        return self.n_ok + self.n_failed

    @property
    def failure_rate(self) -> float:
        if not self.n_attempts:
            raise SchemaError(
                "failure rate over zero attempts is undefined (a 0/0 probe that "
                "reports 0.0 is defect #9)"
            )
        return self.n_failed / self.n_attempts

    def record_ok(self) -> None:
        self.n_ok += 1

    def record_failure(
        self, *, sample_id: str, task_id: str, rollout_index: int, exc: BaseException
    ) -> RolloutFailure:
        """Classify and record a failure, then check the ceiling.

        Raises:
            FailureRateExceeded: the ceiling is crossed. The original exception is
                chained so the first cause is never lost.
        """
        failure = RolloutFailure(
            sample_id=sample_id,
            task_id=task_id,
            rollout_index=rollout_index,
            kind=type(exc).__name__,
            detail=str(exc)[:500],
        )
        self.failures.append(failure)
        if (
            self.n_attempts >= self.min_attempts_before_abort
            and self.failure_rate > self.max_failure_rate
        ):
            raise FailureRateExceeded(
                f"rollout failure rate {self.failure_rate:.1%} "
                f"({self.n_failed}/{self.n_attempts}) exceeded the ceiling "
                f"{self.max_failure_rate:.1%}; aborting instead of writing a "
                f"silently-degraded sample set. Failure kinds: {self.kind_counts()!r}"
            ) from exc
        return failure

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.failures:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_attempts": self.n_attempts,
            "n_resumed": self.n_resumed,
            "failure_rate": self.failure_rate if self.n_attempts else None,
            "max_failure_rate": self.max_failure_rate,
            "kind_counts": self.kind_counts(),
            "failures": [f.__dict__ for f in self.failures[:200]],
        }

    def describe(self) -> str:
        rate = f"{self.failure_rate:.1%}" if self.n_attempts else "n/a (0 attempts)"
        return (
            f"rollouts: ok={self.n_ok} failed={self.n_failed} resumed={self.n_resumed} "
            f"failure_rate={rate} (ceiling {self.max_failure_rate:.1%}) "
            f"kinds={self.kind_counts()!r}"
        )


# ---------------------------------------------------------------------------
# deterministic sharding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleUnit:
    """One unit of sampling work: rollout ``rollout_index`` of ``task_id``."""

    task_id: str
    rollout_index: int
    sample_id: str
    seed: int


def _unit_hash(task_id: str, rollout_index: int, salt: str) -> int:
    key = f"{salt}\x00{task_id}\x00{rollout_index}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def enumerate_units(
    task_ids: Sequence[str],
    *,
    k: int,
    sample_root: str | Path,
    salt: str = "rft-v1",
    app_of: Mapping[str, str] | None = None,
) -> list[SampleUnit]:
    """All ``len(task_ids) * k`` units, in a stable order, with per-unit seeds.

    The seed is a hash of ``(salt, task_id, rollout_index)`` — not a counter — so
    a resumed or re-sharded run reproduces the same seed for the same unit.
    """
    if k < 1:
        raise SchemaError(f"k must be >= 1, got {k}")
    if not task_ids:
        raise SchemaError("no task ids to sample")
    counts = Counter(task_ids)
    dupes = {t for t, n in counts.items() if n > 1}
    if dupes:
        raise SchemaError(f"duplicate task ids in input: {sorted(dupes)[:10]!r}")
    units: list[SampleUnit] = []
    for tid in task_ids:
        for j in range(k):
            units.append(
                SampleUnit(
                    task_id=tid,
                    rollout_index=j,
                    sample_id=make_sample_id(
                        sample_root=sample_root,
                        task_id=tid,
                        rollout_index=j,
                        app=(app_of or {}).get(tid),
                    ),
                    seed=_unit_hash(tid, j, salt) % (2**31 - 1),
                )
            )
    return units


def shard_assignments(
    units: Sequence[SampleUnit], *, num_shards: int, shard_index: int, salt: str = "rft-v1"
) -> list[SampleUnit]:
    """Deterministically select this shard's units.

    Assignment is ``hash(task_id, rollout_index) % num_shards``, so:
      * it does not depend on input order, on how many units were already done,
        or on which shard asks;
      * for a fixed ``num_shards`` the mapping is stable across restarts;
      * all ``k`` rollouts of a task are *not* forced onto one shard, which is
        what lets a single slow task's rollouts run in parallel.
    """
    if num_shards < 1:
        raise SchemaError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise SchemaError(f"shard_index {shard_index} out of range for {num_shards} shards")
    return [
        u
        for u in units
        if _unit_hash(u.task_id, u.rollout_index, salt) % num_shards == shard_index
    ]


# ---------------------------------------------------------------------------
# resumable store
# ---------------------------------------------------------------------------


class RolloutStore:
    """Append-only JSONL rollout store with resumability.

    ``completed_ids()`` reads back what is already on disk so a restart skips it.
    A truncated final line (killed mid-write) is reported, not silently ignored:
    it is dropped from the completed set so the unit is redone, and counted so the
    operator sees it happened.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_truncated_lines = 0

    def completed_ids(self) -> set[str]:
        if not self.path.is_file():
            return set()
        done: set[str] = set()
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    self.n_truncated_lines += 1
                    continue
                if "sample_id" not in obj:
                    raise MissingFieldError(f"{self.path}:<line>.sample_id")
                done.add(str(obj["sample_id"]))
        return done

    def append(self, record: Mapping[str, Any]) -> None:
        if "sample_id" not in record:
            raise MissingFieldError("record.sample_id")
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read_all(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open() as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"{self.path}:{line_no} is not valid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Everything stage 1 needs. All paths must be on /fast/project."""

    task_ids: Sequence[str]
    k: int
    grammar: str
    out_path: Path
    base_url: str
    model: str
    num_shards: int = 1
    shard_index: int = 0
    max_failure_rate: float = 0.05
    min_attempts_before_abort: int = 20
    salt: str = "rft-v1"
    preflight_timeout_s: float = 900.0
    app_of: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.out_path = Path(self.out_path)
        from rft.grammars import get_grammar

        get_grammar(self.grammar)  # fail fast on an unknown grammar name


@dataclass
class SamplingReport:
    """Stage-1 outcome. Written next to the rollouts as ``_sampling_report.json``."""

    config: dict[str, Any]
    preflight: dict[str, Any]
    ledger: dict[str, Any]
    n_units_total: int
    n_units_this_shard: int
    n_skipped_resumed: int
    n_truncated_lines: int
    elapsed_s: float

    def describe(self) -> str:
        return (
            f"stage=sample shard={self.config['shard_index']}/{self.config['num_shards']} "
            f"units={self.n_units_this_shard}/{self.n_units_total} "
            f"resumed={self.n_skipped_resumed} truncated_lines={self.n_truncated_lines} "
            f"elapsed={self.elapsed_s:.0f}s\n  {self.ledger}"
        )


RolloutFn = Callable[[SampleUnit], Mapping[str, Any]]


def run_sampling(
    cfg: SamplingConfig,
    rollout_fn: RolloutFn,
    *,
    preflight: Callable[..., PreflightResult] = preflight_chat_completion,
    http_post: Any = None,
    http_get: Any = None,
    check_caches: bool = True,
) -> SamplingReport:
    """Run stage 1: preflight, then the sharded, resumable, ledgered loop.

    ``rollout_fn`` performs one rollout and returns a JSON-serialisable mapping.
    It may raise; the ledger records and classifies the failure and re-raises only
    when the failure-rate ceiling is crossed. It must **not** catch its own errors
    and return a zero-valued record — that is the failure mode this whole module
    is built to prevent.
    """
    started = time.monotonic()
    if check_caches:
        assert_caches_off_home()

    pf = preflight(
        base_url=cfg.base_url,
        model=cfg.model,
        timeout_s=cfg.preflight_timeout_s,
        http_post=http_post,
        http_get=http_get,
    )

    units = enumerate_units(
        list(cfg.task_ids),
        k=cfg.k,
        sample_root=cfg.out_path.parent,
        salt=cfg.salt,
        app_of=cfg.app_of,
    )
    mine = shard_assignments(
        units, num_shards=cfg.num_shards, shard_index=cfg.shard_index, salt=cfg.salt
    )
    store = RolloutStore(cfg.out_path)
    done = store.completed_ids()
    todo = [u for u in mine if u.sample_id not in done]
    n_skipped = len(mine) - len(todo)

    ledger = ErrorLedger(
        max_failure_rate=cfg.max_failure_rate,
        min_attempts_before_abort=cfg.min_attempts_before_abort,
        n_resumed=n_skipped,
    )

    for unit in todo:
        try:
            record = rollout_fn(unit)
        except Exception as exc:  # noqa: BLE001 - accounted by the ledger, never dropped
            failure = ledger.record_failure(
                sample_id=unit.sample_id,
                task_id=unit.task_id,
                rollout_index=unit.rollout_index,
                exc=exc,
            )
            store.append(
                {
                    "sample_id": unit.sample_id,
                    "task_id": unit.task_id,
                    "rollout_index": unit.rollout_index,
                    "grammar": cfg.grammar,
                    "error": {"kind": failure.kind, "detail": failure.detail},
                }
            )
            continue
        if not isinstance(record, Mapping):
            raise SchemaError(
                f"rollout_fn returned {type(record).__name__}, expected a mapping"
            )
        payload = dict(record)
        payload.setdefault("sample_id", unit.sample_id)
        payload.setdefault("task_id", unit.task_id)
        payload.setdefault("rollout_index", unit.rollout_index)
        payload.setdefault("grammar", cfg.grammar)
        payload.setdefault("seed", unit.seed)
        if "error" in payload:
            raise SchemaError(
                f"rollout_fn returned a record carrying an `error` key for "
                f"{unit.sample_id}: a failed rollout must RAISE so the ledger sees it, "
                "not return an error-shaped success"
            )
        store.append(payload)
        ledger.record_ok()

    report = SamplingReport(
        config={
            "k": cfg.k,
            "grammar": cfg.grammar,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "num_shards": cfg.num_shards,
            "shard_index": cfg.shard_index,
            "salt": cfg.salt,
            "n_tasks": len(cfg.task_ids),
            "out_path": str(cfg.out_path),
        },
        preflight={
            "attempts": pf.attempts,
            "elapsed_s": pf.elapsed_s,
            "completion_preview": pf.completion_preview,
            "served_models": list(pf.served_models),
            "warnings": list(pf.warnings),
        },
        ledger=ledger.as_dict(),
        n_units_total=len(units),
        n_units_this_shard=len(mine),
        n_skipped_resumed=n_skipped,
        n_truncated_lines=store.n_truncated_lines,
        elapsed_s=time.monotonic() - started,
    )
    report_path = cfg.out_path.with_name(
        f"_sampling_report.shard{cfg.shard_index:03d}of{cfg.num_shards:03d}.json"
    )
    report_path.write_text(json.dumps(report.__dict__, indent=2, default=str))
    return report


def merge_shards(paths: Iterable[str | Path], out_path: str | Path) -> int:
    """Concatenate shard JSONLs, asserting sample-id uniqueness (defect #17).

    Returns the number of merged records.
    """
    records: list[dict[str, Any]] = []
    for p in paths:
        records.extend(RolloutStore(p).read_all())
    from rft.splits import assert_unique_sample_ids

    assert_unique_sample_ids(records)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)
