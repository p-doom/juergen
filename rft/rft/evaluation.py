"""Stage 5 - Evaluate: read the validated harnesses, report by bucket.

**This module does not evaluate anything itself.** The harnesses are already
validated and must not be rewritten:

* closed-loop OSWorld task success — ``osworld_parity/split/baseline_eval_shard.py``
  (absolute/native) and ``format_eval_shard.py`` (move_rel / diffabs /
  deltatype_{raw,norm}), launched via their sbatch wrappers. They write
  ``{base_output_dir}/{app}/{task_id}/result.json``.
* single-step and closed-loop mouse *targeting* — the grounding parity harness
  (``parity_harness/run_parity.py`` -> ``report.py``), which reproduces the
  90.5%-class absolute reading (0.9713 on bbox29/crosshair) and the matching
  relative arms in the same run.

What this module owns is **reading them correctly and reporting honestly**:

1. reward from ``scores.reward`` (defect #1), NaN kept distinct from 0 (defect #3),
   and a task that was never written kept distinct from a task that scored 0 —
   the harness deliberately leaves model-unreachable tasks *unwritten*, so
   "missing" and "zero" are different facts and both are reported;
2. **bucketed** results from ``analysis/heldout_buckets.json``, never a single
   aggregate: 74 of the 110 held-out tasks are INFEASIBLE, KEYBOARD_ONLY or
   never-solved-by-any-arm, so the aggregate is ~85% insensitive to mouse control;
3. the three mandatory :mod:`rft.diagnostics` numbers on every report;
4. a hard guard that held-out scores are never used to select a checkpoint.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.diagnostics import DeltaDiagnostics
from rft.errors import LeakError, MissingFieldError, SchemaError
from rft.rewards import CANONICAL_REWARD_PATH, UNSCORED, Reward, read_reward

#: Reward at or above this counts as solved, matching ``aggregate.py``'s
#: ``reward >= 0.999``. Kept as a named constant because partial credit exists
#: (0.9977, 0.8949, 0.1111 in the off-shelf anchor run), so ``reward == 1.0`` is
#: the wrong test.
SOLVED_THRESHOLD: float = 0.999

#: Buckets produced by ``analysis/bucket_partition.py``. Only ``MOUSE_SOLVED`` is
#: sensitive to mouse control; the rest are why a single aggregate is misleading.
BUCKET_MOUSE_SOLVED = "MOUSE_SOLVED"
BUCKET_KEYBOARD_ONLY = "KEYBOARD_ONLY"
BUCKET_INFEASIBLE = "INFEASIBLE"
BUCKET_UNCLASSIFIED = "UNCLASSIFIED_never_solved"
BUCKET_FREEBIE = "FREEBIE_inert_win"
BUCKET_MISSING_TASK_JSON = "MISSING_TASK_JSON"

KNOWN_BUCKETS: tuple[str, ...] = (
    BUCKET_MOUSE_SOLVED,
    BUCKET_KEYBOARD_ONLY,
    BUCKET_INFEASIBLE,
    BUCKET_UNCLASSIFIED,
    BUCKET_FREEBIE,
    BUCKET_MISSING_TASK_JSON,
)

#: Buckets whose outcome can move when mouse control improves. Reporting the
#: aggregate instead of this is how a mouse-control result gets diluted ~3x.
MOUSE_SENSITIVE_BUCKETS: tuple[str, ...] = (BUCKET_MOUSE_SOLVED, BUCKET_UNCLASSIFIED)


# ---------------------------------------------------------------------------
# reading result trees
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskResult:
    """One ``result.json``, read with explicit NaN and missing-field handling."""

    app: str
    task_id: str
    reward: Reward
    path: Path
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """Whether the task counts as solved.

        Raises:
            SchemaError: the task was never scored. "Unscored" is not "unsolved";
                the caller must decide, and :class:`EvalReport` records the count.
        """
        if self.reward is UNSCORED:
            raise SchemaError(
                f"{self.app}/{self.task_id} was never scored (NaN reward: env.evaluate() "
                "raised). It is neither solved nor unsolved - see EvalReport.n_unscored"
            )
        return float(self.reward) >= SOLVED_THRESHOLD


def read_result_tree(
    base_output_dir: str | Path, *, reward_path: str = CANONICAL_REWARD_PATH
) -> list[TaskResult]:
    """Read every ``{app}/{task_id}/result.json`` under ``base_output_dir``.

    Raises:
        SchemaError: the directory does not exist or contains no result files. Zero
            results is never reported as a score of 0.
    """
    base = Path(base_output_dir)
    if not base.is_dir():
        raise SchemaError(f"eval output dir not found: {base}")
    out: list[TaskResult] = []
    for path in sorted(base.glob("*/*/result.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path} is not valid JSON: {exc}") from exc
        out.append(
            TaskResult(
                app=path.parent.parent.name,
                task_id=path.parent.name,
                reward=read_reward(payload, path=reward_path),
                path=path,
                params=payload.get("params", {}) if isinstance(payload, dict) else {},
            )
        )
    if not out:
        raise SchemaError(
            f"no */*/result.json under {base}; an eval that produced no results has no "
            "score (reporting 0.0 for it is defect #9)"
        )
    return out


def load_buckets(path: str | Path) -> dict[str, str]:
    """Load ``heldout_buckets.json`` -> ``{task_id: bucket}``.

    Accepts the real shape ``{"buckets": {...}, "evidence": {...}}`` and a bare
    ``{task_id: bucket}`` map. Unknown bucket names raise rather than being lumped
    into an "other" pile.
    """
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"bucket file not found: {p}")
    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise SchemaError(f"{p} does not contain a JSON object")
    buckets = payload.get("buckets", payload)
    if not isinstance(buckets, dict) or not buckets:
        raise SchemaError(f"{p} has no non-empty `buckets` map")
    unknown = {v for v in buckets.values() if v not in KNOWN_BUCKETS}
    if unknown:
        raise SchemaError(
            f"{p} contains unknown bucket name(s) {sorted(unknown)!r}; known: "
            f"{list(KNOWN_BUCKETS)!r}. Add the new bucket to rft.evaluation "
            "deliberately rather than silently aggregating it."
        )
    return {str(k): str(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketScore:
    bucket: str
    n_tasks: int
    n_written: int
    n_missing: int
    n_unscored: int
    n_solved: int
    reward_sum: float

    @property
    def mean_over_written(self) -> float | None:
        """Mean reward over written results, unscored counted as 0.

        ``None`` when nothing was written — never 0.0.

        Counting NaN as 0 here is deliberate and matches ``osworld_score.py``'s
        ``_mean_count_nan_as_zero``: excluding a crashed evaluator from the
        denominator would let evaluator crashes silently inflate the score. It is
        safe *only* because the harness leaves model-unreachable tasks unwritten
        instead of writing them as 0 - so ``n_missing`` and ``n_unscored`` are both
        reported separately and must be read alongside this number.
        """
        if not self.n_written:
            return None
        return self.reward_sum / self.n_written

    @property
    def mean_over_expected(self) -> float | None:
        """Mean over every task in the bucket, missing counted as 0."""
        if not self.n_tasks:
            return None
        return self.reward_sum / self.n_tasks

    def describe(self) -> str:
        mw = "n/a" if self.mean_over_written is None else f"{self.mean_over_written:.4f}"
        me = "n/a" if self.mean_over_expected is None else f"{self.mean_over_expected:.4f}"
        return (
            f"{self.bucket:&lt;26} n={self.n_tasks:&gt;4} written={self.n_written:&gt;4} "
            f"missing={self.n_missing:&gt;3} unscored={self.n_unscored:&gt;3} "
            f"solved={self.n_solved:&gt;4} sum={self.reward_sum:8.5f} "
            f"mean/written={mw} mean/expected={me}"
        )


@dataclass
class EvalReport:
    """A bucketed evaluation report. There is no single-number accessor.

    Deliberate omission: no ``.score`` property. The aggregate is ~85% insensitive
    to mouse control, so anything that prints one number is reporting the wrong
    thing. Read :attr:`buckets`, and for a mouse-control claim read
    :meth:`mouse_sensitive`.
    """

    run_name: str
    reward_path: str
    n_expected_tasks: int
    overall: BucketScore
    buckets: dict[str, BucketScore] = field(default_factory=dict)
    diagnostics: DeltaDiagnostics | None = None
    excluded_task_ids: tuple[str, ...] = ()
    unbucketed_task_ids: tuple[str, ...] = ()
    parser_path: str = ""

    def mouse_sensitive(self) -> BucketScore:
        """Combined score over the buckets mouse control can actually move."""
        parts = [self.buckets[b] for b in MOUSE_SENSITIVE_BUCKETS if b in self.buckets]
        if not parts:
            raise SchemaError(
                "no mouse-sensitive buckets present; cannot make a mouse-control claim"
            )
        return BucketScore(
            bucket="+".join(p.bucket for p in parts),
            n_tasks=sum(p.n_tasks for p in parts),
            n_written=sum(p.n_written for p in parts),
            n_missing=sum(p.n_missing for p in parts),
            n_unscored=sum(p.n_unscored for p in parts),
            n_solved=sum(p.n_solved for p in parts),
            reward_sum=sum(p.reward_sum for p in parts),
        )

    def describe(self) -> str:
        lines = [
            f"=== eval report: {self.run_name} ===",
            f"reward read from {self.reward_path!r}; expected {self.n_expected_tasks} tasks; "
            f"{len(self.excluded_task_ids)} excluded as unscorable",
            self.overall.describe(),
            "--- by bucket (only MOUSE_SOLVED / UNCLASSIFIED can move with mouse control) ---",
        ]
        for name in KNOWN_BUCKETS:
            if name in self.buckets:
                lines.append(self.buckets[name].describe())
        if self.unbucketed_task_ids:
            lines.append(
                f"WARNING {len(self.unbucketed_task_ids)} written task(s) are not in the "
                f"bucket map: {list(self.unbucketed_task_ids[:5])}"
            )
        try:
            lines.append("--- mouse-sensitive subset ---")
            lines.append(self.mouse_sensitive().describe())
        except SchemaError as exc:
            lines.append(f"--- mouse-sensitive subset unavailable: {exc}")
        lines.append("--- mandatory prediction diagnostics ---")
        lines.append(
            self.diagnostics.describe()
            if self.diagnostics is not None
            else "MISSING: no delta diagnostics were supplied. Every eval must report "
            "distinct-delta count, {0,+-1,+-10,+-100} lattice fraction and "
            "median |pred|/|gold|."
        )
        if self.parser_path:
            lines.append(f"parser: {self.parser_path}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "reward_path": self.reward_path,
            "n_expected_tasks": self.n_expected_tasks,
            "overall": self.overall.__dict__,
            "buckets": {k: v.__dict__ for k, v in self.buckets.items()},
            "diagnostics": self.diagnostics.as_dict() if self.diagnostics else None,
            "excluded_task_ids": list(self.excluded_task_ids),
            "unbucketed_task_ids": list(self.unbucketed_task_ids),
            "parser_path": self.parser_path,
        }


def _score_group(bucket: str, results: Sequence[TaskResult], n_tasks: int) -> BucketScore:
    n_unscored = sum(1 for r in results if r.reward is UNSCORED)
    scored = [float(r.reward) for r in results if r.reward is not UNSCORED]
    return BucketScore(
        bucket=bucket,
        n_tasks=n_tasks,
        n_written=len(results),
        n_missing=max(0, n_tasks - len(results)),
        n_unscored=n_unscored,
        n_solved=sum(1 for v in scored if v >= SOLVED_THRESHOLD),
        reward_sum=math.fsum(scored),
    )


def build_eval_report(
    results: Sequence[TaskResult],
    *,
    run_name: str,
    expected_task_ids: Iterable[str],
    buckets: Mapping[str, str] | None = None,
    excluded_task_ids: Iterable[str] = (),
    diagnostics: DeltaDiagnostics | None = None,
    reward_path: str = CANONICAL_REWARD_PATH,
    parser_path: str = "",
    require_diagnostics: bool = True,
) -> EvalReport:
    """Assemble a bucketed report from harness results.

    Args:
        results: from :func:`read_result_tree`.
        expected_task_ids: every task the eval was supposed to run. Needed to
            distinguish "scored 0" from "never written" — the harness leaves a
            model-unreachable task unwritten on purpose.
        buckets: from :func:`load_buckets`. Omitting it produces an overall-only
            report and is flagged in the output.
        excluded_task_ids: unscorable tasks (e.g. the 3 gdrive tasks), removed from
            both numerator and denominator, and listed in the report.
        diagnostics: the three mandatory numbers. Required by default.

    Raises:
        SchemaError: ``require_diagnostics`` and no diagnostics were supplied, or a
            written result is not in ``expected_task_ids``.
    """
    if require_diagnostics and diagnostics is None:
        raise SchemaError(
            "refusing to build an eval report without prediction diagnostics: the "
            "distinct-delta count, lattice fraction and median |pred|/|gold| are part of "
            "the output contract (they exposed a magnitude-encoding collapse that "
            "aggregate accuracy hid completely). Pass require_diagnostics=False only "
            "for a harness that emits no deltas at all."
        )
    excluded = {str(t) for t in excluded_task_ids}
    expected = {str(t) for t in expected_task_ids} - excluded
    kept = [r for r in results if r.task_id not in excluded]

    unexpected = sorted({r.task_id for r in kept} - expected)
    if unexpected:
        raise SchemaError(
            f"{len(unexpected)} written result(s) are not in the expected task list: "
            f"{unexpected[:5]!r}. Either the split changed or the wrong output dir was "
            "read; both invalidate the denominator."
        )

    report = EvalReport(
        run_name=run_name,
        reward_path=reward_path,
        n_expected_tasks=len(expected),
        overall=_score_group("OVERALL", kept, len(expected)),
        diagnostics=diagnostics,
        excluded_task_ids=tuple(sorted(excluded)),
        parser_path=parser_path,
    )
    if buckets is None:
        return report

    by_bucket: dict[str, list[TaskResult]] = {}
    unbucketed: list[str] = []
    for r in kept:
        bucket = buckets.get(r.task_id)
        if bucket is None:
            unbucketed.append(r.task_id)
            continue
        by_bucket.setdefault(bucket, []).append(r)
    expected_per_bucket: dict[str, int] = {}
    for tid in expected:
        b = buckets.get(tid)
        if b is not None:
            expected_per_bucket[b] = expected_per_bucket.get(b, 0) + 1
    report.buckets = {
        name: _score_group(name, by_bucket.get(name, []), expected_per_bucket.get(name, 0))
        for name in KNOWN_BUCKETS
        if name in by_bucket or name in expected_per_bucket
    }
    report.unbucketed_task_ids = tuple(sorted(unbucketed))
    return report


# ---------------------------------------------------------------------------
# eval-leak guard on checkpoint selection
# ---------------------------------------------------------------------------


def assert_not_selection_input(report: EvalReport, *, purpose: str) -> None:
    """Refuse to let a held-out report be used to select a checkpoint.

    Held-out scores are the parity metric and nothing else. Selecting on them turns
    the held-out set into a validation set, and the next number reported from it is
    no longer an out-of-sample measurement. The bucket file carries the same rule in
    its producer: "DIAGNOSIS ONLY. Must never be used for checkpoint selection."

    Args:
        purpose: what the caller intends to do with the report. Anything that looks
            like model/checkpoint selection raises.
    """
    selection_words = ("select", "selection", "choose", "pick", "best_checkpoint", "early_stop")
    lowered = purpose.lower()
    if any(w in lowered for w in selection_words):
        raise LeakError(
            f"held-out eval report {report.run_name!r} may not be used for {purpose!r}. "
            "The 110 held-out tasks are the parity metric only; select checkpoints on "
            "TRAIN-split val loss (rft.training.ValLossTee) and report held-out once."
        )


def write_eval_report(report: EvalReport, out_path: str | Path) -> Path:
    """Write the report JSON and print the human-readable form."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.as_dict(), indent=2))
    print(report.describe())
    return p


def load_gdrive_exclusions(path: str | Path) -> frozenset[str]:
    """Load ``gdrive_unscorable.txt`` (``app/task_id`` per line) as task ids."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"exclusion file not found: {p}")
    out = {
        line.strip().split("/")[-1]
        for line in p.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not out:
        raise SchemaError(f"{p} listed no tasks")
    return frozenset(out)


def expected_tasks_from_split(path: str | Path) -> frozenset[str]:
    """Task ids from an OSWorld split JSON ``{app: [task_id, ...]}``."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"split file not found: {p}")
    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise SchemaError(f"{p} is not a {{app: [task_id]}} map")
    out: set[str] = set()
    for app, ids in payload.items():
        if not isinstance(ids, list):
            raise SchemaError(f"{p}[{app!r}] is {type(ids).__name__}, expected a list")
        out.update(str(i) for i in ids)
    if not out:
        raise MissingFieldError(f"{p}: no task ids")
    return frozenset(out)
