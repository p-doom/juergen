"""Stage 4 - Train: wrap omegalax correctly. Do not reimplement omegalax.

omegalax is battle-tested; every defect in this stage was in the *invocation*:

* **Defect #12** — ``keep_latest=1`` together with ``keep_period=305`` deleted
  every intermediate checkpoint, so val-based checkpoint selection was impossible
  after the fact. :func:`validate_retention` computes which steps actually survive
  and refuses a config that keeps fewer than the caller says it needs.
* **Defect #11** — ``val_steps=15`` over a 65-record val split with a
  **non-restarting** grain iterator scores a *different* 15 records at each eval,
  so val numbers are only comparable within matched windows.
  :func:`resolve_val_steps` computes the step count that covers the whole split
  and raises if the caller asks for less without explicitly opting in.
* **Defect #13** — omegalax logs ``val/loss`` only to wandb, never to stdout.
  :func:`ValLossTee` scrapes the training log and re-emits val loss to stdout, so a
  run's val curve is readable from the slurm log with no wandb round trip
  (and the offline-wandb key defect, #10, cannot hide it either).

The required-flags list is enforced too: ``origin/main``'s ``train_vlm_sft.py``
hard-requires ``keep_period``, ``keep_latest``, ``log_memory``, ``resume`` and
``num_loss_tiles``, and ``labctl validate`` does not catch their absence.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.errors import RetentionError, SchemaError, ValCoverageError

#: Flags ``train_vlm_sft.py`` on omegalax ``origin/main`` hard-requires. Their
#: absence is a runtime crash, and ``labctl validate`` will not catch it.
REQUIRED_OMEGALAX_FLAGS: tuple[str, ...] = (
    "keep_period",
    "keep_latest",
    "log_memory",
    "resume",
    "num_loss_tiles",
)


def assert_required_flags(flags: Mapping[str, Any]) -> None:
    """Raise unless every hard-required omegalax flag is present.

    Presence, not truthiness: ``resume=False`` and ``log_memory=False`` are
    perfectly valid values, they just have to be *stated*.
    """
    missing = [f for f in REQUIRED_OMEGALAX_FLAGS if f not in flags]
    if missing:
        raise SchemaError(
            f"omegalax train_vlm_sft.py hard-requires these flags and they are absent: "
            f"{missing!r}. labctl validate does not catch this; the job crashes at start."
        )


# ---------------------------------------------------------------------------
# Defect #12: checkpoint retention
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionPlan:
    """Which checkpoint steps survive a given retention config."""

    total_steps: int
    save_interval: int
    keep_period: int | None
    keep_latest: int | None
    surviving_steps: tuple[int, ...]

    @property
    def n_surviving(self) -> int:
        return len(self.surviving_steps)

    def describe(self) -> str:
        return (
            f"total_steps={self.total_steps} save_interval={self.save_interval} "
            f"keep_period={self.keep_period} keep_latest={self.keep_latest} -> "
            f"{self.n_surviving} surviving checkpoint(s): {list(self.surviving_steps)}"
        )


def retention_plan(
    *,
    total_steps: int,
    save_interval: int,
    keep_period: int | None,
    keep_latest: int | None,
) -> RetentionPlan:
    """Compute exactly which saved steps survive.

    orbax semantics as used by omegalax: a checkpoint is kept if it is one of the
    last ``keep_latest`` saves, or if its step is a multiple of ``keep_period``.
    Everything else is garbage-collected. ``keep_latest=1`` with
    ``keep_period=305`` and ``total_steps=300`` therefore keeps exactly one
    checkpoint — the last — which is the defect-#12 configuration.
    """
    if total_steps < 1 or save_interval < 1:
        raise SchemaError(
            f"total_steps and save_interval must be >= 1, got {total_steps}/{save_interval}"
        )
    saved = [s for s in range(save_interval, total_steps + 1, save_interval)]
    if saved and saved[-1] != total_steps:
        saved.append(total_steps)  # omegalax always writes a final checkpoint
    keep: set[int] = set()
    if keep_latest:
        keep.update(saved[-keep_latest:])
    if keep_period:
        keep.update(s for s in saved if s % keep_period == 0)
    if not keep_latest and not keep_period:
        keep.update(saved)
    return RetentionPlan(
        total_steps=total_steps,
        save_interval=save_interval,
        keep_period=keep_period,
        keep_latest=keep_latest,
        surviving_steps=tuple(sorted(keep)),
    )


def validate_retention(
    *,
    total_steps: int,
    save_interval: int,
    keep_period: int | None,
    keep_latest: int | None,
    min_checkpoints_for_selection: int = 3,
) -> RetentionPlan:
    """Refuse a retention config that makes checkpoint selection impossible.

    Raises:
        RetentionError: fewer than ``min_checkpoints_for_selection`` checkpoints
            would survive. Selecting a checkpoint on val loss requires having
            more than one to choose from; discovering otherwise *after* the run
            costs the whole run.
    """
    plan = retention_plan(
        total_steps=total_steps,
        save_interval=save_interval,
        keep_period=keep_period,
        keep_latest=keep_latest,
    )
    if plan.n_surviving < min_checkpoints_for_selection:
        raise RetentionError(
            f"retention would leave only {plan.n_surviving} checkpoint(s) "
            f"({list(plan.surviving_steps)}) but checkpoint selection needs at least "
            f"{min_checkpoints_for_selection} (defect #12: keep_latest=1 + "
            f"keep_period > total_steps deletes every intermediate checkpoint). "
            f"Set keep_period to a multiple of save_interval, or raise keep_latest. "
            f"Plan: {plan.describe()}"
        )
    return plan


# ---------------------------------------------------------------------------
# Defect #11: val coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValPlan:
    n_val_records: int
    global_batch_size: int
    val_steps: int
    covers_full_split: bool
    n_records_scored: int

    def describe(self) -> str:
        return (
            f"val: {self.n_records_scored}/{self.n_val_records} records per eval "
            f"(val_steps={self.val_steps} x gbs={self.global_batch_size}) "
            f"full_split={self.covers_full_split}"
        )


def resolve_val_steps(
    *,
    n_val_records: int,
    global_batch_size: int,
    requested_val_steps: int | None = None,
    allow_partial: bool = False,
) -> ValPlan:
    """Return the ``val_steps`` that covers the whole val split.

    **Defect #11.** With a non-restarting grain iterator, ``val_steps`` smaller
    than ``ceil(n_val / gbs)`` scores a *different* subset at every eval, so the
    val curve mixes measurement noise with a moving denominator and is comparable
    only within matched windows. The default here is full coverage; a partial
    evaluation is available but must be opted into by name.

    Raises:
        ValCoverageError: a partial ``val_steps`` was requested without
            ``allow_partial=True``.
    """
    if n_val_records < 1 or global_batch_size < 1:
        raise SchemaError(
            f"n_val_records and global_batch_size must be >= 1, "
            f"got {n_val_records}/{global_batch_size}"
        )
    full = math.ceil(n_val_records / global_batch_size)
    steps = full if requested_val_steps is None else requested_val_steps
    if steps < 1:
        raise SchemaError(f"val_steps must be >= 1, got {steps}")
    scored = min(steps * global_batch_size, n_val_records)
    covers = steps >= full
    if not covers and not allow_partial:
        raise ValCoverageError(
            f"val_steps={steps} x gbs={global_batch_size} scores {scored} of "
            f"{n_val_records} val records. With a non-restarting iterator each eval "
            f"sees a DIFFERENT subset, so val losses are not comparable across evals "
            f"(defect #11). Use val_steps={full} for full coverage, or pass "
            f"allow_partial=True if you really want a moving window."
        )
    return ValPlan(
        n_val_records=n_val_records,
        global_batch_size=global_batch_size,
        val_steps=steps,
        covers_full_split=covers,
        n_records_scored=scored,
    )


# ---------------------------------------------------------------------------
# Defect #13: surface val loss to stdout
# ---------------------------------------------------------------------------

#: omegalax's own wandb-only val logging is invisible in a slurm log. We scrape
#: whatever val-ish scalar appears and echo it, so the curve is in the log file.
#: ``nan``/``inf`` must be matched too: a diverged run is a real, reportable event,
#: not an unparseable line to skip.
_NUM = r"(?:[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf|Infinity))"
_VAL_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\bval[/_]loss\b\s*[:=]\s*(?P<value>{_NUM})", re.IGNORECASE),
    re.compile(rf"\bvalidation[_ ]loss\b\s*[:=]\s*(?P<value>{_NUM})", re.IGNORECASE),
    re.compile(rf"'val/loss'\s*:\s*(?P<value>{_NUM})", re.IGNORECASE),
)
_STEP_RE = re.compile(r"\bstep\s*[:=]?\s*(?P<step>\d+)")


@dataclass
class ValLossTee:
    """Scrape val loss out of a training log and re-emit it to stdout.

    Wrap omegalax's stdout with :meth:`feed`, or post-process a finished log with
    :meth:`scan`. Either way the run's val curve ends up in the slurm log
    (defect #13) and in :attr:`points`, so checkpoint selection does not depend on
    a wandb datastore read (which has its own defect, #10).
    """

    points: list[tuple[int | None, float]] = field(default_factory=list)
    last_step: int | None = None

    def feed(self, line: str) -> str | None:
        """Consume one log line; return the echo line if it carried a val loss."""
        m_step = _STEP_RE.search(line)
        if m_step:
            self.last_step = int(m_step.group("step"))
        for rx in _VAL_LINE_RES:
            m = rx.search(line)
            if not m:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            if math.isnan(value):
                # NaN val loss is a real event (divergence) and must be visible,
                # not filtered out as "unparseable".
                self.points.append((self.last_step, value))
                return f"[rft.val] step={self.last_step} val/loss=nan  <-- DIVERGED"
            self.points.append((self.last_step, value))
            return f"[rft.val] step={self.last_step} val/loss={value:.6f}"
        return None

    def scan(self, lines: Iterable[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            echo = self.feed(line)
            if echo is not None:
                out.append(echo)
        return out

    def best(self) -> tuple[int | None, float]:
        """Step and value of the lowest non-NaN val loss.

        Raises:
            ValCoverageError: no val point was ever seen. "No val rows exist" was
                itself a defect (#10, offline-wandb ``item.nested_key``); an absent
                val curve must be an error, never an empty selection.
        """
        finite = [(s, v) for s, v in self.points if not math.isnan(v)]
        if not finite:
            raise ValCoverageError(
                f"no finite val/loss points found in the log ({len(self.points)} raw "
                "points). Checkpoint selection cannot proceed; do not fall back to "
                "the last checkpoint silently."
            )
        return min(finite, key=lambda sv: sv[1])


# ---------------------------------------------------------------------------
# invocation building
# ---------------------------------------------------------------------------


@dataclass
class OmegalaxInvocation:
    """A validated omegalax ``train_vlm_sft.py`` invocation.

    This builds a command line; it does not run training itself and it does not
    interpret any omegalax flag beyond the ones whose *misuse* caused a defect.
    """

    omegalax_repo: Path
    script: str
    flags: dict[str, Any]
    retention: RetentionPlan
    val: ValPlan

    def argv(self) -> list[str]:
        args = ["uv", "run", "--project", str(self.omegalax_repo), "python", self.script]
        for key, value in sorted(self.flags.items()):
            if isinstance(value, bool):
                args.append(f"--{key}={'true' if value else 'false'}")
            else:
                args.append(f"--{key}={value}")
        return args

    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv())

    def describe(self) -> str:
        return f"{self.retention.describe()}\n{self.val.describe()}\n{self.command()}"


def build_omegalax_invocation(
    *,
    omegalax_repo: str | Path,
    script: str = "train_vlm_sft.py",
    flags: Mapping[str, Any],
    total_steps: int,
    save_interval: int,
    n_val_records: int,
    global_batch_size: int,
    requested_val_steps: int | None = None,
    allow_partial_val: bool = False,
    min_checkpoints_for_selection: int = 3,
) -> OmegalaxInvocation:
    """Validate then build the omegalax command line.

    Every guard runs *before* the job is submitted, so a misconfigured retention
    or val plan costs a second rather than a training run.
    """
    repo = Path(omegalax_repo)
    if not repo.is_dir():
        raise SchemaError(f"omegalax repo not found: {repo}")
    merged = dict(flags)
    plan = validate_retention(
        total_steps=total_steps,
        save_interval=save_interval,
        keep_period=merged.get("keep_period"),
        keep_latest=merged.get("keep_latest"),
        min_checkpoints_for_selection=min_checkpoints_for_selection,
    )
    val = resolve_val_steps(
        n_val_records=n_val_records,
        global_batch_size=global_batch_size,
        requested_val_steps=requested_val_steps,
        allow_partial=allow_partial_val,
    )
    merged["val_steps"] = val.val_steps
    assert_required_flags(merged)
    return OmegalaxInvocation(
        omegalax_repo=repo, script=script, flags=merged, retention=plan, val=val
    )


def iter_stdout_with_val_tee(lines: Iterable[str], tee: ValLossTee) -> Iterator[str]:
    """Pass a subprocess's stdout through, injecting ``[rft.val]`` echo lines."""
    for line in lines:
        yield line
        echo = tee.feed(line)
        if echo is not None:
            yield echo + "\n"


def count_jsonl_records(path: str | Path) -> int:
    """Number of non-blank lines in a JSONL file (used to size the val plan)."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"record file not found: {p}")
    with p.open() as fh:
        return sum(1 for line in fh if line.strip())


def assert_paths_on_project(paths: Sequence[str | Path]) -> None:
    """Every training output must live on /fast/project (8TB), not /fast/home (95G)."""
    bad = [str(p) for p in paths if not str(Path(p).resolve()).startswith("/fast/project")]
    if bad:
        raise SchemaError(
            "training outputs must be on /fast/project (8TB); these are not: "
            f"{bad!r}. /fast/home is 95G and has run out twice."
        )
