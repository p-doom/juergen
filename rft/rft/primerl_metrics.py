"""Correct reading of prime-rl's on-line RL telemetry.

**Defect #6.** prime-rl's per-step ``Reward`` is the mean over the batch that
*survives* the ``zero_advantage`` filter. Confirmed in the source:
``orchestrator/orchestrator.py:747-749`` prints ``Reward {eff.reward.mean()}`` where
``eff = rollouts.effective.metrics`` and ``effective`` is
``[r for r in rollouts if not r.has_error and not r.is_filtered]``
(``orchestrator/metrics.py:355-357``); ``is_filtered`` is set by any *enforcing*
filter (``orchestrator/filters.py:170-171``), and ``ZeroAdvantageFilterConfig`` is
enforcing **by default** in ``post_batch_filters``
(``configs/orchestrator.py:386,468-472``). Groups whose rollouts all scored
identically have zero advantage and are dropped before the mean is taken, so:

* the reported ``Reward`` is **structurally incapable of showing a climb** — as the
  policy improves, more groups become uniformly-successful and are filtered out,
  which removes exactly the high-reward mass that would have raised the mean;
* it was measured ~2.7x biased high relative to the unfiltered mean;
* ``Trainable N/M`` (``orchestrator.py:749``) is a **rollout** count, not a group
  count: ``n_generated = len(rollouts)`` and
  ``n_trainable = sum(1 for r in rollouts if r.is_trainable)`` where
  ``is_trainable`` is "a nonzero advantage on some token" (``types.py:130-134``).
  Reading it as "N of M groups are trainable" overstates group-level yield by the
  group size.

**The asymmetry is the trap.** In one log line, ``Reward`` is over ``effective``
(errors and filtered rollouts removed) while ``Trainable`` is over raw ``rollouts``
(errors and filtered rollouts included). The two numbers have **different
denominators**, so their ratio means nothing and neither can be compared to a
generation-time mean reward.

This module provides the corrected readings, and refuses to compute the biased one
without saying so. Nothing here re-implements prime-rl; it only interprets its
logs and its rollout dumps.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rft.errors import SchemaError
from rft.rewards import UNSCORED, Reward, UnscoredPolicy, aggregate_rewards

#: ``Trainable 384/512`` style line. Both numbers count ROLLOUTS.
_TRAINABLE_RE = re.compile(r"\bTrainable\s+(?P<n>\d+)\s*/\s*(?P<m>\d+)")
_REWARD_RE = re.compile(r"\bReward\s*[:=]\s*(?P<value>[-+0-9.eE]+)")


@dataclass(frozen=True)
class TrainableCounts:
    """``Trainable N/M`` decoded with its true unit.

    Attributes:
        n_trainable_rollouts: N — rollouts with non-zero advantage.
        n_total_rollouts: M — rollouts in the batch.
        group_size: rollouts per group (``k``), needed to convert to groups. It is
            **not** recoverable from the log line, so the caller must supply it.
    """

    n_trainable_rollouts: int
    n_total_rollouts: int
    group_size: int | None = None

    @property
    def rollout_fraction(self) -> float:
        if not self.n_total_rollouts:
            raise SchemaError("Trainable 0/0: undefined, not 0.0")
        return self.n_trainable_rollouts / self.n_total_rollouts

    @property
    def n_total_groups(self) -> int:
        if not self.group_size:
            raise SchemaError(
                "group_size is required to convert `Trainable N/M` (a ROLLOUT count, "
                "defect #6) into groups; it cannot be inferred from the log line"
            )
        if self.n_total_rollouts % self.group_size:
            raise SchemaError(
                f"total rollouts {self.n_total_rollouts} is not a multiple of "
                f"group_size {self.group_size}"
            )
        return self.n_total_rollouts // self.group_size

    def describe(self) -> str:
        base = (
            f"Trainable {self.n_trainable_rollouts}/{self.n_total_rollouts} ROLLOUTS "
            f"({self.rollout_fraction:.1%})"
        )
        if self.group_size:
            base += f"; {self.n_total_groups} groups of {self.group_size}"
        return base + "  [not a group count - defect #6]"


def parse_trainable_line(line: str, *, group_size: int | None = None) -> TrainableCounts:
    """Parse a ``Trainable N/M`` log line, labelling the unit correctly."""
    m = _TRAINABLE_RE.search(line)
    if not m:
        raise SchemaError(f"no `Trainable N/M` in line: {line!r}")
    return TrainableCounts(
        n_trainable_rollouts=int(m.group("n")),
        n_total_rollouts=int(m.group("m")),
        group_size=group_size,
    )


def parse_logged_reward(line: str) -> float:
    """Parse prime-rl's ``Reward`` scalar, which is the FILTERED-batch mean.

    Provided so the number can be *compared* against the unfiltered mean, never so
    it can be reported as the run's reward. See :func:`filtered_vs_unfiltered`.
    """
    m = _REWARD_RE.search(line)
    if not m:
        raise SchemaError(f"no `Reward` scalar in line: {line!r}")
    return float(m.group("value"))


@dataclass(frozen=True)
class RewardComparison:
    """The two means side by side, with the bias made explicit."""

    unfiltered_mean: float
    filtered_mean: float | None
    n_groups: int
    n_degenerate_groups: int
    n_rollouts: int
    n_rollouts_after_filter: int

    @property
    def bias_factor(self) -> float | None:
        if self.filtered_mean is None or self.unfiltered_mean == 0.0:
            return None
        return self.filtered_mean / self.unfiltered_mean

    def describe(self) -> str:
        bias = "n/a" if self.bias_factor is None else f"{self.bias_factor:.2f}x"
        filt = "n/a (every group degenerate)" if self.filtered_mean is None else (
            f"{self.filtered_mean:.4f}"
        )
        return (
            f"reward(unfiltered, REPORT THIS)={self.unfiltered_mean:.4f}  "
            f"reward(post-zero_advantage, what prime-rl prints)={filt}  bias={bias}\n"
            f"  groups={self.n_groups} degenerate(zero-advantage)={self.n_degenerate_groups} "
            f"rollouts={self.n_rollouts} -> {self.n_rollouts_after_filter} after filter"
        )


def zero_advantage_groups(groups: Sequence[Sequence[float]]) -> list[int]:
    """Indices of groups whose rewards are all equal (zero advantage)."""
    out: list[int] = []
    for i, g in enumerate(groups):
        if not g:
            raise SchemaError(f"group {i} is empty; a group with no rollouts is a bug")
        if len(set(g)) == 1:
            out.append(i)
    return out


def filtered_vs_unfiltered(groups: Sequence[Sequence[Reward]]) -> RewardComparison:
    """Compute both means from per-group rewards.

    ``groups[i]`` is the ``k`` rewards of task *i*. Unscored entries
    (:data:`rft.rewards.UNSCORED`) are excluded from both means and are *not*
    treated as zeros; a group of entirely-unscored rollouts is dropped with its
    count preserved in ``n_groups``.

    The **unfiltered** mean is the one to report. The **filtered** mean is what
    prime-rl prints, reproduced here only so the discrepancy is visible.
    """
    if not groups:
        raise SchemaError("no groups")
    scored_groups: list[list[float]] = []
    n_rollouts = 0
    for g in groups:
        n_rollouts += len(g)
        vals = [float(r) for r in g if r is not UNSCORED]
        scored_groups.append(vals)
    flat = [v for g in scored_groups for v in g]
    if not flat:
        raise SchemaError(
            "no scored rollouts across any group; refusing to report a reward mean "
            "(an unscored batch is not a zero-reward batch)"
        )
    unfiltered = aggregate_rewards(flat, unscored=UnscoredPolicy.RAISE).mean

    non_degenerate = [g for g in scored_groups if len(g) > 1 and len(set(g)) > 1]
    kept = [v for g in non_degenerate for v in g]
    filtered = (
        aggregate_rewards(kept, unscored=UnscoredPolicy.RAISE).mean if kept else None
    )
    return RewardComparison(
        unfiltered_mean=unfiltered,
        filtered_mean=filtered,
        n_groups=len(groups),
        n_degenerate_groups=len(scored_groups) - len(non_degenerate),
        n_rollouts=n_rollouts,
        n_rollouts_after_filter=len(kept),
    )


def groups_from_rollouts(
    rollouts: Sequence[Mapping[str, object]],
    *,
    task_key: str = "task_id",
    reward_path: str = "scores.reward",
) -> list[list[Reward]]:
    """Group rollout payloads by task and read each reward at an explicit path."""
    from rft.rewards import read_reward

    by_task: dict[str, list[Reward]] = {}
    for i, r in enumerate(rollouts):
        if task_key not in r:
            raise SchemaError(f"rollouts[{i}] has no {task_key!r}")
        by_task.setdefault(str(r[task_key]), []).append(read_reward(r, path=reward_path))
    return [by_task[k] for k in sorted(by_task)]
