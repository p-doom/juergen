"""Stage 2 - Score / filter: the accept-reject predicate.

Rejection sampling is one predicate applied to many rollouts, so the predicate is
where a defect does the most damage. This module makes the three historical
mistakes structurally impossible:

* the reward is read through :func:`rft.rewards.read_reward` at an **explicit,
  schema-validated path** (defect #1: it lives at ``scores.reward``);
* an **unscored** rollout (NaN / null reward, i.e. ``evaluate()`` threw) is
  neither accepted nor counted as a rejection — it is a *third* outcome,
  :attr:`Verdict.UNSCORED`, reported separately (defect #3);
* nothing is wrapped in a bare ``except``. A rollout whose payload does not match
  the schema is :attr:`Verdict.MALFORMED` and is reported with its error string
  (defect #9).

An :class:`AcceptReport` carries all four counts, and
:meth:`AcceptReport.assert_healthy` refuses to hand a filtered dataset downstream
when the unscored or malformed fraction is high enough to mean "the instrument
broke", not "the model is bad".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from rft.errors import SchemaError
from rft.rewards import (
    CANONICAL_REWARD_PATH,
    UNSCORED,
    Reward,
    has_field,
    read_reward,
)


class Verdict(Enum):
    """Outcome of applying the accept predicate to one rollout."""

    ACCEPT = "accept"
    REJECT = "reject"
    #: The task was never scored (NaN/null reward). NOT a rejection.
    UNSCORED = "unscored"
    #: The payload did not match the schema. NOT a rejection.
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ScoredRollout:
    sample_id: str
    task_id: str
    reward: Reward
    verdict: Verdict
    error: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_scored(self) -> bool:
        return self.verdict in (Verdict.ACCEPT, Verdict.REJECT)


#: Predicate over a *real* reward value. It never sees UNSCORED, so an
#: implementer cannot accidentally treat NaN as a low reward.
AcceptPredicate = Callable[[float], bool]


def threshold_predicate(min_reward: float) -> AcceptPredicate:
    """Accept iff ``reward >= min_reward``. The default RFT predicate."""

    def _accept(reward: float) -> bool:
        return reward >= min_reward

    _accept.__doc__ = f"accept iff reward >= {min_reward}"
    return _accept


@dataclass
class AcceptReport:
    """Counts for every outcome, plus the health assertion.

    ``unscored`` and ``malformed`` are deliberately *not* folded into
    ``rejected``: a run where they dominate has an instrument problem, and the
    accept rate computed over them is meaningless.
    """

    n_total: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_unscored: int = 0
    n_malformed: int = 0
    malformed_examples: list[str] = field(default_factory=list)
    unscored_examples: list[str] = field(default_factory=list)
    reward_path: str = CANONICAL_REWARD_PATH
    #: How many payloads carried each optional field we were asked about. Exists
    #: so that "field absent from 100% of files" (defect #2) is visible in the
    #: report rather than discovered as a fake verdict.
    field_coverage: dict[str, int] = field(default_factory=dict)

    @property
    def n_scored(self) -> int:
        return self.n_accepted + self.n_rejected

    @property
    def accept_rate(self) -> float:
        """Accept rate over SCORED rollouts only.

        Raises:
            SchemaError: nothing was scored. A 0/0 accept rate is not 0.0.
        """
        if not self.n_scored:
            raise SchemaError(
                f"0 of {self.n_total} rollouts were scored "
                f"({self.n_unscored} unscored, {self.n_malformed} malformed); "
                "an accept rate over an empty denominator is undefined, not 0.0"
            )
        return self.n_accepted / self.n_scored

    def assert_healthy(
        self, *, max_unscored_fraction: float = 0.05, max_malformed_fraction: float = 0.0
    ) -> None:
        """Refuse to pass a degraded filter result downstream."""
        if not self.n_total:
            raise SchemaError("no rollouts scored at all")
        unscored_frac = self.n_unscored / self.n_total
        malformed_frac = self.n_malformed / self.n_total
        problems: list[str] = []
        if unscored_frac > max_unscored_fraction:
            problems.append(
                f"{self.n_unscored}/{self.n_total} ({unscored_frac:.1%}) rollouts were "
                f"NEVER SCORED (> {max_unscored_fraction:.1%}); NaN reward means "
                "evaluate() threw, not reward 0 (defect #3). Examples: "
                f"{self.unscored_examples[:3]!r}"
            )
        if malformed_frac > max_malformed_fraction:
            problems.append(
                f"{self.n_malformed}/{self.n_total} ({malformed_frac:.1%}) rollouts had "
                f"malformed payloads (> {max_malformed_fraction:.1%}). Examples: "
                f"{self.malformed_examples[:3]!r}"
            )
        if problems:
            raise SchemaError("filter result is not healthy:\n  - " + "\n  - ".join(problems))

    def describe(self) -> str:
        rate = f"{self.accept_rate:.3f}" if self.n_scored else "undefined (0 scored)"
        cov = ", ".join(f"{k}={v}/{self.n_total}" for k, v in sorted(self.field_coverage.items()))
        return (
            f"stage=score reward_path={self.reward_path!r} total={self.n_total} "
            f"accepted={self.n_accepted} rejected={self.n_rejected} "
            f"unscored={self.n_unscored} malformed={self.n_malformed} "
            f"accept_rate(scored)={rate}"
            + (f"\n  field coverage: {cov}" if cov else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_unscored": self.n_unscored,
            "n_malformed": self.n_malformed,
            "n_scored": self.n_scored,
            "accept_rate_over_scored": self.accept_rate if self.n_scored else None,
            "reward_path": self.reward_path,
            "field_coverage": self.field_coverage,
            "malformed_examples": self.malformed_examples[:50],
            "unscored_examples": self.unscored_examples[:50],
        }


def score_rollout(
    payload: Mapping[str, Any],
    *,
    accept: AcceptPredicate,
    reward_path: str = CANONICAL_REWARD_PATH,
) -> ScoredRollout:
    """Apply the predicate to one rollout payload.

    Never raises for a data problem: returns a :attr:`Verdict.MALFORMED` rollout
    carrying the error text, so the caller's report keeps the count. Raises only
    for a *programming* error (e.g. ``accept`` is not callable).
    """
    if not callable(accept):
        raise TypeError("accept must be callable")
    sample_id = str(payload.get("sample_id", "<missing sample_id>"))
    task_id = str(payload.get("task_id", "<missing task_id>"))
    if "error" in payload:
        return ScoredRollout(
            sample_id=sample_id,
            task_id=task_id,
            reward=UNSCORED,
            verdict=Verdict.MALFORMED,
            error=f"rollout carried a sampling error: {payload['error']!r}",
            payload=payload,
        )
    try:
        reward = read_reward(payload, path=reward_path)
    except Exception as exc:  # noqa: BLE001 - converted to a counted verdict
        return ScoredRollout(
            sample_id=sample_id,
            task_id=task_id,
            reward=UNSCORED,
            verdict=Verdict.MALFORMED,
            error=f"{type(exc).__name__}: {exc}",
            payload=payload,
        )
    if reward is UNSCORED:
        return ScoredRollout(
            sample_id=sample_id,
            task_id=task_id,
            reward=UNSCORED,
            verdict=Verdict.UNSCORED,
            error="reward is NaN/null: the task was never scored",
            payload=payload,
        )
    verdict = Verdict.ACCEPT if accept(float(reward)) else Verdict.REJECT
    return ScoredRollout(
        sample_id=sample_id, task_id=task_id, reward=reward, verdict=verdict, payload=payload
    )


def score_rollouts(
    payloads: Iterable[Mapping[str, Any]],
    *,
    accept: AcceptPredicate,
    reward_path: str = CANONICAL_REWARD_PATH,
    coverage_fields: Sequence[str] = ("success", "scores.reward", "final_reward"),
) -> tuple[list[ScoredRollout], AcceptReport]:
    """Score every rollout, returning the results and the report.

    ``coverage_fields`` are counted, not required. This is the defect-#2
    tripwire: if a downstream verdict is about to be phrased in terms of
    ``success``, the report already says ``success=0/N`` and the verdict is
    obviously about a nonexistent field.
    """
    report = AcceptReport(reward_path=reward_path)
    report.field_coverage = {f: 0 for f in coverage_fields}
    out: list[ScoredRollout] = []
    for payload in payloads:
        report.n_total += 1
        for f in coverage_fields:
            if has_field(payload, f):
                report.field_coverage[f] += 1
        scored = score_rollout(payload, accept=accept, reward_path=reward_path)
        out.append(scored)
        if scored.verdict is Verdict.ACCEPT:
            report.n_accepted += 1
        elif scored.verdict is Verdict.REJECT:
            report.n_rejected += 1
        elif scored.verdict is Verdict.UNSCORED:
            report.n_unscored += 1
            if len(report.unscored_examples) < 50:
                report.unscored_examples.append(scored.sample_id)
        else:
            report.n_malformed += 1
            if len(report.malformed_examples) < 50:
                report.malformed_examples.append(f"{scored.sample_id}: {scored.error}")
    return out, report


@dataclass(frozen=True)
class GroupStats:
    """Per-task accept statistics — the thing rejection sampling actually needs.

    A task where all ``k`` rollouts are accepted, or all rejected, contributes no
    contrast. ``n_degenerate_groups`` is therefore reported alongside the accept
    rate; a headline accept rate hides it completely.
    """

    n_groups: int
    n_all_accepted: int
    n_all_rejected: int
    n_mixed: int
    n_groups_with_unscored: int

    @property
    def n_degenerate(self) -> int:
        return self.n_all_accepted + self.n_all_rejected

    def describe(self) -> str:
        return (
            f"groups={self.n_groups} mixed={self.n_mixed} all_accept={self.n_all_accepted} "
            f"all_reject={self.n_all_rejected} with_unscored={self.n_groups_with_unscored}"
        )


def group_stats(scored: Sequence[ScoredRollout]) -> GroupStats:
    """Group the scored rollouts by task and summarise contrast."""
    by_task: dict[str, list[ScoredRollout]] = {}
    for s in scored:
        by_task.setdefault(s.task_id, []).append(s)
    n_all_a = n_all_r = n_mixed = n_unsc = 0
    for group in by_task.values():
        verdicts = {s.verdict for s in group}
        if Verdict.UNSCORED in verdicts or Verdict.MALFORMED in verdicts:
            n_unsc += 1
        accepted = sum(1 for s in group if s.verdict is Verdict.ACCEPT)
        rejected = sum(1 for s in group if s.verdict is Verdict.REJECT)
        if accepted and rejected:
            n_mixed += 1
        elif accepted:
            n_all_a += 1
        elif rejected:
            n_all_r += 1
    return GroupStats(
        n_groups=len(by_task),
        n_all_accepted=n_all_a,
        n_all_rejected=n_all_r,
        n_mixed=n_mixed,
        n_groups_with_unscored=n_unsc,
    )


def write_accepted(
    scored: Sequence[ScoredRollout], out_path: str | Path, report: AcceptReport
) -> int:
    """Write accepted rollouts to JSONL and the report beside them.

    Raises:
        SchemaError: nothing was accepted. An empty accepted set is a legitimate
            *finding*, but it must not be written as a silently-empty dataset that
            the next stage trains on; the caller has to handle it deliberately.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    accepted = [s for s in scored if s.verdict is Verdict.ACCEPT]
    report_path = out.with_name(out.stem + "_score_report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2))
    if not accepted:
        raise SchemaError(
            f"0 of {report.n_total} rollouts were accepted; refusing to write an empty "
            f"training set. Report written to {report_path}. "
            f"{report.n_unscored} were never scored and {report.n_malformed} malformed - "
            "check the instrument before concluding the model cannot do the task."
        )
    with out.open("w") as fh:
        for s in accepted:
            fh.write(json.dumps(dict(s.payload), ensure_ascii=False, sort_keys=True) + "\n")
    return len(accepted)
