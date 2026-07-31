"""Separate single-step, multi-step, and matched native-absolute metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence

from stage5_rft.schema import EpisodeTrace, FailureKind
from stage5_rft.util import ContractError


@dataclass(frozen=True)
class ConditionMetrics:
    condition: str
    episodes: int
    task_success_rate: float
    mean_return: float
    mean_steps: float
    action_parse_rate: float
    action_dispatch_rate: float
    first_step_terminal_success_rate: float
    recovery_success_rate: float | None
    infrastructure_failure_rate: float
    failure_taxonomy: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


INFRA_FAILURES = frozenset(
    {
        FailureKind.RESET_FAILED,
        FailureKind.RESET_NONDETERMINISTIC,
        FailureKind.POLICY_TIMEOUT,
        FailureKind.POLICY_ERROR,
        FailureKind.POLICY_PROVENANCE_MISMATCH,
        FailureKind.DISPATCH_ERROR,
        FailureKind.OBSERVATION_ERROR,
        FailureKind.REWARD_ERROR,
        FailureKind.VM_ERROR,
        FailureKind.ACTOR_INTERRUPTED,
        FailureKind.REPLAY_DIVERGENCE,
    }
)


def summarize_condition(episodes: Sequence[EpisodeTrace]) -> ConditionMetrics:
    if not episodes:
        raise ContractError("cannot summarize an empty condition")
    conditions = {episode.condition for episode in episodes}
    if len(conditions) != 1:
        raise ContractError(
            "single-step and multi-step episodes must never be pooled; use summarize_separately"
        )
    for episode in episodes:
        episode.validate()
    condition = next(iter(conditions))
    steps = [step for episode in episodes for step in episode.steps]
    failures = Counter(
        episode.terminal_failure.value
        for episode in episodes
        if episode.terminal_failure != FailureKind.NONE
    )
    infra = sum(episode.terminal_failure in INFRA_FAILURES for episode in episodes)
    first_success = sum(episode.success and len(episode.steps) == 1 for episode in episodes)
    recovery: float | None = None
    if condition == "multi_step":
        at_risk = [episode for episode in episodes if len(episode.steps) > 1]
        recovery = (
            sum(episode.success for episode in at_risk) / len(at_risk) if at_risk else 0.0
        )
    return ConditionMetrics(
        condition=condition,
        episodes=len(episodes),
        task_success_rate=sum(episode.success for episode in episodes) / len(episodes),
        mean_return=fmean(episode.total_reward for episode in episodes),
        mean_steps=fmean(len(episode.steps) for episode in episodes),
        action_parse_rate=sum(step.action.valid for step in steps) / len(steps),
        action_dispatch_rate=sum(step.action.dispatched for step in steps) / len(steps),
        first_step_terminal_success_rate=first_success / len(episodes),
        recovery_success_rate=recovery,
        infrastructure_failure_rate=infra / len(episodes),
        failure_taxonomy=dict(sorted(failures.items())),
    )


def summarize_separately(episodes: Iterable[EpisodeTrace]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeTrace]] = {"single_step": [], "multi_step": []}
    for episode in episodes:
        grouped[episode.condition].append(episode)
    return {
        name: summarize_condition(rows).as_dict()
        for name, rows in grouped.items()
        if rows
    }


@dataclass(frozen=True)
class MatchedParityMetrics:
    condition: str
    pairs: int
    pair_coverage: float
    candidate_success_rate: float
    baseline_success_rate: float
    success_delta_pp: float
    candidate_mean_return: float
    baseline_mean_return: float
    mean_return_delta: float
    candidate_only_successes: int
    baseline_only_successes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def matched_native_absolute_parity(
    candidate: Sequence[EpisodeTrace], baseline: Sequence[EpisodeTrace]
) -> dict[str, dict[str, Any]]:
    if not candidate or not baseline:
        raise ContractError("matched parity needs non-empty candidate and baseline episodes")
    candidate_roles = {episode.policy.role for episode in candidate}
    baseline_roles = {episode.policy.role for episode in baseline}
    if candidate_roles != {"candidate"}:
        raise ContractError(f"candidate policy roles are not pinned: {sorted(candidate_roles)}")
    if baseline_roles != {"native_absolute_baseline"}:
        raise ContractError("baseline must be explicitly marked native_absolute_baseline")
    if any("absolute" not in episode.policy.action_schema.lower() for episode in baseline):
        raise ContractError("matched baseline is not a native absolute action policy")

    candidate_by = {episode.match_key: episode for episode in candidate}
    baseline_by = {episode.match_key: episode for episode in baseline}
    if len(candidate_by) != len(candidate) or len(baseline_by) != len(baseline):
        raise ContractError("matched parity contains duplicate reset/task cells")
    if set(candidate_by) != set(baseline_by):
        missing_candidate = sorted(set(baseline_by) - set(candidate_by))
        missing_baseline = sorted(set(candidate_by) - set(baseline_by))
        raise ContractError(
            "candidate/native-absolute cells are not exactly matched: "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )

    grouped: dict[str, list[tuple[EpisodeTrace, EpisodeTrace]]] = {
        "single_step": [],
        "multi_step": [],
    }
    for key in sorted(candidate_by):
        cand, base = candidate_by[key], baseline_by[key]
        if cand.policy.sampling != base.policy.sampling:
            raise ContractError(
                f"matched cell {key} differs in sampling tuple; baseline is not controlled"
            )
        grouped[cand.condition].append((cand, base))

    out: dict[str, dict[str, Any]] = {}
    total_expected = len(candidate_by)
    for condition, pairs in grouped.items():
        if not pairs:
            continue
        n = len(pairs)
        cand_success = sum(c.success for c, _ in pairs) / n
        base_success = sum(b.success for _, b in pairs) / n
        report = MatchedParityMetrics(
            condition=condition,
            pairs=n,
            pair_coverage=n / sum(e.condition == condition for e in candidate),
            candidate_success_rate=cand_success,
            baseline_success_rate=base_success,
            success_delta_pp=100.0 * (cand_success - base_success),
            candidate_mean_return=fmean(c.total_reward for c, _ in pairs),
            baseline_mean_return=fmean(b.total_reward for _, b in pairs),
            mean_return_delta=fmean(c.total_reward - b.total_reward for c, b in pairs),
            candidate_only_successes=sum(c.success and not b.success for c, b in pairs),
            baseline_only_successes=sum(b.success and not c.success for c, b in pairs),
        )
        out[condition] = report.as_dict()
    if sum(v["pairs"] for v in out.values()) != total_expected:
        raise ContractError("paired metric accounting lost episodes")
    return out
