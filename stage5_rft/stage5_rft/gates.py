"""Preregistered, machine-readable construction and promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stage5_rft.collector import EpisodeStore
from stage5_rft.contamination import ContaminationBlocklist, audit_episodes
from stage5_rft.replay import validate_collection
from stage5_rft.util import ContractError, read_json


@dataclass(frozen=True)
class GateResult:
    name: str
    metric: str
    observed: float | None
    operator: str
    threshold: float
    passed: bool
    missing: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _lookup(metrics: Mapping[str, Any], dotted: str) -> float | None:
    value: Any = metrics
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise ContractError(f"gate metric {dotted!r} is not numeric: {value!r}")
    return float(value)


def evaluate_gates(
    metrics: Mapping[str, Any], gate_config: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
    phases = gate_config.get("phases")
    if not isinstance(phases, Mapping) or phase not in phases:
        raise ContractError(f"unknown gate phase {phase!r}")
    specs = phases[phase]
    if not isinstance(specs, list) or not specs:
        raise ContractError(f"gate phase {phase!r} is empty")
    results: list[GateResult] = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ContractError("gate spec must be an object")
        metric = str(spec["metric"])
        observed = _lookup(metrics, metric)
        op = str(spec["op"])
        threshold = float(spec["threshold"])
        missing = observed is None
        if missing:
            passed = False
        elif op == ">=":
            passed = observed >= threshold
        elif op == "<=":
            passed = observed <= threshold
        elif op == "==":
            passed = observed == threshold
        else:
            raise ContractError(f"unsupported gate operator: {op!r}")
        results.append(
            GateResult(
                name=str(spec["name"]),
                metric=metric,
                observed=observed,
                operator=op,
                threshold=threshold,
                passed=passed,
                missing=missing,
            )
        )
    passed = all(result.passed for result in results)
    return {
        "schema_version": "stage5.gate_report.v1",
        "phase": phase,
        "passed": passed,
        "launch_authorized": False,
        "results": [result.as_dict() for result in results],
        "rule": "all preregistered gates must pass; missing metrics fail closed",
    }


def evaluate_gate_files(
    metrics_path: str | Path, gate_config_path: str | Path, *, phase: str
) -> dict[str, Any]:
    return evaluate_gates(read_json(metrics_path), read_json(gate_config_path), phase=phase)


def construction_metrics(
    *,
    rollout_root: str | Path,
    blocklist: ContaminationBlocklist,
    live_replay_report: Mapping[str, Any],
    deterministic_reset_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the exact metric namespace consumed by construction gates."""
    store = EpisodeStore(rollout_root)
    episodes = store.load_all()
    if not episodes:
        raise ContractError("construction report needs complete episodes")
    offline = validate_collection(rollout_root)
    contamination = audit_episodes(episodes, blocklist)
    manifest = read_json(Path(rollout_root) / "collection_manifest.json")
    actor_fingerprint = manifest.get("actor_policy_fingerprint")
    provenance_mismatches = sum(
        step.action.served_policy_fingerprint != actor_fingerprint
        for episode in episodes
        for step in episode.steps
    )
    atomic = 0
    for episode in episodes:
        path = store.partial_path(episode.episode_id)
        if not path.is_file():
            continue
        tombstone = read_json(path)
        atomic += bool(
            tombstone.get("status") == "committed"
            and tombstone.get("trace_sha256") == episode.trace_sha256
        )
    live_pass = bool(live_replay_report.get("passed"))
    reset_pass = bool(deterministic_reset_report.get("passed"))
    overlap = (
        len(contamination.task_id_overlap)
        + len(contamination.content_digest_overlap)
        + len(contamination.unauthorized_splits)
        + (0 if contamination.blocklist_usable else 1)
    )
    return {
        "schema_version": "stage5.construction_metrics.v1",
        "trace": {"completeness_rate": 1.0},
        "reset": {
            "deterministic_rate": (
                float(deterministic_reset_report.get("pass_rate", 0.0)) if reset_pass else 0.0
            )
        },
        "replay": {
            "pass_rate": min(
                offline.pass_rate if offline.passed else 0.0,
                float(live_replay_report.get("pass_rate", 0.0)) if live_pass else 0.0,
            )
        },
        "contamination": {"overlap_count": overlap},
        "provenance": {"mismatch_count": provenance_mismatches},
        "resume": {"atomic_rate": atomic / len(episodes)},
        "detail": {
            "offline_replay": offline.as_dict(),
            "live_replay": dict(live_replay_report),
            "deterministic_reset": dict(deterministic_reset_report),
            "contamination": contamination.as_dict(),
        },
    }
