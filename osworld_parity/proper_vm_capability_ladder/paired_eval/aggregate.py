from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import ARMS
from .manifest import EvaluationManifest
from .planning import TrialSpec


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("row is not an object")
                    rows.append(value)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot load result {path}: {exc}") from exc
    return rows


def aggregate_results(
    manifest: EvaluationManifest,
    plan: Iterable[TrialSpec],
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    trials = {trial.pair_id: trial for trial in plan}
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        pair_id = row.get("pair_id")
        if pair_id in rows:
            raise ValueError(f"duplicate pair result: {pair_id}")
        if pair_id not in trials:
            raise ValueError(f"result is not in the deterministic plan: {pair_id}")
        _validate_row(manifest, trials[pair_id], row)
        rows[pair_id] = row
    missing = sorted(set(trials) - set(rows))
    if missing:
        raise ValueError(f"missing paired results: {missing[:5]}")

    ordered = [rows[pair_id] for pair_id in sorted(rows)]
    included = [row for row in ordered if not row["exclusion"]["excluded"]]
    excluded = [row for row in ordered if row["exclusion"]["excluded"]]
    arm_values = {
        arm: [_arm(row, arm)["success"] for row in included]
        for arm in ARMS
    }
    for arm, values in arm_values.items():
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"included rows have non-boolean scores for {arm}")
    native = [int(value) for value in arm_values[ARMS[0]]]
    compact = [int(value) for value in arm_values[ARMS[1]]]
    differences = [right - left for left, right in zip(native, compact, strict=True)]

    strata: dict[str, Any] = {}
    stratum_keys = sorted({(row["mode"], row["horizon"]) for row in included})
    for mode, horizon in stratum_keys:
        subset = [
            row for row in included if row["mode"] == mode and row["horizon"] == horizon
        ]
        strata[f"{mode}/h{horizon}"] = _simple_metrics(subset)

    bootstrap = _paired_task_cluster_bootstrap(
        included,
        seed=manifest.bootstrap_seed,
        resamples=manifest.bootstrap_resamples,
    )
    mcnemar = _mcnemar_descriptive(native, compact)
    failure_counts: dict[str, int] = defaultdict(int)
    for row in excluded:
        for failure_class in row["exclusion"]["infra_failure_classes"]:
            failure_counts[failure_class] += 1
    return {
        "schema_version": 1,
        "report_type": "paired_complete_system_development",
        "suite": manifest.suite,
        "split": "development",
        "development_only": True,
        "comparison_scope": "complete_system",
        "comparison_label": manifest.comparison_label,
        "systems": {
            arm.name: {
                "checkpoint": arm.checkpoint,
                "checkpoint_sha256": arm.checkpoint_sha256,
                "prompt_id": arm.prompt_id,
                "prompt_sha256": arm.prompt_sha256,
                "action_interface": arm.action_interface,
            }
            for arm in manifest.arms
        },
        "pair_accounting": {
            "planned": len(trials),
            "complete": len(ordered),
            "included": len(included),
            "excluded_whole_pair": len(excluded),
            "infra_failure_class_counts": dict(sorted(failure_counts.items())),
            "exclusion_policy": "arm_blind_whole_pair_infrastructure_only",
        },
        "overall": {
            "n_pairs": len(included),
            "arm_success_rate": {
                ARMS[0]: _mean(native),
                ARMS[1]: _mean(compact),
            },
            "paired_difference_compact_minus_native": _mean(differences),
            "paired_task_cluster_bootstrap": bootstrap,
            "mcnemar_descriptive": mcnemar,
        },
        "by_mode_horizon": strata,
        "pass_at_k_feasibility": _pass_at_k(manifest, included),
    }


def _validate_row(
    manifest: EvaluationManifest,
    trial: TrialSpec,
    row: dict[str, Any],
) -> None:
    if row.get("schema_version") != 1 or row.get("record_type") != "paired_complete_system_trial":
        raise ValueError(f"{trial.pair_id}: result schema drift")
    if row.get("split") != "development" or row.get("development_only") is not True:
        raise ValueError(f"{trial.pair_id}: non-development result forbidden")
    if row.get("fixture_sha256") != trial.fixture_sha256:
        raise ValueError(f"{trial.pair_id}: fixture hash mismatch")
    if row.get("cell_id") != trial.cell_id or row.get("attempt_id") != trial.attempt_id:
        raise ValueError(f"{trial.pair_id}: attempt identity mismatch")
    if row.get("task_id") != trial.task_id:
        raise ValueError(f"{trial.pair_id}: task mismatch")
    pairing = row.get("pairing")
    expected_pairing = {
        "snapshot_id": trial.snapshot_id,
        "parameter_seed": trial.parameter_seed,
        "initial_cursor": list(trial.initial_cursor),
        "generation_seed": trial.generation_seed,
        "budget": trial.budget,
        "arm_order": list(trial.arm_order),
        "shard_index": trial.shard_index,
        "shard_count": trial.shard_count,
    }
    if pairing != expected_pairing:
        raise ValueError(f"{trial.pair_id}: pair invariants changed")
    arms = row.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError(f"{trial.pair_id}: result must contain two arm rows")
    if {value.get("arm") for value in arms if isinstance(value, dict)} != set(ARMS):
        raise ValueError(f"{trial.pair_id}: missing or duplicate arm")
    exclusion = row.get("exclusion")
    if not isinstance(exclusion, dict):
        raise ValueError(f"{trial.pair_id}: exclusion decision missing")
    if exclusion.get("policy") != "arm_blind_whole_pair_infrastructure_only":
        raise ValueError(f"{trial.pair_id}: exclusion policy drift")
    if exclusion.get("decision_inputs_contain_arm_identity") is not False:
        raise ValueError(f"{trial.pair_id}: arm-aware exclusion forbidden")
    classes = exclusion.get("infra_failure_classes")
    if not isinstance(classes, list) or classes != sorted(set(classes)):
        raise ValueError(f"{trial.pair_id}: invalid infrastructure exclusion classes")
    if exclusion.get("excluded") is not bool(classes):
        raise ValueError(f"{trial.pair_id}: exclusion/class mismatch")
    actual_classes = sorted(
        {
            value.get("infra_failure_class")
            for value in arms
            if value.get("infra_failure_class") is not None
        }
    )
    if classes != actual_classes:
        raise ValueError(f"{trial.pair_id}: exclusion was not derived from both arms")
    for arm in manifest.arms:
        system = row.get("systems", {}).get(arm.name)
        expected = {
            "action_interface": arm.action_interface,
            "checkpoint": arm.checkpoint,
            "checkpoint_sha256": arm.checkpoint_sha256,
            "prompt_id": arm.prompt_id,
            "prompt_sha256": arm.prompt_sha256,
            "generation": arm.generation,
        }
        if system != expected:
            raise ValueError(f"{trial.pair_id}: complete-system provenance drift for {arm.name}")


def _arm(row: dict[str, Any], arm: str) -> dict[str, Any]:
    matches = [value for value in row["arms"] if value["arm"] == arm]
    if len(matches) != 1:
        raise ValueError(f"arm is not unique in result: {arm}")
    return matches[0]


def _simple_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    native = [int(_arm(row, ARMS[0])["success"]) for row in rows]
    compact = [int(_arm(row, ARMS[1])["success"]) for row in rows]
    return {
        "n_pairs": len(rows),
        "arm_success_rate": {ARMS[0]: _mean(native), ARMS[1]: _mean(compact)},
        "paired_difference_compact_minus_native": _mean(
            [right - left for left, right in zip(native, compact, strict=True)]
        ),
    }


def _paired_task_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "resamples": resamples,
            "seed": seed,
            "cluster": "task_id",
            "confidence_interval_95": [None, None],
        }
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        clusters[row["task_id"]].append(
            int(_arm(row, ARMS[1])["success"]) - int(_arm(row, ARMS[0])["success"])
        )
    names = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(names) for _ in names]
        observations = [value for name in sampled for value in clusters[name]]
        values.append(sum(observations) / len(observations))
    values.sort()
    return {
        "resamples": resamples,
        "seed": seed,
        "cluster": "task_id",
        "confidence_interval_95": [_percentile(values, 0.025), _percentile(values, 0.975)],
    }


def _mcnemar_descriptive(native: list[int], compact: list[int]) -> dict[str, Any]:
    native_only = sum(left == 1 and right == 0 for left, right in zip(native, compact, strict=True))
    compact_only = sum(left == 0 and right == 1 for left, right in zip(native, compact, strict=True))
    discordant = native_only + compact_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(native_only, compact_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "label": "descriptive_only_no_independence_claim",
        "native_success_compact_failure": native_only,
        "native_failure_compact_success": compact_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
    }


def _pass_at_k(
    manifest: EvaluationManifest,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[row["cell_id"]].append(row)
    output: dict[str, Any] = {}
    for k in (1, 4, 8):
        eligible = [values for values in cells.values() if len(values) >= k]
        enough = bool(cells) and len(eligible) == len(cells)
        stochastic = {
            arm.name: _stochastic_generation(arm.generation) if k > 1 else True
            for arm in manifest.arms
        }
        feasible = enough and all(stochastic.values())
        estimates: dict[str, float | None] = {name: None for name in ARMS}
        if feasible:
            for arm in ARMS:
                estimates[arm] = _mean(
                    [
                        _unbiased_pass_at_k(
                            len(values),
                            sum(bool(_arm(row, arm)["success"]) for row in values),
                            k,
                        )
                        for values in eligible
                    ]
                )
        output[f"pass@{k}"] = {
            "feasible": feasible,
            "enough_complete_attempts_per_cell": enough,
            "cells_total": len(cells),
            "cells_with_at_least_k_attempts": len(eligible),
            "stochastic_generation_configured": stochastic,
            "paired_task_reset_per_attempt": True,
            "estimate_by_arm": estimates,
        }
    return output


def _stochastic_generation(generation: dict[str, Any]) -> bool:
    if generation.get("do_sample") is True:
        return float(generation.get("temperature", 1.0)) > 0
    return float(generation.get("temperature", 0.0)) > 0


def _unbiased_pass_at_k(n: int, successes: int, k: int) -> float:
    if n < k:
        raise ValueError("pass@k needs at least k attempts")
    failures = n - successes
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n, k)


def _mean(values: list[int] | list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
