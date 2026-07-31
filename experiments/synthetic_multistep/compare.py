#!/usr/bin/env python3
"""Validate and compare a paired absolute/move_rel evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

try:
    from .contract import ContractError, load_frozen, sha256_file
except ImportError:  # direct script execution
    from contract import ContractError, load_frozen, sha256_file


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected object: {path}")
    return value


def _validate(root: Path, semantic: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load(root / "eval_manifest.json")
    report = _load(root / "report.json")
    fixed = {
        "artifact_type": "synthetic_multistep_phasea_eval",
        "status": "complete",
        "semantic": semantic,
    }
    mismatch = {key: (manifest.get(key), expected) for key, expected in fixed.items()
                if manifest.get(key) != expected}
    if mismatch:
        raise ContractError(f"wrong {semantic} artifact: {mismatch}")
    if sha256_file(root / "report.json") != manifest.get("report_sha256"):
        raise ContractError(f"report hash mismatch: {root}")
    if sha256_file(root / "rows.jsonl") != manifest.get("rows_sha256"):
        raise ContractError(f"rows hash mismatch: {root}")
    if report.get("semantic") != semantic or report.get("checkpoint_alias") != manifest.get(
        "checkpoint_alias"
    ):
        raise ContractError(f"report/manifest identity mismatch: {root}")
    return manifest, report


def _rows(root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()
            if line.strip()]
    if len(rows) != 80 or len({row["episode_id"] for row in rows}) != 80:
        raise ContractError(f"expected 80 unique episodes: {root}")
    return rows


def _episode_scalars(row: dict[str, Any]) -> dict[str, float]:
    steps = row["steps"]
    target_count = int(row["target_count"])
    by_target = {i: [] for i in range(target_count)}
    for step in steps:
        by_target[int(step["target_index"])].append(step)
    # After an attempt-budget exhaustion, later planned targets are correctly
    # unattempted and must count as unreached rather than invalidate the arm.
    reached = {
        attempt: sum(any(s["hit"] and int(s["attempt"]) <= attempt for s in values)
                     for values in by_target.values()) / target_count
        for attempt in (1, 2, 3)
    }
    n = len(steps)
    return {
        "first_attempt_reach_rate": reached[1],
        "reach_by_attempt_2": reached[2],
        "reach_by_attempt_3": reached[3],
        "episode_completion_rate": float(bool(row["completed"])),
        "steps_per_target": n / target_count,
        "parse_rate": sum(bool(s["parse_ok"]) for s in steps) / n,
        "strict_schema_rate": sum(bool(s["schema_ok"]) for s in steps) / n,
        "regression_rate": sum(bool(s["regression"]) for s in steps) / n,
        "oscillation_rate": sum(bool(s["oscillation"]) for s in steps) / n,
    }


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _mcnemar(first: dict[tuple[str, int], bool], second: dict[tuple[str, int], bool]) -> dict[str, Any]:
    if first.keys() != second.keys():
        raise ContractError("target keys differ for McNemar test")
    first_only = sum(first[k] and not second[k] for k in first)
    second_only = sum(second[k] and not first[k] for k in first)
    discordant = first_only + second_only
    if discordant:
        lower = min(first_only, second_only)
        tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2 ** discordant)
        p_two_sided = min(1.0, 2 * tail)
    else:
        p_two_sided = 1.0
    return {"n_targets": len(first), "absolute_only": first_only,
            "relative_only": second_only, "discordant": discordant,
            "exact_two_sided_p": p_two_sided}


def paired_uncertainty(abs_rows: list[dict[str, Any]], rel_rows: list[dict[str, Any]],
                       *, n_boot: int = 20000, seed: int = 20260730) -> dict[str, Any]:
    by_abs = {row["episode_id"]: row for row in abs_rows}
    by_rel = {row["episode_id"]: row for row in rel_rows}
    if by_abs.keys() != by_rel.keys() or len(by_abs) != 80:
        raise ContractError("paired episode IDs differ")
    ids = sorted(by_abs)
    for episode_id in ids:
        a, r = by_abs[episode_id], by_rel[episode_id]
        for key in ("episode_index", "kind", "k", "initial_cursor", "target_count"):
            if a[key] != r[key]:
                raise ContractError(f"paired episode mismatch {episode_id}: {key}")
        def targets(row):
            out = {}
            for step in row["steps"]:
                out.setdefault(int(step["target_index"]), (step["bbox"], step["sampling_seed"]))
            return out
        a_targets, r_targets = targets(a), targets(r)
        for observed in (a_targets, r_targets):
            if set(observed) != set(range(len(observed))):
                raise ContractError(f"observed targets are not a contiguous prefix: {episode_id}")
        # A terminal miss legitimately leaves later planned targets unobserved in
        # one arm. The caller has already required the same episode-manifest hash;
        # validate every jointly observed prefix target and score absent ones as
        # misses below instead of rejecting the scientifically relevant failure.
        common = a_targets.keys() & r_targets.keys()
        if any(a_targets[target] != r_targets[target] for target in common):
            raise ContractError(f"paired target geometry/first seed mismatch: {episode_id}")

    scalars_abs = {key: _episode_scalars(row) for key, row in by_abs.items()}
    scalars_rel = {key: _episode_scalars(row) for key, row in by_rel.items()}
    metric_names = tuple(next(iter(scalars_abs.values())))
    rng = random.Random(seed)
    boot = {metric: [] for metric in metric_names}
    for _ in range(n_boot):
        picks = [ids[rng.randrange(len(ids))] for _ in ids]
        for metric in metric_names:
            boot[metric].append(sum(
                scalars_rel[key][metric] - scalars_abs[key][metric] for key in picks
            ) / len(picks))
    metrics = {}
    for metric in metric_names:
        absolute = sum(scalars_abs[key][metric] for key in ids) / len(ids)
        relative = sum(scalars_rel[key][metric] for key in ids) / len(ids)
        metrics[metric] = {"absolute": absolute, "relative": relative,
                           "relative_minus_absolute": relative - absolute,
                           "paired_episode_bootstrap_ci95": [
                               _percentile(boot[metric], .025),
                               _percentile(boot[metric], .975),
                           ]}
    def target_hits(rows, attempt):
        result = {}
        for row in rows:
            for target in range(int(row["target_count"])):
                result[(row["episode_id"], target)] = any(
                    int(s["target_index"]) == target and int(s["attempt"]) <= attempt and s["hit"]
                    for s in row["steps"])
        return result
    return {"pairing": "80 exact episode clusters / 320 exact targets",
            "bootstrap_resamples": n_boot, "bootstrap_seed": seed,
            "metrics": metrics,
            "mcnemar": {
                "first_attempt_reach": _mcnemar(target_hits(abs_rows, 1), target_hits(rel_rows, 1)),
                "reach_by_attempt_2": _mcnemar(target_hits(abs_rows, 2), target_hits(rel_rows, 2)),
                "reach_by_attempt_3": _mcnemar(target_hits(abs_rows, 3), target_hits(rel_rows, 3)),
            }}


def compare(abs_root: Path, rel_root: Path, out: Path, comparison_label: str) -> dict[str, Any]:
    abs_root, rel_root = abs_root.resolve(), rel_root.resolve()
    abs_manifest, abs_report = _validate(abs_root, "absolute_toolcall")
    rel_manifest, rel_report = _validate(rel_root, "move_rel")
    if abs_manifest["episode_manifest_sha256"] != rel_manifest["episode_manifest_sha256"]:
        raise ContractError("paired evaluations used different episode artifacts")
    for key in ("n_episodes", "max_attempts", "history_turns", "sampling"):
        if abs_manifest[key] != rel_manifest[key]:
            raise ContractError(f"paired setting mismatch for {key}")
    frozen = load_frozen()
    if comparison_label == "primary":
        if abs_manifest["comparison_label"] != "primary" or rel_manifest[
            "comparison_label"
        ] != "primary":
            raise ContractError("primary comparison labels do not match")
        for semantic, manifest in (
            ("absolute_toolcall", abs_manifest), ("move_rel", rel_manifest)
        ):
            expected = frozen["primary_checkpoints"][semantic]
            if manifest["checkpoint_alias"] != expected:
                raise ContractError(f"primary checkpoint mismatch: {semantic}")
    elif comparison_label == "capacity_sensitivity":
        capacity = frozen["capacity_sensitivity"]
        if abs_manifest["comparison_label"] != "primary" or rel_manifest[
            "comparison_label"
        ] != "capacity_sensitivity":
            raise ContractError("capacity sensitivity requires primary absolute reference")
        if abs_root.name != capacity["absolute_reference"]:
            raise ContractError("capacity sensitivity absolute artifact is not pinned")
        if rel_manifest["checkpoint_alias"] not in set(
            capacity["candidate_checkpoints"].values()
        ):
            raise ContractError("capacity sensitivity checkpoint is not frozen")
    elif abs_manifest["comparison_label"] != comparison_label or rel_manifest[
        "comparison_label"
    ] != comparison_label:
        raise ContractError("comparison labels do not match requested analysis")

    abs_metrics = abs_report["metrics"]
    rel_metrics = rel_report["metrics"]
    scalar_keys = (
        "first_attempt_reach_rate",
        "episode_completion_rate",
        "first_miss_recovery_rate",
        "miss_event_recovery_rate",
        "normalized_distance_auc",
        "progress_rate",
        "regression_rate",
        "stall_rate",
        "oscillation_rate",
        "parse_rate",
        "strict_schema_rate",
        "coordinate_unit_violation_rate",
        "no_move_rate",
    )
    deltas = {}
    for key in scalar_keys:
        a, r = abs_metrics.get(key), rel_metrics.get(key)
        deltas[key] = None if a is None or r is None else r - a
    reach_delta = {
        attempt: rel_metrics["target_reach_cdf_by_attempt"][attempt]
        - abs_metrics["target_reach_cdf_by_attempt"][attempt]
        for attempt in abs_metrics["target_reach_cdf_by_attempt"]
    }
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_phasea_comparison",
        "status": "complete",
        "comparison_label": comparison_label,
        "checkpoints": {
            "absolute_toolcall": abs_manifest["checkpoint_alias"],
            "move_rel": rel_manifest["checkpoint_alias"],
        },
        "episode_manifest_sha256": abs_manifest["episode_manifest_sha256"],
        "absolute_metrics": abs_metrics,
        "relative_metrics": rel_metrics,
        "relative_minus_absolute": {**deltas, "target_reach_cdf_by_attempt": reach_delta},
        "paired_uncertainty": paired_uncertainty(_rows(abs_root), _rows(rel_root)),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "comparison.json"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", required=True, type=Path)
    parser.add_argument("--relative", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--comparison-label", default="primary",
                        choices=("primary", "capacity_sensitivity", "preamble_sensitivity"))
    args = parser.parse_args()
    print(json.dumps(compare(args.absolute, args.relative, args.out, args.comparison_label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
