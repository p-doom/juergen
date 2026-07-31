#!/usr/bin/env python3
"""Validated descriptive 32→64→256 Phase-A multi-step capacity curve."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .compare import _rows, _validate, paired_uncertainty
    from .contract import ContractError, load_frozen
except ImportError:
    from compare import _rows, _validate, paired_uncertainty
    from contract import ContractError, load_frozen


def _paired(first: Path, second: Path, first_label: str, second_label: str) -> dict[str, Any]:
    result = paired_uncertainty(_rows(first), _rows(second))
    return {
        "first": first_label,
        "second": second_label,
        "effect_direction": f"{second_label}_minus_{first_label}",
        **result,
    }


def analyze(absolute: Path, r32: Path, r64: Path, r256: Path, out: Path) -> dict[str, Any]:
    roots = {
        "absolute": absolute.resolve(),
        "r32": r32.resolve(),
        "r64": r64.resolve(),
        "r256": r256.resolve(),
    }
    manifests = {}
    reports = {}
    for label, root in roots.items():
        semantic = "absolute_toolcall" if label == "absolute" else "move_rel"
        manifests[label], reports[label] = _validate(root, semantic)
    frozen = load_frozen()
    capacity = frozen["capacity_sensitivity"]
    expected_roots = {
        "absolute": capacity["absolute_reference"],
        "r32": capacity["r32_reference"],
    }
    for label, expected in expected_roots.items():
        if roots[label].name != expected:
            raise ContractError(f"{label} artifact is not the frozen capacity reference")
    expected_aliases = {
        "r32": frozen["primary_checkpoints"]["move_rel"],
        "r64": capacity["candidate_checkpoints"]["64"],
        "r256": capacity["candidate_checkpoints"]["256"],
    }
    for label, expected in expected_aliases.items():
        if manifests[label]["checkpoint_alias"] != expected:
            raise ContractError(f"{label} checkpoint does not match preregistration")
        if int(manifests[label]["model_provenance"]["lora_rank"]) != int(label[1:]):
            raise ContractError(f"{label} model provenance has the wrong LoRA rank")
    if manifests["absolute"]["comparison_label"] != "primary" or manifests[
        "r32"
    ]["comparison_label"] != "primary":
        raise ContractError("primary references have drifted")
    if any(
        manifests[label]["comparison_label"] != "capacity_sensitivity"
        for label in ("r64", "r256")
    ):
        raise ContractError("capacity candidate labels have drifted")
    manifest_hashes = {manifest["episode_manifest_sha256"] for manifest in manifests.values()}
    if len(manifest_hashes) != 1:
        raise ContractError("capacity curve evaluations used different episode artifacts")
    for key in ("n_episodes", "max_attempts", "history_turns", "sampling"):
        if len({json.dumps(manifest[key], sort_keys=True) for manifest in manifests.values()}) != 1:
            raise ContractError(f"capacity curve setting mismatch: {key}")

    primary = {}
    for label in ("absolute", "r32", "r64", "r256"):
        metrics = reports[label]["metrics"]
        primary[label] = {
            "first_attempt_reach_rate": metrics["first_attempt_reach_rate"],
            "reach_by_attempt_2": metrics["target_reach_cdf_by_attempt"]["2"],
            "reach_by_attempt_3": metrics["target_reach_cdf_by_attempt"]["3"],
            "episode_completion_rate": metrics["episode_completion_rate"],
        }
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_phasea_capacity_curve",
        "status": "complete",
        "episode_manifest_sha256": next(iter(manifest_hashes)),
        "ordered_primary_endpoints": capacity["ordered_primary_endpoints"],
        "primary_endpoint_curve": primary,
        "all_metrics": {label: reports[label]["metrics"] for label in reports},
        "paired": {
            "absolute_vs_r64": _paired(roots["absolute"], roots["r64"], "absolute", "r64"),
            "absolute_vs_r256": _paired(roots["absolute"], roots["r256"], "absolute", "r256"),
            "r32_vs_r64": _paired(roots["r32"], roots["r64"], "r32", "r64"),
            "r64_vs_r256": _paired(roots["r64"], roots["r256"], "r64", "r256"),
            "r32_vs_r256": _paired(roots["r32"], roots["r256"], "r32", "r256"),
        },
        "provenance": {label: manifests[label] for label in manifests},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "capacity_curve.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", required=True, type=Path)
    parser.add_argument("--r32", required=True, type=Path)
    parser.add_argument("--r64", required=True, type=Path)
    parser.add_argument("--r256", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.absolute, args.r32, args.r64, args.r256, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
