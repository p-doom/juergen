#!/usr/bin/env python3
"""Paired closed-loop B comparison for A→B curriculum versus B→B continuation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .compare import _rows, _validate, paired_uncertainty
    from .contract import ContractError, load_frozen
    from .production_compare import miss_diagnostics
except ImportError:
    from compare import _rows, _validate, paired_uncertainty
    from contract import ContractError, load_frozen
    from production_compare import miss_diagnostics


def _artifact(
    root: Path, alias: str, comparison_label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, report = _validate(root, "deltatype_raw")
    expected = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_phasea_eval",
        "status": "complete", "semantic": "deltatype_raw",
        "checkpoint_alias": alias, "comparison_label": comparison_label,
        "preamble": True, "n_episodes": 80,
    }
    bad = {key: (value.get(key), wanted) for key, wanted in expected.items()
           if value.get(key) != wanted}
    if bad:
        raise ContractError(f"wrong curriculum multistep artifact {root}: {bad}")
    return value, report


def compare(a_root: Path, b_root: Path, out: Path, variant: str = "original") -> dict[str, Any]:
    frozen = load_frozen()["curriculum_transfer"]
    if variant == "original":
        models = frozen["stage2_models"]
        comparison_label = "curriculum_transfer"
    elif variant == "lr5e5":
        models = frozen["low_lr_rescue_prepared"]["models"]
        comparison_label = "curriculum_transfer_lr5e5"
    else:
        raise ContractError(f"unsupported curriculum comparison variant: {variant}")
    a_manifest, a_report = _artifact(
        a_root.resolve(), models["A_to_B"], comparison_label
    )
    b_manifest, b_report = _artifact(
        b_root.resolve(), models["B_to_B"], comparison_label
    )
    if a_manifest["episode_manifest_sha256"] != b_manifest["episode_manifest_sha256"]:
        raise ContractError("curriculum arms used different episode artifacts")
    a_rows, b_rows = _rows(a_root.resolve()), _rows(b_root.resolve())
    # paired_uncertainty's labels are historical: pass B as "absolute" and A as
    # "relative" so every reported difference is the preregistered A→B minus B→B.
    paired = paired_uncertainty(b_rows, a_rows)
    a_metrics, b_metrics = a_report["metrics"], b_report["metrics"]
    scalar_keys = (
        "first_attempt_reach_rate", "episode_completion_rate",
        "first_miss_recovery_rate", "miss_event_recovery_rate",
        "normalized_distance_auc", "progress_rate", "regression_rate",
        "stall_rate", "oscillation_rate", "parse_rate", "strict_schema_rate",
        "coordinate_unit_violation_rate", "no_move_rate",
    )
    deltas = {
        key: None if a_metrics.get(key) is None or b_metrics.get(key) is None
        else a_metrics[key] - b_metrics[key]
        for key in scalar_keys
    }
    deltas["target_reach_cdf_by_attempt"] = {
        attempt: a_metrics["target_reach_cdf_by_attempt"][attempt]
        - b_metrics["target_reach_cdf_by_attempt"][attempt]
        for attempt in a_metrics["target_reach_cdf_by_attempt"]
    }
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_curriculum_comparison",
        "status": "complete", "evidence_scope": frozen["evidence_scope"],
        "effect_direction": "A_to_B_minus_B_to_B",
        "paired_legacy_label_mapping": {"absolute": "B_to_B", "relative": "A_to_B"},
        "ordered_primary_endpoints": frozen["multistep_B_ordered_primary_endpoints"],
        "variant": variant,
        "comparison_label": comparison_label,
        "stage2_models": models,
        "episode_manifest_sha256": a_manifest["episode_manifest_sha256"],
        "A_to_B_metrics": a_metrics,
        "B_to_B_metrics": b_metrics,
        "A_to_B_minus_B_to_B": deltas,
        "paired": paired,
        "miss_diagnostics": {
            "A_to_B": miss_diagnostics(a_rows),
            "B_to_B": miss_diagnostics(b_rows),
        },
        "provenance": {"A_to_B": a_manifest, "B_to_B": b_manifest},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "curriculum_comparison.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-to-b", required=True, type=Path)
    parser.add_argument("--b-to-b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=("original", "lr5e5"), default="original")
    args = parser.parse_args()
    print(json.dumps(compare(args.a_to_b, args.b_to_b, args.out, args.variant), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
