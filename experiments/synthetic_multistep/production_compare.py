#!/usr/bin/env python3
"""Paired r256 production-format movement comparison (A tool vs B raw)."""
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


def miss_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for row in rows for step in row["steps"]]
    misses = [step for step in steps if not step["hit"]]
    first_miss_targets = []
    for row in rows:
        for target_index in range(int(row["target_count"])):
            target_steps = [
                step for step in row["steps"] if int(step["target_index"]) == target_index
            ]
            if target_steps and not target_steps[0]["hit"]:
                first_miss_targets.append(target_steps)
    return {
        "step_misses": len(misses),
        "parse_failures": sum(not step["parse_ok"] for step in misses),
        "strict_schema_failures": sum(not step["schema_ok"] for step in misses),
        "coordinate_unit_violations": sum(not step["unit_range_ok"] for step in misses),
        "geometry_only_misses": sum(
            step["parse_ok"] and step["schema_ok"] and step["unit_range_ok"]
            for step in misses
        ),
        "first_attempt_missed_targets": len(first_miss_targets),
        "first_misses_recovered_by_attempt_2": sum(
            any(step["hit"] and int(step["attempt"]) <= 2 for step in target_steps)
            for target_steps in first_miss_targets
        ),
        "first_misses_recovered_by_attempt_3": sum(
            any(step["hit"] and int(step["attempt"]) <= 3 for step in target_steps)
            for target_steps in first_miss_targets
        ),
        "unrecovered_targets": sum(
            not any(step["hit"] for step in target_steps)
            for target_steps in first_miss_targets
        ),
    }


def compare(tool_root: Path, raw_root: Path, out: Path) -> dict[str, Any]:
    tool_root, raw_root = tool_root.resolve(), raw_root.resolve()
    tool_manifest, tool_report = _validate(tool_root, "move_rel")
    raw_manifest, raw_report = _validate(raw_root, "deltatype_raw")
    frozen = load_frozen()["production_movement_bridge"]
    for semantic, manifest in (("move_rel", tool_manifest), ("deltatype_raw", raw_manifest)):
        if manifest.get("comparison_label") != "production_movement_bridge":
            raise ContractError(f"{semantic}: wrong production comparison label")
        if not manifest.get("preamble"):
            raise ContractError(f"{semantic}: production bridge must preserve preamble")
        if manifest.get("checkpoint_alias") != frozen["checkpoints"][semantic]:
            raise ContractError(f"{semantic}: checkpoint is not preregistered")
        if int(manifest["model_provenance"]["lora_rank"]) != 256:
            raise ContractError(f"{semantic}: expected r256 model provenance")
    if tool_manifest["episode_manifest_sha256"] != raw_manifest["episode_manifest_sha256"]:
        raise ContractError("production arms used different episode artifacts")
    for key in ("n_episodes", "max_attempts", "history_turns", "sampling"):
        if tool_manifest[key] != raw_manifest[key]:
            raise ContractError(f"production arm setting mismatch: {key}")
    tool_rows, raw_rows = _rows(tool_root), _rows(raw_root)
    paired = paired_uncertainty(tool_rows, raw_rows)
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_production_movement_comparison",
        "status": "complete",
        "evidence_scope": frozen["evidence_scope"],
        "effect_direction": "deltatype_raw_B_minus_move_rel_A",
        "ordered_primary_endpoints": frozen["ordered_primary_endpoints"],
        "checkpoints": frozen["checkpoints"],
        "episode_manifest_sha256": tool_manifest["episode_manifest_sha256"],
        "move_rel_A_metrics": tool_report["metrics"],
        "deltatype_raw_B_metrics": raw_report["metrics"],
        "paired": paired,
        "miss_diagnostics": {
            "move_rel_A": miss_diagnostics(tool_rows),
            "deltatype_raw_B": miss_diagnostics(raw_rows),
        },
        "provenance": {"move_rel_A": tool_manifest, "deltatype_raw_B": raw_manifest},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "production_comparison.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--move-rel", required=True, type=Path)
    parser.add_argument("--deltatype-raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.move_rel, args.deltatype_raw, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
