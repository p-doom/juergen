#!/usr/bin/env python3
"""Fail-closed launch gate for the matched curriculum training pair."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEADLINE = datetime(2026, 7, 31, 5, 9, tzinfo=ZoneInfo("Europe/Berlin"))
MAX_PEAK_BYTES = 500_000_000_000
RUN_LIMIT = timedelta(hours=3)


class GateError(RuntimeError):
    pass


def _du(path: Path) -> int:
    output = subprocess.check_output(["du", "-s", "--block-size=1", str(path)], text=True)
    return int(output.split()[0])


def _nonzero(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_nonzero(item) for item in value.values())
    return value != 0


def _recipe_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    allowed = {
        "name",
        "inputs.source_model.artifact",
        "outputs.model.alias",
    }
    differences = []

    def walk(left: Any, right: Any, prefix: str = "") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                differences.append(prefix or "<root keys>")
                return
            for key in sorted(left):
                walk(left[key], right[key], f"{prefix}.{key}".strip("."))
        elif left != right and prefix not in allowed:
            differences.append(prefix)

    walk(a, b)
    return differences


def gate(
    *, dataset: Path, source_a: Path, source_b: Path,
    recipe_a: Path, recipe_b: Path, out: Path | None = None,
) -> dict[str, Any]:
    frozen = json.loads((Path(__file__).parent / "frozen_manifest.json").read_text())[
        "curriculum_transfer"
    ]
    dataset_manifest = json.loads((dataset / "curriculum_dataset_manifest.json").read_text())
    report = json.loads((dataset / dataset_manifest["overlap_report"]).read_text())
    if report.get("status") != "pass" or _nonzero(report.get("overlap_counts", {})):
        raise GateError(f"fresh-geometry overlap gate failed: {report}")
    expected_dataset = frozen["stage2_dataset"]
    if dataset_manifest.get("seeds") != {
        "train": expected_dataset["train_seed"], "val": expected_dataset["validation_seed"]
    }:
        raise GateError(f"dataset seeds differ from preregistration: {dataset_manifest}")
    if (dataset_manifest.get("train_records") != expected_dataset["train_records"]
            or dataset_manifest.get("validation_records") != expected_dataset["validation_records"]):
        raise GateError("dataset counts differ from preregistration")

    recipes = [tomllib.loads(path.read_text()) for path in (recipe_a, recipe_b)]
    unexpected = _recipe_diff(*recipes)
    if unexpected:
        raise GateError(f"training recipes differ outside source/identity fields: {unexpected}")
    if recipes[0]["inputs"]["dataset"]["artifact"] != expected_dataset["dataset_alias"]:
        raise GateError("A recipe does not pin the preregistered dataset")
    if recipes[1]["inputs"]["dataset"]["artifact"] != expected_dataset["dataset_alias"]:
        raise GateError("B recipe does not pin the preregistered dataset")
    expected_sources = frozen["stage1_checkpoints"]
    if recipes[0]["inputs"]["source_model"]["artifact"] != expected_sources["A_to_B"]:
        raise GateError("A→B recipe pins the wrong stage-1 source")
    if recipes[1]["inputs"]["source_model"]["artifact"] != expected_sources["B_to_B"]:
        raise GateError("B→B recipe pins the wrong stage-1 source")
    for recipe in recipes:
        if recipe["resources"]["time"] != "03:00:00":
            raise GateError("training time limit is not exactly three hours")
        if "--deadline=2026-07-31T05:09:00" not in recipe["resources"]["sbatch_extra"]:
            raise GateError("training recipe lacks the frozen SLURM deadline")

    source_sizes = [_du(source_a), _du(source_b)]
    dataset_size = _du(dataset)
    # Bound calibrated from the completed r256 jobs: three retained Orbax saves
    # plus one merged HF export.  Use 4.25x the larger merged source per new root
    # and reserve another 10 GB for both compilation caches/logs.
    output_bound_each = (17 * max(source_sizes) + 3) // 4
    peak_bound = sum(source_sizes) + dataset_size + 2 * output_bound_each + 10_000_000_000
    if peak_bound >= MAX_PEAK_BYTES:
        raise GateError(f"storage peak is not below 500 GB: {peak_bound}")
    free = shutil.disk_usage(dataset).free
    new_bytes = 2 * output_bound_each + 10_000_000_000
    if free <= new_bytes:
        raise GateError(f"insufficient free space: free={free} required>{new_bytes}")

    now = datetime.now(ZoneInfo("Europe/Berlin"))
    projected_latest_end = now + RUN_LIMIT
    if projected_latest_end > DEADLINE:
        raise GateError(
            f"three-hour job would exceed deadline: now={now.isoformat()} "
            f"end={projected_latest_end.isoformat()} deadline={DEADLINE.isoformat()}"
        )
    result = {
        "status": "pass",
        "checked_at": now.isoformat(),
        "deadline": DEADLINE.isoformat(),
        "projected_end_at_full_time_limit": projected_latest_end.isoformat(),
        "recipe_differences_allowed": sorted([
            "name", "inputs.source_model.artifact", "outputs.model.alias"
        ]),
        "fresh_geometry_overlap_counts": report["overlap_counts"],
        "source_sizes_bytes": source_sizes,
        "dataset_size_bytes": dataset_size,
        "output_bound_each_bytes": output_bound_each,
        "aggregate_peak_bound_bytes": peak_bound,
        "aggregate_peak_limit_bytes": MAX_PEAK_BYTES,
        "filesystem_free_bytes": free,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-a", required=True, type=Path)
    parser.add_argument("--source-b", required=True, type=Path)
    parser.add_argument("--recipe-a", required=True, type=Path)
    parser.add_argument("--recipe-b", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = gate(
            dataset=args.dataset, source_a=args.source_a, source_b=args.source_b,
            recipe_a=args.recipe_a, recipe_b=args.recipe_b, out=args.out,
        )
    except (GateError, OSError, ValueError, KeyError) as exc:
        print(f"FATAL curriculum launch gate: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
