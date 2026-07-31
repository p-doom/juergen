#!/usr/bin/env python3
"""Paired comparison of the two frozen teacher-forced B evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def _rows(root: Path) -> list[dict[str, Any]]:
    report = json.loads((root / "teacher_forced_report.json").read_text())
    if (report.get("status") != "complete"
            or report.get("artifact_type") != "synthetic_multistep_curriculum_teacher_forced_eval"):
        raise ValueError(f"invalid teacher-forced artifact: {root}")
    rows_path = root / "teacher_forced_rows.jsonl"
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != report.get("rows_sha256"):
        raise ValueError(f"teacher-forced rows hash mismatch: {root}")
    return [json.loads(line) for line in
            rows_path.read_text().splitlines()]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare(a_root: Path, b_root: Path, out: Path) -> dict[str, Any]:
    a_rows, b_rows = _rows(a_root.resolve()), _rows(b_root.resolve())
    if [row["sample_id"] for row in a_rows] != [row["sample_id"] for row in b_rows]:
        raise ValueError("teacher-forced artifacts are not paired by exact example")
    if len(a_rows) != 200:
        raise ValueError(f"expected 200 paired examples, found {len(a_rows)}")
    rng = random.Random(20260731)
    metrics = {}
    for metric, sum_key, count_key in (
        ("assistant_token_nll", "assistant_nll_sum", "assistant_tokens"),
        ("action_line_token_nll", "action_nll_sum", "action_tokens"),
    ):
        if [row[count_key] for row in a_rows] != [row[count_key] for row in b_rows]:
            raise ValueError(f"token-count mismatch for {metric}")
        a_value = sum(row[sum_key] for row in a_rows) / sum(row[count_key] for row in a_rows)
        b_value = sum(row[sum_key] for row in b_rows) / sum(row[count_key] for row in b_rows)
        differences = [a[metric] - b[metric] for a, b in zip(a_rows, b_rows)]
        boot = []
        for _ in range(20_000):
            indices = [rng.randrange(len(differences)) for _ in differences]
            boot.append(sum(differences[index] for index in indices) / len(indices))
        metrics[metric] = {
            "A_to_B": a_value, "B_to_B": b_value,
            "A_to_B_minus_B_to_B_token_weighted": a_value - b_value,
            "paired_example_mean_difference": sum(differences) / len(differences),
            "paired_example_bootstrap_ci95": [
                _percentile(boot, 0.025), _percentile(boot, 0.975)
            ],
            "A_to_B_better_examples": sum(value < 0 for value in differences),
            "B_to_B_better_examples": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_curriculum_teacher_forced_comparison",
        "status": "complete", "effect_direction": "A_to_B_minus_B_to_B",
        "pairing": "200 exact fresh validation examples",
        "bootstrap_resamples": 20_000, "bootstrap_seed": 20260731,
        "metrics": metrics,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "teacher_forced_comparison.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-to-b", required=True, type=Path)
    parser.add_argument("--b-to-b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.a_to_b, args.b_to_b, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
