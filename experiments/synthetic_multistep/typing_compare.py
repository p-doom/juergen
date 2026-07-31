#!/usr/bin/env python3
"""Paired 2x2 comparison for the frozen typing-format factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


class CompareError(RuntimeError):
    pass


CELLS = ("A_coalesced", "B_coalesced", "A_perkey", "B_perkey")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CompareError(f"expected JSON object: {path}")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_cell(root: Path, cell: str) -> dict[str, Any]:
    lineage, fmt = cell.split("_", 1)
    manifest = load_json(root / "typing_eval_manifest.json")
    fixed = {"artifact_type": "synthetic_typing_factorial_cell_eval", "status": "complete",
             "lineage": lineage, "target_format": fmt, "n_examples": 200}
    bad = {key: (manifest.get(key), value) for key, value in fixed.items() if manifest.get(key) != value}
    if bad:
        raise CompareError(f"wrong typing eval cell {cell}: {bad}")
    paths = {
        "generation_rows": root / "typing_generation_rows.jsonl",
        "generation_report": root / "typing_generation_report.json",
        "generation_manifest": root / "typing_generation_manifest.json",
        "teacher_rows": root / "typing_teacher_forced_rows.jsonl",
        "teacher_report": root / "typing_teacher_forced_report.json",
    }
    expected_hashes = {
        "generation_rows": "generation_rows_sha256",
        "generation_report": "generation_report_sha256",
        "generation_manifest": "generation_manifest_sha256",
        "teacher_rows": "teacher_forced_rows_sha256",
        "teacher_report": "teacher_forced_report_sha256",
    }
    for key, manifest_key in expected_hashes.items():
        if sha256(paths[key]) != manifest.get(manifest_key):
            raise CompareError(f"hash mismatch in {cell}: {key}")
    generation = rows(paths["generation_rows"])
    teacher = rows(paths["teacher_rows"])
    report = load_json(paths["generation_report"])
    teacher_report = load_json(paths["teacher_report"])
    if len(generation) != 200 or len(teacher) != 200:
        raise CompareError(f"wrong row count in {cell}")
    if [row["sample_id"] for row in generation] != [row["sample_id"] for row in teacher]:
        raise CompareError(f"generation/teacher sample order differs in {cell}")
    if report.get("metrics", {}).get("request_error_count") != 0:
        raise CompareError(f"request errors in {cell}")
    return {"manifest": manifest, "generation": generation, "teacher": teacher,
            "report": report, "teacher_report": teacher_report}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def interval(values: list[float], rng: random.Random, n_boot: int = 20_000) -> list[float]:
    boot = []
    for _ in range(n_boot):
        boot.append(mean([values[rng.randrange(len(values))] for _ in values]))
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def contrast_rows(data: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "lineage_effect_coalesced_A_minus_B": [a - b for a, b in zip(data["A_coalesced"], data["B_coalesced"])],
        "lineage_effect_perkey_A_minus_B": [a - b for a, b in zip(data["A_perkey"], data["B_perkey"])],
        "format_effect_A_coalesced_minus_perkey": [a - b for a, b in zip(data["A_coalesced"], data["A_perkey"])],
        "format_effect_B_coalesced_minus_perkey": [a - b for a, b in zip(data["B_coalesced"], data["B_perkey"])],
        "lineage_by_format_interaction": [ac - bc - ap + bp for ac, bc, ap, bp in zip(
            data["A_coalesced"], data["B_coalesced"], data["A_perkey"], data["B_perkey"])],
        "lineage_averaged_coalesced_minus_perkey": [(ac - ap + bc - bp) / 2 for ac, ap, bc, bp in zip(
            data["A_coalesced"], data["A_perkey"], data["B_coalesced"], data["B_perkey"])],
    }


def compare(roots: dict[str, Path], out: Path) -> dict[str, Any]:
    cells = {cell: load_cell(roots[cell].resolve(), cell) for cell in CELLS}
    sample_ids = [row["sample_id"] for row in cells[CELLS[0]]["generation"]]
    for cell in CELLS[1:]:
        if [row["sample_id"] for row in cells[cell]["generation"]] != sample_ids:
            raise CompareError(f"cells are not paired by exact validation example: {cell}")
    success = {cell: [float(row["exact_typed_string_success"]) for row in value["generation"]]
               for cell, value in cells.items()}
    action_nll = {cell: [float(row["action_line_token_nll"]) for row in value["teacher"]]
                  for cell, value in cells.items()}
    rng = random.Random(20260731)
    success_contrasts = {name: {"effect": mean(values), "paired_bootstrap_ci95": interval(values, rng)}
                         for name, values in contrast_rows(success).items()}
    nll_contrasts = {name: {"effect": mean(values), "paired_bootstrap_ci95": interval(values, rng)}
                     for name, values in contrast_rows(action_nll).items()}
    interaction = success_contrasts["lineage_by_format_interaction"]
    coalescing = success_contrasts["lineage_averaged_coalesced_minus_perkey"]
    nll_coalescing = nll_contrasts["lineage_averaged_coalesced_minus_perkey"]
    result = {
        "schema_version": 1, "artifact_type": "synthetic_typing_factorial_comparison",
        "status": "complete", "pairing": "200 exact validation examples across all four cells",
        "bootstrap_resamples": 20_000, "bootstrap_seed": 20260731,
        "cell_metrics": {
            cell: {
                "exact_typed_string_success_rate": mean(success[cell]),
                **cells[cell]["report"]["metrics"],
                **cells[cell]["teacher_report"]["summary"],
            } for cell in CELLS
        },
        "success_contrasts": success_contrasts,
        "action_example_nll_contrasts": nll_contrasts,
        "claim_gates": {
            "lineage_by_format_interaction": {
                "absolute_effect_at_least_5pp": abs(interaction["effect"]) >= 0.05,
                "paired_interval_excludes_zero": (interaction["paired_bootstrap_ci95"][0] > 0
                                                   or interaction["paired_bootstrap_ci95"][1] < 0),
            },
            "coalescing_capacity_benefit": {
                "effect_at_least_5pp": coalescing["effect"] >= 0.05,
                "paired_lower_bound_above_zero": coalescing["paired_bootstrap_ci95"][0] > 0,
                "action_nll_same_direction": nll_coalescing["effect"] < 0,
            },
        },
        "input_manifest_sha256": {cell: sha256(roots[cell] / "typing_eval_manifest.json") for cell in CELLS},
    }
    result["claim_gates"]["lineage_by_format_interaction"]["claim_supported"] = all(
        result["claim_gates"]["lineage_by_format_interaction"].values()
    )
    result["claim_gates"]["coalescing_capacity_benefit"]["claim_supported"] = all(
        result["claim_gates"]["coalescing_capacity_benefit"].values()
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "typing_factorial_comparison.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument("--" + cell.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    roots = {cell: getattr(args, cell) for cell in CELLS}
    try:
        print(json.dumps(compare(roots, args.out), indent=2))
    except CompareError as exc:
        print(f"FATAL typing comparison: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
