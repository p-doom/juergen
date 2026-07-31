#!/usr/bin/env python3
"""Paired-scene uncertainty and geometric failure decomposition for the 2x2x2."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .effects import CELLS, EXPECTED_ROWS, EffectError, _jsonl_rows, _load_cell
except ImportError:  # Direct execution by the labctl recipe.
    from effects import CELLS, EXPECTED_ROWS, EffectError, _jsonl_rows, _load_cell


SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 50_000
FACTOR_NAMES = ("relativity", "grammar", "preamble")
TERM_AXES = {
    "×".join(FACTOR_NAMES[index] for index in axes): axes
    for width in (1, 2, 3)
    for axes in itertools.combinations(range(3), width)
}
RELATIVE_CELLS = (
    "rel_bare_act",
    "rel_bare_pre",
    "rel_tool_act",
    "rel_tool_pre",
)


class AnalysisError(RuntimeError):
    pass


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise AnalysisError("cannot average an empty collection")
    return math.fsum(values) / len(values)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot take a quantile of an empty collection")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": statistics.median(values),
        "p10": _quantile(values, 0.10),
        "p90": _quantile(values, 0.90),
    }


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def factorial_contributions(
    rows_by_cell: dict[str, dict[str, dict[str, Any]]],
    scene_ids: list[str],
) -> dict[str, list[float]]:
    """Return one paired scene-level contribution per factorial estimand."""
    contributions = {"grand_mean": []}
    contributions.update({name: [] for name in TERM_AXES})
    for scene_id in scene_ids:
        values = {
            cell: float(bool(rows_by_cell[cell][scene_id]["in_box"])) for cell in CELLS
        }
        contributions["grand_mean"].append(_mean(values.values()))
        for name, axes in TERM_AXES.items():
            # Four cells have positive product and four negative product. The
            # per-scene contribution therefore uses the signed sum divided by 4.
            signed = math.fsum(
                _product(CELLS[cell][axis] for axis in axes) * value
                for cell, value in values.items()
            )
            contributions[name].append(signed / 4.0)
    return contributions


def exact_sign_flip_p(contributions: list[float]) -> float:
    """Exact two-sided paired randomization p-value conditional on magnitudes.

    Binary 2x2x2 contrasts are multiples of one quarter. Dynamic programming
    enumerates the sign-flip distribution without enumerating 2**n assignments.
    """
    weights = []
    for value in contributions:
        scaled = int(round(float(value) * 4.0))
        if not math.isclose(value * 4.0, scaled, rel_tol=0.0, abs_tol=1e-9):
            raise AnalysisError(f"contrast contribution is not quarter-valued: {value}")
        weights.append(abs(scaled))
    observed = abs(sum(int(round(value * 4.0)) for value in contributions))
    distribution = Counter({0: 1})
    for weight in weights:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + weight] += count
            updated[total - weight] += count
        distribution = updated
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    return extreme / (2 ** len(weights))


def _bootstrap_replicates(
    contributions: dict[str, list[float]],
    *,
    resamples: int,
    rng: random.Random,
) -> dict[str, list[float]]:
    names = list(contributions)
    length = len(contributions[names[0]])
    if length == 0 or any(len(contributions[name]) != length for name in names):
        raise AnalysisError("bootstrap contribution vectors are empty or misaligned")
    replicates = {name: [] for name in names}
    for _ in range(resamples):
        indices = [rng.randrange(length) for _ in range(length)]
        for name in names:
            values = contributions[name]
            replicates[name].append(math.fsum(values[index] for index in indices) / length)
    return replicates


def _effect_section(
    contributions: dict[str, list[float]],
    replicates: dict[str, list[float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_scenes": len(contributions["grand_mean"]),
        "grand_mean": {
            "estimate": _mean(contributions["grand_mean"]),
            "ci95_percentile": [
                _quantile(replicates["grand_mean"], 0.025),
                _quantile(replicates["grand_mean"], 0.975),
            ],
        },
        "effects": {},
    }
    for name in TERM_AXES:
        values = contributions[name]
        result["effects"][name] = {
            "estimate": _mean(values),
            "ci95_percentile": [
                _quantile(replicates[name], 0.025),
                _quantile(replicates[name], 0.975),
            ],
            "exact_paired_sign_flip_p_two_sided": exact_sign_flip_p(values),
            "scene_contributions": values,
        }
    return result


def paired_factorial_analysis(
    rows_by_cell: dict[str, dict[str, dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    scene_ids = sorted(next(iter(rows_by_cell.values())))
    long_ids = [scene_id for scene_id in scene_ids if rows_by_cell["abs_bare_act"][scene_id]["kind"] == "long"]
    short_ids = [scene_id for scene_id in scene_ids if rows_by_cell["abs_bare_act"][scene_id]["kind"] == "short"]
    rng = random.Random(seed)
    contributions = {
        "all": factorial_contributions(rows_by_cell, scene_ids),
        "long": factorial_contributions(rows_by_cell, long_ids),
        "short": factorial_contributions(rows_by_cell, short_ids),
    }
    replicates = {
        stratum: _bootstrap_replicates(values, resamples=resamples, rng=rng)
        for stratum, values in contributions.items()
    }
    sections = {
        stratum: _effect_section(contributions[stratum], replicates[stratum])
        for stratum in ("all", "long", "short")
    }
    sections["long_minus_short"] = {
        "effects": {
            name: {
                "estimate": (
                    sections["long"]["effects"][name]["estimate"]
                    - sections["short"]["effects"][name]["estimate"]
                ),
                "ci95_percentile": [
                    _quantile(
                        [
                            left - right
                            for left, right in zip(
                                replicates["long"][name],
                                replicates["short"][name],
                            )
                        ],
                        0.025,
                    ),
                    _quantile(
                        [
                            left - right
                            for left, right in zip(
                                replicates["long"][name],
                                replicates["short"][name],
                            )
                        ],
                        0.975,
                    ),
                ],
            }
            for name in TERM_AXES
        }
    }
    return sections


def vector_diagnostics(row: dict[str, Any]) -> dict[str, float | str] | None:
    predicted = row.get("coord")
    ideal = row.get("ideal_coord")
    if not (
        isinstance(predicted, list)
        and isinstance(ideal, list)
        and len(predicted) == len(ideal) == 2
        and all(isinstance(value, (int, float)) for value in predicted + ideal)
    ):
        return None
    predicted_norm = math.hypot(*predicted)
    ideal_norm = math.hypot(*ideal)
    if predicted_norm == 0.0 or ideal_norm == 0.0:
        return None
    cosine = max(
        -1.0,
        min(1.0, math.fsum(a * b for a, b in zip(predicted, ideal)) / (predicted_norm * ideal_norm)),
    )
    ratio = predicted_norm / ideal_norm
    radial_component = (ratio - 1.0) ** 2
    angular_component = 2.0 * ratio * (1.0 - cosine)
    if math.isclose(radial_component, angular_component, rel_tol=1e-9, abs_tol=1e-12):
        dominant = "balanced"
    elif radial_component > angular_component:
        dominant = "magnitude"
    else:
        dominant = "direction"
    return {
        "cosine": cosine,
        "angle_degrees": math.degrees(math.acos(cosine)),
        "magnitude_ratio": ratio,
        "normalized_squared_radial_error": radial_component,
        "normalized_squared_angular_error": angular_component,
        "dominant_vector_error": dominant,
    }


def _row_decomposition(rows: list[dict[str, Any]], *, relative: bool) -> dict[str, Any]:
    failures = [row for row in rows if not row["in_box"]]
    result: dict[str, Any] = {
        "n": len(rows),
        "in_box_rate": _mean(float(row["in_box"]) for row in rows),
        "parse_rate": _mean(float(row["parse_ok"]) for row in rows),
        "schema_rate": _mean(float(row["schema_ok"]) for row in rows),
        "raw_in_box_rate": _mean(float(row["raw_in_box"]) for row in rows),
        "endpoint_error_px": _summary(row["endpoint_err_px"] for row in rows),
        "failures": {
            "count": len(failures),
            "parse_failures": sum(not row["parse_ok"] for row in failures),
            "schema_failures": sum(not row["schema_ok"] for row in failures),
            "geometry_only_failures": sum(
                row["parse_ok"] and row["schema_ok"] and not row["in_box"] for row in failures
            ),
            "endpoint_error_px": _summary(row["endpoint_err_px"] for row in failures),
        },
    }
    if relative:
        diagnostics = [diag for row in rows if (diag := vector_diagnostics(row)) is not None]
        failed_diagnostics = [
            diag for row in failures if (diag := vector_diagnostics(row)) is not None
        ]
        result["relative_vector"] = {
            "direction_cosine": _summary(diag["cosine"] for diag in diagnostics),
            "angle_degrees": _summary(diag["angle_degrees"] for diag in diagnostics),
            "magnitude_ratio": _summary(diag["magnitude_ratio"] for diag in diagnostics),
            "failure_direction_cosine": _summary(diag["cosine"] for diag in failed_diagnostics),
            "failure_angle_degrees": _summary(diag["angle_degrees"] for diag in failed_diagnostics),
            "failure_magnitude_ratio": _summary(diag["magnitude_ratio"] for diag in failed_diagnostics),
            "failure_dominant_vector_error": dict(
                sorted(Counter(diag["dominant_vector_error"] for diag in failed_diagnostics).items())
            ),
        }
    return result


def cell_decomposition(
    rows_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for cell in CELLS:
        rows = list(rows_by_cell[cell].values())
        result[cell] = {
            stratum: _row_decomposition(
                [row for row in rows if stratum == "all" or row["kind"] == stratum],
                relative=cell.startswith("rel_"),
            )
            for stratum in ("all", "long", "short")
        }
    return result


def failure_overlap(
    rows_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    scene_ids = sorted(next(iter(rows_by_cell.values())))
    failed = {
        cell: {scene_id for scene_id in scene_ids if not rows_by_cell[cell][scene_id]["in_box"]}
        for cell in RELATIVE_CELLS
    }
    patterns: dict[tuple[str, ...], list[str]] = {}
    per_scene = []
    for scene_id in scene_ids:
        failed_cells = tuple(cell for cell in RELATIVE_CELLS if scene_id in failed[cell])
        if failed_cells:
            patterns.setdefault(failed_cells, []).append(scene_id)
            per_scene.append({"scene_id": scene_id, "failed_cells": list(failed_cells)})
    pairwise = {}
    for left, right in itertools.combinations(RELATIVE_CELLS, 2):
        intersection = failed[left] & failed[right]
        union = failed[left] | failed[right]
        pairwise[f"{left}|{right}"] = {
            "intersection_count": len(intersection),
            "intersection_scene_ids": sorted(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else 1.0,
        }
    failure_rows = {}
    for cell in RELATIVE_CELLS:
        items = []
        for scene_id in sorted(failed[cell]):
            row = rows_by_cell[cell][scene_id]
            items.append({
                "scene_id": scene_id,
                "kind": row["kind"],
                "parse_ok": row["parse_ok"],
                "schema_ok": row["schema_ok"],
                "endpoint_err_px": row["endpoint_err_px"],
                "vector": vector_diagnostics(row),
            })
        failure_rows[cell] = items
    union = set().union(*failed.values())
    return {
        "all_pass_count": len(scene_ids) - len(union),
        "any_relative_failure_count": len(union),
        "any_relative_failure_scene_ids": sorted(union),
        "per_cell_failure_count": {cell: len(failed[cell]) for cell in RELATIVE_CELLS},
        "patterns": [
            {"failed_cells": list(cells), "count": len(ids), "scene_ids": ids}
            for cells, ids in sorted(patterns.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
        "per_scene": per_scene,
        "pairwise": pairwise,
        "failure_rows": failure_rows,
    }


def _load_registry_context(
    path: Path,
    expected_paths: dict[str, Path],
) -> dict[str, Any]:
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read labctl context {path}: {exc}") from exc
    inputs = {item["role"]: item for item in context.get("inputs", [])}
    for role, expected in expected_paths.items():
        item = inputs.get(role)
        if item is None or not item.get("artifact_id"):
            raise AnalysisError(f"labctl context lacks registered artifact input {role}")
        if Path(item["resolved_path"]).resolve() != expected.resolve():
            raise AnalysisError(f"labctl context path mismatch for {role}")
    return {
        "run_id": context.get("run_id"),
        "context_path": str(path.resolve()),
        "inputs": {role: inputs[role] for role in expected_paths},
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_analysis(
    directories: dict[str, Path],
    *,
    labctl_context: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples < 1_000:
        raise AnalysisError("at least 1000 bootstrap resamples are required")
    rows_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    provenance = {}
    expected_scene_ids: set[str] | None = None
    expected_kinds: dict[str, str] | None = None
    for cell in CELLS:
        directory = directories[cell].resolve()
        try:
            _value, provenance[cell] = _load_cell(
                cell, directory, "in_box", require_source_checkpoint=False,
            )
            rows = _jsonl_rows(directory / "rows.jsonl")
        except EffectError as exc:
            raise AnalysisError(str(exc)) from exc
        mapped = {row["scene_id"]: row for row in rows}
        scene_ids = set(mapped)
        kinds = {scene_id: row["kind"] for scene_id, row in mapped.items()}
        if expected_scene_ids is None:
            expected_scene_ids = scene_ids
            expected_kinds = kinds
        elif scene_ids != expected_scene_ids or kinds != expected_kinds:
            raise AnalysisError(f"{cell}: scenes/kinds are not exactly paired across cells")
        rows_by_cell[cell] = mapped
        source_checkpoint = Path(provenance[cell]["model_provenance"]["source_checkpoint"])
        provenance[cell]["source_checkpoint_retention"] = {
            "path": str(source_checkpoint),
            "directory_present": source_checkpoint.is_dir(),
            "completion_marker_present": (
                source_checkpoint / "_CHECKPOINT_METADATA"
            ).is_file(),
            "required_for_this_registered_eval_analysis": False,
        }
    if expected_scene_ids is None or len(expected_scene_ids) != EXPECTED_ROWS:
        raise AnalysisError("factorial does not contain exactly 80 shared scenes")
    registry = _load_registry_context(labctl_context, directories)
    paired = paired_factorial_analysis(rows_by_cell, resamples=resamples, seed=seed)
    return {
        "artifact_type": "synthetic_relative_factorial_paired_uncertainty",
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "metric": "in_box",
        "bootstrap": {
            "method": "nonparametric percentile bootstrap, resampling shared scene_id clusters",
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "stratification": "all, long, and short; long-minus-short uses independent stratum resamples",
        },
        "paired_factorial": paired,
        "cell_decomposition": cell_decomposition(rows_by_cell),
        "failure_overlap": failure_overlap(rows_by_cell),
        "provenance": {
            "labctl": registry,
            "cells": provenance,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}", f"--{cell}", dest=cell, type=Path, required=True)
    parser.add_argument("--labctl-context", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-resamples", "--bootstrap_resamples",
        dest="bootstrap_resamples", type=int, default=BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed", "--bootstrap_seed",
        dest="bootstrap_seed", type=int, default=BOOTSTRAP_SEED,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directories = {cell: getattr(args, cell) for cell in CELLS}
    try:
        payload = run_analysis(
            directories,
            labctl_context=args.labctl_context,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
    except AnalysisError as exc:
        print(f"FATAL paired factorial analysis: {exc}", file=sys.stderr)
        return 2
    _atomic_write(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
