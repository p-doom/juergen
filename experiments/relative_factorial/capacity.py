#!/usr/bin/env python3
"""Paired LoRA-capacity analysis for the four relative-format cells."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from .effects import EXPECTED_ROWS, EffectError, _jsonl_rows, _load_cell
    from .uncertainty import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        _atomic_write,
        _load_registry_context,
        _mean,
        _quantile,
        exact_sign_flip_p,
    )
except ImportError:  # Direct execution by a labctl recipe.
    from effects import EXPECTED_ROWS, EffectError, _jsonl_rows, _load_cell
    from uncertainty import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        _atomic_write,
        _load_registry_context,
        _mean,
        _quantile,
        exact_sign_flip_p,
    )


RANKS = (32, 64, 256)
ARMS = ("relraw_act", "relraw_pre", "reltool_act", "reltool_pre")
CELL_NAMES = {
    "relraw_act": "rel_bare_act",
    "relraw_pre": "rel_bare_pre",
    "reltool_act": "rel_tool_act",
    "reltool_pre": "rel_tool_pre",
}
CODES = {
    "relraw_act": (-1, -1),
    "relraw_pre": (-1, +1),
    "reltool_act": (+1, -1),
    "reltool_pre": (+1, +1),
}
TERMS = {
    "capacity": (),
    "capacity×grammar": (0,),
    "capacity×preamble": (1,),
    "capacity×grammar×preamble": (0, 1),
}


class CapacityError(RuntimeError):
    pass


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def factorial_effect(values: dict[str, float], axes: tuple[int, ...]) -> float:
    """Return the conventional 2-level effect over the four relative cells."""
    if not axes:
        return _mean(values.values())
    positive = [
        values[arm] for arm in ARMS if _product(CODES[arm][axis] for axis in axes) == 1
    ]
    negative = [
        values[arm] for arm in ARMS if _product(CODES[arm][axis] for axis in axes) == -1
    ]
    return _mean(positive) - _mean(negative)


def capacity_contributions(
    rows: dict[int, dict[str, dict[str, dict[str, Any]]]],
    *,
    rank: int,
    scene_ids: list[str],
) -> dict[str, list[float]]:
    if rank == 32:
        raise CapacityError("capacity contrast rank must differ from reference rank 32")
    result = {term: [] for term in TERMS}
    for scene_id in scene_ids:
        reference = {
            arm: float(rows[32][arm][scene_id]["in_box"]) for arm in ARMS
        }
        candidate = {
            arm: float(rows[rank][arm][scene_id]["in_box"]) for arm in ARMS
        }
        for term, axes in TERMS.items():
            result[term].append(
                factorial_effect(candidate, axes) - factorial_effect(reference, axes)
            )
    return result


def _contrast_summary(
    contributions: dict[str, list[float]],
    *,
    resamples: int,
    rng: random.Random,
) -> dict[str, Any]:
    length = len(next(iter(contributions.values())))
    replicates = {term: [] for term in TERMS}
    for _ in range(resamples):
        indices = [rng.randrange(length) for _ in range(length)]
        for term in TERMS:
            values = contributions[term]
            replicates[term].append(_mean(values[index] for index in indices))
    return {
        term: {
            "estimate": _mean(values),
            "ci95_percentile": [
                _quantile(replicates[term], 0.025),
                _quantile(replicates[term], 0.975),
            ],
            "exact_paired_sign_flip_p_two_sided": exact_sign_flip_p(values),
            "scene_contributions": values,
        }
        for term, values in contributions.items()
    }


def analyze_rows(
    rows: dict[int, dict[str, dict[str, dict[str, Any]]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    scene_ids = sorted(rows[32][ARMS[0]])
    strata = {
        "all": scene_ids,
        "long": [scene for scene in scene_ids if rows[32][ARMS[0]][scene]["kind"] == "long"],
        "short": [scene for scene in scene_ids if rows[32][ARMS[0]][scene]["kind"] == "short"],
    }
    cell_rates: dict[str, Any] = {}
    rank_means: dict[str, Any] = {}
    for rank in RANKS:
        cell_rates[str(rank)] = {
            arm: {
                stratum: _mean(float(rows[rank][arm][scene]["in_box"]) for scene in ids)
                for stratum, ids in strata.items()
            }
            for arm in ARMS
        }
        rank_means[str(rank)] = {
            stratum: _mean(cell_rates[str(rank)][arm][stratum] for arm in ARMS)
            for stratum in strata
        }

    decisions = {}
    contrasts = {}
    rng = random.Random(seed)
    for rank in (64, 256):
        overall_change = rank_means[str(rank)]["all"] - rank_means["32"]["all"]
        long_change = rank_means[str(rank)]["long"] - rank_means["32"]["long"]
        decisions[str(rank)] = {
            "overall_change_from_r32": overall_change,
            "long_change_from_r32": long_change,
            "capacity_response_overall_ge_5pp": overall_change >= 0.05,
            "capacity_response_long_ge_10pp": long_change >= 0.10,
            "capacity_response_either_gate": overall_change >= 0.05 or long_change >= 0.10,
            "practical_parity_overall_ge_95pct": rank_means[str(rank)]["all"] >= 0.95,
            "practical_parity_long_ge_90pct": rank_means[str(rank)]["long"] >= 0.90,
            "practical_parity_both_gates": (
                rank_means[str(rank)]["all"] >= 0.95
                and rank_means[str(rank)]["long"] >= 0.90
            ),
        }
        contrasts[f"r{rank}_minus_r32"] = {
            stratum: _contrast_summary(
                capacity_contributions(rows, rank=rank, scene_ids=ids),
                resamples=resamples,
                rng=rng,
            )
            for stratum, ids in strata.items()
        }
    return {
        "cell_accuracy": cell_rates,
        "equal_cell_rank_means": rank_means,
        "preregistered_decisions": decisions,
        "paired_capacity_contrasts": contrasts,
    }


def run_analysis(
    directories: dict[int, dict[str, Path]],
    *,
    labctl_context: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples < 1_000:
        raise CapacityError("at least 1000 bootstrap resamples are required")
    rows: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance: dict[str, Any] = {}
    registry_paths: dict[str, Path] = {}
    expected_scenes: set[str] | None = None
    expected_kinds: dict[str, str] | None = None
    for rank in RANKS:
        rows[rank] = {}
        provenance[str(rank)] = {}
        for arm in ARMS:
            directory = directories[rank][arm].resolve()
            try:
                _value, cell_provenance = _load_cell(
                    CELL_NAMES[arm],
                    directory,
                    "in_box",
                    require_source_checkpoint=False,
                    expected_lora_rank=rank,
                )
                cell_rows = _jsonl_rows(directory / "rows.jsonl")
            except EffectError as exc:
                raise CapacityError(str(exc)) from exc
            mapped = {row["scene_id"]: row for row in cell_rows}
            scenes = set(mapped)
            kinds = {scene: row["kind"] for scene, row in mapped.items()}
            if expected_scenes is None:
                expected_scenes, expected_kinds = scenes, kinds
            elif scenes != expected_scenes or kinds != expected_kinds:
                raise CapacityError(f"r{rank}/{arm}: scenes or strata are not exactly paired")
            rows[rank][arm] = mapped
            provenance[str(rank)][arm] = cell_provenance
            registry_paths[f"r{rank}_{arm}"] = directory
    if expected_scenes is None or len(expected_scenes) != EXPECTED_ROWS:
        raise CapacityError("capacity analysis does not contain exactly 80 paired scenes")
    registry = _load_registry_context(labctl_context, registry_paths)
    return {
        "artifact_type": "synthetic_relative_factorial_lora_capacity",
        "schema_version": 1,
        "status": "complete",
        "metric": "in_box",
        "reference_rank": 32,
        "candidate_ranks": [64, 256],
        "effect_coding": {
            "capacity": "each candidate rank minus rank 32",
            "grammar": {"relraw": -1, "reltool": 1},
            "preamble": {"action_only": -1, "preamble": 1},
            "interaction_scale": (
                "conventional factorial effects: positive-product mean minus "
                "negative-product mean, then candidate minus rank 32"
            ),
        },
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap over shared scene_id",
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
        },
        **analyze_rows(rows, resamples=resamples, seed=seed),
        "provenance": {"labctl": registry, "evaluations": provenance},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for rank in RANKS:
        for arm in ARMS:
            destination = f"r{rank}_{arm}"
            parser.add_argument(
                f"--r{rank}-{arm.replace('_', '-')}",
                f"--r{rank}_{arm}",
                dest=destination,
                type=Path,
                required=True,
            )
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
    directories = {
        rank: {arm: getattr(args, f"r{rank}_{arm}") for arm in ARMS} for rank in RANKS
    }
    try:
        payload = run_analysis(
            directories,
            labctl_context=args.labctl_context,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
    except CapacityError as exc:
        print(f"FATAL capacity analysis: {exc}", file=sys.stderr)
        return 2
    _atomic_write(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
