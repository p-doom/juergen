"""Paired, hierarchical-bootstrap aggregation for ROADMAP 3.5."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .contract import (
    ARMS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    COMPACT_RAW_ARM,
    COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING,
    COMPACT_RAW_SUCCESS_FLOOR,
    EXPECTED_EPISODE_COUNT,
    NATIVE_ABSOLUTE_ARM,
    NONINFERIORITY_MARGIN,
    PAIRED_SEEDS,
    PILOT_TASK_COUNT,
)
from .gates import (
    GateBundle,
    GateError,
    LaunchAuthorization,
    SignedGatePaths,
    verify_gate_bundle,
)
from .io import atomic_json
from .records import EpisodeRow, RecordError, parse_episode_row


class AggregateError(ValueError):
    """Rows are incomplete, unpaired, infrastructure-invalid, or malformed."""


def load_jsonl(path: Path) -> list[Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise AggregateError("rows input must be an existing regular file")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise AggregateError("rows input exceeds the bounded pilot size")
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AggregateError("cannot read rows input") from exc
    if not lines:
        raise AggregateError("rows input is empty")
    result: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AggregateError(f"invalid JSON on row {line_number}") from exc
    return result


def _validated_pairs(
    payloads: Iterable[Any], authorization: LaunchAuthorization
) -> dict[tuple[int, int], dict[str, EpisodeRow]]:
    rows = [
        parse_episode_row(payload, expected_pilot_id=authorization.pilot_id)
        for payload in payloads
    ]
    if len(rows) != EXPECTED_EPISODE_COUNT:
        raise AggregateError(
            f"expected {EXPECTED_EPISODE_COUNT} episode rows, found {len(rows)}"
        )
    pairs: dict[tuple[int, int], dict[str, EpisodeRow]] = {}
    for row in rows:
        if not (row.reset_success and row.setup_success and row.oracle_evaluated):
            raise AggregateError(
                "infrastructure-invalid rows cannot enter scientific aggregation"
            )
        key = (row.cluster_index, row.pair_seed)
        arm_rows = pairs.setdefault(key, {})
        if row.arm in arm_rows:
            raise AggregateError(f"duplicate arm row for pair {row.pair_key}")
        arm_rows[row.arm] = row
    expected_keys = {
        (cluster_index, pair_seed)
        for cluster_index in range(PILOT_TASK_COUNT)
        for pair_seed in PAIRED_SEEDS
    }
    if set(pairs) != expected_keys:
        raise AggregateError("pilot pairs do not cover the frozen cluster/seed grid")
    for key, arm_rows in pairs.items():
        if set(arm_rows) != set(ARMS):
            raise AggregateError(f"pair {key} does not contain exactly the common arms")
        control = arm_rows[NATIVE_ABSOLUTE_ARM]
        treatment = arm_rows[COMPACT_RAW_ARM]
        if (
            control.pair_key != treatment.pair_key
            or control.arm_order != treatment.arm_order
            or {control.reset_ordinal, treatment.reset_ordinal} != {1, 2}
        ):
            raise AggregateError(f"pair {key} violates seed/reset matching")
    return pairs


def _percentile(samples: Sequence[float], probability: float) -> float:
    if not samples:
        raise AggregateError("cannot compute a percentile of no samples")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        raise AggregateError("Wilson interval requires observations")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"rate": rate, "lower": center - radius, "upper": center + radius}


def _bootstrap_difference(
    pairs: dict[tuple[int, int], dict[str, EpisodeRow]],
) -> dict[str, float | int]:
    random_source = random.Random(BOOTSTRAP_SEED)
    clusters = list(range(PILOT_TASK_COUNT))
    seeds = list(PAIRED_SEEDS)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        differences: list[int] = []
        for sampled_cluster in random_source.choices(clusters, k=len(clusters)):
            for sampled_seed in random_source.choices(seeds, k=len(seeds)):
                arm_rows = pairs[(sampled_cluster, sampled_seed)]
                differences.append(
                    int(arm_rows[COMPACT_RAW_ARM].task_success)
                    - int(arm_rows[NATIVE_ABSOLUTE_ARM].task_success)
                )
        estimates.append(sum(differences) / len(differences))
    observed = sum(
        int(arm_rows[COMPACT_RAW_ARM].task_success)
        - int(arm_rows[NATIVE_ABSOLUTE_ARM].task_success)
        for arm_rows in pairs.values()
    ) / len(pairs)
    alpha = 1 - CI_LEVEL
    return {
        "estimate": observed,
        "lower": _percentile(estimates, alpha / 2),
        "upper": _percentile(estimates, 1 - alpha / 2),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def aggregate_rows(
    payloads: Iterable[Any], authorization: LaunchAuthorization
) -> dict[str, Any]:
    pairs = _validated_pairs(payloads, authorization)
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [arm_rows[arm] for arm_rows in pairs.values()]
        task_successes = sum(row.task_success for row in rows)
        parse_successes = sum(row.parse_success for row in rows)
        executor_successes = sum(row.executor_success for row in rows)
        parse_executor_failures = sum(
            not (row.parse_success and row.executor_success) for row in rows
        )
        arm_metrics[arm] = {
            "episodes": len(rows),
            "task_success": _wilson(task_successes, len(rows)),
            "parse_success_rate": parse_successes / len(rows),
            "executor_success_rate": executor_successes / len(rows),
            "parse_executor_failure_rate": parse_executor_failures / len(rows),
        }
    paired = _bootstrap_difference(pairs)
    compact = arm_metrics[COMPACT_RAW_ARM]
    criteria = {
        "paired_lower_above_margin": paired["lower"] > NONINFERIORITY_MARGIN,
        "compact_raw_success_at_least_floor": (
            compact["task_success"]["rate"] >= COMPACT_RAW_SUCCESS_FLOOR
        ),
        "compact_raw_parse_executor_failure_at_most_ceiling": (
            compact["parse_executor_failure_rate"]
            <= COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "pilot_id": authorization.pilot_id,
        "contract_id": authorization.contract_id,
        "task_clusters": PILOT_TASK_COUNT,
        "paired_seeds": list(PAIRED_SEEDS),
        "paired_cells": len(pairs),
        "episodes": EXPECTED_EPISODE_COUNT,
        "arms": arm_metrics,
        "paired_compact_raw_minus_native_absolute": paired,
        "noninferiority": {
            "margin": NONINFERIORITY_MARGIN,
            "compact_raw_success_floor": COMPACT_RAW_SUCCESS_FLOOR,
            "compact_raw_parse_executor_failure_ceiling": (
                COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING
            ),
            "criteria": criteria,
            "pass": all(criteria.values()),
        },
    }


def aggregate_authorized(
    bundle: GateBundle,
    rows_loader: Callable[[], Iterable[Any]],
    *,
    now: Any = None,
) -> dict[str, Any]:
    """Verify release gates before reading even sanitized result rows."""

    authorization = verify_gate_bundle(bundle, now=now)
    return aggregate_rows(rows_loader(), authorization)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prerequisites-gate", type=Path, required=True)
    parser.add_argument("--prerequisites-signature", type=Path, required=True)
    parser.add_argument("--pilot-release-gate", type=Path, required=True)
    parser.add_argument("--pilot-release-signature", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _bundle_from_args(args: argparse.Namespace) -> GateBundle:
    return GateBundle(
        prerequisites=SignedGatePaths(
            args.prerequisites_gate, args.prerequisites_signature
        ),
        pilot_release=SignedGatePaths(
            args.pilot_release_gate, args.pilot_release_signature
        ),
        allowed_signers=args.allowed_signers,
        signer_identity=args.signer_identity,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = aggregate_authorized(
            _bundle_from_args(args), lambda: load_jsonl(args.rows)
        )
        atomic_json(args.output, result)
    except (AggregateError, GateError, RecordError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
