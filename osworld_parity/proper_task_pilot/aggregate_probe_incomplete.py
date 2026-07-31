#!/usr/bin/env python3
"""Seal the preregistered best-of-8 probe as valid but incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path
from typing import Any

try:
    from .aggregate_probe import (
        CANONICAL_TASKS_SHA256,
        CHECKPOINT_MANIFEST_SHA256,
        RUNTIME_TREE_SHA256,
        atomic_json,
        load_object,
        payload_seal_valid,
        sha256,
        validate_seed,
    )
except ImportError:  # Direct script execution from its immutable snapshot.
    from aggregate_probe import (
        CANONICAL_TASKS_SHA256,
        CHECKPOINT_MANIFEST_SHA256,
        RUNTIME_TREE_SHA256,
        atomic_json,
        load_object,
        payload_seal_valid,
        sha256,
        validate_seed,
    )


COMPLETE_SEEDS = (101, 211, 307, 401, 601, 701, 809)
ALL_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
OPERATIONAL_CORRECTION_SHA256 = (
    "776120e37ca78e0bce6c8fefd00380b610e531878f57815de79510db948291ea"
)
INCOMPLETE_AMENDMENT_SHA256 = (
    "dae1f85f42749cd3885f1d5c5d49c45ea95775e49a64377934d618bcbd469465"
)
FAILED_503_FAILURE_SHA256 = (
    "fba50a7f3351f5ace7e4132bd1266994f926832f7ea230d82d892e2481763331"
)
FAILED_503_RESULT_SHA256 = (
    "749685dc53b93dfe4b11cc5ddff9a34629904e06dfe21246885a22e1f894911b",
    "2ba27068e885c149f32d57ed15811b1e7c2ef61c61bc2ec3cecbabac9805300c",
)
FAILED_503_TRACE_SHA256 = (
    "3a97475279f13f21303bb3fd5c5a7cc3540a4c79fca203887b29c37da478efd6",
    "c45007c335adc99312d35afb1b6ff3e068363e77f8c1e63a4dad07c42c02bc5b",
)
EXPECTED_GPU_JOBS = {
    "135517": ("seed101", "COMPLETED", "0:0", 2253),
    "135518": ("seed211", "COMPLETED", "0:0", 1292),
    "135519": ("seed307", "COMPLETED", "0:0", 1674),
    "135520": ("seed401", "COMPLETED", "0:0", 1181),
    "135555": ("seed503_failed_infra", "FAILED", "1:0", 897),
    "135556": ("seed601", "COMPLETED", "0:0", 1074),
    "135558": ("seed701", "COMPLETED", "0:0", 1409),
    "135559": ("seed809", "COMPLETED", "0:0", 1249),
    "135575": ("seed503_continuation_failed_infra", "FAILED", "1:0", 551),
}


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def validate_gpu_accounting(path: Path) -> dict[str, Any]:
    payload = load_object(path)
    if not payload_seal_valid(payload):
        raise ValueError("GPU accounting payload seal mismatch")
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("GPU accounting schema/status mismatch")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(EXPECTED_GPU_JOBS):
        raise ValueError("GPU accounting job set mismatch")
    observed: dict[str, tuple[str, str, str, int]] = {}
    total = 0
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("GPU accounting row is not an object")
        elapsed = job.get("elapsed_raw")
        if (
            job.get("gpu_count") != 1
            or not isinstance(elapsed, int)
            or isinstance(elapsed, bool)
            or elapsed < 0
        ):
            raise ValueError("GPU accounting row has invalid GPU count or elapsed time")
        job_id = str(job.get("job_id"))
        if job_id in observed:
            raise ValueError(f"duplicate GPU job {job_id}")
        observed[job_id] = (
            str(job.get("role")),
            str(job.get("state")),
            str(job.get("exit_code")),
            elapsed,
        )
        total += elapsed
    if observed != EXPECTED_GPU_JOBS:
        raise ValueError(f"GPU accounting identity/state/time mismatch: {observed}")
    if payload.get("first_half_seconds") != 6400:
        raise ValueError("GPU accounting first-half total changed")
    if payload.get("total_gpu_seconds") != total or total != 11_580:
        raise ValueError(f"GPU accounting total mismatch: {total}")
    if total > 14_400:
        raise ValueError(f"GPU cap violated: {total}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "payload_sha256": payload["payload_sha256"],
        "total_gpu_seconds": total,
        "cap_gpu_seconds": 14_400,
        "margin_gpu_seconds": 14_400 - total,
        "jobs": jobs,
    }


def _failure_is_expected(path: Path) -> dict[str, Any]:
    failure = load_object(path)
    if (
        sha256(path) != FAILED_503_FAILURE_SHA256
        or failure.get("schema_version") != 1
        or failure.get("status") != "failed"
        or failure.get("artifact_valid") is not False
        or failure.get("error_type") != "TaskError"
        or "infrastructure-invalid" not in str(failure.get("message"))
        or "no byte screenshot" not in str(failure.get("message"))
    ):
        raise ValueError(f"unexpected seed-503 failure evidence: {path}")
    return failure


def _failed_parent_cell(
    directory: Path, *, index: int, task_id: str
) -> dict[str, Any]:
    task_dir = directory / "tasks" / f"{index:02d}_{task_id}"
    result_path = task_dir / "result.json"
    trace_path = task_dir / "trace.json"
    if (
        sha256(result_path) != FAILED_503_RESULT_SHA256[index]
        or sha256(trace_path) != FAILED_503_TRACE_SHA256[index]
    ):
        raise ValueError(f"seed-503 parent cell {index} seal mismatch")
    result = load_object(result_path)
    trace = load_object(trace_path)
    task_json_path = Path(str(result.get("task_json")))
    if (
        result.get("schema_version") != 1
        or result.get("mode") != "probe_seed"
        or result.get("arm") != "relative_r256"
        or result.get("action_format") != "move_rel"
        or result.get("order_index") != index
        or result.get("task_id") != task_id
        or result.get("session_id") != f"session-{index + 1:06d}"
        or result.get("sampling")
        != {"temperature": 0.7, "top_p": 0.95, "seed": 503}
        or result.get("infra_valid") is not True
        or result.get("infra_error") is not None
        or result.get("trace_error") is not None
        or result.get("raw_reward") != 0.0
        or result.get("full_success") is not False
        or result.get("parse_errors") != 0
        or result.get("trace") != str(trace_path.relative_to(directory))
        or result.get("trace_sha256") != FAILED_503_TRACE_SHA256[index]
        or not task_json_path.is_file()
        or sha256(task_json_path) != result.get("task_json_sha256")
        or trace.get("errors") != []
        or trace.get("is_completed") is not True
        or trace.get("state", {}).get("infra_valid") is not True
        or trace.get("state", {}).get("task_reward") != result.get("raw_reward")
    ):
        raise ValueError(f"seed-503 parent cell {index} content mismatch")
    return {
        "seed": 503,
        "task_id": task_id,
        "raw_reward": result["raw_reward"],
        "full_success": result["full_success"],
        "parse_errors": result["parse_errors"],
        "result_sha256": FAILED_503_RESULT_SHA256[index],
        "trace_sha256": FAILED_503_TRACE_SHA256[index],
        "source_attempt": "failed_parent_valid_cell",
    }


def validate_failed_seed503_parent(
    directory: Path, canonical_ids: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failure_path = directory / "failure.json"
    _failure_is_expected(failure_path)
    if (directory / "run_manifest.json").exists():
        raise ValueError("failed seed-503 parent unexpectedly has a run manifest")
    expected_results = [
        directory / "tasks" / f"{index:02d}_{canonical_ids[index]}" / "result.json"
        for index in (0, 1)
    ]
    if sorted((directory / "tasks").glob("*/result.json")) != expected_results:
        raise ValueError("failed seed-503 parent result set changed")
    cells = [
        _failed_parent_cell(directory, index=index, task_id=canonical_ids[index])
        for index in (0, 1)
    ]
    return (
        {
            "directory": str(directory.resolve()),
            "failure_sha256": FAILED_503_FAILURE_SHA256,
            "failure_classification": "post_action_screenshot_transport_infrastructure_invalid",
            "published_valid_canonical_indices": [0, 1],
            "excluded_failed_canonical_index": 2,
            "result_sha256": list(FAILED_503_RESULT_SHA256),
            "trace_sha256": list(FAILED_503_TRACE_SHA256),
        },
        cells,
    )


def validate_failed_continuation(directory: Path) -> dict[str, Any]:
    failure_path = directory / "failure.json"
    _failure_is_expected(failure_path)
    if (directory / "run_manifest.json").exists():
        raise ValueError("failed seed-503 continuation unexpectedly has a run manifest")
    if list(directory.glob("tasks/*/result.json")):
        raise ValueError("failed seed-503 continuation unexpectedly published a result")
    return {
        "directory": str(directory.resolve()),
        "failure_sha256": FAILED_503_FAILURE_SHA256,
        "failure_classification": "post_action_screenshot_transport_infrastructure_invalid",
        "intended_canonical_indices": list(range(2, 12)),
        "published_valid_cells": 0,
    }


def posterior_predictive_gate_probability(
    alpha: Fraction, beta: Fraction
) -> dict[str, Any]:
    """Exact posterior predictive event over one missing draw for each task."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    high = (Fraction(7) + alpha) / (Fraction(7) + alpha + beta)
    low = alpha / (Fraction(7) + alpha + beta)

    # Exact enumeration makes both gate clauses explicit: at least five of ten
    # missing successes and at least four successful tasks overall.  The one
    # observed successful task is the high-probability task; each low task is
    # a previously unsuccessful task.
    probability = Fraction(0)
    for high_outcome, *low_outcomes in product((0, 1), repeat=10):
        missing_successes = high_outcome + sum(low_outcomes)
        total_successful_tasks = 1 + sum(low_outcomes)
        if missing_successes < 5 or total_successful_tasks < 4:
            continue
        term = high if high_outcome else 1 - high
        for outcome in low_outcomes:
            term *= low if outcome else 1 - low
        probability += term

    # Cross-check the reduced closed form.  The successful-task clause is
    # redundant once five missing successes are required because at most one
    # missing success can occur on the already-successful task.
    def binomial_tail(minimum: int) -> Fraction:
        return sum(
            Fraction(comb(9, count))
            * low**count
            * (1 - low) ** (9 - count)
            for count in range(minimum, 10)
        )

    closed_form = high * binomial_tail(4) + (1 - high) * binomial_tail(5)
    if probability != closed_form:
        raise AssertionError("posterior predictive enumeration disagrees with closed form")
    return {
        "alpha": f"{alpha.numerator}/{alpha.denominator}",
        "beta": f"{beta.numerator}/{beta.denominator}",
        "posterior_predictive_success_probability_observed_7_of_7": {
            "exact": f"{high.numerator}/{high.denominator}",
            "decimal": float(high),
        },
        "posterior_predictive_success_probability_observed_0_of_7": {
            "exact": f"{low.numerator}/{low.denominator}",
            "decimal": float(low),
        },
        "gate_open_probability": {
            "exact": f"{probability.numerator}/{probability.denominator}",
            "decimal": float(probability),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--operational-correction", type=Path, required=True)
    parser.add_argument("--incomplete-amendment", type=Path, required=True)
    parser.add_argument("--gpu-accounting", type=Path, required=True)
    for seed in COMPLETE_SEEDS:
        parser.add_argument(f"--seed{seed}", type=Path, required=True)
    parser.add_argument("--seed503", type=Path, required=True)
    parser.add_argument("--seed503-continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reasons: list[str] = []
    cells: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    canonical_ids: list[str] = []
    gpu_accounting: dict[str, Any] | None = None
    try:
        if sha256(args.operational_correction) != OPERATIONAL_CORRECTION_SHA256:
            raise ValueError("operational-correction hash mismatch")
        if sha256(args.incomplete_amendment) != INCOMPLETE_AMENDMENT_SHA256:
            raise ValueError("incomplete-result amendment hash mismatch")
        if sha256(args.tasks) != CANONICAL_TASKS_SHA256:
            raise ValueError("canonical task-list hash mismatch")
        canonical_ids = args.tasks.read_text(encoding="utf-8").splitlines()
        if len(canonical_ids) != 12 or len(set(canonical_ids)) != 12:
            raise ValueError("canonical task list is not 12 unique IDs")

        gpu_accounting = validate_gpu_accounting(args.gpu_accounting)
        for seed in COMPLETE_SEEDS:
            directory = getattr(args, f"seed{seed}")
            manifest, seed_cells = validate_seed(
                directory, seed=seed, canonical_ids=canonical_ids
            )
            inputs[str(seed)] = {
                "directory": str(directory.resolve()),
                "manifest_sha256": sha256(directory / "run_manifest.json"),
                "payload_sha256": manifest["payload_sha256"],
            }
            cells.extend({**cell, "source_attempt": "complete_seed"} for cell in seed_cells)

        parent_input, parent_cells = validate_failed_seed503_parent(
            args.seed503, canonical_ids
        )
        inputs["503"] = parent_input
        cells.extend(parent_cells)
        inputs["503_continuation"] = validate_failed_continuation(
            args.seed503_continuation
        )

        expected_keys = {
            (seed, task_id)
            for seed in COMPLETE_SEEDS
            for task_id in canonical_ids
        } | {(503, canonical_ids[index]) for index in (0, 1)}
        observed_keys = {(cell["seed"], cell["task_id"]) for cell in cells}
        if observed_keys != expected_keys or len(cells) != 86:
            raise ValueError("valid task/seed crossing is not the exact expected 86 cells")

        positive = [cell for cell in cells if cell["full_success"] is True]
        rejection = [
            cell
            for cell in cells
            if cell["raw_reward"] == 0.0
            and cell["full_success"] is False
            and cell["parse_errors"] == 0
        ]
        if (
            len(positive) != 7
            or {cell["task_id"] for cell in positive} != {canonical_ids[6]}
            or {cell["seed"] for cell in positive} != set(COMPLETE_SEEDS)
            or len(rejection) != 66
            or len({cell["task_id"] for cell in rejection}) != 11
        ):
            raise ValueError("observed success/rejection evidence changed")
    except Exception as exc:  # noqa: BLE001 - publish fail-closed evidence
        reasons.append(f"{type(exc).__name__}: {exc}")

    positive = [cell for cell in cells if cell.get("full_success") is True]
    rejection = [
        cell
        for cell in cells
        if cell.get("raw_reward") == 0.0
        and cell.get("full_success") is False
        and cell.get("parse_errors") == 0
    ]
    positive_tasks = sorted({cell["task_id"] for cell in positive})
    rejection_tasks = sorted({cell["task_id"] for cell in rejection})
    validated = not reasons and len(cells) == 86
    missing_cells = (
        [
            {"seed": 503, "canonical_index": index, "task_id": canonical_ids[index]}
            for index in range(2, 12)
        ]
        if len(canonical_ids) == 12
        else []
    )

    per_task: list[dict[str, Any]] = []
    if validated:
        for index, task_id in enumerate(canonical_ids):
            task_cells = [cell for cell in cells if cell["task_id"] == task_id]
            per_task.append(
                {
                    "canonical_index": index,
                    "task_id": task_id,
                    "observed_attempts": len(task_cells),
                    "observed_successes": sum(
                        cell["full_success"] is True for cell in task_cells
                    ),
                    "missing_seed503": index >= 2,
                }
            )

    bayesian = (
        {
            "label": "post_hoc_sensitivity_diagnostic_not_a_gate_substitute",
            "assumptions": [
                "task success probabilities are independent across tasks",
                "seed outcomes are conditionally Bernoulli within each task",
                "one seed-503 posterior-predictive draw remains for each canonical task index 2 through 11",
                "the nine previously unsuccessful missing tasks share prior hyperparameters but have independent priors",
            ],
            "method": "exact enumeration of all 2^10 posterior-predictive binary outcomes, cross-checked against a two-binomial-tail closed form",
            "event": "at least 5 of 10 missing cells succeed and at least 4 tasks are successful overall",
            "observations": {
                "already_successful_missing_task": "canonical_index_6: 7 successes in 7 observed seeds",
                "other_missing_tasks": "9 tasks: 0 successes in 7 observed seeds each",
            },
            "priors": {
                "jeffreys_beta_half_half": posterior_predictive_gate_probability(
                    Fraction(1, 2), Fraction(1, 2)
                ),
                "uniform_beta_one_one": posterior_predictive_gate_probability(
                    Fraction(1), Fraction(1)
                ),
            },
        }
        if validated
        else None
    )

    status = "incomplete" if validated else "invalid"
    payload = _sealed(
        {
            "schema_version": 1,
            "status": status,
            "artifact_valid": validated,
            "probe_complete": False,
            "expected_cells": 96,
            "valid_cells": len(cells),
            "missing_cells": missing_cells,
            "seeds": list(ALL_SEEDS),
            "canonical_tasks_sha256": CANONICAL_TASKS_SHA256,
            "operational_correction_sha256": OPERATIONAL_CORRECTION_SHA256,
            "incomplete_amendment_sha256": INCOMPLETE_AMENDMENT_SHA256,
            "runtime_tree_sha256": RUNTIME_TREE_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "gpu_accounting": gpu_accounting,
            "observed": {
                "success_count": len(positive),
                "successful_task_count": len(positive_tasks),
                "successful_tasks": positive_tasks,
                "parse_valid_reward_zero_count": len(rejection),
                "parse_valid_reward_zero_task_count": len(rejection_tasks),
                "parse_valid_reward_zero_tasks": rejection_tasks,
            },
            "deterministic_bounds": {
                "success_count": {"minimum": 7, "maximum": 17},
                "successful_task_count": {"minimum": 1, "maximum": 10},
                "best_of_8_success_task_count": {"minimum": 1, "maximum": 10},
            }
            if validated
            else None,
            "yield_gate": {
                "formal_status": "unresolved",
                "open": None,
                "minimum_successes": 12,
                "minimum_success_tasks": 4,
                "minimum_parse_valid_reward_zero": 12,
                "minimum_parse_valid_reward_zero_tasks": 4,
                "rejection_clause_irrevocably_open": True if validated else None,
                "yield_gate_opens_iff_missing_successes_at_least": 5
                if validated
                else None,
            },
            "post_hoc_bayesian_sensitivity": bayesian,
            "per_task": per_task,
            "inputs": inputs,
            "cells": cells,
            "reasons": reasons,
            "recovery_stop_reason": (
                "no exact-state retry is possible after VM teardown; another job would reset "
                "the VM and repeat inference/actions"
            ),
            "gpu_training_authorized": False,
        }
    )
    atomic_json(args.output / "probe_manifest.json", payload)
    print(
        f"best-of-8 aggregate status={status} valid_cells={len(cells)} "
        f"successes={len(positive)} formal_yield_gate=unresolved",
        flush=True,
    )
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
