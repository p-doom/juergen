"""Frozen-pool comparison for the bounded relative-mouse pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from stage5_rft.util import ContractError, read_json


EXPECTED_BASELINE_SHA256 = "782e2fd0e4f6c3943c2cbc02f8e5fdb4028114090ec2ef7a44dfd234779f96e0"
EXPECTED_BASELINE_JOB_ID = "135215"
EXPECTED_TASK_KEYS = {f"val:{index}" for index in range(128)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _validate_probe(path: Path, *, baseline: bool) -> tuple[dict[str, Any], dict[str, int]]:
    payload = read_json(path)
    summary = payload.get("summary")
    per_task = payload.get("per_task")
    if not isinstance(summary, dict) or not isinstance(per_task, list):
        raise ContractError(f"probe has invalid schema: {path}")
    required = {
        "status": "OK",
        "n_tasks": 128,
        "k": 16,
        "n_rollouts_ok": 2048,
        "n_rollouts_err": 0,
        "error_rate": 0.0,
        "pool": "val",
        "val_seed": 7777,
        "offset": 0,
        "temperature": 0.7,
        "max_steps": 8,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ContractError(
                f"probe {path} {key}={summary.get(key)!r}, expected {expected!r}"
            )
    if baseline and _sha256(path) != EXPECTED_BASELINE_SHA256:
        raise ContractError("frozen baseline probe hash differs from preregistration")

    counts: dict[str, int] = {}
    for row in per_task:
        if not isinstance(row, dict):
            raise ContractError("per-task probe row is not an object")
        key = str(row.get("task_key", ""))
        if key in counts:
            raise ContractError(f"duplicate task key: {key}")
        n, c = row.get("n"), row.get("c")
        if isinstance(n, bool) or not isinstance(n, int) or n != 16:
            raise ContractError(f"task {key} does not have fixed n=16")
        if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= n:
            raise ContractError(f"task {key} has invalid success count")
        counts[key] = c
    if set(counts) != EXPECTED_TASK_KEYS:
        raise ContractError("probe does not contain the exact frozen val:0..127 task set")
    accepted = sum(counts.values())
    if summary.get("n_accepted_rollouts") != accepted:
        raise ContractError("summary accepted count disagrees with per-task counts")
    return summary, counts


def _parse_sacct(path: Path, expected_job_id: str) -> dict[str, Any]:
    rows = [line.split("|") for line in path.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row and row[0] == expected_job_id]
    if len(matches) != 1 or len(matches[0]) != 5:
        raise ContractError(f"accounting file lacks one exact row for job {expected_job_id}")
    job_id, state, elapsed_raw, exit_code, alloc_tres = matches[0]
    if state != "COMPLETED" or exit_code != "0:0":
        raise ContractError(f"job {job_id} did not complete successfully: {state}/{exit_code}")
    try:
        elapsed_seconds = int(elapsed_raw)
    except ValueError as exc:
        raise ContractError(f"job {job_id} has invalid ElapsedRaw") from exc
    gpu_match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", alloc_tres)
    if elapsed_seconds <= 0 or gpu_match is None or int(gpu_match.group(1)) != 1:
        raise ContractError(f"job {job_id} does not bind one positive-duration GPU allocation")
    return {
        "job_id": job_id,
        "state": state,
        "elapsed_seconds": elapsed_seconds,
        "gpus": 1,
        "gpu_seconds": elapsed_seconds,
        "alloc_tres": alloc_tres,
    }


def compare_relative_mouse_eval(
    *,
    baseline_probe_path: str | Path,
    candidate_probe_path: str | Path,
    baseline_sacct_path: str | Path,
    candidate_sacct_path: str | Path,
    candidate_manifest_path: str | Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline_probe_path)
    candidate_path = Path(candidate_probe_path)
    manifest = read_json(candidate_manifest_path)
    required_manifest = {
        "status": "complete",
        "data_class": "synthetic_untouched_validation",
        "validation_seed": 7777,
        "tasks": 128,
        "k": 16,
        "attempts": 2048,
        "maximum_rollout_errors": 0,
        "adaptive_resampling": False,
        "contains_official_heldout": False,
        "contains_real_vm_eval": False,
        "contains_crowd_cast": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ContractError(
                f"candidate manifest {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if manifest.get("probe_sha256") != _sha256(candidate_path):
        raise ContractError("candidate probe hash differs from its manifest")

    baseline_summary, baseline_counts = _validate_probe(baseline_path, baseline=True)
    candidate_summary, candidate_counts = _validate_probe(candidate_path, baseline=False)
    candidate_job_id = str(manifest.get("slurm_job_id", ""))
    if not candidate_job_id.isdigit():
        raise ContractError("candidate manifest has invalid Slurm job id")
    baseline_accounting = _parse_sacct(Path(baseline_sacct_path), EXPECTED_BASELINE_JOB_ID)
    candidate_accounting = _parse_sacct(Path(candidate_sacct_path), candidate_job_id)

    def metrics(summary: dict[str, Any], counts: dict[str, int], accounting: dict[str, Any]):
        pass_metrics = {
            f"pass@{k}": sum(_pass_at_k(16, value, k) for value in counts.values()) / 128
            for k in (1, 4, 8)
        }
        for key, computed in pass_metrics.items():
            reported = summary.get(key)
            if not isinstance(reported, (int, float)) or abs(float(reported) - computed) > 5e-4:
                raise ContractError(f"reported {key} disagrees with exact per-task recomputation")
        accepted = sum(counts.values())
        return {
            **pass_metrics,
            "accepted_rollouts": accepted,
            "allocated_gpu_seconds": accounting["gpu_seconds"],
            "accepted_per_allocated_gpu_hour": accepted * 3600 / accounting["gpu_seconds"],
        }

    baseline_metrics = metrics(baseline_summary, baseline_counts, baseline_accounting)
    candidate_metrics = metrics(candidate_summary, candidate_counts, candidate_accounting)
    delta = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in ("pass@1", "pass@4", "pass@8", "accepted_per_allocated_gpu_hour")
    }
    return {
        "schema_version": "stage5.relative_mouse_pilot_comparison.v1",
        "status": "complete",
        "comparison": "descriptive_frozen_synthetic_validation",
        "task_pairing": "exact val:0..127, seed 7777, k=16",
        "baseline": {
            "probe_path": str(baseline_path.resolve()),
            "probe_sha256": _sha256(baseline_path),
            "accounting": baseline_accounting,
            "metrics": baseline_metrics,
        },
        "candidate": {
            "probe_path": str(candidate_path.resolve()),
            "probe_sha256": _sha256(candidate_path),
            "model_artifact_id": manifest.get("model_artifact_id"),
            "checkpoint_digest": manifest.get("checkpoint_digest"),
            "accounting": candidate_accounting,
            "metrics": candidate_metrics,
        },
        "candidate_minus_baseline": delta,
        "contains_official_heldout": False,
        "contains_real_vm_eval": False,
        "contains_crowd_cast": False,
        "promotion_authorized": False,
    }
