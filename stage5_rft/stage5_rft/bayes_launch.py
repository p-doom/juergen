"""Bayesian feasibility gate for the relative-mouse on-policy lane.

The gate consumes an already-completed short-task probe.  It does not load a
checkpoint, issue policy requests, inspect official evaluation examples, or
authorize a learner.  Task heterogeneity is retained with a Bayesian bootstrap;
within-task uncertainty uses independent beta-binomial posteriors.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping

from stage5_rft.util import ContractError, read_json


def _sha256_file(path: str | Path) -> str:
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return number


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _evidence_rows(probe: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = probe.get("summary")
    per_task = probe.get("per_task")
    if not isinstance(summary, Mapping) or not isinstance(per_task, list) or not per_task:
        raise ContractError("probe must contain non-empty summary and per_task records")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(per_task):
        if not isinstance(raw, Mapping):
            raise ContractError(f"per_task[{index}] must be an object")
        key = str(raw.get("task_key", ""))
        if not key or key in seen:
            raise ContractError(f"per_task[{index}] has missing or duplicate task_key")
        seen.add(key)
        n = int(_require_number(raw.get("n"), f"per_task[{index}].n", minimum=1))
        c = int(_require_number(raw.get("c"), f"per_task[{index}].c", minimum=0))
        if c > n:
            raise ContractError(f"per_task[{index}].c exceeds n")
        rows.append({"task_key": key, "n": n, "c": c})
    return dict(summary), rows


def _verify_binding(
    *, probe_path: Path, probe: Mapping[str, Any], attestation: Mapping[str, Any]
) -> dict[str, Any]:
    if attestation.get("schema_version") != "stage5.relative_mouse_evidence.v1":
        raise ContractError("unsupported relative-mouse evidence attestation")
    expected_probe = str(attestation.get("probe_sha256", ""))
    observed_probe = _sha256_file(probe_path)
    if expected_probe != observed_probe:
        raise ContractError("probe digest differs from evidence attestation")

    data = attestation.get("data")
    if not isinstance(data, Mapping):
        raise ContractError("attestation.data must be an object")
    if data.get("class") != "synthetic_train_adjacent_validation":
        raise ContractError("only synthetic train-adjacent validation evidence is eligible")
    if data.get("contains_official_heldout") is not False:
        raise ContractError("attestation must explicitly exclude official heldout data")
    if data.get("contains_crowd_cast") is not False:
        raise ContractError("attestation must explicitly exclude Crowd-Cast data")

    policy = attestation.get("policy")
    if not isinstance(policy, Mapping):
        raise ContractError("attestation.policy must be an object")
    checkpoint_sha256 = str(policy.get("checkpoint_sha256", ""))
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ContractError("attestation policy checkpoint_sha256 is invalid")
    checkpoint_uri = str(policy.get("checkpoint_uri", ""))
    if not checkpoint_uri:
        raise ContractError("attestation policy checkpoint_uri is empty")

    probe_log_path = Path(str(attestation.get("probe_log_path", "")))
    if not probe_log_path.is_file():
        raise ContractError(f"bound probe log is missing: {probe_log_path}")
    observed_log = _sha256_file(probe_log_path)
    if observed_log != attestation.get("probe_log_sha256"):
        raise ContractError("probe log digest differs from evidence attestation")
    log_text = probe_log_path.read_text(errors="replace")
    if checkpoint_uri not in log_text:
        raise ContractError("bound probe log does not name the attested checkpoint")
    label = str(probe.get("summary", {}).get("label", ""))
    if not label or f"label={label}" not in log_text:
        raise ContractError("bound probe log does not name the probe label")

    accounting = attestation.get("resource_accounting")
    if not isinstance(accounting, Mapping):
        raise ContractError("attestation.resource_accounting must be an object")
    gpu_seconds = _require_number(accounting.get("gpu_seconds"), "gpu_seconds", minimum=1)
    return {
        "probe_sha256": observed_probe,
        "probe_log_sha256": observed_log,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "data_class": data["class"],
        "gpu_seconds": gpu_seconds,
        "scheduler_job_id": str(accounting.get("scheduler_job_id", "")),
    }


def evaluate_relative_mouse_launch(
    *, probe_path: str | Path, attestation_path: str | Path, config_path: str | Path
) -> dict[str, Any]:
    """Evaluate the preregistered short-task gate.

    A result with ``threshold_crossed=true`` makes the lane eligible for a human
    launch decision.  It deliberately never sets ``launch_authorized``.
    """

    probe_file = Path(probe_path)
    probe = read_json(probe_file)
    attestation = read_json(attestation_path)
    config = read_json(config_path)
    if config.get("schema_version") != "stage5.relative_mouse_launch_gate.v1":
        raise ContractError("unsupported relative-mouse launch-gate config")
    summary, rows = _evidence_rows(probe)
    binding = _verify_binding(probe_path=probe_file, probe=probe, attestation=attestation)

    evidence_cfg = config.get("evidence")
    posterior_cfg = config.get("posterior")
    thresholds = config.get("thresholds")
    if not all(isinstance(x, Mapping) for x in (evidence_cfg, posterior_cfg, thresholds)):
        raise ContractError("launch-gate config sections are missing")

    min_tasks = int(evidence_cfg["minimum_tasks"])
    min_k = int(evidence_cfg["minimum_samples_per_task"])
    max_error_rate = float(evidence_cfg["maximum_error_rate"])
    checks = {
        "probe_status_ok": summary.get("status") == "OK",
        "minimum_tasks": len(rows) >= min_tasks,
        "minimum_samples_per_task": min(row["n"] for row in rows) >= min_k,
        "summary_task_count_consistent": int(summary.get("n_tasks", -1)) == len(rows),
        "summary_rollout_count_consistent": int(summary.get("n_rollouts_ok", -1))
        == sum(row["n"] for row in rows),
        "summary_accept_count_consistent": int(summary.get("n_accepted_rollouts", -1))
        == sum(row["c"] for row in rows),
        "error_rate_healthy": _require_number(
            summary.get("error_rate"), "summary.error_rate", minimum=0
        )
        <= max_error_rate,
        "no_reported_invalid_reasons": not summary.get("invalid_reasons"),
    }
    if not all(checks.values()):
        return {
            "schema_version": "stage5.relative_mouse_launch_report.v1",
            "threshold_crossed": False,
            "launch_authorized": False,
            "evidence_checks": checks,
            "binding": binding,
            "rule": "all evidence checks and all posterior thresholds must pass",
        }

    alpha = float(posterior_cfg["beta_prior_alpha"])
    beta = float(posterior_cfg["beta_prior_beta"])
    draws = int(posterior_cfg["draws"])
    seed = int(posterior_cfg["seed"])
    scale_efficiency = float(posterior_cfg["scale_efficiency_haircut"])
    required_probability = float(posterior_cfg["required_probability"])
    interval_mass = float(posterior_cfg.get("interval_mass", 0.90))
    if alpha <= 0 or beta <= 0 or draws < 1000:
        raise ContractError("posterior prior must be positive and draws must be >=1000")
    if not 0 < scale_efficiency <= 1 or not 0 < required_probability < 1:
        raise ContractError("invalid posterior efficiency/probability setting")
    if not 0 < interval_mass < 1:
        raise ContractError("posterior interval_mass must be in (0,1)")

    total_attempts = sum(row["n"] for row in rows)
    attempts_per_gpu_hour = total_attempts / (binding["gpu_seconds"] / 3600.0)
    rng = random.Random(seed)
    metric_draws = {"pass@1": [], "pass@4": [], "pass@8": [], "accepted_per_gpu_hour": []}
    for _ in range(draws):
        probabilities = [
            rng.betavariate(row["c"] + alpha, row["n"] - row["c"] + beta)
            for row in rows
        ]
        weights = [rng.expovariate(1.0) for _ in rows]
        weight_sum = sum(weights)
        pass_one = sum(w * p for w, p in zip(weights, probabilities, strict=True)) / weight_sum
        for k in (1, 4, 8):
            estimate = sum(
                w * (1.0 - (1.0 - p) ** k)
                for w, p in zip(weights, probabilities, strict=True)
            ) / weight_sum
            metric_draws[f"pass@{k}"].append(estimate)
        metric_draws["accepted_per_gpu_hour"].append(
            pass_one * attempts_per_gpu_hour * scale_efficiency
        )

    tail = (1.0 - interval_mass) / 2.0
    results: dict[str, Any] = {}
    for name, values in metric_draws.items():
        threshold = float(thresholds[name])
        probability = sum(value >= threshold for value in values) / len(values)
        results[name] = {
            "posterior_mean": sum(values) / len(values),
            "credible_interval": [_quantile(values, tail), _quantile(values, 1.0 - tail)],
            "threshold": threshold,
            "probability_at_or_above_threshold": probability,
            "passed": probability >= required_probability,
        }

    threshold_crossed = all(item["passed"] for item in results.values())
    observed_accepts = sum(row["c"] for row in rows)
    return {
        "schema_version": "stage5.relative_mouse_launch_report.v1",
        "threshold_crossed": threshold_crossed,
        "launch_authorized": False,
        "evidence_checks": checks,
        "binding": binding,
        "observed": {
            "tasks": len(rows),
            "attempts": total_attempts,
            "accepted": observed_accepts,
            "observed_pass_at_1": observed_accepts / total_attempts,
            "observed_accepted_per_gpu_hour": observed_accepts
            / (binding["gpu_seconds"] / 3600.0),
            "attempts_per_gpu_hour": attempts_per_gpu_hour,
        },
        "posterior": {
            "model": "per-task beta-binomial plus Bayesian bootstrap over tasks",
            "beta_prior": {"alpha": alpha, "beta": beta},
            "draws": draws,
            "seed": seed,
            "scale_efficiency_haircut": scale_efficiency,
            "required_probability": required_probability,
            "metrics": results,
        },
        "rule": "all evidence checks and all posterior thresholds must pass",
    }
