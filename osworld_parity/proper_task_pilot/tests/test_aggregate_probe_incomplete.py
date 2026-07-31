from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from osworld_parity.proper_task_pilot.aggregate_probe_incomplete import (
    EXPECTED_GPU_JOBS,
    posterior_predictive_gate_probability,
    validate_gpu_accounting,
)


def test_posterior_predictive_gate_probability_is_exact():
    jeffreys = posterior_predictive_gate_probability(
        Fraction(1, 2), Fraction(1, 2)
    )
    uniform = posterior_predictive_gate_probability(Fraction(1), Fraction(1))
    assert jeffreys["gate_open_probability"] == {
        "exact": "770973263/549755813888",
        "decimal": pytest.approx(0.0014023921958141727),
    }
    assert uniform["gate_open_probability"] == {
        "exact": "1410499/129140163",
        "decimal": pytest.approx(0.010922233387610019),
    }


def test_posterior_predictive_rejects_invalid_beta_prior():
    with pytest.raises(ValueError, match="positive"):
        posterior_predictive_gate_probability(Fraction(0), Fraction(1))


def _write_accounting(path: Path) -> None:
    jobs = [
        {
            "job_id": job_id,
            "role": role,
            "state": state,
            "exit_code": exit_code,
            "gpu_count": 1,
            "elapsed_raw": elapsed,
        }
        for job_id, (role, state, exit_code, elapsed) in EXPECTED_GPU_JOBS.items()
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "first_half_seconds": 6400,
        "jobs": jobs,
        "total_gpu_seconds": 11580,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gpu_accounting_requires_exact_failed_recovery_and_cap(tmp_path: Path):
    path = tmp_path / "gpu.json"
    _write_accounting(path)
    validated = validate_gpu_accounting(path)
    assert validated["total_gpu_seconds"] == 11580
    assert validated["margin_gpu_seconds"] == 2820

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"][-1]["state"] = "COMPLETED"
    unsealed = dict(payload)
    unsealed.pop("payload_sha256")
    canonical = json.dumps(unsealed, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/state/time"):
        validate_gpu_accounting(path)
