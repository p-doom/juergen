from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest

from osworld_parity.proper_task_pilot.aggregate_probe_final import (
    EXPECTED_GPU_JOBS,
    exact_binomial_interval,
    validate_gpu_accounting,
)


def test_exact_binomial_endpoints_and_symmetry():
    zero = exact_binomial_interval(0, 8)
    eight = exact_binomial_interval(8, 8)
    assert zero[0] == 0.0
    assert zero[1] == pytest.approx(1.0 - 0.025 ** (1.0 / 8.0))
    assert eight[0] == pytest.approx(0.025 ** (1.0 / 8.0))
    assert eight[1] == 1.0
    for successes in range(9):
        lower, upper = exact_binomial_interval(successes, 8)
        mirror_lower, mirror_upper = exact_binomial_interval(8 - successes, 8)
        assert 0.0 <= lower <= upper <= 1.0
        assert lower == pytest.approx(1.0 - mirror_upper)
        assert upper == pytest.approx(1.0 - mirror_lower)


def test_exact_binomial_rejects_bad_inputs():
    for args in ((-1, 8), (9, 8), (0, 0), (1, 8, 0.0), (1, 8, 1.0)):
        with pytest.raises(ValueError):
            exact_binomial_interval(*args)


def _accounting(path: Path, *, recovery_seconds: int) -> None:
    elapsed = {
        "135517": 2253,
        "135518": 1292,
        "135519": 1674,
        "135520": 1181,
        "135555": 897,
        "135556": 1074,
        "135558": 1800,
        "135559": 1800,
        "135575": recovery_seconds,
    }
    jobs = [
        {
            "job_id": job_id,
            "role": role,
            "state": state,
            "exit_code": exit_code,
            "gpu_count": 1,
            "elapsed_raw": elapsed[job_id],
        }
        for job_id, (role, state, exit_code) in EXPECTED_GPU_JOBS.items()
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "first_half_seconds": 6400,
        "jobs": jobs,
        "total_gpu_seconds": sum(elapsed.values()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(payload))


def test_gpu_accounting_enforces_cap(tmp_path: Path):
    path = tmp_path / "gpu.json"
    _accounting(path, recovery_seconds=2000)
    result = validate_gpu_accounting(path)
    assert result["total_gpu_seconds"] == 13971
    assert result["margin_gpu_seconds"] == 429
    _accounting(path, recovery_seconds=2430)
    with pytest.raises(ValueError, match="GPU cap"):
        validate_gpu_accounting(path)
