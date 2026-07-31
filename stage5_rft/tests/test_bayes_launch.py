from __future__ import annotations

import hashlib
import json

import pytest

from stage5_rft.bayes_launch import evaluate_relative_mouse_launch
from stage5_rft.util import ContractError


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def _fixture(tmp_path, *, successes: int = 6):
    per_task = [
        {"task_key": f"val:{index}", "n": 8, "c": successes}
        for index in range(12)
    ]
    probe = {
        "summary": {
            "status": "OK",
            "label": "synthetic_probe",
            "n_tasks": len(per_task),
            "n_rollouts_ok": 8 * len(per_task),
            "n_rollouts_err": 0,
            "n_accepted_rollouts": successes * len(per_task),
            "error_rate": 0.0,
            "invalid_reasons": [],
        },
        "per_task": per_task,
    }
    probe_path = _write(tmp_path / "probe.json", probe)
    log_path = tmp_path / "probe.log"
    log_path.write_text("label=synthetic_probe ckpt=/fast/project/mock/checkpoint\n")
    attestation = {
        "schema_version": "stage5.relative_mouse_evidence.v1",
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "probe_log_path": str(log_path),
        "probe_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "policy": {
            "checkpoint_uri": "/fast/project/mock/checkpoint",
            "checkpoint_sha256": "a" * 64,
        },
        "data": {
            "class": "synthetic_train_adjacent_validation",
            "contains_official_heldout": False,
            "contains_crowd_cast": False,
        },
        "resource_accounting": {"gpu_seconds": 3600, "scheduler_job_id": "mock"},
    }
    attestation_path = _write(tmp_path / "attestation.json", attestation)
    config = {
        "schema_version": "stage5.relative_mouse_launch_gate.v1",
        "evidence": {
            "minimum_tasks": 8,
            "minimum_samples_per_task": 8,
            "maximum_error_rate": 0.02,
        },
        "posterior": {
            "beta_prior_alpha": 0.25,
            "beta_prior_beta": 2.25,
            "draws": 1000,
            "seed": 1,
            "interval_mass": 0.9,
            "scale_efficiency_haircut": 0.65,
            "required_probability": 0.95,
        },
        "thresholds": {
            "pass@1": 0.2,
            "pass@4": 0.4,
            "pass@8": 0.5,
            "accepted_per_gpu_hour": 10.0,
        },
    }
    config_path = _write(tmp_path / "config.json", config)
    return probe_path, attestation_path, config_path


def test_bayesian_launch_gate_passes_strong_synthetic_evidence(tmp_path):
    probe, attestation, config = _fixture(tmp_path)
    report = evaluate_relative_mouse_launch(
        probe_path=probe, attestation_path=attestation, config_path=config
    )
    assert report["threshold_crossed"] is True
    assert report["launch_authorized"] is False
    assert set(report["posterior"]["metrics"]) == {
        "pass@1",
        "pass@4",
        "pass@8",
        "accepted_per_gpu_hour",
    }


def test_bayesian_launch_gate_rejects_heldout_attestation(tmp_path):
    probe, attestation_path, config = _fixture(tmp_path)
    attestation = json.loads(attestation_path.read_text())
    attestation["data"]["contains_official_heldout"] = True
    _write(attestation_path, attestation)
    with pytest.raises(ContractError, match="exclude official heldout"):
        evaluate_relative_mouse_launch(
            probe_path=probe, attestation_path=attestation_path, config_path=config
        )


def test_bayesian_launch_gate_rejects_probe_digest_drift(tmp_path):
    probe, attestation, config = _fixture(tmp_path)
    probe.write_text(probe.read_text() + "\n")
    with pytest.raises(ContractError, match="probe digest"):
        evaluate_relative_mouse_launch(
            probe_path=probe, attestation_path=attestation, config_path=config
        )
