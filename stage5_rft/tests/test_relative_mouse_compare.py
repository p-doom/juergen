from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stage5_rft.relative_mouse_compare import compare_relative_mouse_eval
from stage5_rft.util import ContractError


BASELINE = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/rl_scratch/"
    "osworld_rl/rft_star/out/probe_r0_seed.json"
)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_bytes(BASELINE.read_bytes())
    candidate.write_bytes(BASELINE.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
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
                "probe_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "slurm_job_id": "999999",
                "model_artifact_id": "artifact_candidate",
                "checkpoint_digest": "a" * 64,
            }
        )
    )
    baseline_sacct = tmp_path / "baseline.psv"
    baseline_sacct.write_text(
        "135215|COMPLETED|1493|0:0|billing=32,cpu=32,gres/gpu=1,mem=100G,node=1\n"
    )
    candidate_sacct = tmp_path / "candidate.psv"
    candidate_sacct.write_text(
        "999999|COMPLETED|1200|0:0|billing=16,cpu=16,gres/gpu=1,mem=100G,node=1\n"
    )
    return {
        "baseline_probe_path": baseline,
        "candidate_probe_path": candidate,
        "baseline_sacct_path": baseline_sacct,
        "candidate_sacct_path": candidate_sacct,
        "candidate_manifest_path": manifest,
    }


def test_compare_frozen_exact_task_pool_and_accounting(tmp_path):
    report = compare_relative_mouse_eval(**_inputs(tmp_path))
    assert report["status"] == "complete"
    assert report["candidate_minus_baseline"]["pass@8"] == pytest.approx(0.0)
    assert report["baseline"]["metrics"]["accepted_rollouts"] == 318
    assert report["candidate"]["metrics"]["accepted_per_allocated_gpu_hour"] == 954.0
    assert report["promotion_authorized"] is False


def test_compare_rejects_task_pool_drift(tmp_path):
    inputs = _inputs(tmp_path)
    candidate = inputs["candidate_probe_path"]
    payload = json.loads(candidate.read_text())
    payload["per_task"][0]["task_key"] = "val:128"
    candidate.write_text(json.dumps(payload))
    manifest = inputs["candidate_manifest_path"]
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["probe_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(manifest_payload))
    with pytest.raises(ContractError, match="exact frozen"):
        compare_relative_mouse_eval(**inputs)
