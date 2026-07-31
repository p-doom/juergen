from __future__ import annotations

import copy

import pytest

from ..contracts import ARMS
from ..manifest import ManifestError, load_evaluation_manifest, validate_evaluation_manifest
from ..planning import build_plan, task_shard
from .helpers import evaluation_manifest, ready_marker, sealed_file, task_manifest


def _manifest(tmp_path, *, attempts: int = 8):
    marker_path, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    task_raw, task_seal = task_manifest()
    task_path, _ = sealed_file(tmp_path / "tasks.json", task_raw)
    eval_raw = evaluation_manifest(task_seal, marker_sha, attempts=attempts)
    eval_path, _ = sealed_file(tmp_path / "evaluation.json", eval_raw)
    return load_evaluation_manifest(eval_path, task_path), marker_path


def test_manifest_and_deterministic_plan_keep_pairs_together(tmp_path) -> None:
    manifest, _ = _manifest(tmp_path)
    plan = build_plan(manifest)
    # Two semantic prefixes * (one-step + three horizons), plus natural loop.
    assert len(plan) == 9 * 8
    assert all(set(trial.arm_order) == set(ARMS) for trial in plan)
    assert len({trial.pair_id for trial in plan}) == len(plan)
    assert build_plan(manifest) == plan
    assert all(trial.budget == manifest.budget for trial in plan)
    assert all(trial.snapshot_id == "vm-dev" for trial in plan)
    assert all(trial.parameter_seed == 101 for trial in plan)

    assigned = task_shard(manifest, manifest.tasks[0], 7)
    shards = [build_plan(manifest, shard_index=index, shard_count=7) for index in range(7)]
    assert len(shards[assigned]) == len(plan)
    assert sum(len(shard) for shard in shards) == len(plan)


def test_manifest_rejects_arm_specific_pair_fields_and_heldout(tmp_path) -> None:
    _, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    value = evaluation_manifest(task_seal, marker_sha)
    value["arms"][0]["budget"] = {"max_actions": 99}
    with pytest.raises(ManifestError, match="must not vary by arm"):
        validate_evaluation_manifest(value, tasks, task_manifest_payload_sha256=task_seal)

    value = evaluation_manifest(task_seal, marker_sha)
    value["split"] = "heldout"
    with pytest.raises(ManifestError, match="development-only"):
        validate_evaluation_manifest(value, tasks, task_manifest_payload_sha256=task_seal)


def test_manifest_rejects_outcome_aware_exclusion(tmp_path) -> None:
    _, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    value = evaluation_manifest(task_seal, marker_sha)
    value["exclusions"] = [
        {
            "task_id": "writer-dev-1",
            "reason": "arm failed",
            "evidence_sha256": "7" * 64,
            "arm": ARMS[1],
        }
    ]
    with pytest.raises(ManifestError, match="without arm or outcome"):
        validate_evaluation_manifest(value, tasks, task_manifest_payload_sha256=task_seal)


def test_manifest_seals_are_enforced(tmp_path) -> None:
    manifest, _ = _manifest(tmp_path)
    assert manifest.task_manifest_payload_sha256
    path = tmp_path / "evaluation.json"
    raw = path.read_text(encoding="utf-8").replace("paired_real_vm", "tampered")
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ManifestError, match="payload hash mismatch"):
        load_evaluation_manifest(path, tmp_path / "tasks.json")
