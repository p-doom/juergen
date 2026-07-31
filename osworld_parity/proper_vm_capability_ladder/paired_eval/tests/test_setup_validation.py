from __future__ import annotations

import json

import pytest

from ..manifest import load_evaluation_manifest
from ..setup_validation import (
    ConsumedTaskSetupValidation,
    SetupValidationError,
    consume_task_setup_validation,
)
from .helpers import (
    evaluation_manifest,
    labctl_context,
    ready_marker,
    sealed_file,
    task_manifest,
    task_setup_validation,
)


def _setup(tmp_path):
    _, readiness_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    setup_path, setup_sha = task_setup_validation(
        tmp_path / "task_setup_validation.json",
        tasks,
        task_seal,
    )
    task_path, _ = sealed_file(tmp_path / "tasks.json", tasks)
    evaluation_path, _ = sealed_file(
        tmp_path / "evaluation.json",
        evaluation_manifest(
            task_seal,
            readiness_sha,
            setup_validation_sha=setup_sha,
        ),
    )
    manifest = load_evaluation_manifest(evaluation_path, task_path)
    context = labctl_context(tmp_path / "context.json", tmp_path)
    return manifest, setup_path, context


def test_setup_validation_is_explicit_pinned_and_consumed(tmp_path) -> None:
    manifest, path, context = _setup(tmp_path)
    consumed = consume_task_setup_validation(
        path,
        manifest=manifest,
        labctl_context_path=context,
    )
    assert consumed.consumed is True
    assert consumed.artifact_sha256 == manifest.expected_task_setup_validation_sha256
    assert consumed.task_manifest_payload_sha256 == manifest.task_manifest_payload_sha256
    assert consumed.vm_snapshot_id == "osworld_ready"
    with pytest.raises(TypeError):
        ConsumedTaskSetupValidation()  # type: ignore[call-arg]


def test_setup_validation_rejects_raw_or_fixture_tampering(tmp_path) -> None:
    manifest, path, context = _setup(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["fixtures"][0]["fixture_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SetupValidationError, match="hash mismatch"):
        consume_task_setup_validation(
            path,
            manifest=manifest,
            labctl_context_path=context,
        )


def test_setup_validation_requires_exact_second_labctl_input(tmp_path) -> None:
    manifest, path, context = _setup(tmp_path)
    value = json.loads(context.read_text(encoding="utf-8"))
    value["inputs"][1]["role"] = "setup_alias"
    context.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SetupValidationError, match="exactly executor_readiness"):
        consume_task_setup_validation(
            path,
            manifest=manifest,
            labctl_context_path=context,
        )
