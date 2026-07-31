from __future__ import annotations

import json
import hashlib

import pytest

from ..readiness import ConsumedReadiness, ReadinessError, consume_executor_ready
from ..contracts import canonical_json
from .helpers import labctl_context, ready_marker


def _consume(path, seal, tmp_path):
    context = labctl_context(tmp_path / "context.json", path.parent)
    return consume_executor_ready(
        path,
        expected_sha256=seal,
        expected_artifact_id="artifact-executor-ready-test",
        labctl_context_path=context,
    )


def test_readiness_is_explicit_pinned_and_consumed(tmp_path) -> None:
    path, seal = ready_marker(tmp_path / "EXECUTOR_READY.json")
    consumed = _consume(path, seal, tmp_path)
    assert consumed.consumed is True
    assert consumed.marker_sha256 == seal
    assert consumed.certification_schema == "proper_vm_executor_cert_v1"
    assert consumed.vm_snapshot_id == "osworld_ready"

    with pytest.raises(ReadinessError, match="hash mismatch"):
        _consume(path, "0" * 64, tmp_path)
    with pytest.raises(ReadinessError, match="cannot consume"):
        _consume(tmp_path / "missing" / "EXECUTOR_READY.json", seal, tmp_path)
    with pytest.raises(ReadinessError, match="must name"):
        _consume(tmp_path / "ready.json", seal, tmp_path)
    with pytest.raises(TypeError):
        ConsumedReadiness()  # type: ignore[call-arg]


def test_readiness_rejects_failed_check_before_any_runtime(tmp_path) -> None:
    path, _ = ready_marker(tmp_path / "EXECUTOR_READY.json")
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["checks"]["vm_isolation_and_provenance"] = False
    raw = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    seal = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ReadinessError, match="did not pass"):
        _consume(path, seal, tmp_path)


def test_readiness_interface_order_and_check_set_are_frozen(tmp_path) -> None:
    path, _ = ready_marker(tmp_path / "EXECUTOR_READY.json")
    for mutation, message in (
        (
            lambda marker: marker["validated_interfaces"].reverse(),
            "interfaces or order drifted",
        ),
        (
            lambda marker: marker["checks"].update({"unregistered_probe": True}),
            "check set drifted",
        ),
    ):
        marker = json.loads(path.read_text(encoding="utf-8"))
        mutation(marker)
        marker.pop("capability_report_sha256")
        marker["capability_report_sha256"] = hashlib.sha256(
            canonical_json(marker)
        ).hexdigest()
        raw = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(raw)
        with pytest.raises(ReadinessError, match=message):
            _consume(path, hashlib.sha256(raw).hexdigest(), tmp_path)
        ready_marker(path)


def test_readiness_requires_exact_labctl_artifact_binding(tmp_path) -> None:
    path, seal = ready_marker(tmp_path / "EXECUTOR_READY.json")
    context = labctl_context(tmp_path / "context.json", tmp_path)
    value = json.loads(context.read_text(encoding="utf-8"))
    value["inputs"][0]["artifact_id"] = "wrong-artifact"
    context.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReadinessError, match="artifact mismatch"):
        consume_executor_ready(
            path,
            expected_sha256=seal,
            expected_artifact_id="artifact-executor-ready-test",
            labctl_context_path=context,
        )
    value["inputs"][0]["artifact_id"] = "artifact-executor-ready-test"
    value["inputs"][0]["resolved_path"] = str(tmp_path / "wrong")
    context.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReadinessError, match="not the registered"):
        consume_executor_ready(
            path,
            expected_sha256=seal,
            expected_artifact_id="artifact-executor-ready-test",
            labctl_context_path=context,
        )
