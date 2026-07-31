from __future__ import annotations

import json
import hashlib

import pytest

from ..readiness import ReadinessError, consume_executor_ready
from ..contracts import canonical_json
from .helpers import ready_marker


def test_readiness_is_explicit_pinned_and_consumed(tmp_path) -> None:
    path, seal = ready_marker(tmp_path / "EXECUTOR_READY.json")
    consumed = consume_executor_ready(path, expected_sha256=seal)
    assert consumed._consumed is True
    assert consumed.marker_sha256 == seal
    assert consumed.certification_schema == "proper_vm_executor_cert_v1"
    assert consumed.vm_snapshot_id == "osworld_ready"

    with pytest.raises(ReadinessError, match="hash mismatch"):
        consume_executor_ready(path, expected_sha256="0" * 64)
    with pytest.raises(ReadinessError, match="cannot consume"):
        consume_executor_ready(tmp_path / "missing" / "EXECUTOR_READY.json", expected_sha256=seal)
    with pytest.raises(ReadinessError, match="must name"):
        consume_executor_ready(tmp_path / "ready.json", expected_sha256=seal)


def test_readiness_rejects_failed_check_before_any_runtime(tmp_path) -> None:
    path, _ = ready_marker(tmp_path / "EXECUTOR_READY.json")
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["checks"]["vm_isolation_and_provenance"] = False
    raw = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    seal = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ReadinessError, match="did not pass"):
        consume_executor_ready(path, expected_sha256=seal)


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
            consume_executor_ready(
                path,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            )
        ready_marker(path)
