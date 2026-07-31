from __future__ import annotations

import json

import pytest

from ..readiness import ReadinessError, consume_executor_ready
from .helpers import ready_marker


def test_readiness_is_explicit_pinned_and_consumed(tmp_path) -> None:
    path, seal = ready_marker(tmp_path / "EXECUTOR_READY.json")
    consumed = consume_executor_ready(path, expected_sha256=seal)
    assert consumed._consumed is True
    assert consumed.marker_sha256 == seal
    assert consumed.vm_snapshot_id == "vm-dev"

    with pytest.raises(ReadinessError, match="hash mismatch"):
        consume_executor_ready(path, expected_sha256="0" * 64)
    with pytest.raises(ReadinessError, match="cannot consume"):
        consume_executor_ready(tmp_path / "missing" / "EXECUTOR_READY.json", expected_sha256=seal)
    with pytest.raises(ReadinessError, match="must name"):
        consume_executor_ready(tmp_path / "ready.json", expected_sha256=seal)


def test_readiness_rejects_failed_check_before_any_runtime(tmp_path) -> None:
    path, _ = ready_marker(tmp_path / "EXECUTOR_READY.json")
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["checks"]["cursor_readback"] = False
    raw = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    import hashlib

    seal = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ReadinessError, match="did not pass"):
        consume_executor_ready(path, expected_sha256=seal)
