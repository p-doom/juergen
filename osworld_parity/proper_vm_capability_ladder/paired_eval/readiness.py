from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import canonical_json


CERTIFICATION_SCHEMA = "proper_vm_executor_cert_v1"
VALIDATED_INTERFACES = [
    "native_absolute_control",
    "compact_raw_phaseb",
    "shared_atomic_gui_executor",
    "http_vm_transport",
]
READINESS_CHECKS = {
    "clean_build_at_least_109_tests",
    "narrow_click_preflight_10_trials",
    "forced_failure_artifact_probe_with_png",
    "full_click_100_trials_per_arm",
    "rung1a_16_cells",
    "rung1b_12_counterbalanced_cells",
    "sameapp_8_cells",
    "vm_isolation_and_provenance",
}


class ReadinessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsumedReadiness:
    path: str
    marker_sha256: str
    certification_schema: str
    capability_report_sha256: str
    executor_commit: str
    vm_snapshot_id: str
    consumed_at: str
    marker: dict[str, Any]
    _consumed: bool = True


def consume_executor_ready(
    path: Path,
    *,
    expected_sha256: str,
) -> ConsumedReadiness:
    """Consume the registered aggregate executor marker, or fail closed.

    The file must be passed explicitly by the caller.  The evaluator never
    searches a source checkout for a convenient marker and never creates one.
    """

    if path.name != "EXECUTOR_READY.json":
        raise ReadinessError("readiness path must name EXECUTOR_READY.json")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReadinessError(f"cannot consume executor readiness marker {path}: {exc}") from exc
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != expected_sha256:
        raise ReadinessError(
            f"executor readiness marker hash mismatch: {observed_sha} != {expected_sha256}"
        )
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReadinessError(f"executor readiness marker is not JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise ReadinessError("executor readiness marker must be an object")
    if marker.get("schema_version") != 1:
        raise ReadinessError("unsupported executor readiness schema")
    if marker.get("certification_schema") != CERTIFICATION_SCHEMA:
        raise ReadinessError("executor certification schema drift")
    if marker.get("status") != "ready":
        raise ReadinessError("executor status is not ready")
    if marker.get("development_only") is not True:
        raise ReadinessError("executor marker must be development-only")
    if marker.get("scored_execution_completed") is not False:
        raise ReadinessError("readiness marker must precede scored execution")

    interfaces = marker.get("validated_interfaces")
    if interfaces != VALIDATED_INTERFACES:
        raise ReadinessError("validated executor interfaces or order drifted")
    checks = marker.get("checks")
    if not isinstance(checks, dict) or set(checks) != READINESS_CHECKS:
        raise ReadinessError("executor readiness check set drifted")
    if any(value is not True for value in checks.values()):
        failed = sorted(key for key, value in checks.items() if value is not True)
        raise ReadinessError(f"executor readiness checks did not pass: {failed}")

    capability_sha = marker.get("capability_report_sha256")
    if not _sha256(capability_sha):
        raise ReadinessError("capability_report_sha256 binding is invalid")
    unsigned_report = dict(marker)
    unsigned_report.pop("capability_report_sha256", None)
    observed_capability_sha = hashlib.sha256(canonical_json(unsigned_report)).hexdigest()
    if capability_sha != observed_capability_sha:
        raise ReadinessError(
            "capability report canonical hash mismatch: "
            f"{observed_capability_sha} != {capability_sha}"
        )
    commit = marker.get("executor_commit")
    if not isinstance(commit, str) or len(commit) != 40 or commit.lower() != commit:
        raise ReadinessError("executor_commit binding is invalid")
    try:
        int(commit, 16)
    except ValueError as exc:
        raise ReadinessError("executor_commit binding is not hexadecimal") from exc
    snapshot = marker.get("vm_snapshot_id")
    if snapshot != "osworld_ready":
        raise ReadinessError("vm_snapshot_id must be osworld_ready")
    return ConsumedReadiness(
        path=str(path.resolve()),
        marker_sha256=observed_sha,
        certification_schema=CERTIFICATION_SCHEMA,
        capability_report_sha256=capability_sha,
        executor_commit=commit,
        vm_snapshot_id=snapshot,
        consumed_at=datetime.now(timezone.utc).isoformat(),
        marker=dict(marker),
    )


def _sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
