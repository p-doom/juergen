from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
_CONSUMPTION_TOKEN = object()


class ReadinessError(RuntimeError):
    pass


@dataclass(frozen=True, init=False)
class ConsumedReadiness:
    path: str
    artifact_id: str
    labctl_context_path: str
    marker_sha256: str
    certification_schema: str
    capability_report_sha256: str
    executor_commit: str
    vm_snapshot_id: str
    consumed_at: str
    marker: dict[str, Any]
    _token: object = field(repr=False, compare=False)

    def __init__(self, *_: Any, **__: Any) -> None:
        raise TypeError("ConsumedReadiness can only be created by consuming a marker")

    @property
    def consumed(self) -> bool:
        return self._token is _CONSUMPTION_TOKEN


def consume_executor_ready(
    path: Path,
    *,
    expected_sha256: str,
    expected_artifact_id: str,
    labctl_context_path: Path,
) -> ConsumedReadiness:
    """Consume the registered aggregate executor marker, or fail closed.

    The file must be passed explicitly by the caller.  The evaluator never
    searches a source checkout for a convenient marker and never creates one.
    """

    if path.name != "EXECUTOR_READY.json":
        raise ReadinessError("readiness path must name EXECUTOR_READY.json")
    artifact_id = _validate_labctl_binding(
        labctl_context_path,
        marker_path=path,
        expected_artifact_id=expected_artifact_id,
    )
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
    values = {
        "path": str(path.resolve()),
        "artifact_id": artifact_id,
        "labctl_context_path": str(labctl_context_path.resolve()),
        "marker_sha256": observed_sha,
        "certification_schema": CERTIFICATION_SCHEMA,
        "capability_report_sha256": capability_sha,
        "executor_commit": commit,
        "vm_snapshot_id": snapshot,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
        "marker": dict(marker),
        "_token": _CONSUMPTION_TOKEN,
    }
    consumed = object.__new__(ConsumedReadiness)
    for key, value in values.items():
        object.__setattr__(consumed, key, value)
    return consumed


def _sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_labctl_binding(
    context_path: Path,
    *,
    marker_path: Path,
    expected_artifact_id: str,
) -> str:
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot read LABCTL_CONTEXT {context_path}: {exc}") from exc
    if not isinstance(context, dict) or not isinstance(context.get("inputs"), list):
        raise ReadinessError("LABCTL_CONTEXT.inputs must be an array")
    matches = [
        value
        for value in context["inputs"]
        if isinstance(value, dict) and value.get("role") == "executor_readiness"
    ]
    if len(matches) != 1:
        raise ReadinessError(
            "LABCTL_CONTEXT must contain exactly one executor_readiness input"
        )
    binding = matches[0]
    if set(binding) != {"role", "artifact_id", "resolved_path"}:
        raise ReadinessError("executor_readiness input field set drifted")
    artifact_id = binding.get("artifact_id")
    if artifact_id != expected_artifact_id:
        raise ReadinessError(
            f"executor readiness artifact mismatch: {artifact_id} != {expected_artifact_id}"
        )
    resolved_path = binding.get("resolved_path")
    if not isinstance(resolved_path, str) or not resolved_path:
        raise ReadinessError("executor_readiness resolved_path is missing")
    artifact_root = Path(resolved_path).resolve()
    expected_marker = artifact_root / "EXECUTOR_READY.json"
    if marker_path.resolve() != expected_marker:
        raise ReadinessError(
            "explicit EXECUTOR_READY.json is not the registered executor_readiness artifact"
        )
    return artifact_id
