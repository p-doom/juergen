"""Detached-signature verification and fail-closed release gates.

The verifier intentionally knows nothing about the official task filesystem.
Gate payloads are canonical JSON and are verified with OpenSSH signatures.
"""

from __future__ import annotations

import json
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from .contract import (
    ARMS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING,
    COMPACT_RAW_SUCCESS_FLOOR,
    CONTRACT_ID,
    EXPECTED_EPISODE_COUNT,
    GATE_SCOPE,
    NONINFERIORITY_MARGIN,
    PAIRED_SEEDS,
    PILOT_TASK_COUNT,
    REQUIRED_PREREQUISITE_RUNGS,
    SELECTION_POLICY,
    SIGNATURE_NAMESPACE,
    SOURCE_PROTOCOL,
)


class GateError(RuntimeError):
    """A signed prerequisite or pilot-release gate is absent or invalid."""


@dataclass(frozen=True)
class SignedGatePaths:
    payload: Path
    signature: Path


@dataclass(frozen=True)
class GateBundle:
    prerequisites: SignedGatePaths
    pilot_release: SignedGatePaths
    allowed_signers: Path
    signer_identity: str


@dataclass(frozen=True)
class LaunchAuthorization:
    """Sanitized authorization passed to a post-gate source adapter."""

    pilot_id: str
    prerequisites_gate_id: str
    pilot_release_gate_id: str
    contract_id: str = CONTRACT_ID
    source_protocol: str = SOURCE_PROTOCOL
    selection_policy: str = SELECTION_POLICY
    task_count: int = PILOT_TASK_COUNT
    paired_seeds: tuple[int, ...] = PAIRED_SEEDS
    arms: tuple[str, ...] = ARMS
    max_episodes: int = EXPECTED_EPISODE_COUNT

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "authorized",
            "pilot_id": self.pilot_id,
            "prerequisites_gate_id": self.prerequisites_gate_id,
            "pilot_release_gate_id": self.pilot_release_gate_id,
            "contract_id": self.contract_id,
            "source_protocol": self.source_protocol,
            "selection_policy": self.selection_policy,
            "task_count": self.task_count,
            "paired_seeds": list(self.paired_seeds),
            "arms": list(self.arms),
            "max_episodes": self.max_episodes,
        }


_FORBIDDEN_DETAIL_KEYS = {
    "task",
    "tasks",
    "task_id",
    "task_ids",
    "task_file",
    "task_files",
    "task_path",
    "task_paths",
    "task_hash",
    "task_hashes",
    "split_file",
    "split_hash",
    "split_sha256",
    "manifest_hash",
    "manifest_sha256",
    "source_path",
    "heldout_path",
    "root_path",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PILOT_ID_RE = re.compile(r"^pilot-[a-z0-9][a-z0-9-]{2,63}$")
_GATE_ID_RE = re.compile(r"^gate-[a-z0-9][a-z0-9-]{2,95}$")
T = TypeVar("T")


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _load_regular_file(path: Path, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise GateError(f"{label} must be an existing regular file")
        return path.read_bytes()
    except OSError as exc:
        raise GateError(f"cannot read {label}") from exc


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GateError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise GateError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise GateError(f"{label} must be UTC")
    return parsed


def _reject_official_detail(value: Any, *, key: str | None = None) -> None:
    if key is not None and key.lower() in _FORBIDDEN_DETAIL_KEYS:
        raise GateError(f"gate payload contains forbidden official-detail key {key!r}")
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise GateError("gate payload keys must be strings")
            _reject_official_detail(child_value, key=child_key)
    elif isinstance(value, list):
        for child in value:
            _reject_official_detail(child)
    elif isinstance(value, str):
        if value.startswith("/") or value.startswith("file:"):
            raise GateError("gate payload must not contain filesystem locations")
        if _SHA256_RE.fullmatch(value):
            raise GateError("gate payload must not disclose SHA-256-like values")


def _verify_signature(
    message: bytes,
    *,
    signature_path: Path,
    allowed_signers_path: Path,
    signer_identity: str,
) -> None:
    _load_regular_file(signature_path, "gate signature")
    _load_regular_file(allowed_signers_path, "allowed-signers file")
    try:
        if allowed_signers_path.stat().st_mode & stat.S_IWOTH:
            raise GateError("allowed-signers trust anchor must not be world-writable")
    except OSError as exc:
        raise GateError("cannot stat allowed-signers trust anchor") from exc
    if not signer_identity or any(char.isspace() for char in signer_identity):
        raise GateError("invalid signer identity")
    try:
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_signers_path),
                "-I",
                signer_identity,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=message,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise GateError("OpenSSH signature verifier is unavailable") from exc
    if result.returncode != 0:
        raise GateError("gate signature verification failed")


def verify_signed_gate(
    paths: SignedGatePaths,
    *,
    allowed_signers_path: Path,
    signer_identity: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    raw = _load_regular_file(paths.payload, "gate payload")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("gate payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise GateError("gate payload must be an object")
    canonical = canonical_json(payload)
    if raw != canonical:
        raise GateError("gate payload must use canonical JSON encoding")
    _reject_official_detail(payload)
    _verify_signature(
        canonical,
        signature_path=paths.signature,
        allowed_signers_path=allowed_signers_path,
        signer_identity=signer_identity,
    )
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise GateError("verification time must be timezone-aware")
    issued = _parse_time(payload.get("issued_at"), "issued_at")
    expires = _parse_time(payload.get("expires_at"), "expires_at")
    if issued > current or expires <= current or expires <= issued:
        raise GateError("gate is not currently valid")
    return payload


def _require_exact(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: (payload.get(key), expected_value)
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise GateError(f"{label} contract mismatch: {sorted(mismatches)}")


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise GateError(f"{label} keys do not match the frozen schema")


def verify_gate_bundle(
    bundle: GateBundle, *, now: datetime | None = None
) -> LaunchAuthorization:
    """Verify both signatures and the complete preregistered contract."""

    prerequisites = verify_signed_gate(
        bundle.prerequisites,
        allowed_signers_path=bundle.allowed_signers,
        signer_identity=bundle.signer_identity,
        now=now,
    )
    release = verify_signed_gate(
        bundle.pilot_release,
        allowed_signers_path=bundle.allowed_signers,
        signer_identity=bundle.signer_identity,
        now=now,
    )
    _require_keys(
        prerequisites,
        {
            "schema_version",
            "kind",
            "gate_id",
            "scope",
            "decision",
            "contract_id",
            "rungs",
            "issued_at",
            "expires_at",
        },
        "prerequisites gate",
    )
    _require_keys(
        release,
        {
            "schema_version",
            "kind",
            "gate_id",
            "parent_gate_id",
            "pilot_id",
            "scope",
            "decision",
            "contract_id",
            "source_protocol",
            "selection_policy",
            "task_count",
            "paired_seeds",
            "arms",
            "max_episodes",
            "analysis",
            "issued_at",
            "expires_at",
        },
        "pilot-release gate",
    )
    _require_exact(
        prerequisites,
        {
            "schema_version": 1,
            "kind": "proper-vm-prerequisites",
            "scope": GATE_SCOPE,
            "decision": "pass",
            "contract_id": CONTRACT_ID,
            "rungs": {rung: "pass" for rung in REQUIRED_PREREQUISITE_RUNGS},
        },
        "prerequisites gate",
    )
    _require_exact(
        release,
        {
            "schema_version": 1,
            "kind": "official-pilot-release",
            "scope": GATE_SCOPE,
            "decision": "authorize",
            "contract_id": CONTRACT_ID,
            "parent_gate_id": prerequisites.get("gate_id"),
            "source_protocol": SOURCE_PROTOCOL,
            "selection_policy": SELECTION_POLICY,
            "task_count": PILOT_TASK_COUNT,
            "paired_seeds": list(PAIRED_SEEDS),
            "arms": list(ARMS),
            "max_episodes": EXPECTED_EPISODE_COUNT,
            "analysis": {
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "ci_level": CI_LEVEL,
                "noninferiority_margin": NONINFERIORITY_MARGIN,
                "compact_raw_success_floor": COMPACT_RAW_SUCCESS_FLOOR,
                "compact_raw_parse_executor_failure_ceiling": (
                    COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING
                ),
            },
        },
        "pilot-release gate",
    )
    prerequisites_gate_id = prerequisites.get("gate_id")
    release_gate_id = release.get("gate_id")
    pilot_id = release.get("pilot_id")
    if not isinstance(prerequisites_gate_id, str) or not _GATE_ID_RE.fullmatch(
        prerequisites_gate_id
    ):
        raise GateError("invalid prerequisites gate_id")
    if not isinstance(release_gate_id, str) or not _GATE_ID_RE.fullmatch(release_gate_id):
        raise GateError("invalid pilot-release gate_id")
    if not isinstance(pilot_id, str) or not _PILOT_ID_RE.fullmatch(pilot_id):
        raise GateError("invalid pilot_id")
    if _parse_time(release["issued_at"], "pilot issued_at") < _parse_time(
        prerequisites["issued_at"], "prerequisites issued_at"
    ):
        raise GateError("pilot-release gate predates prerequisites gate")
    return LaunchAuthorization(
        pilot_id=pilot_id,
        prerequisites_gate_id=prerequisites_gate_id,
        pilot_release_gate_id=release_gate_id,
    )


def with_authorized_source(
    bundle: GateBundle,
    source_factory: Callable[[], T],
    operation: Callable[[LaunchAuthorization, T], Any],
    *,
    now: datetime | None = None,
) -> Any:
    """Create/use a source only after both gates validate.

    The ordering is the security boundary: build/dev callers can provide a
    poison source factory and prove it is never reached on any gate failure.
    """

    authorization = verify_gate_bundle(bundle, now=now)
    source = source_factory()
    return operation(authorization, source)
