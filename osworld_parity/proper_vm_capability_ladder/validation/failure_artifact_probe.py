from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..rung1.transport import ATOMIC_RESULT_PREFIX, Operation
from ..rung1.vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    KvmFixtureSession,
    sha256_file,
)
from .artifact_index import PINNED_SUBSTRATE_SHA256


SCHEMA_VERSION = "proper_vm_executor_failure_probe_v1"
INJECTED_MESSAGE = "executor-certification-injected-failure"
SEMANTIC_OPERATIONS = (
    Operation("mouse_down", ("left",)),
    Operation("raise_for_test", (INJECTED_MESSAGE,)),
)


class FailureProbeError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _operation_payload(operations: tuple[Operation, ...]) -> list[dict[str, Any]]:
    return [{"kind": item.kind, "args": list(item.args)} for item in operations]


def validate_injected_failure(
    *,
    raw_guest: dict[str, Any],
    atomic_state: dict[str, Any],
    forbidden_success_markers: list[Path],
    screenshot: dict[str, Any],
) -> dict[str, Any]:
    expected_semantic = _operation_payload(SEMANTIC_OPERATIONS)
    raw_returncode = raw_guest.get("returncode")
    if (
        not isinstance(raw_returncode, int)
        or isinstance(raw_returncode, bool)
        or raw_returncode == 0
    ):
        raise FailureProbeError("forced guest failure did not propagate a nonzero exit")
    if atomic_state.get("guest_returncode") != raw_returncode:
        raise FailureProbeError("atomic failure return code differs from raw guest evidence")
    raw_marker = atomic_state.get("raw_result_marker")
    if not isinstance(raw_marker, str) or not raw_marker.startswith(
        ATOMIC_RESULT_PREFIX
    ):
        raise FailureProbeError("atomic failure omitted its raw result marker")
    raw_output = raw_guest.get("output")
    if not isinstance(raw_output, str) or [
        line
        for line in raw_output.splitlines()
        if line.startswith(ATOMIC_RESULT_PREFIX)
    ] != [raw_marker]:
        raise FailureProbeError("raw guest output does not contain exactly the bound marker")
    if atomic_state.get("ok") is not False:
        raise FailureProbeError("forced failure was reported as successful")
    if atomic_state.get("failure_kind") != "injected":
        raise FailureProbeError("forced failure did not retain injected classification")
    error = atomic_state.get("error")
    if not isinstance(error, str) or INJECTED_MESSAGE not in error:
        raise FailureProbeError("forced failure has no exact durable error message")
    semantic = atomic_state.get("semantic_operations")
    lowered = atomic_state.get("lowered_operations")
    executed = atomic_state.get("operations")
    if semantic != expected_semantic:
        raise FailureProbeError("requested semantic operation stream drifted")
    if not isinstance(lowered, list) or not isinstance(executed, list):
        raise FailureProbeError("failure artifact omitted lowered/executed streams")
    if not executed or len(executed) >= len(lowered) or executed != lowered[: len(executed)]:
        raise FailureProbeError("executed trace is not a strict lowered-operation prefix")
    if any(item.get("kind") == "raise_for_test" for item in executed):
        raise FailureProbeError("executed trace falsely claims the injected op completed")
    cursor_before = atomic_state.get("cursor_before")
    cursor_after = atomic_state.get("cursor_after")
    if (
        not isinstance(cursor_before, list)
        or len(cursor_before) != 2
        or not all(isinstance(value, int) for value in cursor_before)
        or not isinstance(cursor_after, list)
        or len(cursor_after) != 2
        or not all(isinstance(value, int) for value in cursor_after)
        or atomic_state.get("cursor") != cursor_after
    ):
        raise FailureProbeError("failure artifact omitted consistent cursor readback")
    if atomic_state.get("cleanup_attempted") is not True:
        raise FailureProbeError("forced failure did not attempt input cleanup")
    final_mask = atomic_state.get("pointer_button_mask")
    expected_mask = atomic_state.get("expected_pointer_button_mask")
    observed_mask = atomic_state.get("observed_pointer_button_mask")
    if final_mask != 0 or expected_mask != (1 << 8) or not isinstance(observed_mask, int):
        raise FailureProbeError("forced failure button-mask evidence is inconsistent")
    if atomic_state.get("guest_process_count") != 1:
        raise FailureProbeError("forced failure did not use exactly one guest process")
    if any(path.exists() for path in forbidden_success_markers):
        raise FailureProbeError("executor success marker exists after forced failure")
    screenshot_path = Path(str(screenshot.get("path", "")))
    if not screenshot_path.is_file() or screenshot_path.is_symlink():
        raise FailureProbeError("forced failure screenshot artifact is missing")
    screenshot_bytes = screenshot_path.read_bytes()
    if (
        not screenshot_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        or len(screenshot_bytes) <= 8
        or screenshot.get("bytes") != len(screenshot_bytes)
        or screenshot.get("sha256") != hashlib.sha256(screenshot_bytes).hexdigest()
    ):
        raise FailureProbeError("forced failure screenshot artifact is invalid")
    return {
        "guest_nonzero_propagated": True,
        "raw_marker_bound": True,
        "classification_preserved": True,
        "requested_stream_preserved": True,
        "executed_strict_prefix": True,
        "cursor_readback_preserved": True,
        "cleanup_verified": True,
        "button_mask_verified": True,
        "success_marker_absent": True,
        "failure_screenshot_verified": True,
    }


def run_probe(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider_path: Path,
    expected_provider_sha256: str,
) -> dict[str, Any]:
    provider_sha = sha256_file(provider_path)
    if provider_sha != expected_provider_sha256:
        raise FailureProbeError("pinned KVM provider hash mismatch")
    forbidden = [
        output / "executor_success.json",
        output / "selfcheck.json",
        output / "transport_diagnostic.json",
        output / "replay.json",
    ]
    raw_calls: list[dict[str, Any]] = []
    screenshot: dict[str, Any]
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider_path,
        vm_log_dir=output / "vm_logs",
    ) as session:
        transport = session.reset_to_ready()
        original: Callable[..., dict[str, Any]] = transport.execute_argv

        def recording_execute(argv: list[str], *, check: bool = True) -> dict[str, Any]:
            raw = original(argv, check=check)
            raw_calls.append(raw)
            return raw

        transport.execute_argv = recording_execute  # type: ignore[method-assign]
        result = transport.execute_atomic(SEMANTIC_OPERATIONS)
        atomic_state = result.as_dict()
        if transport.audit.held_buttons or transport.audit.held_keys:
            raise FailureProbeError("host input audit retained held state after cleanup")
        with urllib.request.urlopen(transport.base_url + "/screenshot", timeout=15) as response:
            screenshot_bytes = response.read()
        if not screenshot_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(screenshot_bytes) <= 8:
            raise FailureProbeError("VM screenshot endpoint did not return a nonempty PNG")
        screenshot_path = output / "injected_failure.png"
        screenshot_path.write_bytes(screenshot_bytes)
        screenshot = {
            "path": str(screenshot_path.resolve()),
            "bytes": len(screenshot_bytes),
            "sha256": hashlib.sha256(screenshot_bytes).hexdigest(),
        }
    if len(raw_calls) != 1:
        raise FailureProbeError(f"forced failure executed {len(raw_calls)} guest processes")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": "expected_failure_observed",
        "semantic_operations": _operation_payload(SEMANTIC_OPERATIONS),
        "raw_guest": raw_calls[0],
        "atomic_state": atomic_state,
        "screenshot": screenshot,
        "provider_sha256": provider_sha,
        "forbidden_success_markers": [str(path) for path in forbidden],
    }
    # The raw evidence is durable even if one of the assertions below rejects
    # it.  The success marker is written only after every negative gate passes.
    _atomic_json(output / "injected_failure.json", evidence)
    checks = validate_injected_failure(
        raw_guest=raw_calls[0],
        atomic_state=atomic_state,
        forbidden_success_markers=forbidden,
        screenshot=screenshot,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "expected_outcome": "injected_executor_failure",
        "failure_artifact": "injected_failure.json",
        "failure_artifact_sha256": sha256_file(output / "injected_failure.json"),
        "failure_screenshot": screenshot,
        "checks": checks,
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--expected-provider-sha256",
        "--expected_provider_sha256",
        default=PINNED_SUBSTRATE_SHA256["provider"],
        dest="expected_provider_sha256",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "failure_probe.json"
    marker.unlink(missing_ok=True)
    try:
        value = run_probe(
            output=args.output,
            qcow=args.qcow,
            qemu=args.qemu,
            provider_path=args.provider,
            expected_provider_sha256=args.expected_provider_sha256,
        )
        _atomic_json(marker, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output := args.output / "probe_failure.json", failure)
        print(json.dumps({**failure, "artifact": str(output)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
