from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from ..rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ..rung1.vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    KvmFixtureSession,
    sha256_file,
)
from .curriculum.manifests import TaskManifest, load_manifest
from .curriculum.oracle import as_sameapp_fixture, as_vscode_fixture, verify_fixture_contract
from .curriculum.program import (
    CompiledProgram,
    CompiledSegment,
    ExecutedSegmentReceipt,
    aggregate_executed_segments,
    compile_semantic_step,
    record_executed_segment,
)
from .curriculum.runtime import (
    RuntimeEvidenceLedger,
    RuntimeProbe,
    ValidatedRuntimeBinding,
    bind_repeated_runtime_probes,
    probe_runtime,
    refresh_binding_after_step,
)
from .curriculum.schema import SemanticTask
from .curriculum.setup_validation import load_task_setup_validation
from .fixtures import canonical_json
from .vm import AppReadinessError


ARMS = ("native_absolute_control", "compact_raw_phaseb")
ARM_SCHEMAS = {
    "native_absolute_control": "native_absolute_sequence_v1",
    "compact_raw_phaseb": "compact_raw_phaseb_v1",
}
ARTIFACT_MARKER = "RUNG2_ARTIFACT_JSON="


def _dummy_geometry(app: str) -> dict[str, tuple[int, int]]:
    """Retain the executor-v4 deterministic compiler fixture for its unit tests."""

    del app
    return {
        "editor": (820, 520),
        "cell": (640, 420),
        "source": (500, 300),
        "destination": (500, 420),
        "decoy": (500, 360),
        "moved": (500, 300),
        "nav": (260, 140),
        "decoy_nav": (460, 140),
        "toggle": (340, 760),
        "decoy_toggle": (340, 820),
        "scroll_surface": (960, 540),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_setup_dependency(
    manifest: TaskManifest,
    *,
    path: Path,
    expected_artifact_id: str,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    if len(expected_raw_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_raw_sha256
    ):
        raise ValueError("task setup validation SHA must be lowercase 64-hex")
    observed = _sha256_file(path)
    if observed != expected_raw_sha256:
        raise ValueError("task setup validation raw SHA mismatch")
    artifact = load_task_setup_validation(path, manifest)
    if artifact["artifact_id"] != expected_artifact_id:
        raise ValueError("task setup validation artifact ID mismatch")
    return artifact


def _setup_after_reset(transport: Any, task: SemanticTask) -> None:
    if task.app == "vscode":
        from ..rung1b_realapps.vm import setup_fixture

        setup_fixture(transport, as_vscode_fixture(task))
    else:
        from .vm import setup_fixture

        setup_fixture(transport, as_sameapp_fixture(task))


def _reset_probe(
    session: KvmFixtureSession,
    task: SemanticTask,
    ledger: RuntimeEvidenceLedger,
) -> tuple[Any, RuntimeProbe]:
    transport, provider_reset_receipt = session.reset_to_ready_with_receipt()
    _setup_after_reset(transport, task)
    probe = probe_runtime(transport, task)
    attributed = ledger.issue_reset_probe(
        task,
        probe,
        provider_reset_receipt=provider_reset_receipt,
        transport_endpoint=str(transport.base_url),
    )
    return transport, attributed


def _guest_root(transport: Any, task: SemanticTask) -> PurePosixPath:
    if task.app == "vscode":
        from ..rung1b_realapps.vm import resolve_guest_root

        return resolve_guest_root(transport) / task.task_id
    from .vm import resolve_guest_root

    return resolve_guest_root(transport) / task.task_id


def _export_guest_artifact(
    transport: Any, task: SemanticTask, destination: Path
) -> Path:
    """Copy real guest files to a host root for independent extraction."""

    root = _guest_root(transport, task)
    if task.app in {"writer", "calc", "vscode"}:
        relative_paths = [str(task.params["file_name"])]
        directory_paths: list[str] = []
    elif task.app == "chrome":
        relative_paths = ["state.json"]
        directory_paths = []
    else:
        relative_paths = []
        directory_paths = [
            str(task.params["destination_name"]),
            str(task.params["decoy_name"]),
        ]
    code = f"""
import base64,json,pathlib
root=pathlib.Path({str(root)!r}).resolve(strict=True)
files={relative_paths!r}; dirs={directory_paths!r}
for directory in dirs:
 path=(root/directory).resolve(strict=True)
 if path.parent != root: raise RuntimeError('artifact directory escaped root')
 files.extend(str(item.relative_to(root)) for item in path.iterdir() if item.is_file())
if {task.app!r} == 'files':
 source=(root/{str(task.params.get('source_name', ''))!r}).resolve()
 if source.is_file(): files.append(str(source.relative_to(root)))
payload={{'directories':dirs,'files':{{}}}}
for relative in sorted(set(files)):
 path=(root/relative).resolve(strict=True)
 if root not in path.parents: raise RuntimeError('artifact file escaped root')
 payload['files'][relative]=base64.b64encode(path.read_bytes()).decode('ascii')
print({ARTIFACT_MARKER!r}+json.dumps(payload,sort_keys=True))
""".strip()
    result = transport.execute_argv(["python3", "-c", code])
    output = result.get("output")
    if not isinstance(output, str):
        raise RuntimeError("guest artifact export returned no output")
    lines = [line for line in output.splitlines() if line.startswith(ARTIFACT_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("guest artifact export marker mismatch")
    payload = json.loads(lines[0][len(ARTIFACT_MARKER) :])
    destination.mkdir(parents=True, exist_ok=False)
    for relative in payload["directories"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("unsafe exported artifact directory")
        (destination / path).mkdir(parents=True, exist_ok=False)
    for relative, encoded in payload["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("unsafe exported artifact file")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded, validate=True))
    return destination


def _execute_native(transport: Any, payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    executor = NativeAbsoluteExecutor(transport)
    results: list[dict[str, Any]] = []
    for operation_index, operation in enumerate(payload["operations"]):
        cursor_before = tuple(transport.cursor_position())
        action = dict(operation)
        action["action"] = {
            "click": "left_click",
            "key_chord": "key",
        }.get(action["action"], action["action"])
        raw = asdict(executor.execute(action))
        results.append(
            _seal_dispatch_result(
                raw,
                compiled_payload=operation,
                compiled_operation_index=operation_index,
                cursor_before=cursor_before,
                cursor_after=tuple(transport.cursor_position()),
            )
        )
    return tuple(results)


def _seal_dispatch_result(
    result: dict[str, Any],
    *,
    compiled_payload: Any,
    compiled_operation_index: int,
    cursor_before: tuple[int, int],
    cursor_after: tuple[int, int],
) -> dict[str, Any]:
    value = dict(result)
    if (
        value.get("adapter") == "native_absolute_control"
        and value.get("executor_dispatch_status") == "ok"
    ):
        operations, atomic_operations = _approved_native_receipt_view(
            compiled_payload
        )
        value["executor_v4_dispatch_evidence"] = dict(result)
        value["operations"] = operations
        if atomic_operations is None:
            value["atomic_state"] = None
        else:
            atomic = value.get("atomic_state")
            if not isinstance(atomic, dict) or atomic.get("ok") is not True:
                raise RuntimeError("executor v4 returned invalid atomic click evidence")
            value["atomic_state"] = {**atomic, "operations": atomic_operations}
    value["compiled_payload_sha256"] = hashlib.sha256(
        canonical_json(compiled_payload)
    ).hexdigest()
    value["compiled_operation_index"] = compiled_operation_index
    value["cursor_before"] = list(cursor_before)
    value["cursor_after"] = list(cursor_after)
    atomic = value.get("atomic_state")
    value["atomic_state_sha256"] = (
        hashlib.sha256(canonical_json(atomic)).hexdigest()
        if isinstance(atomic, dict)
        else None
    )
    value["dispatch_result_sha256"] = hashlib.sha256(
        canonical_json(value)
    ).hexdigest()
    return value


def _approved_native_receipt_view(
    operation: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Expose the c603 receipt vocabulary while retaining v4 evidence above it."""

    if not isinstance(operation, dict):
        raise TypeError("native compiled operation must be an object")
    kind = operation.get("action")
    coordinate = operation.get("coordinate")
    rows: list[dict[str, Any]] = []
    if coordinate is not None and kind in {
        "click",
        "mouse_down",
        "mouse_move",
        "mouse_up",
    }:
        rows.append({"kind": "move_to", "args": list(coordinate)})
    atomic: list[dict[str, Any]] | None = None
    if kind == "click":
        atomic = [
            {"kind": "mouse_down", "args": ["left"]},
            {"kind": "mouse_up", "args": ["left"]},
        ]
        rows.extend(atomic)
    elif kind == "mouse_down":
        rows.append(
            {"kind": "mouse_down", "args": [operation.get("button", "left")]}
        )
    elif kind == "mouse_up":
        rows.append(
            {"kind": "mouse_up", "args": [operation.get("button", "left")]}
        )
    elif kind == "mouse_move":
        pass
    elif kind == "scroll":
        rows.append({"kind": "scroll", "args": [operation["clicks"]]})
    elif kind == "key_chord":
        rows.append({"kind": "key_chord", "args": list(operation["keys"])})
    elif kind == "type":
        rows.append({"kind": "coalesced_type", "args": [operation["text"]]})
    else:
        raise ValueError(f"unsupported native compiled operation: {kind!r}")
    return rows, atomic


def _dispatch_compiled_action(
    transport: Any, action_schema: str, action: dict[str, Any] | str
) -> tuple[dict[str, Any], ...]:
    if action_schema == "native_absolute_sequence_v1":
        if not isinstance(action, dict):
            raise TypeError("native compiled action must be an object")
        return _execute_native(transport, action)
    if not isinstance(action, str):
        raise TypeError("compact compiled action must be text")
    cursor_before = tuple(transport.cursor_position())
    raw = asdict(CompactRawExecutor(transport).execute(action))
    return (
        _seal_dispatch_result(
            raw,
            compiled_payload=action,
            compiled_operation_index=0,
            cursor_before=cursor_before,
            cursor_after=tuple(transport.cursor_position()),
        ),
    )


def _screenshot(transport: Any, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(transport.base_url + "/screenshot", timeout=15) as response:
        payload = response.read()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("VM screenshot endpoint did not return PNG")
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _assert_released(transport: Any, task: SemanticTask) -> None:
    audit = transport.audit
    if audit.held_buttons or audit.held_keys:
        raise RuntimeError(
            f"{task.task_id}: execution left held inputs "
            f"buttons={sorted(audit.held_buttons)} keys={sorted(audit.held_keys)}"
        )


def _require_fixture_contract(
    task: SemanticTask, arm: str, fixture_contract: dict[str, Any]
) -> None:
    required = (
        "reset_rejected",
        "near_miss_rejected",
        "gold_passed",
        "reset_reproducible",
        "fresh_process_final_oracle",
        "zero_held_inputs",
    )
    if any(fixture_contract.get(key) is not True for key in required):
        raise RuntimeError(f"{task.task_id}/{arm}: fixture contract failed")


def _execute_bound_trajectory(
    transport: Any,
    task: SemanticTask,
    binding: ValidatedRuntimeBinding,
    ledger: RuntimeEvidenceLedger,
    *,
    action_schema: str,
    near_miss: bool,
    frame_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], CompiledProgram]:
    """Compile, execute, and receipt segments in causal semantic-step order."""

    journal: list[dict[str, Any]] = []
    executed: list[ExecutedSegmentReceipt] = []
    for semantic_step in range(1, task.semantic_step_count + 1):
        binding_receipt = binding.receipt()
        segment: CompiledSegment = compile_semantic_step(
            task,
            action_schema,
            binding=binding,
            semantic_step_index=semantic_step,
            near_miss=near_miss,
        )
        started = time.monotonic_ns()
        dispatches: list[tuple[dict[str, Any], ...]] = []
        action_rows: list[dict[str, Any]] = []
        for action_index, action in enumerate(segment.actions):
            screenshot = _screenshot(
                transport,
                (
                    frame_dir / f"step_{semantic_step:02d}_action_{action_index:02d}.png"
                    if frame_dir is not None
                    else None
                ),
            )
            result = _dispatch_compiled_action(transport, action_schema, action)
            dispatches.append(result)
            action_rows.append(
                {
                    "action_index": action_index,
                    "screenshot": screenshot,
                    "action": action,
                    "dispatch": list(result),
                }
            )
        completed = time.monotonic_ns()
        receipt = record_executed_segment(
            segment,
            tuple(dispatches),
            execution_started_monotonic_ns=started,
            execution_completed_monotonic_ns=completed,
        )
        ledger.record_executed_segment(
            task,
            binding,
            segment,
            tuple(dispatches),
            receipt,
            near_miss=near_miss,
        )
        executed.append(receipt)
        journal.append(
            {
                "semantic_step": semantic_step,
                "binding_receipt": binding_receipt,
                "binding_sha256": segment.binding_sha256,
                "compiled_segment": asdict(segment),
                "executed_receipt": asdict(receipt),
                "actions": action_rows,
            }
        )
        if task.app == "chrome" and semantic_step == 2:
            time.sleep(0.35)
            probe_started = time.monotonic_ns()
            refreshed = probe_runtime(transport, task, expect_initial_state=False)
            probe_completed = time.monotonic_ns()
            refreshed = ledger.issue_refresh_probe(
                task,
                binding,
                refreshed,
                completed_step=2,
                executed_segment=receipt,
                action_started_monotonic_ns=started,
                action_completed_monotonic_ns=completed,
                probe_started_monotonic_ns=probe_started,
                probe_completed_monotonic_ns=probe_completed,
            )
            binding = refresh_binding_after_step(
                task,
                binding,
                completed_step=2,
                probe=refreshed,
                executed_segment=receipt,
                ledger=ledger,
            )
            journal[-1]["post_scroll_refresh"] = asdict(refreshed.refresh_evidence)
            journal[-1]["refreshed_binding_sha256"] = binding.binding_sha256
            journal[-1]["refreshed_binding_receipt"] = binding.receipt()
    _assert_released(transport, task)
    aggregate = aggregate_executed_segments(
        task, action_schema, segments=tuple(executed)
    )
    return journal, aggregate


def run_build_replay(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "build-only scripted replay is disabled; production requires live VM bindings, "
        "real artifacts, and task_setup_validation.json"
    )


def run_vm_replay(
    split: str,
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
    task_setup_validation: Path,
    task_setup_validation_artifact_id: str,
    task_setup_validation_sha256: str,
    expected_provider_sha256: str,
    fixture_id: str | None = None,
) -> dict[str, Any]:
    if split != "development":
        raise ValueError("hardened production replay is development-only")
    manifest = load_manifest(split)
    setup = _load_setup_dependency(
        manifest,
        path=task_setup_validation,
        expected_artifact_id=task_setup_validation_artifact_id,
        expected_raw_sha256=task_setup_validation_sha256,
    )
    provider_sha256 = sha256_file(provider)
    rows: list[dict[str, Any]] = []
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=output / "vm_logs",
        smp=int(os.environ.get("OSWORLD_VM_SMP", "4")),
        memory=os.environ.get("OSWORLD_VM_MEM", "8G"),
        expected_provider_sha256=expected_provider_sha256,
    ) as session:
        tasks = manifest.tasks
        if fixture_id is not None:
            tasks = (manifest.by_id(fixture_id),)
        ledger = RuntimeEvidenceLedger(
            setup_commit=setup["setup_commit"],
            vm_snapshot_id=setup["vm_snapshot_id"],
            reset_provider=str(provider.resolve()),
            reset_attestor=session,
        )
        for task in tasks:
            for arm in ARMS:
                action_schema = ARM_SCHEMAS[arm]
                artifact_root = output / "artifacts" / task.task_id / arm

                transport_a, probe_a = _reset_probe(session, task, ledger)
                reset_root = _export_guest_artifact(
                    transport_a, task, artifact_root / "reset"
                )
                transport_b, probe_b = _reset_probe(session, task, ledger)
                reset_repeat_root = _export_guest_artifact(
                    transport_b, task, artifact_root / "reset_repeat"
                )
                near_binding = bind_repeated_runtime_probes(
                    task, (probe_a, probe_b), ledger=ledger
                )
                near_journal, near_receipt = _execute_bound_trajectory(
                    transport_b,
                    task,
                    near_binding,
                    ledger,
                    action_schema=action_schema,
                    near_miss=True,
                )
                near_root = _export_guest_artifact(
                    transport_b, task, artifact_root / "near"
                )

                _, probe_c = _reset_probe(session, task, ledger)
                transport_d, probe_d = _reset_probe(session, task, ledger)
                gold_binding = bind_repeated_runtime_probes(
                    task, (probe_c, probe_d), ledger=ledger
                )
                gold_journal, gold_receipt = _execute_bound_trajectory(
                    transport_d,
                    task,
                    gold_binding,
                    ledger,
                    action_schema=action_schema,
                    near_miss=False,
                    frame_dir=output / "frames" / task.task_id / arm / "gold",
                )
                gold_root = _export_guest_artifact(
                    transport_d, task, artifact_root / "gold"
                )
                fixture_contract = verify_fixture_contract(
                    task,
                    artifact_roots={
                        "reset": reset_root,
                        "reset_repeat": reset_repeat_root,
                        "near": near_root,
                        "gold": gold_root,
                    },
                )
                _require_fixture_contract(task, arm, fixture_contract)
                rows.append(
                    {
                        "fixture_id": task.task_id,
                        "fixture_sha256": task.fixture_sha256,
                        "app": task.app,
                        "arm": arm,
                        "action_schema": action_schema,
                        "semantic_steps": task.semantic_step_count,
                        "reset_cycle_evidence": [
                            asdict(value.reset_cycle_evidence)
                            for value in (probe_a, probe_b, probe_c, probe_d)
                        ],
                        "near_miss_journal": near_journal,
                        "near_miss_receipt": asdict(near_receipt),
                        "gold_journal": gold_journal,
                        "gold_receipt": asdict(gold_receipt),
                        "fixture_contract": fixture_contract,
                    }
                )
                _atomic_json(
                    output / "progress.json",
                    {
                        "schema_version": 2,
                        "status": "running",
                        "split": split,
                        "sealed_eval_executed": False,
                        "task_setup_validation_artifact_id": setup["artifact_id"],
                        "task_setup_validation_sha256": task_setup_validation_sha256,
                        "completed_cells": len(rows),
                        "rows": rows,
                    },
                )
    return {
        "schema_version": 2,
        "status": "passed",
        "mode": "vm",
        "split": split,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "task_setup_validation_artifact_id": setup["artifact_id"],
        "task_setup_validation_sha256": task_setup_validation_sha256,
        "sealed_eval_executed": False,
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "provider": {
            "path": str(provider.resolve()),
            "sha256": provider_sha256,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vm",), required=True)
    parser.add_argument("--split", choices=("development",), default="development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--task-setup-validation", type=Path, required=True)
    parser.add_argument("--task-setup-validation-artifact-id", required=True)
    parser.add_argument("--task-setup-validation-sha256", required=True)
    parser.add_argument("--expected-provider-sha256", required=True)
    parser.add_argument("--fixture-id")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "replay.json"
    marker.unlink(missing_ok=True)
    try:
        payload = run_vm_replay(
            args.split,
            output=args.output,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
            task_setup_validation=args.task_setup_validation,
            task_setup_validation_artifact_id=args.task_setup_validation_artifact_id,
            task_setup_validation_sha256=args.task_setup_validation_sha256,
            expected_provider_sha256=args.expected_provider_sha256,
            fixture_id=args.fixture_id,
        )
        _atomic_json(marker, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 2,
            "status": "failed",
            "mode": args.mode,
            "split": args.split,
            "sealed_eval_executed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, AppReadinessError):
            failure["readiness"] = {
                "fixture_id": exc.fixture_id,
                "failed_phase": exc.failed_phase,
                "evidence": exc.evidence,
            }
        _atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
