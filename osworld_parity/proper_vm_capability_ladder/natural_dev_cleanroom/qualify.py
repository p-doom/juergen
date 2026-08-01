from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import tempfile
import time
import traceback
from itertools import groupby
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..rung1.executor import NativeAbsoluteExecutor
from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU, KvmFixtureSession, sha256_file
from ..rung2_sameapp.actions import compile_native
from ..rung2_sameapp.trajectory import build_trajectory
from ..rung2_sameapp.vm import AppReadinessError, probe_geometry, probe_state, setup_fixture
from .oracle import evaluate_in_fresh_process, evaluate_state, initial_state, reset_signature, scripted_state
from .schema import APPS, Corpus, Task, canonical_json, load_corpus
from .smoke_schema import SmokeInventory, SmokeTask, load_smoke


EXPECTED_PROVIDER_SHA256 = "76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7"


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


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": hashlib.sha256(canonical_json(value)).hexdigest()}


def _static_task(task: Task | SmokeTask) -> dict[str, Any]:
    reset_a = initial_state(task)
    reset_b = initial_state(task)
    reset_a_result = evaluate_state(task, reset_a)
    near_result = evaluate_in_fresh_process(task, scripted_state(task, near_miss=True))
    gold_result = evaluate_in_fresh_process(task, scripted_state(task, near_miss=False))
    signature_a = reset_signature(task, reset_a)
    signature_b = reset_signature(task, reset_b)
    passed = (
        not reset_a_result.MOUSE_SOLVED
        and not near_result.MOUSE_SOLVED
        and gold_result.MOUSE_SOLVED
        and signature_a == signature_b
        and near_result.oracle_pid != os.getpid()
        and gold_result.oracle_pid != os.getpid()
    )
    return {
        "task_id": task.id,
        "app": task.app,
        "difficulty": task.difficulty,
        "fixture_sha256": task.fixture_sha256,
        "task_sha256": task.task_sha256,
        "reset_signature": signature_a,
        "reset_reject": not reset_a_result.MOUSE_SOLVED,
        "near_miss_reject": not near_result.MOUSE_SOLVED,
        "gold_pass": gold_result.MOUSE_SOLVED,
        "fresh_oracle_pids": [near_result.oracle_pid, gold_result.oracle_pid],
        "status": "pass" if passed else "fail",
    }


def qualify_static(corpus: Corpus | SmokeInventory) -> dict[str, Any]:
    rows = [_static_task(task) for task in corpus.tasks]
    plumbing_smoke = isinstance(corpus, SmokeInventory)
    payload = {
        "schema_version": 1,
        "qualification": "host_contract",
        "inventory_role": (
            "plumbing_smoke_only" if plumbing_smoke else "auxiliary_development_only"
        ),
        "eligibility": corpus.eligibility,
        "suite_manifest_sha256": corpus.manifest_payload_sha256,
        "model_runs": False,
        "task_count": len(rows),
        "passed_count": sum(row["status"] == "pass" for row in rows),
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "tasks": rows,
    }
    return _seal(payload)


def _selected_tasks(corpus: Corpus, per_app: int) -> tuple[Task, ...]:
    if not 1 <= per_app <= 10:
        raise ValueError("per-app qualification count must be in [1, 10]")
    return tuple(
        task
        for app in APPS
        for task in [item for item in corpus.tasks if item.app == app][:per_app]
    )


def _dispatch_gold(
    transport: Any,
    task: Task | SmokeTask,
    geometry: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executor = NativeAbsoluteExecutor(transport)
    receipts: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    if task.app == "vscode":
        actions = (
            {"action": "left_click", "coordinate": list(geometry["editor"])},
            {"action": "key", "keys": ["ControlLeft", "KeyA"]},
            {"action": "type", "text": str(task.expected["text"])},
            {"action": "key", "keys": ["ControlLeft", "KeyS"]},
        )
        for operation_index, action in enumerate(actions):
            result = executor.execute(action)
            receipts.append(
                {
                    "turn_index": operation_index,
                    "operation_index": 0,
                    "semantic_step": min(operation_index + 1, task.semantic_steps),
                    "action_class": result.action_class,
                    "operation_count": len(result.operations),
                    "dispatch_status": result.executor_dispatch_status,
                    "atomic_ok": bool(result.atomic_state and result.atomic_state.get("ok") is True),
                }
            )
            transport.wait(0.5)
        transport.wait(2.0)
        return receipts, bindings
    fixture = task.as_fixture()
    indexed_turns = tuple(enumerate(build_trajectory(fixture).turns))
    for semantic_step, grouped in groupby(
        indexed_turns, key=lambda item: item[1].semantic_step
    ):
        for turn_index, turn in grouped:
            payload = compile_native(turn, geometry)
            for operation_index, operation in enumerate(payload["operations"]):
                action = dict(operation)
                action["action"] = {"click": "left_click", "key_chord": "key"}.get(
                    action["action"], action["action"]
                )
                result = executor.execute(action)
                receipts.append(
                    {
                        "turn_index": turn_index,
                        "operation_index": operation_index,
                        "semantic_step": turn.semantic_step,
                        "action_class": result.action_class,
                        "operation_count": len(result.operations),
                        "dispatch_status": result.executor_dispatch_status,
                        "atomic_ok": bool(result.atomic_state and result.atomic_state.get("ok") is True),
                    }
                )
                transport.wait(0.5)
        transport.wait(1.0)
        # Calc's formula text is deliberately unconfirmed between semantic
        # steps 2 and 3.  Host-side state/geometry probing at that boundary
        # activates the window and can cancel the in-progress cell edit.  Only
        # Files (folder transition) and Chrome (scroll-relative controls) have
        # geometry that genuinely changes between steps.
        if semantic_step < task.semantic_steps and task.app in {"files", "chrome"}:
            rebound_state = probe_state(transport, fixture)
            geometry = probe_geometry(transport, fixture, rebound_state)
            bindings.append(
                {
                    "completed_semantic_step": semantic_step,
                    "state_sha256": hashlib.sha256(canonical_json(rebound_state)).hexdigest(),
                    "geometry": {key: list(value) for key, value in geometry.items()},
                }
            )
    transport.wait(2.0)
    return receipts, bindings


def _vm_task(session: KvmFixtureSession, task: Task | SmokeTask) -> dict[str, Any]:
    started = time.monotonic()
    transport, provider_reset = session.reset_to_ready_with_receipt()
    session.consume_provider_reset_receipt(provider_reset)
    if task.app == "vscode":
        from ..rung1b_realapps.vm import probe_fixture as probe_vscode
        from ..rung1b_realapps.vm import probe_geometry as probe_vscode_geometry
        from ..rung1b_realapps.vm import setup_fixture as setup_vscode

        fixture = task.as_vscode_fixture()
        original_timeout = transport.timeout_s
        transport.timeout_s = max(original_timeout, 90.0)
        try:
            guest = setup_vscode(transport, fixture)
        finally:
            transport.timeout_s = original_timeout
        transport.execute_argv(
            [
                "bash",
                "-lc",
                "win=$(wmctrl -lx | awk 'tolower($0) ~ /(code|codium)/ {print $1; exit}'); test -n \"$win\"; wmctrl -ia \"$win\"; sleep 1",
            ]
        )
        rebound_geometry = probe_vscode_geometry(transport, fixture)
        geometry = {"editor": rebound_geometry.editor}
    else:
        fixture = task.as_fixture()
        original_timeout = transport.timeout_s
        transport.timeout_s = max(original_timeout, 90.0)
        try:
            guest = setup_fixture(transport, fixture)
        finally:
            transport.timeout_s = original_timeout
        if task.app == "chrome":
            transport.execute_argv(
                ["bash", "-lc", "wmctrl -a 'Same-app settings'; sleep 1"]
            )
        rebound_state = probe_state(transport, fixture)
        geometry = probe_geometry(transport, fixture, rebound_state)
    initial_result = evaluate_state(task, guest.state)
    initial_signature = reset_signature(task, guest.state)
    actions, bindings = _dispatch_gold(transport, task, geometry)
    state = probe_vscode(transport, fixture) if task.app == "vscode" else probe_state(transport, fixture)
    final = evaluate_in_fresh_process(task, state)
    audit = transport.audit
    passed = (
        not initial_result.MOUSE_SOLVED
        and final.MOUSE_SOLVED
        and not audit.held_buttons
        and not audit.held_keys
        and all(row["dispatch_status"] == "ok" and row["atomic_ok"] for row in actions)
    )
    return {
        "task_id": task.id,
        "app": task.app,
        "difficulty": task.difficulty,
        "fixture_sha256": task.fixture_sha256,
        "status": "pass" if passed else "fail",
        "duration_s": round(time.monotonic() - started, 3),
        "provider_reset_receipt": asdict(provider_reset),
        "reset_signature": initial_signature,
        "initial_reject": not initial_result.MOUSE_SOLVED,
        "gold_pass": final.MOUSE_SOLVED,
        "oracle_pid": final.oracle_pid,
        "oracle_status": final.oracle_status,
        "oracle_reason": final.reason,
        "final_state": state,
        "readiness": guest.readiness,
        "actions": actions,
        "runtime_bindings": bindings,
        "input_audit": {
            "operation_count": len(audit.operations),
            "held_buttons": sorted(audit.held_buttons),
            "held_keys": sorted(audit.held_keys),
            "scroll_total": audit.scroll_total,
            "coalesced_type_count": len(audit.typed_texts),
        },
    }


def qualify_vm(
    corpus: Corpus | SmokeInventory,
    *,
    per_app: int,
    plumbing_smoke: bool,
    shard_index: int | None,
    task_id: str | None,
    qcow: Path,
    qemu: Path,
    provider: Path,
    work_dir: Path,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if not 1 <= per_app <= 10:
        raise ValueError("per-app qualification count must be in [1, 10]")
    if max_attempts not in {1, 2}:
        raise ValueError("max-attempts must be 1 or 2")
    if task_id is not None:
        if shard_index is not None:
            raise ValueError("task-id and shard-index are mutually exclusive")
        selected = tuple(task for task in corpus.tasks if task.id == task_id)
    elif plumbing_smoke:
        if shard_index is not None:
            raise ValueError("the five-task plumbing smoke is not shardable")
        selected = corpus.tasks
    elif shard_index is None:
        selected = _selected_tasks(corpus, per_app)
    else:
        if not 0 <= shard_index < len(APPS):
            raise ValueError(f"shard index must be in [0, {len(APPS) - 1}]")
        app = APPS[shard_index]
        selected = tuple(task for task in corpus.tasks if task.app == app)[:per_app]
    if not selected:
        raise ValueError("qualification selection is empty")
    work_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    session = KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        expected_provider_sha256=EXPECTED_PROVIDER_SHA256,
        vm_log_dir=work_dir / "vm_logs",
        scratch_root=work_dir / "scratch",
    )
    session_start_failure: dict[str, Any] | None = None
    cleanup_error: str | None = None
    try:
        session.start()
        for task in selected:
            attempt_failures: list[dict[str, Any]] = []
            for attempt_index in range(1, max_attempts + 1):
                try:
                    row = _vm_task(session, task)
                    row["attempt_index"] = attempt_index
                    row["prior_attempt_failures"] = attempt_failures
                    break
                except Exception as exc:
                    failure = {
                        "attempt_index": attempt_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    attempt_failures.append(failure)
                    if not isinstance(exc, (TimeoutError, AppReadinessError)) or attempt_index == max_attempts:
                        row = {
                            "task_id": task.id,
                            "app": task.app,
                            "difficulty": task.difficulty,
                            "fixture_sha256": task.fixture_sha256,
                            "status": "fail",
                            **failure,
                            "prior_attempt_failures": attempt_failures[:-1],
                        }
                        break
            rows.append(row)
            print(json.dumps({"task_id": task.id, "status": row["status"]}, sort_keys=True), flush=True)
    except Exception as exc:
        session_start_failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        rows = [
            {
                "task_id": task.id,
                "app": task.app,
                "difficulty": task.difficulty,
                "fixture_sha256": task.fixture_sha256,
                "status": "fail",
                "failure_phase": "session_start",
                **session_start_failure,
            }
            for task in selected
        ]
    finally:
        try:
            session.close()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
    metadata_path = work_dir / "vm_metadata.json"
    payload = {
        "schema_version": 1,
        "qualification": "cpu_kvm_native_absolute_gold",
        "shard_index": shard_index,
        "task_id_filter": task_id,
        "max_attempts": max_attempts,
        "suite_manifest_sha256": corpus.manifest_payload_sha256,
        "model_runs": False,
        "task_count": len(rows),
        "passed_count": sum(row["status"] == "pass" for row in rows),
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "inventory_role": (
            "plumbing_smoke_only" if plumbing_smoke else "auxiliary_development_only"
        ),
        "eligibility": corpus.eligibility,
        "application_counts": {
            app: sum(row["app"] == app for row in rows)
            for app in sorted({row["app"] for row in rows})
        },
        "platform": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "kvm_readable_writable": os.access("/dev/kvm", os.R_OK | os.W_OK),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "provider_path": str(provider.resolve()),
            "provider_sha256": sha256_file(provider),
            "qemu_path": str(qemu.resolve()),
            "qemu_sha256": sha256_file(qemu),
            "qcow_path": str(qcow.resolve()),
            "qcow_sha256": sha256_file(qcow),
        },
        "session_start_failure": session_start_failure,
        "cleanup_error": cleanup_error,
        "vm_metadata_path": str(metadata_path),
        "vm_metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        "tasks": rows,
    }
    return _seal(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "vm"), required=True)
    parser.add_argument("--inventory", choices=("corpus", "plumbing-smoke"), default="corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-app", type=int, default=10)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--task-id")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    corpus = load_smoke() if args.inventory == "plumbing-smoke" else load_corpus()
    if args.mode == "static":
        receipt = qualify_static(corpus)
    else:
        work_dir = args.work_dir or args.output.with_suffix(".work")
        receipt = qualify_vm(
            corpus,
            per_app=args.per_app,
            plumbing_smoke=args.inventory == "plumbing-smoke",
            shard_index=args.shard_index,
            task_id=args.task_id,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
            work_dir=work_dir,
            max_attempts=args.max_attempts,
        )
    _atomic_json(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("qualification", "status", "task_count", "passed_count")}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
