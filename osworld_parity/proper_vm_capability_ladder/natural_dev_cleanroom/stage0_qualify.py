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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..rung1.executor import NativeAbsoluteExecutor
from ..rung1.vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    KvmFixtureSession,
    sha256_file,
)
from ..rung2_sameapp.vm import probe_state, setup_fixture
from .oracle import initial_state, reset_signature, scripted_state
from .qualify import EXPECTED_PROVIDER_SHA256, _dispatch_gold
from .stage0_loader import (
    ANCHOR_APPS,
    RECORD_ELIGIBILITY,
    Stage0Inventory,
    Stage0Record,
    Stage0SourceTask,
    canonical_json,
    load_stage0_inventory,
)
from .stage0_oracle import evaluate_composed_in_fresh_process


REPEATABILITY_RUNS = 2


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


def _static_record(record: Stage0Record) -> dict[str, Any]:
    tasks = record.component_tasks
    reset_a = [initial_state(task) for task in tasks]
    reset_b = [initial_state(task) for task in tasks]
    reset_signatures_a = [
        reset_signature(task, state) for task, state in zip(tasks, reset_a, strict=True)
    ]
    reset_signatures_b = [
        reset_signature(task, state) for task, state in zip(tasks, reset_b, strict=True)
    ]
    reset_result = evaluate_composed_in_fresh_process(record, reset_a)
    gold_states = [scripted_state(task, near_miss=False) for task in tasks]
    gold_result = evaluate_composed_in_fresh_process(record, gold_states)
    near_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        states = list(gold_states)
        states[index] = scripted_state(task, near_miss=True)
        result = evaluate_composed_in_fresh_process(record, states)
        near_rows.append(
            {
                "order": index + 1,
                "source_task_id": task.id,
                "rejected": not result.MOUSE_SOLVED,
                "oracle_status": result.oracle_status,
                "oracle_pid": result.oracle_pid,
            }
        )
    passed = bool(
        reset_result.oracle_status == "ok"
        and not reset_result.MOUSE_SOLVED
        and all(row["rejected"] and row["oracle_status"] == "ok" for row in near_rows)
        and gold_result.oracle_status == "ok"
        and gold_result.MOUSE_SOLVED
        and reset_signatures_a == reset_signatures_b
        and all(
            pid != os.getpid()
            for pid in [reset_result.oracle_pid, gold_result.oracle_pid]
            + [row["oracle_pid"] for row in near_rows]
        )
    )
    return {
        "record_id": record.id,
        "anchor_app": record.anchor_app,
        "mode": record.mode,
        "record_sha256": record.record_sha256,
        "reset_signatures": reset_signatures_a,
        "reset_reject": not reset_result.MOUSE_SOLVED,
        "each_component_near_miss_reject": near_rows,
        "gold_pass": gold_result.MOUSE_SOLVED,
        "fresh_oracle_pids": [reset_result.oracle_pid, gold_result.oracle_pid]
        + [row["oracle_pid"] for row in near_rows],
        "status": "pass" if passed else "fail",
    }


def qualify_static(inventory: Stage0Inventory) -> dict[str, Any]:
    rows = [_static_record(record) for record in inventory.tasks]
    return _seal(
        {
            "schema_version": 1,
            "qualification": "stage0_host_composed_contract",
            "inventory_role": "natural_dev_stage0",
            "eligibility": dict(RECORD_ELIGIBILITY),
            "suite_manifest_sha256": inventory.manifest_payload_sha256,
            "model_runs": False,
            "paired_runtime": False,
            "task_count": len(rows),
            "passed_count": sum(row["status"] == "pass" for row in rows),
            "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
            "tasks": rows,
        }
    )


def _setup_source(transport: Any, task: Stage0SourceTask) -> dict[str, Any]:
    old_timeout = transport.timeout_s
    transport.timeout_s = max(old_timeout, 90.0)
    try:
        if task.app == "vscode":
            from ..rung1b_realapps.vm import setup_fixture as setup_vscode

            fixture = task.as_vscode_fixture()
            guest = setup_vscode(transport, fixture)
            return {
                "task": task,
                "fixture": fixture,
                "initial_state": guest.state,
                "geometry": {"editor": guest.geometry.editor},
                "readiness": guest.readiness,
            }
        fixture = task.as_fixture()
        guest = setup_fixture(transport, fixture)
        return {
            "task": task,
            "fixture": fixture,
            "initial_state": guest.state,
            "geometry": guest.geometry,
            "readiness": guest.readiness,
        }
    finally:
        transport.timeout_s = old_timeout


def _window_token(task: Stage0SourceTask) -> str:
    if task.app in {"writer", "calc", "vscode"}:
        return str(task.params["file_name"])
    if task.app == "files":
        return task.id
    return "Same-app settings"


def _active_window(transport: Any) -> dict[str, str]:
    code = """
import json,subprocess
active=subprocess.run(['xprop','-root','_NET_ACTIVE_WINDOW'],capture_output=True,text=True,check=True).stdout.strip().split()[-1]
lines=subprocess.run(['wmctrl','-l'],capture_output=True,text=True,check=True).stdout.splitlines()
line=next((item for item in lines if int(item.split()[0],16)==int(active,16)),'')
print('STAGE0_ACTIVE='+json.dumps({'window_id':active,'window_line':line},sort_keys=True))
""".strip()
    output = str(transport.execute_argv(["python3", "-c", code]).get("output", ""))
    lines = [line for line in output.splitlines() if line.startswith("STAGE0_ACTIVE=")]
    if len(lines) != 1:
        raise RuntimeError("active-window evidence missing")
    return json.loads(lines[0].removeprefix("STAGE0_ACTIVE="))


def _activate_anchor_for_setup(transport: Any, task: Stage0SourceTask) -> dict[str, Any]:
    token = _window_token(task)
    script = f"""
set -euo pipefail
token={json.dumps(token)}
win=$(wmctrl -l | awk -v token="$token" 'index($0,token){{print $1; exit}}')
test -n "$win"
wmctrl -ia "$win"
sleep 1
""".strip()
    transport.execute_argv(["bash", "-lc", script])
    active = _active_window(transport)
    if token not in active["window_line"]:
        raise RuntimeError(f"setup anchor activation mismatch: {active}")
    return {"phase": "setup_only", "target_token": token, **active}


def _probe_source(transport: Any, task: Stage0SourceTask, fixture: Any) -> dict[str, Any]:
    if task.app == "vscode":
        from ..rung1b_realapps.vm import probe_fixture

        return probe_fixture(transport, fixture)
    return probe_state(transport, fixture)


def _vm_repetition(
    session: KvmFixtureSession,
    record: Stage0Record,
    repetition_index: int,
) -> dict[str, Any]:
    started = time.monotonic()
    transport, provider_reset = session.reset_to_ready_with_receipt()
    session.consume_provider_reset_receipt(provider_reset)
    components = [_setup_source(transport, task) for task in record.component_tasks]
    setup_activation = _activate_anchor_for_setup(transport, record.component_tasks[0])
    initial_states = [dict(component["initial_state"]) for component in components]
    initial_result = evaluate_composed_in_fresh_process(record, initial_states)
    reset_signatures = [
        reset_signature(component["task"], component["initial_state"])
        for component in components
    ]
    action_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for order, component in enumerate(components, start=1):
        task = component["task"]
        if order == 2:
            before = _active_window(transport)
            switch = NativeAbsoluteExecutor(transport).execute(
                {"action": "key", "keys": ["Alt", "Tab"]}
            )
            transport.wait(1.5)
            after = _active_window(transport)
            target_token = _window_token(task)
            if target_token not in after["window_line"]:
                raise RuntimeError(
                    f"visible Alt+Tab did not activate ordered partner {target_token!r}: {after}"
                )
            switch_rows.append(
                {
                    "record_semantic_step": 2,
                    "policy_visible": True,
                    "input": {"action": "key", "keys": ["Alt", "Tab"]},
                    "action_class": switch.action_class,
                    "dispatch_status": switch.executor_dispatch_status,
                    "atomic_ok": bool(switch.atomic_state and switch.atomic_state.get("ok") is True),
                    "active_before": before,
                    "active_after": after,
                    "target_token": target_token,
                }
            )
        actions, bindings = _dispatch_gold(
            transport,
            task,
            component["geometry"],
        )
        action_rows.append(
            {
                "record_semantic_step": order if record.mode == "multi" else None,
                "source_task_id": task.id,
                "app": task.app,
                "source_semantic_steps": task.semantic_steps,
                "actions": actions,
                "runtime_bindings": bindings,
            }
        )
    final_states = [
        _probe_source(transport, component["task"], component["fixture"])
        for component in components
    ]
    final_result = evaluate_composed_in_fresh_process(record, final_states)
    audit = transport.audit
    dispatch_ok = all(
        row["dispatch_status"] == "ok" and row["atomic_ok"]
        for component in action_rows
        for row in component["actions"]
    ) and all(
        row["dispatch_status"] == "ok" and row["atomic_ok"] for row in switch_rows
    )
    passed = bool(
        initial_result.oracle_status == "ok"
        and not initial_result.MOUSE_SOLVED
        and final_result.oracle_status == "ok"
        and final_result.MOUSE_SOLVED
        and dispatch_ok
        and not audit.held_buttons
        and not audit.held_keys
        and len(switch_rows) == (1 if record.mode == "multi" else 0)
    )
    return {
        "repetition_index": repetition_index,
        "status": "pass" if passed else "fail",
        "duration_s": round(time.monotonic() - started, 3),
        "provider_reset_receipt": asdict(provider_reset),
        "reset_signatures": reset_signatures,
        "setup_activation": setup_activation,
        "readiness": [component["readiness"] for component in components],
        "initial_reject": not initial_result.MOUSE_SOLVED,
        "initial_oracle_pid": initial_result.oracle_pid,
        "gold_pass": final_result.MOUSE_SOLVED,
        "gold_oracle_pid": final_result.oracle_pid,
        "gold_oracle_status": final_result.oracle_status,
        "gold_oracle_reason": final_result.reason,
        "component_results": list(final_result.component_results),
        "final_states": final_states,
        "app_switches": switch_rows,
        "component_actions": action_rows,
        "input_audit": {
            "operation_count": len(audit.operations),
            "coalesced_type_count": len(audit.typed_texts),
            "held_buttons": sorted(audit.held_buttons),
            "held_keys": sorted(audit.held_keys),
            "scroll_total": audit.scroll_total,
        },
    }


def _vm_record(session: KvmFixtureSession, record: Stage0Record) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    for repetition_index in range(1, REPEATABILITY_RUNS + 1):
        try:
            repetitions.append(_vm_repetition(session, record, repetition_index))
        except Exception as exc:
            repetitions.append(
                {
                    "repetition_index": repetition_index,
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break
    reset_signatures = [row.get("reset_signatures") for row in repetitions]
    reset_repeatable = bool(
        len(reset_signatures) == REPEATABILITY_RUNS
        and reset_signatures[0] is not None
        and all(value == reset_signatures[0] for value in reset_signatures)
    )
    reset_receipts = [row.get("provider_reset_receipt", {}) for row in repetitions]
    distinct_resets = bool(
        len(reset_receipts) == REPEATABILITY_RUNS
        and len({row.get("reset_id") for row in reset_receipts}) == REPEATABILITY_RUNS
        and len({row.get("new_generation_id") for row in reset_receipts}) == REPEATABILITY_RUNS
    )
    passed = bool(
        len(repetitions) == REPEATABILITY_RUNS
        and all(row["status"] == "pass" for row in repetitions)
        and reset_repeatable
        and distinct_resets
    )
    return {
        "record_id": record.id,
        "anchor_app": record.anchor_app,
        "mode": record.mode,
        "difficulty": record.difficulty,
        "record_sha256": record.record_sha256,
        "status": "pass" if passed else "fail",
        "repeatability_runs_required": REPEATABILITY_RUNS,
        "repeatability_runs_observed": len(repetitions),
        "reset_signatures_repeatable": reset_repeatable,
        "provider_resets_distinct": distinct_resets,
        "repetitions": repetitions,
    }


def _selected_records(
    inventory: Stage0Inventory,
    *,
    shard_index: int | None,
    record_id: str | None,
) -> tuple[Stage0Record, ...]:
    if record_id is not None:
        if shard_index is not None:
            raise ValueError("record-id and shard-index are mutually exclusive")
        return (inventory.by_id(record_id),)
    if shard_index is None:
        return inventory.tasks
    if not 0 <= shard_index < len(ANCHOR_APPS):
        raise ValueError(f"shard-index must be in [0, {len(ANCHOR_APPS) - 1}]")
    anchor = ANCHOR_APPS[shard_index]
    return tuple(record for record in inventory.tasks if record.anchor_app == anchor)


def qualify_vm(
    inventory: Stage0Inventory,
    *,
    shard_index: int | None,
    record_id: str | None,
    qcow: Path,
    qemu: Path,
    provider: Path,
    work_dir: Path,
) -> dict[str, Any]:
    selected = _selected_records(inventory, shard_index=shard_index, record_id=record_id)
    if not selected:
        raise ValueError("Stage0 qualification selection is empty")
    work_dir.mkdir(parents=True, exist_ok=False)
    session = KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        expected_provider_sha256=EXPECTED_PROVIDER_SHA256,
        vm_log_dir=work_dir / "vm_logs",
        scratch_root=work_dir / "scratch",
    )
    rows: list[dict[str, Any]] = []
    start_failure: dict[str, Any] | None = None
    cleanup_error: str | None = None
    try:
        session.start()
        for record in selected:
            row = _vm_record(session, record)
            rows.append(row)
            print(json.dumps({"record_id": record.id, "status": row["status"]}, sort_keys=True), flush=True)
    except Exception as exc:
        start_failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not rows:
            rows = [
                {
                    "record_id": record.id,
                    "anchor_app": record.anchor_app,
                    "mode": record.mode,
                    "difficulty": record.difficulty,
                    "record_sha256": record.record_sha256,
                    "status": "fail",
                    "failure_phase": "session_start",
                    **start_failure,
                }
                for record in selected
            ]
    finally:
        try:
            session.close()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
    metadata_path = work_dir / "vm_metadata.json"
    return _seal(
        {
            "schema_version": 1,
            "qualification": "cpu_kvm_stage0_composed_repeatability_gold",
            "inventory_role": "natural_dev_stage0",
            "eligibility": dict(RECORD_ELIGIBILITY),
            "suite_manifest_sha256": inventory.manifest_payload_sha256,
            "model_runs": False,
            "paired_runtime": False,
            "native_gold_does_not_substitute_for_paired_adapter": True,
            "retries": 0,
            "repeatability_runs_per_record": REPEATABILITY_RUNS,
            "shard_index": shard_index,
            "record_id_filter": record_id,
            "task_count": len(rows),
            "passed_count": sum(row["status"] == "pass" for row in rows),
            "status": "pass" if len(rows) == len(selected) and all(row["status"] == "pass" for row in rows) else "fail",
            "anchor_counts": {
                app: sum(row.get("anchor_app") == app for row in rows)
                for app in sorted({row.get("anchor_app") for row in rows if row.get("anchor_app")})
            },
            "mode_counts": {
                mode: sum(row.get("mode") == mode for row in rows)
                for mode in ("single", "multi")
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
            "session_start_failure": start_failure,
            "cleanup_error": cleanup_error,
            "vm_metadata_path": str(metadata_path),
            "vm_metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
            "tasks": rows,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify the clean-room Stage0 inventory")
    parser.add_argument("--mode", choices=("static", "vm"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--record-id")
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    inventory = load_stage0_inventory()
    if args.mode == "static":
        receipt = qualify_static(inventory)
    else:
        receipt = qualify_vm(
            inventory,
            shard_index=args.shard_index,
            record_id=args.record_id,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
            work_dir=args.work_dir or args.output.with_suffix(".work"),
        )
    _atomic_json(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("qualification", "status", "task_count", "passed_count")}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
