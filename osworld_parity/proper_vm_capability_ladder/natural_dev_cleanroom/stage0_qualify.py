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
from itertools import groupby
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
from ..rung2_sameapp.actions import compile_native
from ..rung2_sameapp.trajectory import build_trajectory
from ..rung2_sameapp.vm import probe_geometry, probe_state, setup_fixture
from .oracle import initial_state, reset_signature, scripted_state
from .qualify import EXPECTED_PROVIDER_SHA256, _dispatch_gold
from .stage0_actions import compile_multi_native, compile_visible_app_switch_native
from .stage0_loader import (
    ANCHOR_APPS,
    RECORD_ELIGIBILITY,
    Stage0Inventory,
    Stage0Record,
    Stage0SourceTask,
    canonical_json,
    load_stage0_inventory,
    sha256_value,
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
            setup_preparation = None
            if "-multi-" in task.id:
                x, y = guest.geometry.editor
                transport.execute_argv(
                    [
                        "python3",
                        "-c",
                        f"import pyautogui,time;pyautogui.click({x},{y});time.sleep(0.75)",
                    ]
                )
                setup_preparation = {
                    "kind": "fixture_editor_focus",
                    "target": "editor",
                    "coordinate": [x, y],
                    "qualification_action": False,
                }
            return {
                "task": task,
                "fixture": fixture,
                "initial_state": guest.state,
                "geometry": {"editor": guest.geometry.editor},
                "readiness": guest.readiness,
                "setup_preparation": setup_preparation,
            }
        fixture = task.as_fixture()
        guest = setup_fixture(transport, fixture)
        setup_preparation = None
        if "-multi-" in task.id:
            if task.app in {"writer", "files"}:
                target = "editor" if task.app == "writer" else "source"
                x, y = guest.geometry[target]
                transport.execute_argv(
                    [
                        "python3",
                        "-c",
                        f"import pyautogui,time;pyautogui.click({x},{y});time.sleep(0.75)",
                    ]
                )
                setup_preparation = {
                    "kind": (
                        "fixture_editor_focus"
                        if task.app == "writer"
                        else "fixture_initial_selection"
                    ),
                    "target": target,
                    "coordinate": [x, y],
                    "qualification_action": False,
                }
            elif task.app == "calc":
                transport.execute_argv(
                    [
                        "python3",
                        "-c",
                        "import pyautogui,time;pyautogui.hotkey('ctrl','home');time.sleep(0.75)",
                    ]
                )
                setup_preparation = {
                    "kind": "fixture_initial_selection",
                    "target": "A1",
                    "qualification_action": False,
                }
        return {
            "task": task,
            "fixture": fixture,
            "initial_state": guest.state,
            "geometry": guest.geometry,
            "readiness": guest.readiness,
            "setup_preparation": setup_preparation,
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
    if task.app == "files" and task.params.get("stage0_program") == "files_rename_selected_source":
        from ..rung2_sameapp.vm import resolve_guest_root

        root = resolve_guest_root(transport) / task.id
        source = root / str(task.params["source_name"])
        gold = root / str(task.expected["final_name"])
        near = root / str(task.near_miss["final_name"])
        code = f"""
import hashlib,json,pathlib
source=pathlib.Path({str(source)!r}); gold=pathlib.Path({str(gold)!r}); near=pathlib.Path({str(near)!r})
present=[path for path in (gold,near,source) if path.is_file()]
assert len(present)==1, 'short Files state is ambiguous'
path=present[0]; renamed=path in (gold,near)
data=path.read_bytes() if path.is_file() else b''
value={{
 'schema_version':1,
 'fixture_id':{task.id!r},
 'fixture_sha256':{task.fixture_sha256!r},
 'app':'files',
 'source_exists':source.is_file(),
 'destination':'root' if renamed else None,
 'final_name':path.name,
 'content_sha256':hashlib.sha256(data).hexdigest(),
 'saved':renamed,
}}
print('STAGE0_STATE='+json.dumps(value,sort_keys=True))
""".strip()
        output = str(transport.execute_argv(["python3", "-c", code]).get("output", ""))
        rows = [line for line in output.splitlines() if line.startswith("STAGE0_STATE=")]
        if len(rows) != 1:
            raise RuntimeError("short Files state evidence missing")
        return json.loads(rows[0].removeprefix("STAGE0_STATE="))
    state = probe_state(transport, fixture)
    state.pop("_geometry", None)
    return state


def _active_window_frame(transport: Any) -> dict[str, Any]:
    code = """
import json,subprocess
active=subprocess.run(['xprop','-root','_NET_ACTIVE_WINDOW'],capture_output=True,text=True,check=True).stdout.strip().split()[-1]
active_value=int(active,16)
lines=subprocess.run(['wmctrl','-lGx'],capture_output=True,text=True,check=True).stdout.splitlines()
matches=[]
for line in lines:
 parts=line.split(None,8)
 if len(parts)==9 and int(parts[0],16)==active_value:
  x,y,w,h=map(int,parts[2:6]); matches.append({'window_id':parts[0],'x':x,'y':y,'width':w,'height':h,'window_class':parts[6],'window_line':line})
assert len(matches)==1, 'active window frame missing or ambiguous'
print('STAGE0_ACTIVE_FRAME='+json.dumps(matches[0],sort_keys=True))
""".strip()
    output = str(transport.execute_argv(["python3", "-c", code]).get("output", ""))
    rows = [
        line
        for line in output.splitlines()
        if line.startswith("STAGE0_ACTIVE_FRAME=")
    ]
    if len(rows) != 1:
        raise RuntimeError("active-window frame evidence missing")
    return json.loads(rows[0].removeprefix("STAGE0_ACTIVE_FRAME="))


def _passive_rebind_active_geometry(
    transport: Any,
    task: Stage0SourceTask,
    fixture: Any,
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    before = _active_window_frame(transport)
    token = _window_token(task)
    if token not in before["window_line"]:
        raise RuntimeError(f"passive geometry target mismatch for {task.id}: {before}")
    x, y, width, height = (
        int(before["x"]),
        int(before["y"]),
        int(before["width"]),
        int(before["height"]),
    )
    if x < 0 or y < 0 or width <= 500 or height <= 400:
        raise RuntimeError(f"active partner is not visibly mapped: {before}")
    if task.app == "writer":
        geometry = {"editor": (x + width // 2, y + int(height * 0.58))}
    elif task.app == "calc":
        if width <= 1000 or height <= 600:
            raise RuntimeError(f"active Calc geometry is not normalized: {before}")
        geometry = {"cell": (x + 55, y + 84)}
    elif task.app == "files":
        if width <= 700 or height <= 450:
            raise RuntimeError(f"active Files geometry is not normalized: {before}")
        geometry = {
            "decoy": (x + 250, y + 15),
            "destination": (x + 250, y + 63),
            "source": (x + 250, y + 131),
            "moved": (x + 250, y + 15),
        }
    elif task.app == "chrome":
        state = probe_state(transport, fixture)
        raw = state.get("_geometry")
        required = {"nav", "decoy_nav", "toggle", "decoy_toggle", "scroll_surface"}
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise RuntimeError(f"passive Chrome geometry is incomplete: {raw}")
        geometry = {
            name: (int(point[0]), int(point[1]))
            for name, point in raw.items()
            if name in required
        }
    elif task.app == "vscode":
        geometry = {"editor": (x + width // 2, y + height // 2)}
    else:  # pragma: no cover - loader fixes the app set
        raise RuntimeError(f"unsupported passive geometry app: {task.app}")
    after = _active_window_frame(transport)
    if before["window_id"].lower() != after["window_id"].lower():
        raise RuntimeError(
            f"passive geometry probe changed the active window: {before} -> {after}"
        )
    return geometry, {
        "probe": "active_window_geometry_read_only_v1",
        "activation_commands": 0,
        "active_before": before,
        "active_after": after,
    }


_EXPECTED_SWITCH_EVENTS = [
    {"kind": "key_down", "args": ["AltLeft"]},
    {"kind": "key_down", "args": ["Tab"]},
    {"kind": "key_up", "args": ["Tab"]},
    {"kind": "key_up", "args": ["AltLeft"]},
]
_SWITCH_REBIND_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "record_id",
    "record_sha256",
    "task_id",
    "target_task_sha256",
    "source_task_sha256s",
    "fixture_sha256",
    "record_semantic_step",
    "app",
    "arm",
    "policy_visible",
    "input",
    "symbolic_switch_payload_sha256",
    "exact_dispatched_events",
    "exact_dispatched_event_sha256",
    "action_class",
    "dispatch_status",
    "atomic_ok",
    "active_before",
    "active_after",
    "target_token",
    "partner_geometry_binding",
    "partner_geometry_sha256",
    "geometry_binding_phase",
    "geometry_binding_source",
    "geometry_probe_evidence",
    "switch_rebind_receipt_sha256",
}

_ACTIVE_WINDOW_FIELDS = {"window_id", "window_line"}
_ACTIVE_FRAME_FIELDS = {
    "window_id",
    "x",
    "y",
    "width",
    "height",
    "window_class",
    "window_line",
}
_GEOMETRY_EVIDENCE_FIELDS = {
    "probe",
    "activation_commands",
    "active_before",
    "active_after",
}
_GEOMETRY_FIELDS_BY_APP = {
    "writer": {"editor"},
    "calc": {"cell"},
    "files": {"decoy", "destination", "source", "moved"},
    "chrome": {"nav", "decoy_nav", "toggle", "decoy_toggle", "scroll_surface"},
    "vscode": {"editor"},
}


def _verify_window_object(
    value: Any, *, frame: bool, label: str
) -> dict[str, Any]:
    expected = _ACTIVE_FRAME_FIELDS if frame else _ACTIVE_WINDOW_FIELDS
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"{label} schema mismatch")
    if (
        not isinstance(value["window_id"], str)
        or not isinstance(value["window_line"], str)
        or not value["window_line"]
    ):
        raise RuntimeError(f"{label} identity mismatch")
    try:
        parsed_window_id = int(value["window_id"], 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} window ID mismatch") from exc
    if not value["window_id"].lower().startswith("0x") or parsed_window_id <= 0:
        raise RuntimeError(f"{label} window ID mismatch")
    if frame:
        if (
            not isinstance(value["window_class"], str)
            or not value["window_class"]
            or any(
                not isinstance(value[field], int)
                or isinstance(value[field], bool)
                for field in ("x", "y", "width", "height")
            )
            or value["x"] < 0
            or value["y"] < 0
            or value["width"] <= 0
            or value["height"] <= 0
        ):
            raise RuntimeError(f"{label} frame mismatch")
    return value


def _verify_geometry_object(value: Any, app: str) -> dict[str, Any]:
    expected = _GEOMETRY_FIELDS_BY_APP[app]
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("switch/rebind app-specific geometry schema mismatch")
    if any(
        not isinstance(point, list)
        or len(point) != 2
        or any(
            not isinstance(coordinate, int) or isinstance(coordinate, bool)
            for coordinate in point
        )
        for point in value.values()
    ):
        raise RuntimeError("switch/rebind geometry coordinate mismatch")
    return value


def _verify_switch_rebind_receipt(
    receipt: dict[str, Any], record: Stage0Record, task: Stage0SourceTask
) -> None:
    if set(receipt) != _SWITCH_REBIND_RECEIPT_FIELDS:
        raise RuntimeError("switch/rebind receipt schema mismatch")
    unsigned = dict(receipt)
    seal = unsigned.pop("switch_rebind_receipt_sha256")
    if seal != sha256_value(unsigned):
        raise RuntimeError("switch/rebind receipt seal mismatch")
    expected_target_token = _window_token(task)
    if (
        receipt["schema_version"] != 1
        or receipt["receipt_type"] != "stage0_visible_switch_passive_rebind_v1"
        or receipt["record_id"] != record.id
        or receipt["record_sha256"] != record.record_sha256
        or receipt["task_id"] != task.id
        or receipt["target_task_sha256"] != task.task_sha256
        or receipt["source_task_sha256s"]
        != [source.task_sha256 for source in record.component_tasks]
        or receipt["fixture_sha256"] != task.fixture_sha256
        or receipt["record_semantic_step"] != 2
        or receipt["app"] != task.app
        or receipt["arm"] != "native_absolute_control"
        or not isinstance(receipt["target_token"], str)
        or not receipt["target_token"]
        or receipt["target_token"] != expected_target_token
    ):
        raise RuntimeError("switch/rebind receipt identity binding mismatch")
    if (
        receipt["exact_dispatched_events"] != _EXPECTED_SWITCH_EVENTS
        or receipt["exact_dispatched_event_sha256"]
        != sha256_value(_EXPECTED_SWITCH_EVENTS)
        or receipt["symbolic_switch_payload_sha256"]
        != sha256_value(compile_visible_app_switch_native())
    ):
        raise RuntimeError("switch/rebind exact-event binding mismatch")
    if (
        receipt["policy_visible"] is not True
        or receipt["input"] != {"action": "key", "keys": ["AltLeft", "Tab"]}
        or receipt["action_class"] != "key_chord"
        or receipt["dispatch_status"] != "ok"
        or receipt["atomic_ok"] is not True
        or receipt["geometry_binding_phase"]
        != "after_partner_active_attestation"
        or receipt["geometry_binding_source"]
        != "fresh_passive_post_switch_probe"
    ):
        raise RuntimeError("switch/rebind execution contract mismatch")
    if receipt["partner_geometry_sha256"] != sha256_value(
        receipt["partner_geometry_binding"]
    ):
        raise RuntimeError("switch/rebind geometry seal mismatch")
    _verify_window_object(
        receipt["active_before"], frame=False, label="switch active-before"
    )
    _verify_window_object(
        receipt["active_after"], frame=False, label="switch active-after"
    )
    _verify_geometry_object(receipt["partner_geometry_binding"], task.app)
    evidence = receipt["geometry_probe_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != _GEOMETRY_EVIDENCE_FIELDS:
        raise RuntimeError("switch/rebind geometry evidence schema mismatch")
    evidence_before = _verify_window_object(
        evidence["active_before"], frame=True, label="geometry active-before"
    )
    evidence_after = _verify_window_object(
        evidence["active_after"], frame=True, label="geometry active-after"
    )
    if (
        evidence.get("probe") != "active_window_geometry_read_only_v1"
        or evidence.get("activation_commands") != 0
        or int(receipt["active_after"]["window_id"], 16)
        != int(evidence_before["window_id"], 16)
        or int(evidence_before["window_id"], 16)
        != int(evidence_after["window_id"], 16)
        or receipt["target_token"] not in receipt["active_after"]["window_line"]
        or receipt["target_token"] not in evidence_before["window_line"]
        or receipt["target_token"] not in evidence_after["window_line"]
    ):
        raise RuntimeError("switch/rebind passive-active evidence mismatch")


def _switch_rebind_receipt(
    record: Stage0Record,
    task: Stage0SourceTask,
    switch_payload: dict[str, Any],
    switch: Any,
    active_before: dict[str, Any],
    active_after: dict[str, Any],
    geometry: dict[str, tuple[int, int]],
    geometry_evidence: dict[str, Any],
) -> dict[str, Any]:
    dispatched_events = [
        {"kind": operation.kind, "args": list(operation.args)}
        for operation in switch.operations
    ]
    if dispatched_events != _EXPECTED_SWITCH_EVENTS:
        raise RuntimeError(
            f"canonical visible-switch event drift: {dispatched_events}"
        )
    if int(active_after["window_id"], 16) != int(
        geometry_evidence["active_before"]["window_id"], 16
    ):
        raise RuntimeError("switch attestation and passive binding target disagree")
    geometry_payload = {
        name: list(point) for name, point in sorted(geometry.items())
    }
    payload = {
        "schema_version": 1,
        "receipt_type": "stage0_visible_switch_passive_rebind_v1",
        "record_id": record.id,
        "record_sha256": record.record_sha256,
        "task_id": task.id,
        "target_task_sha256": task.task_sha256,
        "source_task_sha256s": [
            source.task_sha256 for source in record.component_tasks
        ],
        "fixture_sha256": task.fixture_sha256,
        "record_semantic_step": 2,
        "app": task.app,
        "arm": switch.adapter,
        "policy_visible": True,
        "input": {"action": "key", "keys": ["AltLeft", "Tab"]},
        "symbolic_switch_payload_sha256": sha256_value(switch_payload),
        "exact_dispatched_events": dispatched_events,
        "exact_dispatched_event_sha256": sha256_value(dispatched_events),
        "action_class": switch.action_class,
        "dispatch_status": switch.executor_dispatch_status,
        "atomic_ok": bool(
            switch.atomic_state and switch.atomic_state.get("ok") is True
        ),
        "active_before": active_before,
        "active_after": active_after,
        "target_token": _window_token(task),
        "partner_geometry_binding": geometry_payload,
        "partner_geometry_sha256": sha256_value(geometry_payload),
        "geometry_binding_phase": "after_partner_active_attestation",
        "geometry_binding_source": "fresh_passive_post_switch_probe",
        "geometry_probe_evidence": geometry_evidence,
    }
    receipt = {
        **payload,
        "switch_rebind_receipt_sha256": sha256_value(payload),
    }
    _verify_switch_rebind_receipt(receipt, record, task)
    return receipt


def _dispatch_multi_program(
    transport: Any,
    task: Stage0SourceTask,
    geometry: dict[str, tuple[int, int]],
    *,
    near_miss: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executor = NativeAbsoluteExecutor(transport)
    receipts: list[dict[str, Any]] = []
    for turn_index, payload in enumerate(
        compile_multi_native(task, geometry, near_miss=near_miss)
    ):
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
                    "semantic_step": 1,
                    "action_class": result.action_class,
                    "operation_count": len(result.operations),
                    "dispatch_status": result.executor_dispatch_status,
                    "atomic_ok": bool(
                        result.atomic_state and result.atomic_state.get("ok") is True
                    ),
                }
            )
            transport.wait(0.5)
        transport.wait(0.75)
    return receipts, []


def _dispatch_single_near_miss(
    transport: Any,
    task: Stage0SourceTask,
    geometry: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executor = NativeAbsoluteExecutor(transport)
    receipts: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    if task.app == "vscode":
        actions = (
            {"action": "left_click", "coordinate": list(geometry["editor"])},
            {"action": "key", "keys": ["ControlLeft", "KeyA"]},
            {"action": "type", "text": str(task.near_miss["text"])},
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
    indexed_turns = tuple(enumerate(build_trajectory(fixture, near_miss=True).turns))
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
        if semantic_step < task.semantic_steps and task.app in {"files", "chrome"}:
            rebound_state = probe_state(transport, fixture)
            geometry = probe_geometry(transport, fixture, rebound_state)
            bindings.append(
                {
                    "completed_semantic_step": semantic_step,
                    "geometry": {key: list(value) for key, value in geometry.items()},
                }
            )
    transport.wait(2.0)
    return receipts, bindings


def _vm_repetition(
    session: KvmFixtureSession,
    record: Stage0Record,
    repetition_index: int,
    *,
    near_miss_order: int | None = None,
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
            switch_payload = compile_visible_app_switch_native()
            switch_operation = dict(switch_payload["operations"][0])
            switch_operation["action"] = "key"
            switch = NativeAbsoluteExecutor(transport).execute(switch_operation)
            transport.wait(1.5)
            after = _active_window(transport)
            target_token = _window_token(task)
            if target_token not in after["window_line"]:
                raise RuntimeError(
                    f"visible Alt+Tab did not activate ordered partner {target_token!r}: {after}"
                )
            partner_geometry, geometry_evidence = _passive_rebind_active_geometry(
                transport, task, component["fixture"]
            )
            component["geometry"] = partner_geometry
            switch_rows.append(
                _switch_rebind_receipt(
                    record,
                    task,
                    switch_payload,
                    switch,
                    before,
                    after,
                    partner_geometry,
                    geometry_evidence,
                )
            )
        is_near_miss = near_miss_order == order
        if record.mode == "multi":
            actions, bindings = _dispatch_multi_program(
                transport,
                task,
                component["geometry"],
                near_miss=is_near_miss,
            )
        elif is_near_miss:
            actions, bindings = _dispatch_single_near_miss(
                transport, task, component["geometry"]
            )
        else:
            actions, bindings = _dispatch_gold(
                transport, task, component["geometry"]
            )
        action_rows.append(
            {
                "record_semantic_step": order if record.mode == "multi" else None,
                "source_task_id": task.id,
                "app": task.app,
                "source_semantic_steps": task.semantic_steps,
                "near_miss_program": is_near_miss,
                "actions": actions,
                "runtime_bindings": bindings,
            }
        )
    final_states = [
        _probe_source(transport, component["task"], component["fixture"])
        for component in components
    ]
    final_result = evaluate_composed_in_fresh_process(record, final_states)
    near_miss_exact = None
    trial_state_exact = None
    if near_miss_order is not None:
        target_index = near_miss_order - 1
        expected_states = [
            scripted_state(task, near_miss=index == target_index)
            for index, task in enumerate(record.component_tasks)
        ]
        near_miss_exact = final_states[target_index] == expected_states[target_index]
        trial_state_exact = final_states == expected_states
    audit = transport.audit
    dispatch_ok = all(
        row["dispatch_status"] == "ok" and row["atomic_ok"]
        for component in action_rows
        for row in component["actions"]
    ) and all(
        row["dispatch_status"] == "ok" and row["atomic_ok"] for row in switch_rows
    )
    expected_outcome = (
        final_result.MOUSE_SOLVED
        if near_miss_order is None
        else not final_result.MOUSE_SOLVED and trial_state_exact is True
    )
    passed = bool(
        initial_result.oracle_status == "ok"
        and not initial_result.MOUSE_SOLVED
        and final_result.oracle_status == "ok"
        and expected_outcome
        and dispatch_ok
        and not audit.held_buttons
        and not audit.held_keys
        and len(switch_rows) == (1 if record.mode == "multi" else 0)
    )
    return {
        "repetition_index": repetition_index,
        "trial_kind": "gold" if near_miss_order is None else "component_near_miss",
        "near_miss_order": near_miss_order,
        "status": "pass" if passed else "fail",
        "duration_s": round(time.monotonic() - started, 3),
        "provider_reset_receipt": asdict(provider_reset),
        "reset_signatures": reset_signatures,
        "setup_activation": setup_activation,
        "readiness": [component["readiness"] for component in components],
        "setup_preparations": [component["setup_preparation"] for component in components],
        "initial_reject": not initial_result.MOUSE_SOLVED,
        "initial_oracle_pid": initial_result.oracle_pid,
        "gold_pass": final_result.MOUSE_SOLVED,
        "near_miss_reject": not final_result.MOUSE_SOLVED if near_miss_order is not None else None,
        "near_miss_exact": near_miss_exact,
        "trial_state_exact": trial_state_exact,
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
    def run_trial(
        repetition_index: int, *, near_miss_order: int | None
    ) -> dict[str, Any]:
        try:
            return _vm_repetition(
                session,
                record,
                repetition_index,
                near_miss_order=near_miss_order,
            )
        except Exception as exc:
            return {
                "repetition_index": repetition_index,
                "trial_kind": (
                    "gold" if near_miss_order is None else "component_near_miss"
                ),
                "near_miss_order": near_miss_order,
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    near_miss_trials = [
        run_trial(order, near_miss_order=order)
        for order in range(1, len(record.component_tasks) + 1)
    ]
    repetitions: list[dict[str, Any]] = []
    for repetition_index in range(1, REPEATABILITY_RUNS + 1):
        row = run_trial(repetition_index, near_miss_order=None)
        repetitions.append(row)
        if row["status"] != "pass":
            break
    reset_signatures = [row.get("reset_signatures") for row in repetitions]
    reset_repeatable = bool(
        len(reset_signatures) == REPEATABILITY_RUNS
        and reset_signatures[0] is not None
        and all(value == reset_signatures[0] for value in reset_signatures)
    )
    all_trials = near_miss_trials + repetitions
    all_reset_signatures = [row.get("reset_signatures") for row in all_trials]
    all_trial_resets_repeatable = bool(
        all_reset_signatures
        and all_reset_signatures[0] is not None
        and all(value == all_reset_signatures[0] for value in all_reset_signatures)
    )
    reset_receipts = [row.get("provider_reset_receipt", {}) for row in all_trials]
    distinct_resets = bool(
        len(reset_receipts) == REPEATABILITY_RUNS + len(record.component_tasks)
        and None not in {row.get("reset_id") for row in reset_receipts}
        and len({row.get("reset_id") for row in reset_receipts}) == len(reset_receipts)
        and len({row.get("new_generation_id") for row in reset_receipts}) == len(reset_receipts)
    )
    live_near_misses_pass = bool(
        len(near_miss_trials) == len(record.component_tasks)
        and all(
            row["status"] == "pass"
            and row.get("near_miss_reject") is True
            and row.get("near_miss_exact") is True
            and row.get("trial_state_exact") is True
            for row in near_miss_trials
        )
    )
    passed = bool(
        len(repetitions) == REPEATABILITY_RUNS
        and all(row["status"] == "pass" for row in repetitions)
        and reset_repeatable
        and distinct_resets
        and live_near_misses_pass
        and all_trial_resets_repeatable
    )
    return {
        "record_id": record.id,
        "anchor_app": record.anchor_app,
        "mode": record.mode,
        "difficulty": record.difficulty,
        "record_sha256": record.record_sha256,
        "program_budget": record.program_budget,
        "status": "pass" if passed else "fail",
        "repeatability_runs_required": REPEATABILITY_RUNS,
        "repeatability_runs_observed": len(repetitions),
        "reset_signatures_repeatable": reset_repeatable,
        "all_trial_reset_signatures_repeatable": all_trial_resets_repeatable,
        "provider_resets_distinct": distinct_resets,
        "live_each_component_near_miss_reject": live_near_misses_pass,
        "near_miss_trials": near_miss_trials,
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
