#!/usr/bin/env python3
"""Prepared, launch-disabled runner for true roadmap stage-2 closed loop."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "PROTOCOL_DRAFT.json"
CHUNK0_AUTHORIZATION_PATH = HERE / "CHUNK0_AUTHORIZATION.json"
DYNAMIC_GUEST_APP = HERE / "dynamic_guest_app.py"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from contract import Contract, strict_schema_ok, unit_range_ok  # type: ignore  # noqa: E402
from evaluate import _call_model  # type: ignore  # noqa: E402

sys.path.insert(0, str(HERE.parent / "proper_vm_stage2"))
from gate import (  # type: ignore  # noqa: E402
    GateError,
    actuation_plan,
    load_cells,
    rgb_sha256,
    sha256_file,
)
from live_smoke import (  # type: ignore  # noqa: E402
    _assert_no_guest_processes,
    _guest_exec,
    _plan_code,
    _stop_guest_process,
    _x_client_inventory,
    leased_ports,
)

from closed_loop_contract import (  # noqa: E402
    AttemptEvidence,
    ClosedLoopState,
    advance,
    initial_state,
    reference_png,
    request_seed,
)


ARM_NAMES = ("absolute_matched_control", "normalized_relative", "raw_relative")
EPISODES_PER_VM_CHUNK = 5
VM_CHUNK_COUNT = 16
SENTINEL_REQUEST_SLOTS_PER_EPISODE = 4
MULTISTEP_REQUEST_SLOTS_PER_EPISODE = 12
REQUEST_SLOTS_PER_EPISODE = (
    SENTINEL_REQUEST_SLOTS_PER_EPISODE + MULTISTEP_REQUEST_SLOTS_PER_EPISODE
)
MAX_REQUEST_SLOTS_PER_VM = EPISODES_PER_VM_CHUNK * REQUEST_SLOTS_PER_EPISODE
MAX_X_CLIENT_SLACK = 8
GUEST_SOURCE = "/tmp/roadmap_stage2_dynamic_guest.py"
GUEST_IMAGE = "/tmp/roadmap_stage2_dynamic_scene.png"
GUEST_CONFIG = "/tmp/roadmap_stage2_dynamic_config.json"
GUEST_COMMAND = "/tmp/roadmap_stage2_dynamic_command.json"
GUEST_STATE = "/tmp/roadmap_stage2_dynamic_state.json"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def load_protocol(path: Path, *, require_authorized: bool | None) -> dict[str, Any]:
    protocol = _load_object(path)
    authorized = protocol.get("launch_authorized")
    status = protocol.get("status")
    if status == "design_draft_not_launch_authorized" and authorized is False:
        pass
    elif status == "authorized_ready" and authorized is True:
        pass
    else:
        raise GateError("stage-2 protocol status/authorization mismatch")
    if require_authorized is True and authorized is not True:
        raise GateError("roadmap stage-2 GPU launch is not authorized")
    if require_authorized is False and authorized is not False:
        raise GateError("preparation gate unexpectedly authorizes stage-2 launch")
    scope = protocol.get("scope", {})
    required_scope = {
        "roadmap_stage": "2",
        "model_actions_change_next_observation": True,
        "model_actions_change_next_cursor": True,
        "target_advances_only_after_verified_hit": True,
    }
    if any(scope.get(key) != value for key, value in required_scope.items()):
        raise GateError("roadmap stage-2 scope drift")
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    source_paths = {
        "CHUNK0_AUTHORIZATION.json": CHUNK0_AUTHORIZATION_PATH,
        "aggregate.py": HERE / "aggregate.py",
        "closed_loop_contract.py": HERE / "closed_loop_contract.py",
        "dynamic_guest_app.py": HERE / "dynamic_guest_app.py",
        "live_dynamic_smoke.py": HERE / "live_dynamic_smoke.py",
        "synthetic_contract.py": HERE.parent / "contract.py",
        "synthetic_evaluate.py": HERE.parent / "evaluate.py",
        "stage1_5_gate.py": HERE.parent / "proper_vm_stage2" / "gate.py",
        "stage1_5_live_smoke.py": HERE.parent / "proper_vm_stage2" / "live_smoke.py",
        "model_readiness.py": HERE.parents[1] / "relative_factorial" / "readiness.py",
        "run_arm_stage.sh": HERE / "run_arm_stage.sh",
        "runner.py": HERE / "runner.py",
        "proper_vm_roadmap_stage2_dynamic_smoke_prepared.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_dynamic_smoke_prepared.toml"
        ),
        "proper_vm_roadmap_stage2_absolute_prepared.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_absolute_prepared.toml"
        ),
        "proper_vm_roadmap_stage2_normalized_prepared.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_normalized_prepared.toml"
        ),
        "proper_vm_roadmap_stage2_raw_prepared.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_raw_prepared.toml"
        ),
        "proper_vm_roadmap_stage2_absolute_one_cell_preflight.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_absolute_one_cell_preflight.toml"
        ),
        "proper_vm_roadmap_stage2_normalized_one_cell_preflight.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_normalized_one_cell_preflight.toml"
        ),
        "proper_vm_roadmap_stage2_raw_one_cell_preflight.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_raw_one_cell_preflight.toml"
        ),
        "proper_vm_roadmap_stage2_absolute_chunk0_pilot.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_absolute_chunk0_pilot.toml"
        ),
        "proper_vm_roadmap_stage2_normalized_chunk0_pilot.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_normalized_chunk0_pilot.toml"
        ),
        "proper_vm_roadmap_stage2_raw_chunk0_pilot.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_roadmap_stage2_raw_chunk0_pilot.toml"
        ),
    }
    if set(protocol.get("implementation_sources", {})) != set(source_paths):
        raise GateError("stage-2 implementation source set drift")
    for name, path in source_paths.items():
        if sha256_file(path) != protocol["implementation_sources"][name]:
            raise GateError(f"stage-2 implementation source hash drift: {name}")
    vm = protocol.get("vm", {})
    provider = Path(str(vm.get("provider_source", "")))
    if not provider.is_file() or sha256_file(provider) != vm.get("provider_sha256"):
        raise GateError("stage-2 provider evidence drift")
    if not Path(str(vm.get("qcow", ""))).is_file():
        raise GateError("stage-2 qcow is absent")
    qemu = Path(str(vm.get("qemu_bin", "")))
    if not qemu.is_file() or not os.access(qemu, os.X_OK):
        raise GateError("stage-2 QEMU is absent/nonexecutable")
    resume = protocol.get("partial_resumability", {})
    if resume.get("atomic_units") != [
        "one complete single-step sentinel cell",
        "one complete multi-step episode",
    ]:
        raise GateError("stage-2 resumability atomic-unit drift")
    if resume.get("mid_episode_resume_allowed") is not False:
        raise GateError("stage-2 mid-episode resume must remain forbidden")
    upper = protocol.get("resource_upper_bound_draft", {})
    if upper.get("single_step_requests_per_arm") != 320 or upper.get(
        "multi_step_requests_per_arm"
    ) != 960:
        raise GateError("stage-2 request upper-bound drift")
    if upper.get("total_model_requests_all_arms") != 3840:
        raise GateError("stage-2 total request upper-bound drift")
    chunking = protocol.get("bounded_vm_chunking", {})
    required_chunking = {
        "episode_aligned": True,
        "episodes_per_chunk": EPISODES_PER_VM_CHUNK,
        "chunks_per_arm": VM_CHUNK_COUNT,
        "atomic_units_per_chunk": 25,
        "maximum_request_slots_per_chunk": MAX_REQUEST_SLOTS_PER_VM,
        "fresh_vm_per_chunk": True,
        "full_arm_launch_authorized": False,
    }
    if any(chunking.get(key) != value for key, value in required_chunking.items()):
        raise GateError("stage-2 bounded-VM chunk contract drift")
    model_smoke = protocol.get("model_kvm_one_cell_preflight", {})
    if (
        model_smoke.get("cell") != {"episode_index": 0, "target_index": 0}
        or model_smoke.get("requests_per_arm") != 1
        or model_smoke.get("independent_arms") != list(ARM_NAMES)
    ):
        raise GateError("stage-2 model/KVM one-cell preflight drift")
    smoke = protocol.get("dynamic_kvm_smoke", {})
    if smoke.get("status") == "prepared_not_run":
        if protocol.get("launch_authorized") is not False:
            raise GateError("stage-2 launch cannot precede dynamic KVM smoke")
    elif smoke.get("status") == "pass":
        manifest_path = Path(str(smoke.get("manifest", "")))
        if not manifest_path.is_file() or sha256_file(manifest_path) != smoke.get(
            "manifest_sha256"
        ):
            raise GateError("stage-2 dynamic KVM smoke evidence drift")
        manifest = _load_object(manifest_path)
        required = {
            "artifact_type": "synthetic_proper_vm_roadmap_stage2_dynamic_kvm_smoke",
            "status": "pass",
            "cpu_only": True,
            "gpu_used": False,
            "semantic_sequences": 3,
            "retry_exhaustion_sequences": 1,
            "provider_sha256": vm["provider_sha256"],
            "guest_teardown_proven": True,
            "exact_button_counts_verified": True,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise GateError("stage-2 dynamic KVM smoke contract drift")
    else:
        raise GateError("unknown stage-2 dynamic KVM smoke state")


def _validate_chunk0_authorization(protocol: dict[str, Any]) -> dict[str, Any]:
    authorization = _load_object(CHUNK0_AUTHORIZATION_PATH)
    required = {
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_chunk0_authorization",
        "status": "authorized_ready",
        "chunk_index": 0,
        "episode_start_inclusive": 0,
        "episode_end_exclusive": EPISODES_PER_VM_CHUNK,
        "episodes_per_arm": EPISODES_PER_VM_CHUNK,
        "atomic_units_per_arm": 25,
        "maximum_request_slots_per_arm": MAX_REQUEST_SLOTS_PER_VM,
        "fresh_vm_per_arm": True,
        "chunks_1_15_launch_authorized": False,
        "full_arm_launch_authorized": False,
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise GateError("stage-2 chunk-0 authorization scope drift")
    gate = protocol.get("chunk0_closed_loop_pilot", {})
    if (
        gate.get("status") != "authorized_ready"
        or gate.get("launch_authorized") is not True
        or gate.get("chunk_index") != 0
        or gate.get("chunks_1_15_launch_authorized") is not False
        or gate.get("full_arm_launch_authorized") is not False
        or gate.get("authorization_artifact_sha256")
        != sha256_file(CHUNK0_AUTHORIZATION_PATH)
    ):
        raise GateError("stage-2 chunk-0 protocol authorization drift")
    records = authorization.get("preflight_manifests")
    if not isinstance(records, dict) or set(records) != set(ARM_NAMES):
        raise GateError("stage-2 chunk-0 preflight evidence set drift")
    preflight_protocol_hash = authorization.get("preflight_protocol_sha256")
    for arm_name, record in records.items():
        if not isinstance(record, dict):
            raise GateError("stage-2 chunk-0 preflight record is not an object")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise GateError(f"stage-2 chunk-0 preflight evidence drift: {arm_name}")
        manifest = _load_object(path)
        expected = {
            "status": "pass",
            "arm": arm_name,
            "protocol_sha256": preflight_protocol_hash,
            "episode_index": 0,
            "target_index": 0,
            "model_requests": 1,
            "model_generated": True,
            "history": [],
            "guest_hit": True,
            "button_presses": 1,
            "button_releases": 1,
            "guest_teardown_proven": True,
            "host_ports_released_after_vm": True,
            "request_errors": 0,
            "infrastructure_mismatches": 0,
            "full_arm_launch_authorized": False,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise GateError(f"stage-2 chunk-0 preflight gate failed: {arm_name}")
        if manifest.get("final_x_clients", 10**9) > manifest.get(
            "baseline_x_clients", -10**9
        ) + manifest.get("x_client_slack_bound", -1):
            raise GateError(f"stage-2 chunk-0 X-client gate failed: {arm_name}")
    return authorization


def validate_arm_inputs(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    arm = protocol["arms_draft"].get(args.arm)
    if not isinstance(arm, dict) or args.arm not in ARM_NAMES:
        raise GateError(f"unknown stage-2 arm: {args.arm}")
    root = Path(arm["checkpoint_root"])
    if args.model_dir.resolve() != (root / "hf").resolve():
        raise GateError("stage-2 model directory drift")
    manifest_path = root / arm["checkpoint_manifest"]
    if sha256_file(manifest_path) != arm["checkpoint_manifest_sha256"]:
        raise GateError("stage-2 checkpoint manifest hash drift")
    manifest = _load_object(manifest_path)
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in arm["expected_manifest"].items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise GateError(f"stage-2 checkpoint manifest mismatch: {mismatches}")
    hf = root / "hf"
    if sha256_file(hf / "config.json") != arm["config_sha256"]:
        raise GateError("stage-2 model config hash drift")
    weights = hf / "model.safetensors"
    if weights.stat().st_size != arm["model_weights_bytes"]:
        raise GateError("stage-2 model weight size drift")
    if sha256_file(weights) != arm["model_weights_sha256"]:
        raise GateError("stage-2 model weight hash drift")
    return arm


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    launch_scope = getattr(args, "launch_scope", "full")
    if launch_scope == "full":
        protocol = load_protocol(args.protocol, require_authorized=True)
    elif launch_scope == "one_cell_preflight":
        protocol = load_protocol(args.protocol, require_authorized=False)
        model_smoke = protocol.get("model_kvm_one_cell_preflight", {})
        if (
            model_smoke.get("status") != "authorized_ready"
            or model_smoke.get("launch_authorized") is not True
        ):
            raise GateError("stage-2 one-cell model/KVM preflight is not authorized")
    elif launch_scope == "chunk0_pilot":
        protocol = load_protocol(args.protocol, require_authorized=False)
        _validate_chunk0_authorization(protocol)
    else:
        raise GateError(f"unknown stage-2 launch scope: {launch_scope}")
    validate_protocol(protocol)
    arm = validate_arm_inputs(args, protocol)
    vm = protocol["vm"]
    if args.provider_source.resolve() != Path(vm["provider_source"]).resolve():
        raise GateError("stage-2 provider path drift")
    if sha256_file(args.provider_source) != vm["provider_sha256"]:
        raise GateError("stage-2 provider hash drift")
    if args.qcow.resolve() != Path(vm["qcow"]).resolve() or not args.qcow.is_file():
        raise GateError("stage-2 qcow path drift")
    if args.qemu_bin.resolve() != Path(vm["qemu_bin"]).resolve():
        raise GateError("stage-2 QEMU path drift")
    if not args.qemu_bin.is_file() or not os.access(args.qemu_bin, os.X_OK):
        raise GateError("stage-2 QEMU executable missing")
    if not args.osworld_root.is_dir():
        raise GateError("stage-2 OSWorld root missing")
    return protocol, arm


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cursor(controller: Any) -> tuple[int, int]:
    output = _guest_exec(controller, "import pyautogui; print(list(pyautogui.position()))")
    value = json.loads(output.splitlines()[-1])
    return int(value[0]), int(value[1])


def _wait_for_exact_rgb(
    controller: Any,
    expected_png: bytes,
    *,
    label: str,
    timeout_s: float = 10.0,
) -> str:
    """Wait for the asynchronous VNC frame to publish, without relaxing equality."""
    expected = rgb_sha256(expected_png)
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        last = rgb_sha256(controller.get_screenshot())
        if last == expected:
            return last
        time.sleep(0.1)
    raise GateError(
        f"{label} pixel mismatch after {timeout_s}s: expected {expected}, got {last}"
    )


def _read_guest_state(
    controller: Any,
    episode_revision: str,
    *,
    render_revision: str | None = None,
    minimum_releases: int = 0,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        output = _guest_exec(
            controller,
            f"import pathlib; p=pathlib.Path({GUEST_STATE!r}); print(p.read_text() if p.is_file() else '')",
        )
        if output:
            try:
                state = json.loads(output.splitlines()[-1])
            except json.JSONDecodeError:
                state = None
            last = state
            if (
                isinstance(state, dict)
                and state.get("episode_revision") == episode_revision
                and state.get("ready") is True
                and state.get("error") is None
                and int(state.get("button_releases", -1)) >= minimum_releases
                and (render_revision is None or state.get("render_revision") == render_revision)
            ):
                return state
        time.sleep(0.1)
    raise GateError(f"dynamic guest state timeout: {last}")


def _install_episode(
    controller: Any,
    *,
    episode_revision: str,
    targets: list[tuple[int, int, int, int]],
    cursor: tuple[int, int],
    png: bytes,
) -> int:
    _assert_no_guest_processes(controller, GUEST_SOURCE)
    source = base64.b64encode(DYNAMIC_GUEST_APP.read_bytes()).decode("ascii")
    image = base64.b64encode(png).decode("ascii")
    image_sha = __import__("hashlib").sha256(png).hexdigest()
    config = {
        "episode_revision": episode_revision,
        "screen": [1920, 1080],
        "targets": [list(bbox) for bbox in targets],
        "initial_cursor": list(cursor),
        "initial_image_sha256": image_sha,
        "image_path": GUEST_IMAGE,
        "command_path": GUEST_COMMAND,
        "state_path": GUEST_STATE,
    }
    encoded_config = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
    code = (
        "import base64,pathlib,subprocess,sys;"
        f"pathlib.Path({GUEST_SOURCE!r}).write_bytes(base64.b64decode({source!r}));"
        f"pathlib.Path({GUEST_IMAGE!r}).write_bytes(base64.b64decode({image!r}));"
        f"pathlib.Path({GUEST_CONFIG!r}).write_bytes(base64.b64decode({encoded_config!r}));"
        f"pathlib.Path({GUEST_COMMAND!r}).unlink(missing_ok=True);"
        f"pathlib.Path({GUEST_STATE!r}).unlink(missing_ok=True);"
        f"p=subprocess.Popen([sys.executable,{GUEST_SOURCE!r},{GUEST_CONFIG!r}],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "print(p.pid)"
    )
    output = _guest_exec(controller, code)
    pid = int(output.splitlines()[-1])
    try:
        _read_guest_state(controller, episode_revision)
        _guest_exec(
            controller,
            "import pyautogui; pyautogui.FAILSAFE=False; "
            f"pyautogui.moveTo({cursor[0]},{cursor[1]}); print(list(pyautogui.position()))",
        )
        if _cursor(controller) != cursor:
            raise GateError("dynamic guest failed initial cursor placement")
        _wait_for_exact_rgb(controller, png, label="dynamic initial screenshot")
        return pid
    except BaseException:
        _stop_guest_process(controller, pid, GUEST_SOURCE)
        raise


def _stop_episode(controller: Any, pid: int) -> None:
    try:
        _guest_exec(controller, "import pyautogui; pyautogui.mouseUp(button='left')")
    finally:
        _stop_guest_process(controller, pid, GUEST_SOURCE)


def _update_render(
    controller: Any,
    *,
    episode_revision: str,
    command_sequence: int,
    render_revision: str,
    state: ClosedLoopState,
    targets: list[tuple[int, int, int, int]],
    png: bytes,
) -> dict[str, Any]:
    import hashlib

    image = base64.b64encode(png).decode("ascii")
    command = {
        "episode_revision": episode_revision,
        "sequence": command_sequence,
        "target_index": state.target_index,
        "bbox": list(targets[state.target_index]),
        "cursor": list(state.cursor),
        "image_sha256": hashlib.sha256(png).hexdigest(),
        "render_revision": render_revision,
    }
    encoded = base64.b64encode(json.dumps(command).encode("utf-8")).decode("ascii")
    code = (
        "import base64,os,pathlib;"
        f"i=pathlib.Path({GUEST_IMAGE!r});it=i.with_suffix('.png.tmp');"
        f"it.write_bytes(base64.b64decode({image!r}));os.replace(it,i);"
        f"c=pathlib.Path({GUEST_COMMAND!r});ct=c.with_suffix('.json.tmp');"
        f"ct.write_bytes(base64.b64decode({encoded!r}));os.replace(ct,c)"
    )
    _guest_exec(controller, code)
    guest_state = _read_guest_state(
        controller, episode_revision, render_revision=render_revision
    )
    _wait_for_exact_rgb(controller, png, label="dynamic next observation")
    if _cursor(controller) != state.cursor:
        raise GateError("dynamic next-observation cursor mismatch")
    return guest_state


def _single_user_text(
    contract: Contract,
    semantic: str,
    cursor: tuple[int, int],
    target: tuple[int, int],
    preamble: bool,
) -> str:
    scene = {"cursor": list(cursor), "target_center": list(target)}
    return str(
        contract.rung2.build_user_text(
            contract.rung2.GRAMMARS[semantic], scene, False, preamble
        )
    )


def _serialized_tool_calls(raw: str) -> list[dict[str, Any]]:
    marker = " | tool_calls="
    if marker not in raw:
        return []
    try:
        value = json.loads(raw.split(marker, 1)[1])
    except json.JSONDecodeError as exc:
        raise GateError("model tool-call audit suffix is not valid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise GateError("model tool-call audit suffix is not a list of objects")
    return value


def _run_unit(
    *,
    controller: Any,
    client: OpenAI,
    served_model: str,
    contract: Contract,
    protocol: dict[str, Any],
    arm_name: str,
    arm: dict[str, Any],
    condition: str,
    episode_id: str,
    episode_index: int,
    targets: list[tuple[int, int, int, int]],
    target_centers: list[tuple[int, int]],
    initial_cursor: tuple[int, int],
) -> dict[str, Any]:
    import hashlib

    semantic = arm["semantic"]
    maximum = 1 if condition == "single_step_sentinel" else int(
        protocol["sampling"]["attempts_per_target"]
    )
    state = initial_state(episode_id, initial_cursor)
    episode_revision = hashlib.sha256(
        f"{arm_name}|{condition}|{episode_id}".encode("utf-8")
    ).hexdigest()
    initial_png = reference_png(contract, state, targets)
    pid = _install_episode(
        controller,
        episode_revision=episode_revision,
        targets=targets,
        cursor=initial_cursor,
        png=initial_png,
    )
    prior: list[str] = []
    rows: list[dict[str, Any]] = []
    command_sequence = 0
    releases = 0
    try:
        while not state.terminated:
            target_index = state.target_index
            target = target_centers[target_index]
            attempt = state.attempts_on_target + 1
            png = reference_png(contract, state, targets)
            screenshot = controller.get_screenshot()
            if rgb_sha256(screenshot) != rgb_sha256(png):
                raise GateError("pre-request dynamic screenshot drift")
            guest_state = _read_guest_state(
                controller, episode_revision, minimum_releases=releases
            )
            render_revision_before = guest_state.get("render_revision")
            if condition == "single_step_sentinel":
                user_text = _single_user_text(
                    contract, semantic, state.cursor, target, bool(arm["preamble"])
                )
            else:
                user_text = contract.user_text(
                    semantic,
                    state.cursor,
                    target,
                    target_index=target_index,
                    target_count=len(targets),
                    preamble=bool(arm["preamble"]),
                    prior=prior[-3:] or None,
                )
            seed = request_seed(condition, episode_id, target_index, attempt)
            raw, tool_calls, meta = _call_model(
                client,
                model=served_model,
                system=contract.system_prompt(semantic),
                user_text=user_text,
                png=screenshot,
                history=[],
                seed=seed,
                max_tokens=int(protocol["sampling"]["max_tokens"]),
            )
            parse_text = raw.split(" | tool_calls=", 1)[0]
            move = contract.parse(semantic, parse_text, tool_calls)
            schema_ok = strict_schema_ok(semantic, parse_text, move.coord)
            units_ok = unit_range_ok(semantic, move.coord)
            valid = bool(move.parse_ok and move.coord is not None and schema_ok and units_ok)
            endpoint = (
                contract.apply_coord(semantic, state.cursor, move.coord)
                if move.coord is not None
                else None
            )
            before = state
            if valid:
                assert endpoint is not None
                _guest_exec(
                    controller,
                    _plan_code(actuation_plan(semantic, "click", state.cursor, endpoint)),
                )
                releases += 1
                guest_state = _read_guest_state(
                    controller, episode_revision, minimum_releases=releases
                )
                actual_cursor = _cursor(controller)
                evidence = AttemptEvidence(
                    raw_output=raw,
                    parse_ok=True,
                    schema_ok=True,
                    unit_range_ok=True,
                    dispatched=True,
                    endpoint=endpoint,
                    actual_cursor_after=actual_cursor,
                    guest_hit=guest_state["last_hit"],
                )
            else:
                evidence = AttemptEvidence(
                    raw_output=raw,
                    parse_ok=bool(move.parse_ok and move.coord is not None),
                    schema_ok=schema_ok,
                    unit_range_ok=units_ok,
                    dispatched=False,
                    endpoint=endpoint,
                    actual_cursor_after=None,
                    guest_hit=None,
                )
            transition = advance(
                state, evidence, targets, max_attempts_per_target=maximum
            )
            state = transition.after
            if guest_state.get("target_index") != state.target_index:
                raise GateError("guest/host active target mismatch")
            if guest_state.get("completed") is not state.success:
                raise GateError("guest/host completion mismatch")
            if guest_state.get("down") is not False:
                raise GateError("guest button remained down")
            if int(guest_state.get("button_presses", -1)) != releases or int(
                guest_state.get("button_releases", -1)
            ) != releases:
                raise GateError("guest button press/release count mismatch")
            row = {
                "condition": condition,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "target_index": target_index,
                "attempt": attempt,
                "request_seed": seed,
                "history_length": len(prior[-3:]),
                "history_raw_output_sha256": [
                    hashlib.sha256(output.encode("utf-8")).hexdigest()
                    for output in prior[-3:]
                ],
                "user_text_sha256": hashlib.sha256(
                    user_text.encode("utf-8")
                ).hexdigest(),
                "raw_output": raw,
                "tool_calls": _serialized_tool_calls(raw),
                "completion_tokens": meta["completion_tokens"],
                "parse_ok": evidence.parse_ok,
                "schema_ok": schema_ok,
                "unit_range_ok": units_ok,
                "dispatched": evidence.dispatched,
                "coord": list(move.coord) if move.coord is not None else None,
                "endpoint": list(endpoint) if endpoint is not None else None,
                "cursor_before": list(before.cursor),
                "cursor_after": list(state.cursor),
                "active_bbox": list(targets[target_index]),
                "observation_rgb_sha256": rgb_sha256(png),
                "render_revision_before": render_revision_before,
                "guest_hit": transition.hit,
                "target_advanced": transition.target_advanced,
                "attempts_on_target_after": state.attempts_on_target,
                "terminated": state.terminated,
                "success": state.success,
                "terminal_reason": transition.terminal_reason,
                "guest_state": guest_state,
                "next_observation_rgb_sha256": None,
                "next_render_revision": None,
            }
            prior.append(raw)
            if not state.terminated:
                command_sequence += 1
                next_png = reference_png(contract, state, targets)
                render_revision = hashlib.sha256(
                    f"{episode_revision}|{command_sequence}|{state.target_index}|{state.cursor}".encode()
                ).hexdigest()
                next_guest_state = _update_render(
                    controller,
                    episode_revision=episode_revision,
                    command_sequence=command_sequence,
                    render_revision=render_revision,
                    state=state,
                    targets=targets,
                    png=next_png,
                )
                if next_guest_state.get("render_revision") != render_revision:
                    raise GateError("dynamic next render revision mismatch")
                row["next_observation_rgb_sha256"] = rgb_sha256(next_png)
                row["next_render_revision"] = render_revision
            rows.append(row)
    finally:
        _stop_episode(controller, pid)
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_complete_unit",
        "status": "complete",
        "condition": condition,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "arm": arm_name,
        "rows": rows,
        "summary": {
            "success": state.success,
            "terminated": state.terminated,
            "attempts_total": state.attempts_total,
            "target_hit_attempts": list(state.target_hit_attempts),
            "targets_reached": len(state.target_hit_attempts),
            "final_target_index": state.target_index,
            "final_cursor": list(state.cursor),
        },
    }


def _unit_path(root: Path, condition: str, episode_index: int, target_index: int | None) -> Path:
    if condition == "single_step_sentinel":
        assert target_index is not None
        return root / "units" / "single_step_sentinel" / f"{episode_index:03d}_{target_index:02d}.json"
    return root / "units" / "multi_step_closed_loop" / f"{episode_index:03d}.json"


def _copy_resume_units(resume_from: Path, out: Path, arm: str, protocol_hash: str) -> int:
    count = 0
    source = resume_from / "units"
    if not source.is_dir():
        raise GateError("resume artifact has no atomic units directory")
    if (resume_from / "arm_manifest.json").exists():
        raise GateError("completed arm must not be used as a partial resume source")
    for path in sorted(source.rglob("*.json")):
        value = _load_object(path)
        if not _basic_unit_trusted(value, arm, protocol_hash):
            raise GateError(f"untrusted resume unit: {path}")
        destination = out / "units" / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise GateError(f"resume unit collision: {destination}")
        shutil.copy2(path, destination)
        count += 1
    return count


def _basic_unit_trusted(value: dict[str, Any], arm: str, protocol_hash: str) -> bool:
    return bool(
        value.get("schema_version") == 1
        and value.get("artifact_type")
        == "synthetic_proper_vm_roadmap_stage2_complete_unit"
        and value.get("status") == "complete"
        and value.get("arm") == arm
        and value.get("protocol_sha256") == protocol_hash
        and value.get("condition")
        in {"single_step_sentinel", "multi_step_closed_loop"}
        and isinstance(value.get("rows"), list)
        and len(value["rows"]) > 0
        and isinstance(value.get("summary"), dict)
    )


def _load_completed_unit(path: Path, arm: str, protocol_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _load_object(path)
    if not _basic_unit_trusted(value, arm, protocol_hash):
        raise GateError(f"existing atomic unit failed validation: {path}")
    return value


def _chunk_ranges() -> tuple[tuple[int, int], ...]:
    chunks = tuple(
        (start, start + EPISODES_PER_VM_CHUNK)
        for start in range(0, 80, EPISODES_PER_VM_CHUNK)
    )
    if len(chunks) != VM_CHUNK_COUNT or chunks[-1][1] != 80:
        raise GateError("bounded-VM chunk partition drift")
    return chunks


def _chunk_unit_paths(root: Path, start: int, end: int) -> list[Path]:
    paths: list[Path] = []
    for episode_index in range(start, end):
        for target_index in range(4):
            paths.append(
                _unit_path(
                    root,
                    "single_step_sentinel",
                    episode_index,
                    target_index,
                )
            )
        paths.append(
            _unit_path(root, "multi_step_closed_loop", episode_index, None)
        )
    if len(paths) != (end - start) * 5 or len(paths) != len(set(paths)):
        raise GateError("chunk unit partition is not disjoint")
    return paths


def _wait_ports_released(ports: dict[str, int], timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_busy: list[int] = []
    while time.monotonic() < deadline:
        probes: list[socket.socket] = []
        busy: list[int] = []
        for port in ports.values():
            probe = socket.socket()
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                probes.append(probe)
            except OSError:
                probe.close()
                busy.append(port)
        for probe in probes:
            probe.close()
        if not busy:
            return
        last_busy = busy
        time.sleep(0.1)
    raise GateError(f"fresh-VM teardown left host ports busy: {last_busy}")


def _desktop_env(args: argparse.Namespace, DesktopEnv: Any, label: str) -> Any:
    vm_logs = args.out / "vm_logs" / label
    vm_logs.mkdir(parents=True, exist_ok=True)
    os.environ["OSWORLD_VM_LOG_DIR"] = str(vm_logs)
    return DesktopEnv(
        provider_name="docker",
        path_to_vm=str(args.qcow),
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
        cache_dir=str(args.out / "cache" / label),
    )


def _require_screen(controller: Any) -> None:
    if controller.get_vm_screen_size() != {"width": 1920, "height": 1080}:
        raise GateError("stage-2 live VM screen mismatch")


def _chunk_lifecycle_start(controller: Any) -> int:
    _assert_no_guest_processes(controller, GUEST_SOURCE)
    return int(_x_client_inventory(controller)["count"])


def _chunk_lifecycle_finish(controller: Any, baseline: int) -> int:
    _assert_no_guest_processes(controller, GUEST_SOURCE)
    final = int(_x_client_inventory(controller)["count"])
    if final > baseline + MAX_X_CLIENT_SLACK:
        raise GateError(
            f"bounded-VM X-client evidence exceeded {baseline}+{MAX_X_CLIENT_SLACK}: {final}"
        )
    return final


def _configure_vm_runtime(args: argparse.Namespace) -> None:
    os.environ["OSWORLD_QEMU_BIN"] = str(args.qemu_bin)
    os.environ["OSWORLD_QCOW2"] = str(args.qcow)
    sys.path[:0] = [str(args.osworld_root), str(args.provider_source.parent)]


def _unit_hash_records(paths: list[Path], root: Path) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in paths
    ]


def run_one_cell_preflight(args: argparse.Namespace) -> dict[str, Any]:
    protocol, arm = preflight(args)
    protocol_hash = sha256_file(args.protocol)
    if getattr(args, "launch_scope", None) != "one_cell_preflight":
        raise GateError("one-cell preflight requires its exact launch scope")
    if len([item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]) != 1:
        raise GateError("one-cell preflight requires exactly one GPU")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("one-cell preflight requires readable/writable /dev/kvm")
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "preflight_manifest.json"
    unit_path = args.out / "one_cell_unit.json"
    if marker.exists() or unit_path.exists():
        raise GateError("refusing to overwrite one-cell preflight evidence")

    contract = Contract()
    cells = load_cells(
        _load_object(HERE.parent / "proper_vm_stage2" / "protocol.json"), contract
    )
    candidates = [
        cell
        for cell in cells
        if cell.episode_index == 0 and cell.target_index == 0
    ]
    if len(candidates) != 1:
        raise GateError("frozen one-cell preflight selection drift")
    cell = candidates[0]
    _configure_vm_runtime(args)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/") + "/")
    with leased_ports(args.port_lock_dir) as ports:
        for name, variable in {
            "server": "OSWORLD_APPTAINER_SERVER_PORT",
            "chromium": "OSWORLD_APPTAINER_CHROMIUM_PORT",
            "vnc": "OSWORLD_APPTAINER_VNC_PORT",
            "vlc": "OSWORLD_APPTAINER_VLC_PORT",
        }.items():
            os.environ[variable] = str(ports[name])
        import qemu_kvm_provider

        qemu_kvm_provider.install()
        from desktop_env.desktop_env import DesktopEnv

        env = _desktop_env(args, DesktopEnv, "one_cell_preflight")
        try:
            controller = env.controller
            _require_screen(controller)
            baseline_x_clients = _chunk_lifecycle_start(controller)
            unit = _run_unit(
                controller=controller,
                client=client,
                served_model=args.served_model,
                contract=contract,
                protocol=protocol,
                arm_name=args.arm,
                arm=arm,
                condition="single_step_sentinel",
                episode_id=f"{cell.episode_id}:t{cell.target_index:02d}",
                episode_index=0,
                targets=[cell.bbox],
                target_centers=[cell.target],
                initial_cursor=cell.cursor,
            )
            unit["protocol_sha256"] = protocol_hash
            unit["sentinel_target_index"] = 0
            rows = unit["rows"]
            if (
                len(rows) != 1
                or unit["summary"].get("success") is not True
                or rows[0].get("parse_ok") is not True
                or rows[0].get("schema_ok") is not True
                or rows[0].get("unit_range_ok") is not True
                or rows[0].get("dispatched") is not True
                or rows[0].get("guest_hit") is not True
                or rows[0].get("target_advanced") is not True
            ):
                raise GateError("one-cell model output did not pass every action gate")
            final_x_clients = _chunk_lifecycle_finish(
                controller, baseline_x_clients
            )
        finally:
            env.close()
        _wait_ports_released(ports)

    _atomic_json(unit_path, unit)
    row = unit["rows"][0]
    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_one_cell_model_kvm_preflight",
        "status": "pass",
        "arm": args.arm,
        "semantic": arm["semantic"],
        "checkpoint_alias": arm["checkpoint_alias"],
        "model_weights_sha256": arm["model_weights_sha256"],
        "protocol_sha256": protocol_hash,
        "episode_index": 0,
        "target_index": 0,
        "model_requests": 1,
        "model_generated": True,
        "history": [],
        "raw_output_sha256": hashlib.sha256(
            row["raw_output"].encode("utf-8")
        ).hexdigest(),
        "request_seed": row["request_seed"],
        "observation_rgb_sha256": row["observation_rgb_sha256"],
        "endpoint": row["endpoint"],
        "actual_cursor_after": row["cursor_after"],
        "guest_hit": row["guest_hit"],
        "button_presses": row["guest_state"]["button_presses"],
        "button_releases": row["guest_state"]["button_releases"],
        "unit_path": str(unit_path),
        "unit_sha256": sha256_file(unit_path),
        "guest_teardown_proven": True,
        "host_ports_released_after_vm": True,
        "baseline_x_clients": baseline_x_clients,
        "final_x_clients": final_x_clients,
        "x_client_slack_bound": MAX_X_CLIENT_SLACK,
        "request_errors": 0,
        "infrastructure_mismatches": 0,
        "full_arm_launch_authorized": False,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(marker, manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, arm = preflight(args)
    protocol_hash = sha256_file(args.protocol)
    chunk0_pilot = bool(getattr(args, "chunk0_pilot", False))
    if chunk0_pilot != (getattr(args, "launch_scope", None) == "chunk0_pilot"):
        raise GateError("chunk-0 pilot flag and launch scope must match exactly")
    if len([item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]) != 1:
        raise GateError("stage-2 runner requires exactly one GPU allocation")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("stage-2 runner requires readable/writable /dev/kvm")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(
        (args.out / name).exists()
        for name in ("arm_manifest.json", "chunk0_pilot_manifest.json", "rows.jsonl")
    ):
        raise GateError("refusing to overwrite trusted stage-2 output")
    copied = 0
    if chunk0_pilot and args.resume_from:
        raise GateError("chunk-0 pilot does not accept resume input")
    if args.resume_from:
        copied = _copy_resume_units(args.resume_from, args.out, args.arm, protocol_hash)
    contract = Contract()
    cells = load_cells(
        _load_object(HERE.parent / "proper_vm_stage2" / "protocol.json"), contract
    )
    episodes: dict[int, list[Any]] = {}
    for cell in cells:
        episodes.setdefault(cell.episode_index, []).append(cell)
    for episode_cells in episodes.values():
        episode_cells.sort(key=lambda cell: cell.target_index)
    _configure_vm_runtime(args)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/") + "/")
    started = time.time()
    created = 0
    with leased_ports(args.port_lock_dir) as ports:
        for name, variable in {
            "server": "OSWORLD_APPTAINER_SERVER_PORT",
            "chromium": "OSWORLD_APPTAINER_CHROMIUM_PORT",
            "vnc": "OSWORLD_APPTAINER_VNC_PORT",
            "vlc": "OSWORLD_APPTAINER_VLC_PORT",
        }.items():
            os.environ[variable] = str(ports[name])
        import qemu_kvm_provider

        qemu_kvm_provider.install()
        from desktop_env.desktop_env import DesktopEnv

        chunk_manifest_paths: list[Path] = []
        selected_chunks = tuple(enumerate(_chunk_ranges()))
        if chunk0_pilot:
            selected_chunks = selected_chunks[:1]
        for chunk_index, (start, end) in selected_chunks:
            expected_paths = _chunk_unit_paths(args.out, start, end)
            missing_before = [
                path
                for path in expected_paths
                if _load_completed_unit(path, args.arm, protocol_hash) is None
            ]
            baseline_x_clients: int | None = None
            final_x_clients: int | None = None
            if missing_before:
                env = _desktop_env(args, DesktopEnv, f"chunk_{chunk_index:02d}")
                try:
                    controller = env.controller
                    _require_screen(controller)
                    baseline_x_clients = _chunk_lifecycle_start(controller)
                    for episode_index in range(start, end):
                        episode_cells = episodes[episode_index]
                        for cell in episode_cells:
                            path = _unit_path(
                                args.out,
                                "single_step_sentinel",
                                episode_index,
                                cell.target_index,
                            )
                            if _load_completed_unit(path, args.arm, protocol_hash) is None:
                                unit = _run_unit(
                                    controller=controller,
                                    client=client,
                                    served_model=args.served_model,
                                    contract=contract,
                                    protocol=protocol,
                                    arm_name=args.arm,
                                    arm=arm,
                                    condition="single_step_sentinel",
                                    episode_id=f"{cell.episode_id}:t{cell.target_index:02d}",
                                    episode_index=episode_index,
                                    targets=[cell.bbox],
                                    target_centers=[cell.target],
                                    initial_cursor=cell.cursor,
                                )
                                unit["protocol_sha256"] = protocol_hash
                                unit["sentinel_target_index"] = cell.target_index
                                _atomic_json(path, unit)
                                created += 1
                            _assert_no_guest_processes(controller, GUEST_SOURCE)
                        path = _unit_path(
                            args.out, "multi_step_closed_loop", episode_index, None
                        )
                        if _load_completed_unit(path, args.arm, protocol_hash) is None:
                            unit = _run_unit(
                                controller=controller,
                                client=client,
                                served_model=args.served_model,
                                contract=contract,
                                protocol=protocol,
                                arm_name=args.arm,
                                arm=arm,
                                condition="multi_step_closed_loop",
                                episode_id=episode_cells[0].episode_id,
                                episode_index=episode_index,
                                targets=[cell.bbox for cell in episode_cells],
                                target_centers=[cell.target for cell in episode_cells],
                                initial_cursor=episode_cells[0].cursor,
                            )
                            unit["protocol_sha256"] = protocol_hash
                            _atomic_json(path, unit)
                            created += 1
                        _assert_no_guest_processes(controller, GUEST_SOURCE)
                    final_x_clients = _chunk_lifecycle_finish(
                        controller, baseline_x_clients
                    )
                finally:
                    env.close()
                _wait_ports_released(ports)

            for path in expected_paths:
                if _load_completed_unit(path, args.arm, protocol_hash) is None:
                    raise GateError(f"bounded chunk is incomplete after VM teardown: {path}")
            chunk_manifest = {
                "schema_version": 1,
                "artifact_type": "synthetic_proper_vm_roadmap_stage2_vm_chunk",
                "status": "complete",
                "arm": args.arm,
                "protocol_sha256": protocol_hash,
                "chunk_index": chunk_index,
                "episode_start_inclusive": start,
                "episode_end_exclusive": end,
                "episodes": end - start,
                "atomic_units": len(expected_paths),
                "maximum_request_slots": (end - start) * REQUEST_SLOTS_PER_EPISODE,
                "fresh_vm_booted": bool(missing_before),
                "new_units": len(missing_before),
                "resumed_units": len(expected_paths) - len(missing_before),
                "guest_teardown_proven": bool(missing_before),
                "host_ports_released_after_vm": bool(missing_before),
                "baseline_x_clients": baseline_x_clients,
                "final_x_clients": final_x_clients,
                "x_client_slack_bound": MAX_X_CLIENT_SLACK,
                "units": _unit_hash_records(expected_paths, args.out),
            }
            chunk_path = args.out / "chunks" / f"chunk_{chunk_index:02d}.json"
            _atomic_json(chunk_path, chunk_manifest)
            chunk_manifest_paths.append(chunk_path)
    unit_paths = sorted((args.out / "units").rglob("*.json"))
    expected_unit_count = 25 if chunk0_pilot else 400
    if len(unit_paths) != expected_unit_count:
        raise GateError(f"stage-2 complete-unit count drift: {len(unit_paths)}")
    units = [_load_object(path) for path in unit_paths]
    all_rows = [row for unit in units for row in unit["rows"]]
    rows_temporary = args.out / "rows.jsonl.tmp"
    rows_temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    rows_path = args.out / "rows.jsonl"
    os.replace(rows_temporary, rows_path)
    if chunk0_pilot:
        if len(chunk_manifest_paths) != 1:
            raise GateError("chunk-0 pilot emitted the wrong chunk-manifest count")
        chunk_manifest = _load_object(chunk_manifest_paths[0])
        required_chunk = {
            "status": "complete",
            "arm": args.arm,
            "protocol_sha256": protocol_hash,
            "chunk_index": 0,
            "episode_start_inclusive": 0,
            "episode_end_exclusive": EPISODES_PER_VM_CHUNK,
            "episodes": EPISODES_PER_VM_CHUNK,
            "atomic_units": 25,
            "maximum_request_slots": MAX_REQUEST_SLOTS_PER_VM,
            "fresh_vm_booted": True,
            "guest_teardown_proven": True,
            "host_ports_released_after_vm": True,
        }
        if any(chunk_manifest.get(key) != value for key, value in required_chunk.items()):
            raise GateError("chunk-0 pilot manifest failed its bounded-VM gate")
        multi_step_units = [
            unit for unit in units if unit["condition"] == "multi_step_closed_loop"
        ]
        if len(multi_step_units) != EPISODES_PER_VM_CHUNK or any(
            not any(row.get("history_length", 0) > 0 for row in unit["rows"])
            for unit in multi_step_units
        ):
            raise GateError("chunk-0 pilot lacks model-generated-history evidence")
        authorization = _validate_chunk0_authorization(protocol)
        pilot_manifest = {
            "schema_version": 1,
            "artifact_type": "synthetic_proper_vm_roadmap_stage2_chunk0_closed_loop_pilot",
            "status": "complete",
            "arm": args.arm,
            "protocol_sha256": protocol_hash,
            "authorization_artifact": str(CHUNK0_AUTHORIZATION_PATH),
            "authorization_artifact_sha256": sha256_file(CHUNK0_AUTHORIZATION_PATH),
            "preflight_protocol_sha256": authorization["preflight_protocol_sha256"],
            "checkpoint_alias": arm["checkpoint_alias"],
            "model_weights_sha256": arm["model_weights_sha256"],
            "chunk_index": 0,
            "episode_start_inclusive": 0,
            "episode_end_exclusive": EPISODES_PER_VM_CHUNK,
            "episodes": EPISODES_PER_VM_CHUNK,
            "atomic_units": len(units),
            "sentinel_units": sum(
                unit["condition"] == "single_step_sentinel" for unit in units
            ),
            "multi_step_units": len(multi_step_units),
            "rows": len(all_rows),
            "rows_sha256": sha256_file(rows_path),
            "new_units": created,
            "resumed_units": copied,
            "vm_chunks": 1,
            "episodes_per_vm_chunk": EPISODES_PER_VM_CHUNK,
            "maximum_request_slots_per_vm": MAX_REQUEST_SLOTS_PER_VM,
            "chunk_manifest": {
                "path": str(chunk_manifest_paths[0].relative_to(args.out)),
                "sha256": sha256_file(chunk_manifest_paths[0]),
            },
            "model_generated_history_proven": True,
            "multi_step_units_with_model_generated_history": len(multi_step_units),
            "matched_seed_rule": authorization["matched_seed_rule"],
            "elapsed_seconds": time.time() - started,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "request_errors": 0,
            "infrastructure_mismatches": 0,
            "guest_teardown_proven": True,
            "host_ports_released_after_vm": True,
            "baseline_x_clients": chunk_manifest["baseline_x_clients"],
            "final_x_clients": chunk_manifest["final_x_clients"],
            "x_client_slack_bound": MAX_X_CLIENT_SLACK,
            "chunks_1_15_launched": False,
            "full_arm_launch_authorized": False,
        }
        _atomic_json(args.out / "chunk0_pilot_manifest.json", pilot_manifest)
        return pilot_manifest
    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_closed_loop_arm",
        "status": "complete",
        "arm": args.arm,
        "protocol_sha256": protocol_hash,
        "checkpoint_alias": arm["checkpoint_alias"],
        "model_weights_sha256": arm["model_weights_sha256"],
        "atomic_units": len(units),
        "sentinel_units": sum(unit["condition"] == "single_step_sentinel" for unit in units),
        "multi_step_units": sum(unit["condition"] == "multi_step_closed_loop" for unit in units),
        "rows": len(all_rows),
        "rows_sha256": sha256_file(rows_path),
        "resumed_units": copied,
        "new_units": created,
        "vm_chunks": len(chunk_manifest_paths),
        "episodes_per_vm_chunk": EPISODES_PER_VM_CHUNK,
        "maximum_request_slots_per_vm": MAX_REQUEST_SLOTS_PER_VM,
        "chunk_manifests": _unit_hash_records(chunk_manifest_paths, args.out),
        "elapsed_seconds": time.time() - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "request_errors": 0,
        "infrastructure_mismatches": 0,
        "guest_teardown_proven": True,
    }
    _atomic_json(args.out / "arm_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", default="policy")
    parser.add_argument("--api-key", default="x")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--provider-source", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, required=True)
    parser.add_argument("--qemu-bin", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--port-lock-dir", type=Path, default=Path("/tmp/osworld_port_locks"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--launch-scope",
        choices=("full", "one_cell_preflight", "chunk0_pilot"),
        default="full",
    )
    parser.add_argument("--one-cell-preflight", action="store_true")
    parser.add_argument("--chunk0-pilot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        protocol, arm = preflight(parsed)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mutated": False,
                    "arm": parsed.arm,
                    "checkpoint_alias": arm["checkpoint_alias"],
                    "protocol_sha256": sha256_file(parsed.protocol),
                    "launch_authorized": protocol["launch_authorized"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif parsed.one_cell_preflight:
        print(
            json.dumps(
                run_one_cell_preflight(parsed), indent=2, sort_keys=True
            )
        )
    else:
        print(json.dumps(run(parsed), indent=2, sort_keys=True))
