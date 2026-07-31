#!/usr/bin/env python3
"""Run one roadmap stage-1.5 endpoint-actuation conformance arm."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from .gate import (
        PROTOCOL_PATH,
        GateError,
        actuation_plan,
        load_cells,
        load_protocol,
        rgb_sha256,
        sha256_file,
        validate_protocol,
    )
    from .live_smoke import (
        _guest_exec,
        _install_guest_scene,
        _plan_code,
        _read_state,
        _stop_guest,
        leased_ports,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate import (  # type: ignore
        PROTOCOL_PATH,
        GateError,
        actuation_plan,
        load_cells,
        load_protocol,
        rgb_sha256,
        sha256_file,
        validate_protocol,
    )
    from live_smoke import (  # type: ignore
        _guest_exec,
        _install_guest_scene,
        _plan_code,
        _read_state,
        _stop_guest,
        leased_ports,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import (  # type: ignore  # noqa: E402
    Contract,
    serialize_action,
    sha256_bytes,
    strict_schema_ok,
    unit_range_ok,
)
from evaluate import _call_model  # type: ignore  # noqa: E402


ARM_NAMES = ("absolute_phase_a", "normalized_phase_a", "raw_a_to_b")
CHUNK_BOUNDS = ((0, 80), (80, 160), (160, 240), (240, 320))
SCREENSHOT_READY_TIMEOUT_S = 5.0
SCREENSHOT_READY_POLL_S = 0.1
SCREENSHOT_HASH_HISTORY = 8


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _release_and_stop(controller: Any, pid: int) -> None:
    """Best-effort button release must never bypass proven process teardown."""
    try:
        _guest_exec(controller, "import pyautogui; pyautogui.mouseUp(button='left')")
    finally:
        _stop_guest(controller, pid)


def _cursor_position(controller: Any) -> tuple[int, int]:
    output = _guest_exec(controller, "import pyautogui; print(list(pyautogui.position()))")
    try:
        value = json.loads(output.splitlines()[-1])
        return int(value[0]), int(value[1])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"could not parse guest cursor: {output!r}") from exc


def _wait_for_exact_screenshot(
    controller: Any,
    expected_png: bytes,
    *,
    timeout_s: float = SCREENSHOT_READY_TIMEOUT_S,
    poll_s: float = SCREENSHOT_READY_POLL_S,
) -> tuple[bytes, int, tuple[str, ...]]:
    """Wait only for compositor convergence; exact decoded pixels remain mandatory."""
    expected_hash = rgb_sha256(expected_png)
    started = time.monotonic()
    attempts = 0
    last_hashes: list[str] = []
    while True:
        attempts += 1
        screenshot = controller.get_screenshot()
        if not screenshot:
            observed_hash = "<missing>"
        else:
            try:
                observed_hash = rgb_sha256(screenshot)
            except BaseException as exc:
                observed_hash = f"<invalid:{type(exc).__name__}>"
        last_hashes.append(observed_hash)
        del last_hashes[:-SCREENSHOT_HASH_HISTORY]
        if observed_hash == expected_hash:
            return screenshot, attempts, tuple(last_hashes)
        if time.monotonic() - started >= timeout_s:
            raise GateError(
                "live screenshot pixels did not converge to the frozen canonical image: "
                f"expected={expected_hash} attempts={attempts} last_hashes={last_hashes}"
            )
        time.sleep(poll_s)


def _start_scene(controller: Any, cell: Any, operation: str, revision: str) -> tuple[int, bytes]:
    pid = _install_guest_scene(controller, cell, operation, revision)
    try:
        _read_state(controller, revision)
        _guest_exec(
            controller,
            "import pyautogui; pyautogui.FAILSAFE=False; "
            f"pyautogui.moveTo({cell.cursor[0]},{cell.cursor[1]}); "
            "print(list(pyautogui.position()))",
        )
        if _cursor_position(controller) != cell.cursor:
            raise GateError(f"guest failed to assume frozen cursor: {cell.cell_id}")
        screenshot, attempts, last_hashes = _wait_for_exact_screenshot(
            controller, cell.image_path.read_bytes()
        )
        if attempts > 1:
            print(
                json.dumps(
                    {
                        "event": "exact_screenshot_ready",
                        "cell_id": cell.cell_id,
                        "operation": operation,
                        "attempts": attempts,
                        "last_hashes": list(last_hashes),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return pid, screenshot
    except BaseException:
        _release_and_stop(controller, pid)
        raise


def _finish_scene(
    controller: Any,
    cell: Any,
    semantic: str,
    operation: str,
    revision: str,
    pid: int,
    endpoint: tuple[int, int],
    expected_hit: bool,
) -> dict[str, Any]:
    try:
        plan = actuation_plan(semantic, operation, cell.cursor, endpoint)
        _guest_exec(controller, _plan_code(plan))
        state = _read_state(controller, revision, min_releases=1)
        success_key = "click_success" if operation == "click" else "drag_success"
        guest_success = bool(state.get(success_key))
        cursor = _cursor_position(controller)
        if state.get("down") or int(state.get("button_presses", -1)) != 1:
            raise GateError(f"guest button state mismatch: {cell.cell_id}/{operation}: {state}")
        if int(state.get("button_releases", -1)) != 1:
            raise GateError(f"guest release count mismatch: {cell.cell_id}/{operation}: {state}")
        if cursor != endpoint:
            raise GateError(
                f"guest cursor endpoint mismatch: {cell.cell_id}/{operation}: {cursor} != {endpoint}"
            )
        if guest_success != expected_hit:
            raise GateError(
                f"guest/geometry outcome mismatch: {cell.cell_id}/{operation}: "
                f"{guest_success} != {expected_hit}"
            )
        return {
            "success": guest_success,
            "cursor_after": list(cursor),
            "plan": [list(command) for command in plan],
            "state": state,
        }
    finally:
        _release_and_stop(controller, pid)


def _validate_selected_arm(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    if args.arm not in ARM_NAMES:
        raise GateError(f"unknown arm: {args.arm}")
    arm = protocol["arms"][args.arm]
    expected_model = (Path(arm["checkpoint_root"]) / "hf").resolve()
    if args.model_dir.resolve() != expected_model:
        raise GateError(f"model directory drift: {args.model_dir} != {expected_model}")
    if args.live_smoke_manifest.resolve() != Path(
        protocol["live_smoke_evidence"]["manifest"]
    ).resolve():
        raise GateError("live-smoke evidence path drift")
    if sha256_file(args.live_smoke_manifest) != protocol["live_smoke_evidence"][
        "manifest_sha256"
    ]:
        raise GateError("live-smoke evidence hash drift")
    return arm, expected_model


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Validate every immutable input and the explicit launch bit without mutation."""
    protocol = load_protocol(args.protocol, require_launch_authorized=True)
    validate_protocol(protocol)
    arm, model_dir = _validate_selected_arm(args, protocol)
    fallback = protocol.get("execution_recovery", {}).get("active_fallback", {})
    expected_chunks = [list(bounds) for bounds in CHUNK_BOUNDS]
    if (
        fallback.get("kind") != "four_disjoint_fresh_vm_chunks"
        or fallback.get("bounds") != expected_chunks
        or args.chunk_index not in range(len(CHUNK_BOUNDS))
        or (args.chunk_start, args.chunk_stop) != CHUNK_BOUNDS[args.chunk_index]
    ):
        raise GateError("chunk does not match the authorized fixed fallback plan")
    vm = protocol["vm"]
    if args.provider_source.resolve() != Path(vm["provider_source"]).resolve():
        raise GateError("provider path differs from protocol")
    if sha256_file(args.provider_source) != vm["provider_sha256"]:
        raise GateError("provider hash drift")
    if args.qcow.resolve() != Path(vm["qcow"]).resolve():
        raise GateError("qcow path differs from protocol")
    if args.qemu_bin.resolve() != Path(vm["qemu_bin"]).resolve():
        raise GateError("QEMU path differs from protocol")
    if not args.qemu_bin.is_file() or not os.access(args.qemu_bin, os.X_OK):
        raise GateError("pinned QEMU executable is absent")
    if not args.osworld_root.is_dir():
        raise GateError("OSWorld root is absent")
    return protocol, arm, model_dir


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, arm, model_dir = preflight(args)
    semantic = arm["semantic"]
    contract = Contract()
    all_cells = load_cells(protocol, contract)
    cells = all_cells[args.chunk_start : args.chunk_stop]
    if len(cells) != 80 or any(
        cell.episode_index // 20 != args.chunk_index for cell in cells
    ):
        raise GateError("fixed chunk is not 80 complete ordered episodes")
    visible_gpus = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible_gpus) != 1:
        raise GateError(f"expected exactly one allocated GPU, got {visible_gpus}")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("/dev/kvm is not readable and writable")
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "chunk_manifest.json"
    partial = args.out / "rows.partial.jsonl"
    rows_path = args.out / "rows.jsonl"
    existing = [path for path in (marker, partial, rows_path) if path.exists()]
    if existing:
        raise GateError(f"refusing to overwrite existing arm outputs: {existing}")
    os.environ["OSWORLD_QEMU_BIN"] = str(args.qemu_bin)
    os.environ["OSWORLD_QCOW2"] = str(args.qcow)
    os.environ["OSWORLD_VM_LOG_DIR"] = str(args.out / "vm_logs")
    sys.path.insert(0, str(args.osworld_root))
    sys.path.insert(0, str(args.provider_source.parent))
    started = time.time()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/") + "/")
    records: list[dict[str, Any]] = []
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

        env = DesktopEnv(
            provider_name="docker",
            path_to_vm=str(args.qcow),
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False,
            cache_dir=str(args.out / "cache"),
        )
        try:
            controller = env.controller
            screen_size = controller.get_vm_screen_size()
            if not isinstance(screen_size, dict) or (
                int(screen_size.get("width", -1)), int(screen_size.get("height", -1))
            ) != (1920, 1080):
                raise GateError("live VM screen-size mismatch")
            prior: list[str] = []
            previous_episode = None
            for cell in cells:
                if cell.episode_id != previous_episode:
                    prior = []
                    previous_episode = cell.episode_id
                revision = sha256_bytes(f"{args.arm}|{cell.cell_id}|click".encode("utf-8"))
                click_pid, screenshot = _start_scene(controller, cell, "click", revision)
                user_text = contract.user_text(
                    semantic,
                    cell.cursor,
                    cell.target,
                    target_index=cell.target_index,
                    target_count=protocol["geometry"]["targets_per_episode"],
                    preamble=bool(arm["preamble"]),
                    prior=prior[-int(protocol["context"]["history_turns"]):] or None,
                )
                try:
                    raw, tool_calls, meta = _call_model(
                        client,
                        model=args.served_model,
                        system=contract.system_prompt(semantic),
                        user_text=user_text,
                        png=screenshot,
                        history=[],
                        seed=cell.request_seed,
                        max_tokens=int(protocol["sampling"]["max_tokens"]),
                    )
                except BaseException:
                    _release_and_stop(controller, click_pid)
                    raise
                parse_text = raw.split(" | tool_calls=", 1)[0]
                move = contract.parse(semantic, parse_text, tool_calls)
                schema_ok = strict_schema_ok(semantic, parse_text, move.coord)
                units_ok = unit_range_ok(semantic, move.coord)
                endpoint = (
                    contract.apply_coord(semantic, cell.cursor, move.coord)
                    if move.coord is not None
                    else None
                )
                endpoint_hit = bool(endpoint is not None and contract.in_bbox(endpoint, cell.bbox))
                click_result = None
                drag_result = None
                drag_rgb_hash = None
                if endpoint is None:
                    _release_and_stop(controller, click_pid)
                else:
                    click_result = _finish_scene(
                        controller,
                        cell,
                        semantic,
                        "click",
                        revision,
                        click_pid,
                        endpoint,
                        endpoint_hit,
                    )
                    drag_revision = sha256_bytes(
                        f"{args.arm}|{cell.cell_id}|drag".encode("utf-8")
                    )
                    drag_pid, drag_screenshot = _start_scene(
                        controller, cell, "drag", drag_revision
                    )
                    drag_rgb_hash = rgb_sha256(drag_screenshot)
                    drag_result = _finish_scene(
                        controller,
                        cell,
                        semantic,
                        "drag",
                        drag_revision,
                        drag_pid,
                        endpoint,
                        endpoint_hit,
                    )
                row = {
                    "cell_id": cell.cell_id,
                    "episode_id": cell.episode_id,
                    "episode_index": cell.episode_index,
                    "target_index": cell.target_index,
                    "cursor_before": list(cell.cursor),
                    "bbox": list(cell.bbox),
                    "target_center": list(cell.target),
                    "canonical_png_sha256": cell.image_sha256,
                    "vm_observation_rgb_sha256": rgb_sha256(screenshot),
                    "vm_drag_rgb_sha256": drag_rgb_hash,
                    "request_seed": cell.request_seed,
                    "raw_output": raw,
                    "completion_tokens": meta["completion_tokens"],
                    "parse_ok": bool(move.parse_ok and move.coord is not None),
                    "schema_ok": schema_ok,
                    "unit_range_ok": units_ok,
                    "coord": list(move.coord) if move.coord is not None else None,
                    "endpoint": list(endpoint) if endpoint is not None else None,
                    "endpoint_in_bbox": endpoint_hit,
                    "click": click_result,
                    "drag": drag_result,
                    "compound_success": bool(
                        move.parse_ok
                        and move.coord is not None
                        and schema_ok
                        and units_ok
                        and endpoint_hit
                        and click_result
                        and click_result["success"]
                        and drag_result
                        and drag_result["success"]
                    ),
                }
                records.append(row)
                with partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                oracle_coord = contract.ideal_coord(semantic, cell.cursor, cell.target)
                prior.append(serialize_action(semantic, oracle_coord))
        finally:
            env.close()
    if len(records) != args.chunk_stop - args.chunk_start:
        raise GateError("chunk did not produce every registered cell in its interval")
    os.replace(partial, rows_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_stage1_5_endpoint_actuation_chunk",
        "status": "complete",
        "scope_classification": protocol["scope_classification"],
        "arm": args.arm,
        "chunk_index": args.chunk_index,
        "chunk_start": args.chunk_start,
        "chunk_stop": args.chunk_stop,
        "cell_ids_sha256": sha256_bytes(
            json.dumps([cell.cell_id for cell in cells], separators=(",", ":")).encode()
        ),
        "semantic": semantic,
        "preamble": arm["preamble"],
        "checkpoint_alias": arm["checkpoint_alias"],
        "checkpoint_manifest_sha256": arm["checkpoint_manifest_sha256"],
        "model_weights_sha256": arm["model_weights_sha256"],
        "model_dir": str(model_dir),
        "protocol_sha256": sha256_file(args.protocol),
        "live_smoke_manifest_sha256": protocol["live_smoke_evidence"]["manifest_sha256"],
        "provider_sha256": protocol["vm"]["provider_sha256"],
        "n_cells": len(records),
        "n_compound_success": sum(row["compound_success"] for row in records),
        "n_parse_ok": sum(row["parse_ok"] for row in records),
        "n_schema_ok": sum(row["schema_ok"] for row in records),
        "n_unit_range_ok": sum(row["unit_range_ok"] for row in records),
        "n_endpoint_in_bbox": sum(row["endpoint_in_bbox"] for row in records),
        "sampling": protocol["sampling"],
        "context": protocol["context"],
        "rows_sha256": sha256_file(rows_path),
        "elapsed_seconds": time.time() - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_visible": visible_gpus,
        "request_errors": 0,
        "infrastructure_mismatches": 0,
    }
    _atomic_json(marker, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--chunk-stop", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", default="policy")
    parser.add_argument("--api-key", default="x")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--live-smoke-manifest", type=Path, required=True)
    parser.add_argument("--provider-source", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, required=True)
    parser.add_argument("--qemu-bin", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--port-lock-dir", type=Path, default=Path("/tmp/osworld_port_locks"))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        parsed = parse_args()
        if parsed.preflight_only:
            protocol, arm, model_dir = preflight(parsed)
            result = {
                "status": "pass",
                "arm": parsed.arm,
                "chunk_index": parsed.chunk_index,
                "chunk_start": parsed.chunk_start,
                "chunk_stop": parsed.chunk_stop,
                "semantic": arm["semantic"],
                "model_dir": str(model_dir),
                "protocol_sha256": sha256_file(parsed.protocol),
                "launch_authorized": protocol["launch_gate"]["authorized"],
                "mutated": False,
            }
        else:
            result = run(parsed)
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        print(f"FATAL proper-VM arm: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
