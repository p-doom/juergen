#!/usr/bin/env python3
"""CPU/KVM lifecycle stress beyond the historical 230-scene X-client failure."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .gate import (
        PROTOCOL_PATH,
        GateError,
        load_cells,
        load_protocol,
        sha256_file,
        validate_protocol,
    )
    from .live_smoke import (
        GUEST_SOURCE_PATH,
        _guest_process_snapshot,
        _x_client_inventory,
        leased_ports,
    )
    from .run_arm import _finish_scene, _start_scene
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate import (  # type: ignore
        PROTOCOL_PATH,
        GateError,
        load_cells,
        load_protocol,
        sha256_file,
        validate_protocol,
    )
    from live_smoke import (  # type: ignore
        GUEST_SOURCE_PATH,
        _guest_process_snapshot,
        _x_client_inventory,
        leased_ports,
    )
    from run_arm import _finish_scene, _start_scene  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Contract  # type: ignore  # noqa: E402


SEMANTICS = ("absolute_toolcall", "move_rel", "deltatype_raw")
DEFAULT_CELLS = 131
HISTORICAL_FAILURE_CELLS = 115
CHECKPOINT_CELLS = frozenset({1, 64, 115, 116, DEFAULT_CELLS})
X_CLIENT_SLACK = 8


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cells < DEFAULT_CELLS:
        raise GateError(f"lifecycle stress requires at least {DEFAULT_CELLS} cells")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise GateError("lifecycle stress must not receive a GPU")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("lifecycle stress requires readable/writable /dev/kvm")
    protocol = load_protocol(args.protocol, require_launch_authorized=False)
    validate_protocol(protocol)
    vm = protocol["vm"]
    if args.provider_source.resolve() != Path(vm["provider_source"]).resolve():
        raise GateError("lifecycle-stress provider path drift")
    if sha256_file(args.provider_source) != vm["provider_sha256"]:
        raise GateError("lifecycle-stress provider hash drift")
    if args.qcow.resolve() != Path(vm["qcow"]).resolve():
        raise GateError("lifecycle-stress qcow path drift")
    if args.qemu_bin.resolve() != Path(vm["qemu_bin"]).resolve():
        raise GateError("lifecycle-stress QEMU path drift")

    contract = Contract()
    cell = load_cells(protocol, contract)[0]
    os.environ["OSWORLD_QEMU_BIN"] = str(args.qemu_bin)
    os.environ["OSWORLD_QCOW2"] = str(args.qcow)
    os.environ["OSWORLD_VM_LOG_DIR"] = str(args.out / "vm_logs")
    sys.path[:0] = [str(args.osworld_root), str(args.provider_source.parent)]
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "lifecycle_stress_manifest.json"
    if marker.exists():
        raise GateError("refusing to overwrite lifecycle stress marker")

    started = time.time()
    checkpoints: list[dict[str, Any]] = []
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
            if controller.get_vm_screen_size() != {"width": 1920, "height": 1080}:
                raise GateError("lifecycle-stress screen-size mismatch")
            baseline = _x_client_inventory(controller)
            baseline_count = int(baseline["count"])
            for index in range(args.cells):
                semantic = SEMANTICS[index % len(SEMANTICS)]
                endpoint = cell.target
                for operation in ("click", "drag"):
                    revision = f"lifecycle-stress-{index:03d}-{semantic}-{operation}"
                    pid, _ = _start_scene(controller, cell, operation, revision)
                    result = _finish_scene(
                        controller,
                        cell,
                        semantic,
                        operation,
                        revision,
                        pid,
                        endpoint,
                        True,
                    )
                    if not result["success"]:
                        raise GateError("lifecycle-stress replay unexpectedly missed")
                    # _finish_scene proves teardown; this independent scan proves no peer remains.
                    remaining = _guest_process_snapshot(controller, GUEST_SOURCE_PATH)
                    if remaining:
                        raise GateError(f"guest process leaked after scene {index}/{operation}: {remaining}")
                completed = index + 1
                if completed in CHECKPOINT_CELLS or completed == args.cells:
                    inventory = _x_client_inventory(controller)
                    count = int(inventory["count"])
                    if count > baseline_count + X_CLIENT_SLACK:
                        raise GateError(
                            f"X-client inventory grew beyond bound at cell {completed}: "
                            f"{count} > {baseline_count}+{X_CLIENT_SLACK}"
                        )
                    checkpoints.append(
                        {
                            "cells_completed": completed,
                            "scenes_completed": completed * 2,
                            "exact_source_processes": 0,
                            "x_client_count": count,
                        }
                    )
            final_inventory = _x_client_inventory(controller)
            if _guest_process_snapshot(controller, GUEST_SOURCE_PATH):
                raise GateError("guest process remained after lifecycle stress")
        finally:
            env.close()

    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_stage1_5_guest_lifecycle_stress",
        "status": "pass",
        "cpu_only": True,
        "gpu_used": False,
        "cells": args.cells,
        "scenes": args.cells * 2,
        "historical_failure_cells": HISTORICAL_FAILURE_CELLS,
        "historical_failure_scenes": HISTORICAL_FAILURE_CELLS * 2,
        "exceeded_historical_boundary": args.cells > HISTORICAL_FAILURE_CELLS,
        "exact_source_processes_after_each_scene": 0,
        "baseline_x_clients": baseline_count,
        "final_x_clients": int(final_inventory["count"]),
        "x_client_slack_bound": X_CLIENT_SLACK,
        "checkpoints": checkpoints,
        "protocol_sha256": sha256_file(args.protocol),
        "live_smoke_source_sha256": sha256_file(Path(__file__).with_name("live_smoke.py")),
        "run_arm_source_sha256": sha256_file(Path(__file__).with_name("run_arm.py")),
        "provider_sha256": sha256_file(args.provider_source),
        "elapsed_seconds": time.time() - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(marker, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider-source", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, required=True)
    parser.add_argument("--qemu-bin", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--port-lock-dir", type=Path, default=Path("/tmp/osworld_port_locks"))
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
