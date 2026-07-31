#!/usr/bin/env python3
"""CPU/KVM-only dynamic closed-loop smoke; never starts a model server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from contract import Contract  # type: ignore  # noqa: E402

sys.path.insert(0, str(HERE.parent / "proper_vm_stage2"))
from gate import actuation_plan, load_cells, rgb_sha256, sha256_file  # type: ignore  # noqa: E402
from live_smoke import (  # type: ignore  # noqa: E402
    _assert_no_guest_processes,
    _guest_exec,
    _plan_code,
    _x_client_inventory,
    leased_ports,
)

from closed_loop_contract import AttemptEvidence, advance, initial_state, reference_png  # noqa: E402
from runner import (  # noqa: E402
    GUEST_SOURCE,
    PROTOCOL_PATH,
    GateError,
    _cursor,
    _install_episode,
    _read_guest_state,
    _stop_episode,
    _update_render,
    load_protocol,
    validate_protocol,
)


X_CLIENT_SLACK = 8


def _dispatch(
    controller: Any,
    *,
    semantic: str,
    state: Any,
    endpoint: tuple[int, int],
    targets: list[tuple[int, int, int, int]],
    episode_revision: str,
    releases: int,
    maximum: int = 3,
) -> tuple[Any, dict[str, Any], int]:
    _guest_exec(
        controller,
        _plan_code(actuation_plan(semantic, "click", state.cursor, endpoint)),
    )
    releases += 1
    guest = _read_guest_state(
        controller, episode_revision, minimum_releases=releases
    )
    if int(guest.get("button_presses", -1)) != releases or int(
        guest.get("button_releases", -1)
    ) != releases:
        raise GateError("dynamic smoke button-count mismatch")
    actual = _cursor(controller)
    transition = advance(
        state,
        AttemptEvidence(
            raw_output="smoke",
            parse_ok=True,
            schema_ok=True,
            unit_range_ok=True,
            dispatched=True,
            endpoint=endpoint,
            actual_cursor_after=actual,
            guest_hit=guest["last_hit"],
        ),
        targets,
        max_attempts_per_target=maximum,
    )
    if guest.get("target_index") != transition.after.target_index:
        raise GateError("dynamic smoke target state mismatch")
    if guest.get("completed") is not transition.after.success or guest.get("down") is not False:
        raise GateError("dynamic smoke completion/button mismatch")
    return transition, guest, releases


def _render_next(
    controller: Any,
    contract: Contract,
    state: Any,
    targets: list[tuple[int, int, int, int]],
    episode_revision: str,
    sequence: int,
) -> str:
    revision = hashlib.sha256(
        f"{episode_revision}|{sequence}|{state.target_index}|{state.cursor}".encode()
    ).hexdigest()
    _update_render(
        controller,
        episode_revision=episode_revision,
        command_sequence=sequence,
        render_revision=revision,
        state=state,
        targets=targets,
        png=reference_png(contract, state, targets),
    )
    return revision


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol, require_authorized=False)
    validate_protocol(protocol)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise GateError("dynamic smoke must not receive a GPU")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("dynamic smoke requires readable/writable /dev/kvm")
    stage1_protocol = json.loads(
        (HERE.parent / "proper_vm_stage2" / "protocol.json").read_text(encoding="utf-8")
    )
    vm = stage1_protocol["vm"]
    if args.provider_source.resolve() != Path(vm["provider_source"]).resolve():
        raise GateError("dynamic smoke provider path drift")
    if sha256_file(args.provider_source) != vm["provider_sha256"]:
        raise GateError("dynamic smoke provider hash drift")
    if args.qcow.resolve() != Path(vm["qcow"]).resolve():
        raise GateError("dynamic smoke qcow drift")
    contract = Contract()
    cells = load_cells(stage1_protocol, contract)
    first_episode = sorted(
        [cell for cell in cells if cell.episode_index == 0], key=lambda cell: cell.target_index
    )
    targets = [cell.bbox for cell in first_episode[:2]]
    centers = [cell.target for cell in first_episode[:2]]
    initial_cursor = first_episode[0].cursor
    os.environ["OSWORLD_QEMU_BIN"] = str(args.qemu_bin)
    os.environ["OSWORLD_QCOW2"] = str(args.qcow)
    os.environ["OSWORLD_VM_LOG_DIR"] = str(args.out / "vm_logs")
    sys.path[:0] = [str(args.osworld_root), str(args.provider_source.parent)]
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "dynamic_smoke_manifest.json"
    if marker.exists():
        raise GateError("refusing to overwrite dynamic smoke marker")
    records: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
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
                raise GateError("dynamic smoke screen mismatch")
            baseline_inventory = _x_client_inventory(controller)
            baseline_x_clients = int(baseline_inventory["count"])

            def stop_and_prove(label: str, pid: int) -> None:
                _stop_episode(controller, pid)
                _assert_no_guest_processes(controller, GUEST_SOURCE)
                inventory = _x_client_inventory(controller)
                count = int(inventory["count"])
                if count > baseline_x_clients + X_CLIENT_SLACK:
                    raise GateError(
                        f"dynamic smoke X-client bound exceeded after {label}: "
                        f"{count} > {baseline_x_clients}+{X_CLIENT_SLACK}"
                    )
                lifecycle.append(
                    {
                        "scene": label,
                        "exact_source_processes_after": 0,
                        "x_client_count_after": count,
                    }
                )

            for semantic in ("absolute_toolcall", "move_rel", "deltatype_raw"):
                state = initial_state(f"smoke-{semantic}", initial_cursor)
                episode_revision = hashlib.sha256(state.episode_id.encode()).hexdigest()
                initial_png = reference_png(contract, state, targets)
                pid = _install_episode(
                    controller,
                    episode_revision=episode_revision,
                    targets=targets,
                    cursor=state.cursor,
                    png=initial_png,
                )
                releases = 0
                sequence = 0
                try:
                    before_hash = rgb_sha256(controller.get_screenshot())
                    invalid = advance(
                        state,
                        AttemptEvidence("bad", False, False, False, False, None, None, None),
                        targets,
                        max_attempts_per_target=3,
                    )
                    state = invalid.after
                    guest = _read_guest_state(controller, episode_revision)
                    if guest["button_releases"] != 0 or rgb_sha256(controller.get_screenshot()) != before_hash:
                        raise GateError("dynamic invalid-output no-op contract failed")
                    miss_endpoint = (0, 0)
                    miss, guest, releases = _dispatch(
                        controller,
                        semantic=semantic,
                        state=state,
                        endpoint=miss_endpoint,
                        targets=targets,
                        episode_revision=episode_revision,
                        releases=releases,
                    )
                    state = miss.after
                    if miss.hit or state.target_index != 0 or state.cursor != miss_endpoint:
                        raise GateError("dynamic miss transition failed")
                    sequence += 1
                    miss_revision = _render_next(
                        controller, contract, state, targets, episode_revision, sequence
                    )
                    hit, guest, releases = _dispatch(
                        controller,
                        semantic=semantic,
                        state=state,
                        endpoint=centers[0],
                        targets=targets,
                        episode_revision=episode_revision,
                        releases=releases,
                    )
                    state = hit.after
                    if not hit.hit or state.target_index != 1 or state.cursor != centers[0]:
                        raise GateError("dynamic hit/advance transition failed")
                    sequence += 1
                    hit_revision = _render_next(
                        controller, contract, state, targets, episode_revision, sequence
                    )
                    final, guest, releases = _dispatch(
                        controller,
                        semantic=semantic,
                        state=state,
                        endpoint=centers[1],
                        targets=targets,
                        episode_revision=episode_revision,
                        releases=releases,
                    )
                    state = final.after
                    if not state.success or not state.terminated:
                        raise GateError("dynamic final completion transition failed")
                    records.append(
                        {
                            "semantic": semantic,
                            "invalid_no_op": True,
                            "miss_retained_target": True,
                            "miss_cursor": list(miss_endpoint),
                            "miss_render_revision": miss_revision,
                            "hit_advanced_target": True,
                            "hit_cursor": list(centers[0]),
                            "hit_render_revision": hit_revision,
                            "all_targets_completed": True,
                            "button_releases": releases,
                        }
                    )
                finally:
                    stop_and_prove(f"semantic-{semantic}", pid)

            state = initial_state("smoke-exhaustion", initial_cursor)
            one_target = [targets[0]]
            episode_revision = hashlib.sha256(state.episode_id.encode()).hexdigest()
            pid = _install_episode(
                controller,
                episode_revision=episode_revision,
                targets=one_target,
                cursor=state.cursor,
                png=reference_png(contract, state, one_target),
            )
            releases = 0
            try:
                for attempt, endpoint in enumerate(((0, 0), (10, 10), (20, 20)), 1):
                    transition, guest, releases = _dispatch(
                        controller,
                        semantic="absolute_toolcall",
                        state=state,
                        endpoint=endpoint,
                        targets=one_target,
                        episode_revision=episode_revision,
                        releases=releases,
                    )
                    state = transition.after
                    if attempt < 3:
                        _render_next(
                            controller, contract, state, one_target, episode_revision, attempt
                        )
                if not state.terminated or state.success or guest["target_index"] != 0:
                    raise GateError("dynamic retry-exhaustion contract failed")
                records.append(
                    {
                        "semantic": "absolute_toolcall",
                        "retry_exhaustion": True,
                        "attempts": 3,
                        "target_retained": True,
                        "button_releases": releases,
                    }
                )
            finally:
                stop_and_prove("retry-exhaustion", pid)
            final_inventory = _x_client_inventory(controller)
            final_x_clients = int(final_inventory["count"])
            if final_x_clients > baseline_x_clients + X_CLIENT_SLACK:
                raise GateError("dynamic smoke final X-client bound exceeded")
        finally:
            env.close()
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_dynamic_kvm_smoke",
        "status": "pass",
        "cpu_only": True,
        "gpu_used": False,
        "protocol_sha256": sha256_file(args.protocol),
        "dynamic_guest_app_sha256": sha256_file(HERE / "dynamic_guest_app.py"),
        "provider_sha256": sha256_file(args.provider_source),
        "records": records,
        "semantic_sequences": 3,
        "retry_exhaustion_sequences": 1,
        "guest_teardown_proven": True,
        "exact_guest_process_absence_after_each_scene": True,
        "exact_button_counts_verified": True,
        "scene_count": len(lifecycle),
        "baseline_x_clients": baseline_x_clients,
        "maximum_x_clients_after_scene": max(
            [baseline_x_clients]
            + [int(item["x_client_count_after"]) for item in lifecycle]
        ),
        "final_x_clients": final_x_clients,
        "x_client_slack_bound": X_CLIENT_SLACK,
        "lifecycle": lifecycle,
        "model_server_started": False,
        "model_checkpoint_loaded": False,
        "model_generated_history": False,
        "scripted_endpoint_smoke_only": True,
        "full_stage2_arms_authorized": False,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, marker)
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
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
