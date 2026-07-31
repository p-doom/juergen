#!/usr/bin/env python3
"""No-model native-QEMU/KVM smoke for exact pixels and real click/drag replay.

This script is prepared for a CPU-only labctl job but is not run by the CPU
selftest.  It boots the same pinned provider used by the proper-task pilot.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Contract  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent
GUEST_APP = HERE / "guest_app.py"
GUEST_SOURCE_PATH = "/tmp/proper_vm_stage2_guest.py"
GUEST_IMAGE_PATH = "/tmp/proper_vm_stage2_scene.png"
GUEST_CONFIG_PATH = "/tmp/proper_vm_stage2_config.json"
GUEST_STATE_PATH = "/tmp/proper_vm_stage2_state.json"


def _encoded_guest_script(source: str) -> str:
    """Render a multiline guest script without shell quoting or pyautogui imports."""
    payload = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"import base64;exec(base64.b64decode({payload!r}))"


def _guest_process_snapshot(controller: Any, source_path: str) -> list[dict[str, Any]]:
    """Return exact argv matches for a detached guest application."""
    script = f"""
import json
import pathlib

expected = {source_path!r}.encode()
matches = []
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    command = entry / 'cmdline'
    try:
        data = command.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    argv = [part for part in data.split(b'\\0') if part]
    if expected in argv:
        matches.append({{'pid': int(entry.name), 'argv': [part.decode('utf-8', 'replace') for part in argv]}})
print(json.dumps(sorted(matches, key=lambda item: item['pid']), sort_keys=True))
"""
    output = _guest_exec(controller, _encoded_guest_script(script))
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GateError(f"could not parse guest-process snapshot: {output!r}") from exc
    if not isinstance(value, list):
        raise GateError(f"guest-process snapshot is not a list: {value!r}")
    return value


def _assert_no_guest_processes(controller: Any, source_path: str) -> None:
    matches = _guest_process_snapshot(controller, source_path)
    if matches:
        raise GateError(f"stale guest application processes: {matches}")


def _stop_guest_process(
    controller: Any,
    pid: int,
    source_path: str,
    *,
    term_timeout_s: float = 3.0,
    kill_timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Stop one detached app and prove that its PID and all exact-source peers exited."""
    script = f"""
import json
import os
import pathlib
import signal
import time

pid = {pid!r}
expected = {source_path!r}.encode()
term_timeout_s = {term_timeout_s!r}
kill_timeout_s = {kill_timeout_s!r}

def proc_path():
    return pathlib.Path('/proc') / str(pid)

def argv_for(target):
    try:
        data = (pathlib.Path('/proc') / str(target) / 'cmdline').read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [part for part in data.split(b'\\0') if part]

def wait_absent(timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not proc_path().exists():
            return True
        current = argv_for(pid)
        if current and expected not in current:
            raise RuntimeError('guest PID identity changed during teardown')
        time.sleep(0.02)
    return not proc_path().exists()

before = argv_for(pid)
action = 'already_absent'
if before is not None:
    if expected not in before:
        raise RuntimeError('refusing to signal PID with mismatched exact argv')
    action = 'term'
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not wait_absent(term_timeout_s):
        current = argv_for(pid)
        if current and expected not in current:
            raise RuntimeError('guest PID identity changed before SIGKILL')
        action = 'kill'
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not wait_absent(kill_timeout_s):
            raise RuntimeError('guest PID remained present after SIGKILL')

remaining = []
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    argv = argv_for(int(entry.name))
    if argv and expected in argv:
        remaining.append(int(entry.name))
if remaining:
    raise RuntimeError('guest application peers remain: ' + repr(sorted(remaining)))
print(json.dumps({{'pid': pid, 'action': action, 'remaining_exact_source_processes': 0}}, sort_keys=True))
"""
    output = _guest_exec(controller, _encoded_guest_script(script))
    try:
        result = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GateError(f"could not parse guest teardown evidence: {output!r}") from exc
    if not isinstance(result, dict) or result.get("remaining_exact_source_processes") != 0:
        raise GateError(f"guest teardown did not prove process absence: {result!r}")
    return result


def _x_client_inventory(controller: Any) -> dict[str, Any]:
    """Collect an observational xlsclients inventory for lifecycle stress evidence."""
    script = """
import json
import os
import subprocess

display = os.environ.get('DISPLAY', ':0')
result = subprocess.run(
    ['xlsclients', '-display', display],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=10,
)
lines = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
print(json.dumps({
    'display': display,
    'returncode': result.returncode,
    'count': len(lines),
    'clients': lines,
    'stderr': result.stderr.strip(),
}, sort_keys=True))
"""
    output = _guest_exec(controller, _encoded_guest_script(script))
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GateError(f"could not parse X-client inventory: {output!r}") from exc
    if not isinstance(value, dict) or value.get("returncode") != 0:
        raise GateError(f"xlsclients inventory failed: {value!r}")
    return value


@contextmanager
def leased_ports(lock_dir: Path) -> Iterator[dict[str, int]]:
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if lock_dir.is_symlink() or not os.access(lock_dir, os.W_OK):
        raise GateError(f"unsafe port lock directory: {lock_dir}")
    seed = int(os.environ.get("SLURM_JOB_ID", str(os.getpid())))
    names = ("server", "chromium", "vnc", "vlc")
    held = None
    ports = None
    for offset in range(1000):
        block = (seed + offset) % 1000
        candidate = [30000 + block * 10 + index for index in range(4)]
        path = lock_dir / f"proper_vm_{candidate[0]}.lock"
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        probes: list[socket.socket] = []
        try:
            for port in candidate:
                probe = socket.socket()
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # QEMU's ``hostfwd=tcp::PORT`` binds all host interfaces, so
                # the lease probe must use the same scope.  A loopback-only
                # probe can miss a listener bound to another node address and
                # let QEMU fail after the lease has already been granted.
                probe.bind(("0.0.0.0", port))
                probes.append(probe)
        except OSError:
            for probe in probes:
                probe.close()
            handle.close()
            continue
        for probe in probes:
            probe.close()
        held = handle
        ports = dict(zip(names, candidate, strict=True))
        break
    if held is None or ports is None:
        raise GateError("could not lease a collision-free OSWorld port block")
    try:
        yield ports
    finally:
        held.close()


def _guest_exec(controller: Any, code: str) -> str:
    result = controller.execute_python_command(code)
    if (
        not isinstance(result, dict)
        or result.get("status") != "success"
        or result.get("returncode") != 0
    ):
        raise GateError(f"guest command failed: {result!r}")
    return str(result.get("output", "")).strip()


def _install_guest_scene(controller: Any, cell: Any, operation: str, revision: str) -> int:
    _assert_no_guest_processes(controller, GUEST_SOURCE_PATH)
    source = base64.b64encode(GUEST_APP.read_bytes()).decode("ascii")
    image = base64.b64encode(cell.image_path.read_bytes()).decode("ascii")
    config = {
        "screen": [1920, 1080],
        "bbox": list(cell.bbox),
        "cursor": list(cell.cursor),
        "operation": operation,
        "revision": revision,
        "image_path": GUEST_IMAGE_PATH,
        "state_path": GUEST_STATE_PATH,
    }
    config_b64 = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
    code = (
        "import base64,pathlib,subprocess,sys;"
        f"pathlib.Path({GUEST_SOURCE_PATH!r}).write_bytes(base64.b64decode({source!r}));"
        f"pathlib.Path({GUEST_IMAGE_PATH!r}).write_bytes(base64.b64decode({image!r}));"
        f"pathlib.Path({GUEST_CONFIG_PATH!r}).write_bytes(base64.b64decode({config_b64!r}));"
        f"pathlib.Path({GUEST_STATE_PATH!r}).unlink(missing_ok=True);"
        f"p=subprocess.Popen([sys.executable,{GUEST_SOURCE_PATH!r},{GUEST_CONFIG_PATH!r}],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "print(p.pid)"
    )
    output = _guest_exec(controller, code)
    try:
        return int(output.splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise GateError(f"guest app did not return a pid: {output!r}") from exc


def _read_state(
    controller: Any,
    revision: str,
    timeout_s: float = 10.0,
    min_releases: int = 0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        output = _guest_exec(
            controller,
            f"import pathlib; p=pathlib.Path({GUEST_STATE_PATH!r}); print(p.read_text() if p.is_file() else '')",
        )
        last = output
        if output:
            try:
                state = json.loads(output.splitlines()[-1])
            except json.JSONDecodeError:
                state = None
            if (
                isinstance(state, dict)
                and state.get("revision") == revision
                and state.get("ready")
                and int(state.get("button_releases", 0)) >= min_releases
            ):
                return state
        time.sleep(0.1)
    raise GateError(f"guest app did not become ready: {last!r}")


def _plan_code(plan: tuple[tuple[Any, ...], ...]) -> str:
    rendered = ["import pyautogui,time; pyautogui.FAILSAFE=False; pyautogui.PAUSE=0"]
    for command in plan:
        if command[0] == "moveTo":
            rendered.append(f"pyautogui.moveTo({command[1]},{command[2]},duration={command[3]})")
        elif command[0] == "moveRel":
            rendered.append(f"pyautogui.moveRel({command[1]},{command[2]},duration={command[3]})")
        elif command[0] == "click":
            rendered.append("pyautogui.click(button='left')")
        elif command[0] == "mouseDown":
            rendered.append("pyautogui.mouseDown(button='left')")
        elif command[0] == "mouseUp":
            rendered.append("pyautogui.mouseUp(button='left')")
        else:
            raise GateError(f"unsupported live command: {command}")
    rendered.append("time.sleep(0.25)")
    return ";".join(rendered)


def _stop_guest(controller: Any, pid: int) -> None:
    _stop_guest_process(controller, pid, GUEST_SOURCE_PATH)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    validate_protocol(protocol)
    contract = Contract()
    cell = load_cells(protocol, contract)[0]
    vm = protocol["vm"]
    if Path(args.provider_source).resolve() != Path(vm["provider_source"]).resolve():
        raise GateError("provider path differs from protocol")
    if sha256_file(args.provider_source) != vm["provider_sha256"]:
        raise GateError("provider hash drift")
    if Path(args.qcow).resolve() != Path(vm["qcow"]).resolve():
        raise GateError("qcow path differs from protocol")
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise GateError("/dev/kvm is not readable and writable")
    os.environ["OSWORLD_QEMU_BIN"] = str(args.qemu_bin)
    os.environ["OSWORLD_QCOW2"] = str(args.qcow)
    os.environ["OSWORLD_VM_LOG_DIR"] = str(args.out / "vm_logs")
    sys.path.insert(0, str(args.osworld_root))
    sys.path.insert(0, str(args.provider_source.parent))
    args.out.mkdir(parents=True, exist_ok=True)
    with leased_ports(args.port_lock_dir) as ports:
        env_map = {
            "server": "OSWORLD_APPTAINER_SERVER_PORT",
            "chromium": "OSWORLD_APPTAINER_CHROMIUM_PORT",
            "vnc": "OSWORLD_APPTAINER_VNC_PORT",
            "vlc": "OSWORLD_APPTAINER_VLC_PORT",
        }
        for name, variable in env_map.items():
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
        records = []
        try:
            controller = env.controller
            screen_size = controller.get_vm_screen_size()
            if not isinstance(screen_size, dict) or (
                int(screen_size.get("width", -1)), int(screen_size.get("height", -1))
            ) != (1920, 1080):
                raise GateError("live VM screen-size mismatch")
            for semantic in ("absolute_toolcall", "move_rel", "deltatype_raw"):
                coord = contract.ideal_coord(semantic, cell.cursor, cell.target)
                endpoint = contract.apply_coord(semantic, cell.cursor, coord)
                for operation in ("click", "drag"):
                    revision = hashlib.sha256(
                        f"{cell.cell_id}|{semantic}|{operation}".encode("utf-8")
                    ).hexdigest()
                    pid = _install_guest_scene(controller, cell, operation, revision)
                    try:
                        _read_state(controller, revision)
                        _guest_exec(
                            controller,
                            f"import pyautogui; pyautogui.FAILSAFE=False; pyautogui.moveTo({cell.cursor[0]},{cell.cursor[1]}); print(tuple(pyautogui.position()))",
                        )
                        screenshot = controller.get_screenshot()
                        if not screenshot or rgb_sha256(screenshot) != rgb_sha256(cell.image_path.read_bytes()):
                            raise GateError("live screenshot pixels differ from frozen canonical PNG")
                        plan = actuation_plan(semantic, operation, cell.cursor, endpoint)
                        _guest_exec(controller, _plan_code(plan))
                        state = _read_state(controller, revision, min_releases=1)
                        success_key = "click_success" if operation == "click" else "drag_success"
                        if not state.get(success_key) or state.get("down"):
                            raise GateError(f"guest {semantic}/{operation} state mismatch: {state}")
                        cursor_text = _guest_exec(controller, "import pyautogui; print(list(pyautogui.position()))")
                        cursor = tuple(json.loads(cursor_text.splitlines()[-1]))
                        if cursor != endpoint:
                            raise GateError(f"guest cursor mismatch: {cursor} != {endpoint}")
                        records.append(
                            {
                                "semantic": semantic,
                                "operation": operation,
                                "revision": revision,
                                "endpoint": list(endpoint),
                                "state": state,
                                "plan": [list(command) for command in plan],
                            }
                        )
                    finally:
                        try:
                            _guest_exec(
                                controller,
                                "import pyautogui; pyautogui.mouseUp(button='left')",
                            )
                        finally:
                            _stop_guest(controller, pid)
        finally:
            env.close()
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_stage2_live_smoke",
        "status": "pass",
        "cpu_only": True,
        "gpu_used": False,
        "kvm_read_write": True,
        "protocol_sha256": sha256_file(args.protocol),
        "guest_app_sha256": sha256_file(GUEST_APP),
        "provider_sha256": sha256_file(args.provider_source),
        "cell_id": cell.cell_id,
        "canonical_rgb_sha256": rgb_sha256(cell.image_path.read_bytes()),
        "replays": records,
        "replay_count": len(records),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
    }
    (args.out / "live_smoke_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
