"""On-policy rollout collection with a COMPETENT ABSOLUTE-format teacher.

Path (a) of the on-policy program: serve an off-the-shelf Qwen3-VL teacher in
ABSOLUTE native ``computer_use`` (the format where it is strong, ~22-34% OSWorld),
roll it out closed-loop in the freeroll qemu+KVM VM, and log every step's parsed
absolute tool-call + cursor_before + resolved absolute target. A downstream
converter turns those into native-RELATIVE training records via diff-of-absolute.

Non-invasive w.r.t. shared freeroll.py: this reuses the shared runtime helpers
(osworld_runtime / osworld_vm_client / action_parser / osworld_system_prompts)
but implements its own rollout loop so we can (1) add the 0-999-grid -> screen
coordinate scaling the off-the-shelf computer_use prompt requires, and (2) log
per-step distillation metadata. It does NOT modify freeroll.py or the RL track's
freeroll usage.

COORDINATE HANDLING (the crux): the off-the-shelf ``computer_use_v1`` system
prompt declares "resolution is 1000x1000", i.e. Qwen3-VL emits coordinates on a
fixed 1000x1000 normalized grid (the canonical Qwen3-VL cookbook convention:
x_real = coord/1000 * screen_w, y_real = coord/1000 * screen_h, per-axis). freeroll's
absolute dispatch treats coordinates as LITERAL pixels, so on a 1920x1080 VM
everything would collapse into the top-left. We scale each emitted coordinate by
(screen_w/grid, screen_h/grid) before dispatch (--coord_grid, default 1000).
NOTE smart_resize needs NO separate handling here: coords are normalized to the
DECLARED 1000x1000 display, so the resized-pixel dims cancel and scaling by the
real VM dims is exactly right. Pass --coord_grid 0 for identity to A/B.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

_JUERGEN_EVAL = "/fast/home/franz.srambical/juergen/eval"
if _JUERGEN_EVAL not in sys.path:
    sys.path.insert(0, _JUERGEN_EVAL)

from action_parser import parse_computer_use_tool_call  # noqa: E402
from osworld_vm_client import OSWorldClient, StepResult  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from osworld_runtime import (  # noqa: E402
    _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _EVAL_DIR,
    _call_model, _wait_for, append_turn, build_loggable_messages,
    window_frame_labels,
)

_LOGGER = logging.getLogger("collect_abs")
_TERMINATE = "TERMINATE"


class ScaledOSWorldClient(OSWorldClient):
    """OSWorldClient that pre-scales ABSOLUTE computer_use coordinates from a
    0..grid normalized grid to real screen pixels before dispatch.

    Only the ``coordinate`` field is scaled; scroll/keys/text pass through. With
    ``coord_grid <= 0`` this is exactly the base client (identity)."""

    def __init__(self, base_url: str, *, coord_grid: int = 1000,
                 coord_grid_x: int | None = None, coord_grid_y: int | None = None, **kw):
        super().__init__(base_url, **kw)
        self.coord_grid = coord_grid
        self._sw, self._sh = self.screen_size()
        # per-axis grids: relative(0-999) -> both=coord_grid; absolute -> processed_w/h
        self._grid_x = coord_grid_x or coord_grid
        self._grid_y = coord_grid_y or coord_grid

    def dispatch_computer_use(self, arguments: dict, *, relative: bool = False) -> StepResult:
        if not relative and self._grid_x and self._grid_x > 0 and "coordinate" in arguments:
            raw = arguments.get("coordinate")
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                try:
                    x = float(raw[0]) * self._sw / self._grid_x
                    y = float(raw[1]) * self._sh / self._grid_y
                    arguments = dict(arguments)
                    arguments["coordinate"] = [int(round(x)), int(round(y))]
                except (TypeError, ValueError):
                    pass
        return super().dispatch_computer_use(arguments, relative=relative)


@dataclass
class StepLog:
    step: int
    action_text: str
    parsed: dict | None
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    intended_target: tuple[int, int] | None
    events_dispatched: list[str]
    parse_error: str | None
    elapsed_s: float


def _computer_use_terminate_status(text: str) -> str | None:
    try:
        call = parse_computer_use_tool_call(text)
    except (TypeError, ValueError):
        return None
    if str(call.arguments.get("action", "")).strip().lower() != "terminate":
        return None
    return str(call.arguments.get("status", "success")).strip().lower() or "success"


def _run_rollout(
    *, sglang_url, api_key, model, osworld_url, output_dir, max_steps, instruction,
    system_prompt, n_history_frames, persist_instruction, max_tokens, temperature,
    coord_grid, settle_s, settle_stable_timeout_s, settle_poll_s,
    coord_grid_x=None, coord_grid_y=None,
) -> dict:
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    traj_path = output_dir / "trajectory.jsonl"
    conv_path = output_dir / "conversation.jsonl"
    gif_path = output_dir / "rollout.gif"

    client = ScaledOSWorldClient(osworld_url, coord_grid=coord_grid,
                                 coord_grid_x=coord_grid_x, coord_grid_y=coord_grid_y)
    client.wait_ready()
    sw, sh = client.screen_size()
    _LOGGER.info("VM %dx%d max_steps=%d coord_grid=%d instr=%r", sw, sh, max_steps, coord_grid, instruction)

    frame = client.screenshot()
    frames_for_gif = [frame.copy()]
    frame.save(steps_dir / "step_000.png")
    recent_frames = [frame]
    recent_actions: list[str] = []

    t_start = time.time()
    steps: list[StepLog] = []
    stop_reason = "max_steps"
    parse_errors = 0

    with traj_path.open("w") as traj_f, conv_path.open("w") as conv_f:
        traj_f.write(json.dumps({"step_num": 0, "action": "<reset>", "response": "<reset>",
                                 "reward": 0.0, "done": False, "info": {}}) + "\n")
        traj_f.flush()
        for step in range(1, max_steps + 1):
            t0 = time.time()
            instr_used = instruction if (step == 1 or persist_instruction) else None
            frame_labels = window_frame_labels(step, len(recent_frames))
            loggable = build_loggable_messages(system_prompt=system_prompt, instruction=instr_used,
                                               recent_actions=recent_actions, frame_labels=frame_labels)
            (steps_dir / f"prompt_{step:03d}.json").write_text(json.dumps(loggable, indent=2))
            try:
                action_text = _call_model(
                    sglang_url=sglang_url, api_key=api_key, model=model,
                    system_prompt=system_prompt, instruction=instr_used,
                    recent_frames=recent_frames, recent_actions=recent_actions,
                    max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                _LOGGER.error("step %d model call failed: %s", step, e)
                stop_reason = "model_error"
                break

            conv_f.write(json.dumps({"step": step, "messages": loggable, "response": action_text}) + "\n")
            conv_f.flush()
            _LOGGER.info("step %d | frame %s | response=%r", step, frame_labels[-1], action_text[:300])

            cu_status = _computer_use_terminate_status(action_text)
            if cu_status is not None:
                stop_reason = "terminate" if cu_status == "success" else f"terminate_{cu_status}"
                try:
                    cursor = client.cursor_position()
                    frame = client.screenshot()
                except Exception as e:
                    _LOGGER.error("terminate snapshot failed: %s", e)
                    stop_reason = "screenshot_error"
                    break
                frames_for_gif.append(frame.copy())
                frame.save(steps_dir / f"step_{step:03d}.png")
                append_turn(recent_frames, recent_actions, frame, action_text, n_history_frames=n_history_frames)
                sl = StepLog(step, action_text, {"terminate": True, "computer_use_status": cu_status},
                             cursor, cursor, cursor, [], None, time.time() - t0)
                steps.append(sl)
                traj_f.write(json.dumps({"step_num": step, "action": action_text, "response": action_text,
                                         "reward": 0.0, "done": True, "info": asdict(sl)}) + "\n")
                traj_f.flush()
                break

            parse_err = None
            parsed = None
            sr_dict = None
            try:
                computer_call = parse_computer_use_tool_call(action_text)
                parsed = {"computer_use": computer_call.arguments,
                          "no_op": str(computer_call.arguments.get("action", "")).strip().lower() == "answer"}
                sr = client.dispatch_computer_use(computer_call.arguments)
                sr_dict = {"cursor_before": list(sr.cursor_before), "cursor_after": list(sr.cursor_after),
                           "intended_target": list(sr.intended_target), "events_dispatched": sr.events_dispatched}
            except (TypeError, ValueError) as e:
                parse_err = str(e)
                parse_errors += 1
                _LOGGER.warning("step %d parse/dispatch error %s on %r", step, e, action_text[:200])

            try:
                frame = client.screenshot_settled(min_delay_s=settle_s,
                                                   stability_timeout_s=settle_stable_timeout_s,
                                                   poll_s=settle_poll_s)
            except Exception as e:
                _LOGGER.error("screenshot failed: %s", e)
                stop_reason = "screenshot_error"
                break
            frames_for_gif.append(frame.copy())
            frame.save(steps_dir / f"step_{step:03d}.png")
            append_turn(recent_frames, recent_actions, frame, action_text, n_history_frames=n_history_frames)

            sl = StepLog(step, action_text, parsed,
                         tuple(sr_dict["cursor_before"]) if sr_dict else (-1, -1),
                         tuple(sr_dict["cursor_after"]) if sr_dict else (-1, -1),
                         tuple(sr_dict["intended_target"]) if sr_dict else None,
                         sr_dict["events_dispatched"] if sr_dict else [], parse_err, time.time() - t0)
            steps.append(sl)
            traj_f.write(json.dumps({"step_num": step, "action": action_text, "response": action_text,
                                     "reward": 0.0, "done": False, "info": asdict(sl)}) + "\n")
            traj_f.flush()

    elapsed_s = time.time() - t_start
    if len(frames_for_gif) > 1:
        small = [f.resize((min(960, f.width), int(f.height * min(960, f.width) / f.width))) for f in frames_for_gif]
        small[0].save(gif_path, save_all=True, append_images=small[1:], duration=300, loop=0, optimize=True)

    return {
        "schema_version": 1, "model": model, "screen_size": [sw, sh], "coord_grid": coord_grid,
        "n_steps": len(steps), "max_steps": max_steps, "stop_reason": stop_reason,
        "parse_errors": parse_errors, "elapsed_s": elapsed_s, "instruction": instruction,
        "system_prompt": system_prompt, "n_history_frames": n_history_frames,
        "persist_instruction": persist_instruction, "max_tokens": max_tokens, "temperature": temperature,
        "traj_path": str(traj_path), "conversation_path": str(conv_path), "gif_path": str(gif_path),
    }


def _slug(text, idx):
    import re
    base = re.sub(r"[^a-z0-9]+", "-", (text or "no-instruction").lower()).strip("-")
    return f"task_{idx:03d}_{base[:40] or 'task'}"


def _boot_vm(*, qemu_bin, qcow2, vm_port, vnc_port, log_path):
    return subprocess.Popen(
        [qemu_bin, "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
         "-machine", "type=q35,accel=kvm",
         "-drive", f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
         "-netdev", f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
         "-device", "virtio-net-pci,netdev=net0", "-display", "none", "-nographic"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


def _parse_instructions(path):
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--instructions_file", required=True, help="one instruction per line")
    p.add_argument("--run_prefix", default="",
                   help="prefix for run-slug dirs so PARALLEL sharded collect jobs can "
                        "write into ONE shared output_dir without task-index collisions "
                        "(e.g. 'sh0_'). Filter/convert then scan the single dir unchanged.")
    p.add_argument("--samples_per_instruction", type=int, default=1,
                   help=">1 with temperature>0 gives best-of-N candidates for the filter")
    p.add_argument("--max_steps", type=int, default=15)
    p.add_argument("--system_prompt_id", default="computer_use_v1")
    p.add_argument("--coord_grid", type=int, default=1000)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_history_frames", type=int, default=6)
    p.add_argument("--persist_instruction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--settle_s", type=float, default=0.4)
    p.add_argument("--settle_stable_timeout_s", type=float, default=2.0)
    p.add_argument("--settle_poll_s", type=float, default=0.1)
    p.add_argument("--sglang_port", type=int, default=30000)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--mem_fraction_static", type=float, default=0.80)
    p.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    p.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    args = p.parse_args()

    if args.system_prompt_id not in SYSTEM_PROMPTS:
        print(f"Unknown system_prompt_id {args.system_prompt_id!r}; have {list(SYSTEM_PROMPTS)}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(output_dir / "collect.log"))
    instructions = _parse_instructions(args.instructions_file)

    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    vm_port = 5000 + job_mod
    vnc_port = 5900 + job_mod
    sglang_port = (30000 + job_mod) if args.sglang_port == 30000 else args.sglang_port

    _procs: list[subprocess.Popen] = []

    def _cleanup():
        for proc in _procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))

    _LOGGER.info("starting sglang model=%s port=%d", args.model_path, sglang_port)
    sglang_proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path, "--host", "0.0.0.0", "--port", str(sglang_port),
         "--api-key", args.sglang_api_key, "--mem-fraction-static", str(args.mem_fraction_static),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR), stdout=open(output_dir / "sglang.log", "w"), stderr=subprocess.STDOUT)
    _procs.append(sglang_proc)

    runs = []
    first = True
    idx = -1
    for instr in instructions:
        for s in range(args.samples_per_instruction):
            idx += 1
            slug = args.run_prefix + _slug(instr, idx) + (f"_s{s}" if args.samples_per_instruction > 1 else "")
            run_dir = output_dir / slug
            run_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info("=== %s : %r (sample %d) ===", slug, instr, s)
            vm_proc = _boot_vm(qemu_bin=args.qemu_bin, qcow2=args.qcow2, vm_port=vm_port,
                               vnc_port=vnc_port, log_path=run_dir / "qemu.log")
            _procs.append(vm_proc)
            try:
                _wait_for(f"http://localhost:{vm_port}/screenshot", proc=vm_proc,
                          poll_s=5, max_polls=60, label="VM")
                if first:
                    _wait_for(f"http://localhost:{sglang_port}/health_generate",
                              headers={"Authorization": f"Bearer {args.sglang_api_key}"},
                              proc=sglang_proc, poll_s=10, max_polls=180, label="sglang")
                    first = False
                result = _run_rollout(
                    sglang_url=f"http://localhost:{sglang_port}/v1", api_key=args.sglang_api_key,
                    model=args.model_path, osworld_url=f"http://localhost:{vm_port}", output_dir=run_dir,
                    max_steps=args.max_steps, instruction=instr,
                    system_prompt=SYSTEM_PROMPTS[args.system_prompt_id],
                    n_history_frames=args.n_history_frames, persist_instruction=args.persist_instruction,
                    max_tokens=args.max_tokens, temperature=args.temperature, coord_grid=args.coord_grid,
                    settle_s=args.settle_s, settle_stable_timeout_s=args.settle_stable_timeout_s,
                    settle_poll_s=args.settle_poll_s)
                result["instruction"] = instr
                result["sample"] = s
                result["slug"] = slug
                (run_dir / "result.json").write_text(json.dumps(result, indent=2))
                runs.append({"slug": slug, "instruction": instr, "sample": s,
                             "stop_reason": result["stop_reason"], "n_steps": result["n_steps"],
                             "parse_errors": result["parse_errors"]})
            except Exception as e:
                _LOGGER.error("rollout %s failed: %s", slug, e)
                runs.append({"slug": slug, "instruction": instr, "sample": s, "error": str(e)})
            finally:
                if vm_proc.poll() is None:
                    vm_proc.terminate()
                    try:
                        vm_proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        vm_proc.kill()
                if vm_proc in _procs:
                    _procs.remove(vm_proc)

    (output_dir / "index.json").write_text(json.dumps(
        {"schema_version": 1, "model": args.model_path, "system_prompt_id": args.system_prompt_id,
         "coord_grid": args.coord_grid, "n_runs": len(runs), "runs": runs}, indent=2))
    _LOGGER.info("done. %d rollouts -> %s", len(runs), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
