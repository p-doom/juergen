"""OSWorld freeroll — single entry point for labctl freeroll recipes.

Boots the VM via native qemu+KVM, starts sglang, runs a closed-loop
rollout, writes result.json for labctl.

Replaces the former run_freeroll_recipe.sh + rollout.py split.
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
from typing import Any

from PIL import Image

# Eval primitives now live in juergen/eval/ (see osworld_grounding_runner.py
# for the rationale: the shared modules are pre-existing crowd-cast BC
# infrastructure, not experiment-specific glue). Add juergen/eval to
# sys.path so freeroll runs from its slurm/dev/franz/.../crowd-cast-bc/
# location can still pull these in. Labctl recipes that point at this
# script already use `uv run --project=/fast/.../juergen/eval`, so the
# venv has the same Python; only the import path needs hinting.
_JUERGEN_EVAL = str(Path(__file__).resolve().parent)
if _JUERGEN_EVAL not in sys.path:
    sys.path.insert(0, _JUERGEN_EVAL)

from action_parser import parse_action_tolerant  # noqa: E402
from osworld_vm_client import OSWorldClient  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from osworld_runtime import (  # noqa: E402
    _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _EVAL_DIR,
    _call_model, _pil_to_data_url, _wait_for,
)

_LOGGER = logging.getLogger(__name__)


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


# _pil_to_data_url / _call_model / _wait_for moved to
# juergen/eval/osworld_runtime.py; imported above.


def _run_rollout(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    osworld_url: str,
    output_dir: Path,
    max_steps: int,
    instruction: str | None,
    system_prompt: str,
    n_history_frames: int,
    max_tokens: int,
    temperature: float,
    save_frames: bool,
    stop_on_click: bool,
) -> dict:
    steps_dir = output_dir / "steps"
    if save_frames:
        steps_dir.mkdir(exist_ok=True)
    traj_path = output_dir / "trajectory.jsonl"
    gif_path = output_dir / "rollout.gif"

    client = OSWorldClient(osworld_url)
    client.wait_ready()
    sw, sh = client.screen_size()
    _LOGGER.info("VM screen %dx%d; max_steps=%d; instruction=%r", sw, sh, max_steps, instruction)

    frame = client.screenshot()
    frames_for_gif: list[Image.Image] = [frame.copy()]
    if save_frames:
        frame.save(steps_dir / "step_000.png")
    recent_frames: list[Image.Image] = [frame]

    t_start = time.time()
    steps: list[StepLog] = []
    stop_reason = "max_steps"
    parse_errors = 0

    with traj_path.open("w") as traj_f:
        traj_f.write(json.dumps({
            "step_num": 0, "action": "<reset>", "response": "<reset>",
            "reward": 0.0, "done": False, "info": {},
        }) + "\n")
        traj_f.flush()

        for step in range(1, max_steps + 1):
            t0 = time.time()
            try:
                action_text = _call_model(
                    sglang_url=sglang_url, api_key=api_key, model=model,
                    system_prompt=system_prompt,
                    instruction=instruction if step == 1 else None,
                    recent_frames=recent_frames,
                    max_tokens=max_tokens, temperature=temperature,
                )
            except Exception as e:
                _LOGGER.error("step %d: model call failed: %s", step, e)
                stop_reason = "model_error"
                break

            parse_err: str | None = None
            parsed = None
            sr_dict: dict | None = None
            try:
                action = parse_action_tolerant(action_text)
                parsed = {
                    "dx": action.dx, "dy": action.dy,
                    "scroll": action.scroll, "no_op": action.no_op,
                    "events": [{"kind": e.kind, "what": e.what, "mouse_button": e.mouse_button}
                               for e in action.events],
                }
                sr = client.dispatch_action(action)
                sr_dict = {
                    "cursor_before": list(sr.cursor_before),
                    "cursor_after": list(sr.cursor_after),
                    "intended_target": list(sr.intended_target),
                    "events_dispatched": sr.events_dispatched,
                }
            except ValueError as e:
                parse_err = str(e)
                parse_errors += 1
                _LOGGER.warning("step %d: parse error %s on %r", step, e, action_text)

            try:
                frame = client.screenshot()
            except Exception as e:
                _LOGGER.error("step %d: screenshot failed: %s", step, e)
                stop_reason = "screenshot_error"
                break

            frames_for_gif.append(frame.copy())
            if save_frames:
                frame.save(steps_dir / f"step_{step:03d}.png")
            recent_frames.append(frame)
            if len(recent_frames) > n_history_frames:
                recent_frames = recent_frames[-n_history_frames:]

            step_log = StepLog(
                step=step, action_text=action_text, parsed=parsed,
                cursor_before=tuple(sr_dict["cursor_before"]) if sr_dict else (-1, -1),
                cursor_after=tuple(sr_dict["cursor_after"]) if sr_dict else (-1, -1),
                intended_target=tuple(sr_dict["intended_target"]) if sr_dict else None,
                events_dispatched=sr_dict["events_dispatched"] if sr_dict else [],
                parse_error=parse_err, elapsed_s=time.time() - t0,
            )
            steps.append(step_log)
            traj_f.write(json.dumps({
                "step_num": step, "action": action_text, "response": action_text,
                "reward": 0.0, "done": False, "info": asdict(step_log),
            }) + "\n")
            traj_f.flush()

            if (stop_on_click and parsed and not parsed["no_op"]
                    and any(e["what"] == "LMB" and e["kind"] == "press"
                            for e in parsed["events"])):
                stop_reason = "click"
                _LOGGER.info("step %d: model clicked, stopping", step)
                break

    elapsed_s = time.time() - t_start

    if len(frames_for_gif) > 1:
        small = [f.resize((min(960, f.width), int(f.height * min(960, f.width) / f.width)))
                 for f in frames_for_gif]
        small[0].save(gif_path, save_all=True, append_images=small[1:],
                      duration=300, loop=0, optimize=True)
        _LOGGER.info("wrote GIF %s (%d frames)", gif_path, len(small))

    return {
        "schema_version": 1,
        "model": model,
        "osworld_url": osworld_url,
        "sglang_url": sglang_url,
        "screen_size": [sw, sh],
        "n_steps": len(steps),
        "max_steps": max_steps,
        "stop_reason": stop_reason,
        "parse_errors": parse_errors,
        "click": stop_reason == "click",
        "elapsed_s": elapsed_s,
        "instruction": instruction,
        "system_prompt": system_prompt,
        "n_history_frames": n_history_frames,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "traj_path": str(traj_path),
        "gif_path": str(gif_path),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--instruction", default=None)
    p.add_argument("--system_prompt_id", default="training_v1")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_history_frames", type=int, default=1)
    p.add_argument("--no_frames", action="store_true")
    p.add_argument("--stop_on_click", action="store_true")
    p.add_argument("--sglang_port", type=int, default=30000)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--mem_fraction_static", type=float, default=0.40)
    p.add_argument("--qcow2", default=_DEFAULT_QCOW2)
    p.add_argument("--qemu_bin", default=_DEFAULT_QEMU_BIN)
    args = p.parse_args()

    if args.system_prompt_id not in SYSTEM_PROMPTS:
        print(f"Unknown --system_prompt_id {args.system_prompt_id!r}. "
              f"Available: {list(SYSTEM_PROMPTS)}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger().addHandler(
        logging.FileHandler(output_dir / "freeroll.log")
    )

    # Port isolation: shift from SLURM_JOB_ID so concurrent jobs on the same
    # node don't collide. Range: 5000 + (job_id % 200) * 10 → 5000..6990.
    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    vm_port = 5000 + job_mod
    vnc_port = 5900 + job_mod
    sglang_port = (30000 + job_mod) if args.sglang_port == 30000 else args.sglang_port

    _LOGGER.info("model=%s instruction=%s max_steps=%d vm_port=%d sglang_port=%d system_prompt_id=%s",
                 args.model_path, args.instruction or "<none>", args.max_steps,
                 vm_port, sglang_port, args.system_prompt_id)

    # Cleanup: terminate VM and sglang on exit.
    _procs: list[subprocess.Popen] = []

    def _cleanup() -> None:
        _LOGGER.info("cleanup...")
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

    # Boot VM via native qemu+KVM (apptainer userns strips KVM ioctls on hai-*).
    _LOGGER.info("starting VM...")
    vm_proc = subprocess.Popen(
        [args.qemu_bin,
         "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
         "-machine", "type=q35,accel=kvm",
         "-drive", f"file={args.qcow2},if=virtio,format=qcow2,snapshot=on",
         "-netdev", f"user,id=net0,hostfwd=tcp::{vm_port}-:5000,hostfwd=tcp::{vnc_port}-:5900",
         "-device", "virtio-net-pci,netdev=net0",
         "-display", "none", "-nographic"],
        stdout=open(output_dir / "qemu.log", "w"),
        stderr=subprocess.STDOUT,
    )
    _procs.append(vm_proc)
    _LOGGER.info("qemu pid=%d", vm_proc.pid)

    # Start sglang.
    _LOGGER.info("starting sglang...")
    sglang_proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path,
         "--host", "0.0.0.0",
         "--port", str(sglang_port),
         "--api-key", args.sglang_api_key,
         "--mem-fraction-static", str(args.mem_fraction_static),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR),
        stdout=open(output_dir / "sglang.log", "w"),
        stderr=subprocess.STDOUT,
    )
    _procs.append(sglang_proc)
    _LOGGER.info("sglang pid=%d", sglang_proc.pid)

    # Wait for VM (up to 5 min).
    _wait_for(f"http://localhost:{vm_port}/screenshot",
              proc=vm_proc, poll_s=5, max_polls=60, label="VM")

    # Wait for sglang (up to 20 min).
    _wait_for(f"http://localhost:{sglang_port}/health_generate",
              headers={"Authorization": f"Bearer {args.sglang_api_key}"},
              proc=sglang_proc, poll_s=10, max_polls=120, label="sglang")

    result = _run_rollout(
        sglang_url=f"http://localhost:{sglang_port}/v1",
        api_key=args.sglang_api_key,
        model=args.model_path,
        osworld_url=f"http://localhost:{vm_port}",
        output_dir=output_dir,
        max_steps=args.max_steps,
        instruction=args.instruction,
        system_prompt=SYSTEM_PROMPTS[args.system_prompt_id],
        n_history_frames=args.n_history_frames,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_frames=not args.no_frames,
        stop_on_click=args.stop_on_click,
    )

    result_path = output_dir / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    _LOGGER.info("done. outputs under %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
