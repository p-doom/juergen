"""Grounding eval — closed-loop OSWorld cursor-landing benchmark.

For each (target, cursor-start-regime) in bboxes.jsonl × {near, medium, far}:
  - boot OSWorld VM (qemu+KVM, fresh snapshot per rollout)
  - load the target's OSWorld task JSON and run SetupController.setup(...) so
    the desktop is in the right app/state for the labeled bbox
  - move cursor to a deterministic starting position based on regime
  - roll out for K=100 frames at the model's training fps
  - score reach: did the cursor enter the labeled bbox at any frame?

One sglang server is shared across all rollouts; a fresh VM is launched
per rollout so the in-VM state stays clean. Cursor history is autoregressive
(model sees N most recent frames + the system prompt + the bbox's
natural-language instruction once at turn 1).

Task setup uses OSWorld's SetupController against our manually-launched VM
(rather than going through DesktopEnv, which uses the apptainer provider
that strips KVM ioctls on hai-* nodes). The qemu hostfwd is extended with
a chromium-port forward so SetupController's playwright-over-CDP path can
reach Chrome's debug interface inside the VM.

Known limitations:
  - No macOS-cursor sprite compositing. If the cursor-OOD between training
    (macOS) and inference (Linux) confounds the model, we'll see it.
  - Setup commands that need a non-empty client_password (proxy, sudo) will
    fail; our 29 targets shouldn't trip these.

Usage:
    python3 grounding_eval.py \\
        --bboxes_jsonl /fast/.../osworld_grounding_eval_v0/bboxes.jsonl \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --output_dir /fast/.../grounding_results/<run_id> \\
        --limit 1     # smoke test on first target before scaling up
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
import logging
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

# Sibling modules in juergen/eval/.
from action_parser import parse_action_tolerant
from osworld_vm_client import OSWorldClient
from osworld_system_prompts import SYSTEM_PROMPTS
from osworld_runtime import (
    _DEFAULT_QCOW2, _DEFAULT_QEMU_BIN, _EVAL_DIR,
    _call_model, _wait_for, append_turn,
    build_loggable_messages, window_frame_labels,
)

# OSWorld imports — SetupController + task JSONs. Importing the module has
# the side effect of trying to load a proxy config; we don't use proxies,
# the warning at import time is harmless.
_OSWORLD_ROOT = Path(
    os.environ.get("OSWORLD_ROOT", "/fast/home/franz.srambical/OSWorld")
)
if str(_OSWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_OSWORLD_ROOT))
from desktop_env.controllers.setup import SetupController  # noqa: E402

_LOGGER = logging.getLogger(__name__)

REGIMES: tuple[str, ...] = ("near", "medium", "far")


# ---------------------------------------------------------------------------
# bboxes.jsonl → grounding targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A single labeled grounding target."""
    idx: int                # bbox-list index
    task_id: str            # OSWorld task uuid (extracted from image_path)
    app: str
    instruction: str        # NL instruction the model receives at turn 1
    bbox: tuple[int, int, int, int]  # xyxy in 1920×1080
    image_path: Path        # the original labeled screenshot (for debugging)


def _parse_target(label: dict) -> Target:
    """Extract (task_id, app) from a bbox entry's image_path.

    Expected path shape (from the freeroll cache):
      .../<eval_set>/<app>/<task_id>/steps/step_NNN.png
    """
    parts = Path(label["image_path"]).parts
    # parts[-1] is step_NNN.png; parts[-2] is "steps"; parts[-3] is task_id.
    if parts[-2] != "steps":
        raise ValueError(f"unexpected image_path shape: {label['image_path']!r}")
    return Target(
        idx=int(label["idx"]),
        task_id=parts[-3],
        app=str(label["app"]),
        instruction=str(label["instruction"]),
        bbox=tuple(int(v) for v in label["bbox_xyxy"]),
        image_path=Path(label["image_path"]),
    )


def load_targets(jsonl_path: Path) -> list[Target]:
    out: list[Target] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(_parse_target(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# OSWorld task setup
# ---------------------------------------------------------------------------


def load_osworld_task(app: str, task_id: str) -> dict:
    """Load the OSWorld task JSON at evaluation_examples/examples/<app>/<id>.json."""
    p = _OSWORLD_ROOT / "evaluation_examples" / "examples" / app / f"{task_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"OSWorld task JSON not found: {p}")
    return json.loads(p.read_text())


def trajectory_path_for(target: Target) -> Path | None:
    """Cached freeroll/fullbench traj.jsonl associated with a target.

    Our labeled bboxes were sampled from step_001.png frames sitting next
    to the original rollout's traj.jsonl. The path shape is:
      .../<app>/<task_id>/steps/step_001.png  →  .../<app>/<task_id>/traj.jsonl
    Returns None if the file isn't present (then replay is skipped and the
    rollout starts from the post-task-setup state, which is fine for
    targets whose labeled state IS step_000).
    """
    p = target.image_path.parent.parent / "traj.jsonl"
    return p if p.is_file() else None


class _GroundingSetupController(SetupController):
    """SetupController with a working ``_replay_setup`` for the cached
    OSWorld fullbench / freeroll trajectories our labels were sampled from.

    Upstream OSWorld's ``_replay_setup`` is a ``NotImplementedError`` stub.
    Our implementation reads the first N non-reset rows of a ``traj.jsonl``
    and dispatches each row's ``action`` field. ``action`` may be either:
      - a raw pyautogui expression like ``pyautogui.hotkey('ctrl','p')``
        (the format OSWorld's qwen3vl_agent emits to disk), or
      - one of our BC model's delta tokens (``100 -3 0 ; +LMB -LMB``).
    We try the delta-token parse first via ``parse_action_tolerant`` and
    fall back to raw ``client.execute`` — so the same replay step works
    against either cache origin.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Reuse the same in-VM Flask agent the rollout uses; bound to
        # self.http_server (== f"http://{vm_ip}:{server_port}").
        self._osworld_client = OSWorldClient(self.http_server)

    def _replay_setup(
        self,
        trajectory: str,
        n_steps: int = 1,
        sleep_after_each_s: float = 0.5,
    ) -> None:
        traj_path = Path(trajectory)
        replayed = 0
        for raw_line in traj_path.read_text().splitlines():
            if not raw_line.strip():
                continue
            entry = json.loads(raw_line)
            if entry.get("step_num", 0) == 0:
                continue
            if replayed >= n_steps:
                break
            action_text = (entry.get("action") or "").strip()
            if not action_text or action_text == "<reset>":
                continue
            dispatched = False
            try:
                parsed = parse_action_tolerant(action_text)
                self._osworld_client.dispatch_action(parsed)
                dispatched = True
            except (ValueError, TypeError):
                # Not delta-token format; try as raw pyautogui expression.
                try:
                    self._osworld_client.execute(action_text)
                    dispatched = True
                except Exception as e:
                    _LOGGER.warning(
                        "replay: skipping unexecutable action %r: %s",
                        action_text[:120], e,
                    )
            if dispatched:
                replayed += 1
                # Give the UI time to render the post-action state. Matches
                # the OSWorld fullbench runner's default sleep_after_execution.
                time.sleep(sleep_after_each_s)
        _LOGGER.info(
            "replay: dispatched %d action(s) from %s", replayed, traj_path,
        )


def run_task_setup(
    *,
    task: dict,
    vm_port: int,
    chromium_port: int,
    vlc_port: int,
    cache_dir: Path,
    screen_w: int,
    screen_h: int,
    replay_trajectory: Path | None = None,
    replay_n_steps: int = 1,
) -> None:
    """Run the task's `config` setup commands against our manually-launched VM.

    Reuses OSWorld's ``SetupController`` (no DesktopEnv dependency, so the
    VM lifecycle stays under our qemu+KVM control). When ``replay_trajectory``
    is provided, we append a ``replay`` setup step that dispatches the first
    ``replay_n_steps`` cached actions — bringing the desktop from the
    post-task-setup state to the post-replay state our labeled bboxes were
    drawn against (typically step_001).
    """
    cfg = list(task.get("config", []))
    if replay_trajectory is not None:
        cfg.append({
            "type": "replay",
            "parameters": {
                "trajectory": str(replay_trajectory),
                "n_steps": int(replay_n_steps),
            },
        })
    if not cfg:
        _LOGGER.info("task %s: empty config, skipping setup", task.get("id", "?"))
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    setup_controller = _GroundingSetupController(
        vm_ip="localhost",
        server_port=vm_port,
        chromium_port=chromium_port,
        vlc_port=vlc_port,
        cache_dir=str(cache_dir),
        client_password="",  # not used by the command types we exercise
        screen_width=screen_w,
        screen_height=screen_h,
    )
    setup_controller.setup(cfg)


# ---------------------------------------------------------------------------
# Stratified cursor-start sampling
# ---------------------------------------------------------------------------


def _seed_for(target: Target, regime: str) -> int:
    """Deterministic 32-bit seed so reruns place the cursor identically."""
    h = (hash((target.task_id, regime, "v0")) & 0xFFFFFFFF)
    return int(h)


def cursor_start(
    target: Target, screen_w: int, screen_h: int, regime: str,
) -> tuple[int, int]:
    """Deterministic starting cursor position by distance regime.

      near:    >= 200 px from bbox center at a seeded angle, always outside bbox
      medium:  >= 500 px from bbox center at a seeded angle, always outside bbox
      far:     screen-mirrored: cursor at (sw-cx, sh-cy)

    Always clipped to the screen. The minimum radius for near/medium is
    raised dynamically when the bbox is large enough that a fixed-radius
    sample would land inside it — otherwise a giant target like a full
    terminal window would get a degenerate "reach at step 0" credit.
    """
    cx = (target.bbox[0] + target.bbox[2]) // 2
    cy = (target.bbox[1] + target.bbox[3]) // 2
    if regime == "far":
        sx = max(0, min(screen_w - 1, screen_w - cx))
        sy = max(0, min(screen_h - 1, screen_h - cy))
        return sx, sy

    rng = random.Random(_seed_for(target, regime))
    base = {"near": 200, "medium": 500}[regime]
    bw = target.bbox[2] - target.bbox[0]
    bh = target.bbox[3] - target.bbox[1]
    # Half-diagonal + 30 px buffer ensures the *unclipped* sample is outside
    # the bbox at any angle. Doesn't help against screen-edge clipping,
    # which is handled by the retry loop below.
    min_dist = int(math.hypot(bw, bh) / 2) + 30
    dist = max(base, min_dist)
    # Retry: for targets near a screen edge, some angles produce
    # off-screen points that, once clipped, fall back inside the bbox.
    # Try 8 deterministic angles before giving up.
    for k in range(8):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        sx = max(0, min(screen_w - 1, cx + int(round(dist * math.cos(angle)))))
        sy = max(0, min(screen_h - 1, cy + int(round(dist * math.sin(angle)))))
        if not in_bbox((sx, sy), target.bbox):
            return sx, sy
    # All 8 angles failed — bbox must be in a corner with very little
    # outside-of-bbox screen near it. Push as far from the bbox as
    # possible along the screen diagonal away from bbox center.
    fx = 0 if cx > screen_w // 2 else screen_w - 1
    fy = 0 if cy > screen_h // 2 else screen_h - 1
    return fx, fy


def in_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= pos[0] < bbox[2] and bbox[1] <= pos[1] < bbox[3]


# ---------------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    step: int
    action_text: str
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    in_bbox: bool
    parse_error: str | None
    elapsed_s: float


def _run_grounding_rollout(
    *,
    client: OSWorldClient,
    target: Target,
    regime: str,
    cursor_start_pos: tuple[int, int],
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_steps: int,
    n_history_frames: int,
    persist_instruction: bool,
    max_tokens: int,
    temperature: float,
    output_dir: Path,
    save_frames: bool,
) -> dict:
    """Roll out the model and score reach."""
    steps_dir = output_dir / "steps"
    if save_frames:
        steps_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / "trajectory.jsonl"

    sw, sh = client.screen_size()

    # Move cursor to the stratified starting position before any frame is
    # captured for the model. The model sees this as the cursor's "initial
    # state" — and the bbox reach test uses cursor positions AFTER each
    # delta application.
    client.execute(f"pyautogui.moveTo({cursor_start_pos[0]}, {cursor_start_pos[1]})")
    # Verify the move (in case the VM ignored it for any reason — e.g., the
    # in-VM agent hadn't fully bound to its display when we hit /execute).
    pos = client.cursor_position()
    if pos != cursor_start_pos:
        _LOGGER.warning(
            "cursor_start: requested %s, got %s — using actual position",
            cursor_start_pos, pos,
        )

    frame = client.screenshot()
    if save_frames:
        frame.save(steps_dir / "step_000.png")
    recent_frames: list[Image.Image] = [frame]
    recent_actions: list[str] = []

    t_start = time.time()
    steps: list[StepLog] = []
    reach_frame: int = -1
    stop_reason = "max_steps"
    parse_errors = 0

    with traj_path.open("w") as traj_f:
        traj_f.write(json.dumps({
            "step_num": 0, "action": "<reset>",
            "cursor_after": list(pos), "in_bbox": in_bbox(pos, target.bbox),
        }) + "\n")
        traj_f.flush()

        # Edge case: cursor starts already inside the bbox. Record but keep
        # going so the trajectory length stays consistent across regimes.
        if in_bbox(pos, target.bbox):
            reach_frame = 0

        for step in range(1, max_steps + 1):
            t0 = time.time()
            instr_used = target.instruction if (step == 1 or persist_instruction) else None
            if save_frames:
                (steps_dir / f"prompt_{step:03d}.json").write_text(json.dumps(
                    build_loggable_messages(
                        system_prompt=system_prompt, instruction=instr_used,
                        recent_actions=recent_actions,
                        frame_labels=window_frame_labels(step, len(recent_frames)),
                    ), indent=2))
            try:
                action_text, _finish_reason = _call_model(
                    sglang_url=sglang_url, api_key=api_key, model=model,
                    system_prompt=system_prompt,
                    instruction=instr_used,
                    recent_frames=recent_frames,
                    recent_actions=recent_actions,
                    max_tokens=max_tokens, temperature=temperature,
                )
            except Exception as e:
                _LOGGER.error("step %d: model call failed: %s", step, e)
                stop_reason = "model_error"
                break

            parse_err: str | None = None
            cursor_before = pos
            cursor_after = pos
            try:
                action = parse_action_tolerant(action_text)
                sr = client.dispatch_action(action)
                cursor_before = sr.cursor_before
                cursor_after = sr.cursor_after
                pos = cursor_after
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

            if save_frames:
                frame.save(steps_dir / f"step_{step:03d}.png")
            append_turn(recent_frames, recent_actions, frame, action_text,
                        n_history_frames=n_history_frames)

            hit = in_bbox(cursor_after, target.bbox)
            if hit and reach_frame < 0:
                reach_frame = step

            step_log = StepLog(
                step=step, action_text=action_text,
                cursor_before=cursor_before, cursor_after=cursor_after,
                in_bbox=hit, parse_error=parse_err,
                elapsed_s=time.time() - t0,
            )
            steps.append(step_log)
            traj_f.write(json.dumps({
                "step_num": step, "action": action_text,
                "cursor_before": list(cursor_before),
                "cursor_after": list(cursor_after),
                "in_bbox": hit, "parse_error": parse_err,
                "elapsed_s": step_log.elapsed_s,
            }) + "\n")
            traj_f.flush()

    return {
        "schema_version": 1,
        "task_id": target.task_id,
        "app": target.app,
        "bbox": list(target.bbox),
        "instruction": target.instruction,
        "regime": regime,
        "cursor_start": list(cursor_start_pos),
        "screen_size": [sw, sh],
        "model": model,
        "system_prompt": system_prompt,
        "n_history_frames": n_history_frames,
        "persist_instruction": persist_instruction,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_steps": max_steps,
        "n_steps": len(steps),
        "stop_reason": stop_reason,
        "parse_errors": parse_errors,
        # Headline metric:
        "reach": reach_frame >= 0,
        "reach_frame": reach_frame,
        "elapsed_s": time.time() - t_start,
        # Absolute path read by labctl's RolloutViewer
        # (server.rs:get_artifact_rollout reads metadata.result.traj_path
        # and walks siblings/steps/ for the per-frame PNGs).
        "traj_path": str(traj_path),
    }


# ---------------------------------------------------------------------------
# VM lifecycle (one VM per rollout)
# ---------------------------------------------------------------------------


def _launch_vm(
    *, qemu_bin: str, qcow2: str,
    vm_port: int, vnc_port: int, chromium_port: int,
    log_path: Path,
) -> subprocess.Popen:
    # The chromium hostfwd is required by SetupController.chrome_open_tabs
    # — playwright connects to host:chromium_port over CDP; inside the VM
    # the task's `launch socat ...` step listens on :9222 and forwards to
    # Chrome on :1337 (the --remote-debugging-port).
    return subprocess.Popen(
        [qemu_bin,
         "-enable-kvm", "-cpu", "host", "-smp", "4", "-m", "4G",
         "-machine", "type=q35,accel=kvm",
         "-drive", f"file={qcow2},if=virtio,format=qcow2,snapshot=on",
         "-netdev", (
             f"user,id=net0,"
             f"hostfwd=tcp::{vm_port}-:5000,"
             f"hostfwd=tcp::{vnc_port}-:5900,"
             f"hostfwd=tcp::{chromium_port}-:9222"
         ),
         "-device", "virtio-net-pci,netdev=net0",
         "-display", "none", "-nographic"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


def _terminate_proc(proc: subprocess.Popen, *, timeout_s: float = 5.0) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()


def _register_artifact(alias: str, path: Path) -> bool:
    """Register a completed rollout dir as a labctl ``eval_result`` artifact.

    Returns True on success. Logs (not raises) on failure so a registry
    hiccup doesn't tank a multi-hour rollout run.
    """
    try:
        proc = subprocess.run(
            ["labctl", "register-external",
             "--alias", alias, "--kind", "eval_result",
             "--path", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        _LOGGER.warning("register-external invocation failed for %s: %s", alias, e)
        return False
    if proc.returncode != 0:
        _LOGGER.warning(
            "register-external failed (rc=%d) for %s: %s",
            proc.returncode, alias, proc.stderr.strip()[:500],
        )
        return False
    _LOGGER.info("registered artifact: alias=%s", alias)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--bboxes_jsonl", required=True,
                   help="Path to bboxes.jsonl (the labeled targets).")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True,
                   help="Eval-results root. For labctl-visible runs, point at "
                        "/fast/.../labctl/eval_logs/<user>/. Per-rollout dirs "
                        "are flat siblings under this root so each can be "
                        "registered as its own eval_result artifact.")
    p.add_argument("--run_alias", default=None,
                   help="Prefix for per-rollout artifact aliases. Defaults to "
                        "grounding_<SLURM_JOB_ID> (or grounding_local_<pid>).")
    p.add_argument("--register_artifacts", action="store_true", default=True,
                   help="Call `labctl register-external` after each rollout so "
                        "it appears in the labctl UI's RolloutViewer.")
    p.add_argument("--no_register_artifacts", dest="register_artifacts",
                   action="store_false",
                   help="Skip labctl registration (useful for ephemeral smokes).")
    p.add_argument("--max_steps", type=int, default=100,
                   help="K — rollout length per (target, regime). 100 = 10s at 10fps.")
    p.add_argument("--regimes", nargs="+", default=list(REGIMES),
                   choices=REGIMES, help="Cursor-start regimes to run.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only run the first N targets (for smoke testing).")
    p.add_argument("--target_idxs", type=int, nargs="+", default=None,
                   help="If set, only run targets with these idx values.")
    p.add_argument("--system_prompt_id", default="training_v1")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_history_frames", type=int, default=16)
    p.add_argument(
        "--persist_instruction", action=argparse.BooleanOptionalAction, default=True,
        help="Re-anchor the goal on the earliest in-window user turn every step "
             "so it stays in context after the first frame is evicted. On by "
             "default; --no-persist_instruction reverts to goal-on-step-1.",
    )
    p.add_argument("--no_frames", action="store_true",
                   help="Skip saving per-step PNGs (saves disk).")
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

    bboxes_path = Path(args.bboxes_jsonl)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Run-level alias: prefixes every per-rollout artifact alias so a single
    # batch's outputs are grouped and findable in the labctl UI.
    run_alias = args.run_alias or (
        f"grounding_{os.environ.get('SLURM_JOB_ID', f'local_{os.getpid()}')}"
    )
    # Non-artifact scratch dir for sglang.log / aggregate summary / runner log.
    # Leading underscore keeps it out of the way of artifact dirs (labctl's
    # register-external is the only thing that registers; this dir won't be
    # auto-indexed).
    scratch_dir = output_root / f"_grounding_run_{run_alias}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(scratch_dir / "grounding_eval.log")
    )

    targets = load_targets(bboxes_path)
    if args.target_idxs is not None:
        keep = set(args.target_idxs)
        targets = [t for t in targets if t.idx in keep]
    if args.limit > 0:
        targets = targets[: args.limit]

    if not targets:
        print("no targets selected — check --target_idxs / --limit", file=sys.stderr)
        return 1

    _LOGGER.info(
        "model=%s output_root=%s run_alias=%s n_targets=%d regimes=%s register=%s",
        args.model_path, output_root, run_alias,
        len(targets), args.regimes, args.register_artifacts,
    )

    # Port isolation across concurrent SLURM array jobs.
    job_mod = (int(os.environ.get("SLURM_JOB_ID", "0")) % 200) * 10
    sglang_port = (30000 + job_mod) if args.sglang_port == 30000 else args.sglang_port

    # Sglang lifecycle: spawn once, share across all rollouts.
    sglang_log = scratch_dir / "sglang.log"
    _LOGGER.info("starting sglang on port %d ...", sglang_port)
    sglang_proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path,
         "--host", "0.0.0.0",
         "--port", str(sglang_port),
         "--api-key", args.sglang_api_key,
         "--mem-fraction-static", str(args.mem_fraction_static),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR),
        stdout=open(sglang_log, "w"),
        stderr=subprocess.STDOUT,
    )

    def _global_cleanup() -> None:
        _LOGGER.info("global cleanup ...")
        _terminate_proc(sglang_proc)

    atexit.register(_global_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))

    _wait_for(
        f"http://localhost:{sglang_port}/health_generate",
        headers={"Authorization": f"Bearer {args.sglang_api_key}"},
        proc=sglang_proc, poll_s=10, max_polls=180, label="sglang",
    )
    sglang_url = f"http://localhost:{sglang_port}/v1"

    # Per-(target, regime) loop. Fresh VM each time so the desktop state is
    # clean and rollouts can't pollute each other.
    base_vm_port = 5000 + job_mod
    base_vnc_port = 5900 + job_mod
    base_chromium_port = 9200 + job_mod

    summary: list[dict] = []
    for ti, target in enumerate(targets):
        for ri, regime in enumerate(args.regimes):
            # Per-rollout artifact alias + dir. Flat layout (sibling of other
            # rollouts) so each dir sits at <eval_root>/<user>/<alias>/, which
            # is what labctl's register-external requires.
            task_short = target.task_id[:12]
            rollout_alias = f"{run_alias}_{target.app}_{task_short}_{regime}"
            rollout_id = f"{target.app}/{target.task_id}/{regime}"
            _LOGGER.info(
                "[%d/%d] %s  (alias=%s)",
                ti * len(args.regimes) + ri + 1,
                len(targets) * len(args.regimes), rollout_id, rollout_alias,
            )
            rollout_dir = output_root / rollout_alias
            rollout_dir.mkdir(parents=True, exist_ok=True)

            # Per-rollout port offset (so retries / parallel runs don't collide).
            offset = (ti * len(args.regimes) + ri) % 50
            vm_port = base_vm_port + offset
            vnc_port = base_vnc_port + offset
            chromium_port = base_chromium_port + offset

            vm_proc = _launch_vm(
                qemu_bin=args.qemu_bin, qcow2=args.qcow2,
                vm_port=vm_port, vnc_port=vnc_port,
                chromium_port=chromium_port,
                log_path=rollout_dir / "qemu.log",
            )
            try:
                client = OSWorldClient(f"http://localhost:{vm_port}")
                client.wait_ready(timeout_s=300)
                sw, sh = client.screen_size()
                # OSWorld task setup: load the per-task config and run its
                # setup commands so the desktop matches the labeled bbox's
                # screen state before we score grounding. Setup runs in the
                # VM via /setup/* HTTP calls; chrome_open_tabs uses host-
                # side playwright against the forwarded chromium_port.
                task = load_osworld_task(target.app, target.task_id)
                traj = trajectory_path_for(target)
                _LOGGER.info(
                    "task setup for %s (%d task steps + %s replay) ...",
                    rollout_id, len(task.get("config", [])),
                    "1 cached action" if traj else "no",
                )
                t_setup = time.time()
                run_task_setup(
                    task=task,
                    vm_port=vm_port,
                    chromium_port=chromium_port,
                    vlc_port=base_vnc_port + 100 + offset,  # unused for non-vlc tasks
                    cache_dir=rollout_dir / "setup_cache",
                    screen_w=sw, screen_h=sh,
                    replay_trajectory=traj,
                    replay_n_steps=1,
                )
                _LOGGER.info("setup done in %.1fs", time.time() - t_setup)
                start = cursor_start(target, sw, sh, regime)
                result = _run_grounding_rollout(
                    client=client,
                    target=target,
                    regime=regime,
                    cursor_start_pos=start,
                    sglang_url=sglang_url,
                    api_key=args.sglang_api_key,
                    model=args.model_path,
                    system_prompt=SYSTEM_PROMPTS[args.system_prompt_id],
                    max_steps=args.max_steps,
                    n_history_frames=args.n_history_frames,
                    persist_instruction=args.persist_instruction,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    output_dir=rollout_dir,
                    save_frames=not args.no_frames,
                )
            except Exception as e:
                _LOGGER.exception("rollout %s failed", rollout_id)
                result = {
                    "schema_version": 1,
                    "task_id": target.task_id, "app": target.app,
                    "bbox": list(target.bbox), "regime": regime,
                    "reach": False, "reach_frame": -1,
                    "stop_reason": f"exception: {type(e).__name__}: {e}",
                }
            finally:
                _terminate_proc(vm_proc)

            with (rollout_dir / "result.json").open("w") as f:
                json.dump(result, f, indent=2)
            registered = False
            if args.register_artifacts:
                registered = _register_artifact(rollout_alias, rollout_dir)
            summary.append({
                "alias": rollout_alias,
                "task_id": target.task_id, "app": target.app,
                "regime": regime, "idx": target.idx,
                "reach": result.get("reach", False),
                "reach_frame": result.get("reach_frame", -1),
                "registered": registered,
            })

    summary_path = scratch_dir / "summary.json"
    overall_reach = sum(1 for r in summary if r["reach"]) / max(1, len(summary))
    with summary_path.open("w") as f:
        json.dump({
            "n_rollouts": len(summary),
            "overall_reach": overall_reach,
            "run_alias": run_alias,
            "per_rollout": summary,
        }, f, indent=2)

    # Drop a one-liner script for retrying registration from the login node
    # (where labctl's Postgres is reachable). Useful when in-job registration
    # failed (compute-node PG unreachable, or labctl binary/schema skew).
    n_unregistered = sum(1 for r in summary if not r.get("registered"))
    if n_unregistered:
        script = scratch_dir / "register_artifacts.sh"
        with script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Re-register grounding-eval rollouts as labctl artifacts.\n")
            f.write("# Run on the login node (where labctl's Postgres is reachable).\n")
            f.write("set -e\n")
            for r in summary:
                if r.get("registered"):
                    continue
                path = output_root / r["alias"]
                f.write(
                    f"labctl register-external "
                    f"--alias {r['alias']} "
                    f"--kind eval_result "
                    f"--path {path}\n"
                )
        script.chmod(0o755)
        _LOGGER.info(
            "%d of %d rollouts were NOT registered; retry from login node:\n  bash %s",
            n_unregistered, len(summary), script,
        )

    _LOGGER.info(
        "done: %d rollouts, overall reach=%.2f, summary=%s",
        len(summary), overall_reach, summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
