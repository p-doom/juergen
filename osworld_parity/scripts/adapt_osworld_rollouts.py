"""Adapt a4781827 baseline-runner OSWorld rollouts -> the on_policy_distill converter format.

Track-2 rollout-gen (collect_osworld_train.sbatch = baseline_eval_shard.py) saves, per task:
  {app}/{task_id}/result.json  (params.task_instruction, scores.reward, params.stop_reason)
  {app}/{task_id}/traj.jsonl   (per step: step_num, action=pyautogui, response=raw model text, reward, done)
  {app}/{task_id}/steps/step_NNN.png

This reconstructs, per task, a rollout dir in the format convert_abs_to_{absolute,moverel,diffabs}.py
consume (result.json + trajectory.jsonl with info.parsed.computer_use / cursor_before /
intended_target / parse_error + steps/), so the SAME tested encoders produce all 3 formats.

Reconstruction (teacher = off-shelf Qwen3VLAgent, coordinate_type=relative => model coordinate in
the 0-1000 grid, exactly the v3 collector's coord_grid=1000 convention):
  * parse `response` for computer_use <tool_call>s (action + 0-1000 coordinate); first call per step.
  * intended_target(px) = coordinate * screen / 1000 (per axis).
  * cursor_before telescopes: seeded at screen center; after a coordinate action the cursor := target;
    type/key/scroll/wait leave it unchanged (matches the diff-of-absolute telescoping).
  * screenshots are symlinked (same step_NNN.png naming; converter reads step_{step_num-1}.png).
NOTE: the absolute positive-control system_prompt is set to osworld_system_prompts["computer_use_v1"];
for a true format-shift-0 vs a4781827's eval harness, this MUST match the qwen3vl_agent eval prompt
byte-for-byte -- flagged for coordination before trusting the absolute-control parity number.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

_EVAL_DIR = "/fast/home/franz.srambical/juergen/eval"
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from action_parser import parse_computer_use_tool_calls  # noqa: E402

# Re-baked from a4781827's EXACT eval system prompt (sha256 f0aa74a3…, len 4051) -- read the
# file literally (NOT transcribed / NOT the computer_use_v1 stand-in, which materially differs).
# Screen-size-independent ("1000x1000" for relative), safe to bake as a literal.
_EVAL_PROMPT_FILE = "/fast/home/franz.srambical/osworld_parity_split/eval_system_prompt.txt"
ABS_SYSTEM_PROMPT = Path(_EVAL_PROMPT_FILE).read_text()
_COORD_ACTIONS = {"mouse_move", "left_click", "right_click", "middle_click",
                  "double_click", "triple_click", "left_click_drag", "mouse_down"}


def adapt_task(task_dir: Path, out_root: Path, app: str, task_id: str,
               sw: int, sh: int) -> bool:
    res_p = task_dir / "result.json"
    traj_p = task_dir / "traj.jsonl"
    steps_src = task_dir / "steps"
    if not (res_p.is_file() and traj_p.is_file() and steps_src.is_dir()):
        return False
    res = json.loads(res_p.read_text())
    params = res.get("params", {}) or {}
    scores = res.get("scores", {}) or {}
    instruction = params.get("task_instruction")
    stop_reason = params.get("stop_reason")
    reward = scores.get("reward")
    slug = f"{app}__{task_id}"

    # group traj entries by step_num (a predict with multiple pyautogui actions writes several
    # lines with the same step_num + same response); keep first response per step.
    by_step = collections.OrderedDict()
    for line in traj_p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        k = e.get("step_num", 0)
        if k == 0:
            continue
        if k not in by_step:
            by_step[k] = e

    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dst = out_dir / "steps"
    if not steps_dst.exists():
        os.symlink(steps_src.resolve(), steps_dst)

    cursor = [sw / 2.0, sh / 2.0]   # seed at screen center
    traj_out = []
    for k, e in by_step.items():
        response = e.get("response") or ""
        info = {"action_text": response, "cursor_before": [int(round(cursor[0])), int(round(cursor[1]))]}
        try:
            calls = parse_computer_use_tool_calls(response)
        except Exception:  # noqa: BLE001
            calls = []
        if not calls:
            info["parse_error"] = True
            info["parsed"] = None
            traj_out.append({"step_num": k, "info": info})
            continue
        args = dict(calls[0].arguments)
        action = str(args.get("action", "")).strip().lower()
        if action == "terminate":
            info["parsed"] = {"terminate": True,
                              "computer_use_status": args.get("status", "success")}
            info["intended_target"] = info["cursor_before"]
            info["parse_error"] = False
            traj_out.append({"step_num": k, "info": info})
            continue
        # resolve intended_target (px) from the 0-1000 coordinate; advance the telescoped cursor
        target = list(cursor)
        co = args.get("coordinate")
        if action in _COORD_ACTIONS and isinstance(co, (list, tuple)) and len(co) == 2:
            try:
                target = [float(co[0]) * sw / 1000.0, float(co[1]) * sh / 1000.0]
                cursor = list(target)
            except (TypeError, ValueError):
                pass
        info["intended_target"] = [int(round(target[0])), int(round(target[1]))]
        info["parse_error"] = False
        info["parsed"] = {"computer_use": args, "no_op": False}
        traj_out.append({"step_num": k, "info": info})

    if not any(t["info"].get("parsed") for t in traj_out):
        return False
    (out_dir / "trajectory.jsonl").write_text("\n".join(json.dumps(t) for t in traj_out) + "\n")
    (out_dir / "result.json").write_text(json.dumps({
        "instruction": instruction, "system_prompt": ABS_SYSTEM_PROMPT,
        "screen_size": [sw, sh], "coord_grid": 1000, "slug": slug,
        "stop_reason": stop_reason, "task_success": reward, "app": app, "task_id": task_id,
    }, indent=2))
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collected_root", required=True, help="baseline-runner output root ({app}/{task_id}/...)")
    p.add_argument("--out_root", required=True, help="adapted rollout dirs (converter-consumable)")
    p.add_argument("--screen_width", type=int, default=1920)
    p.add_argument("--screen_height", type=int, default=1080)
    args = p.parse_args()

    root = Path(args.collected_root)
    out = Path(args.out_root)
    n_ok = n_skip = 0
    for app_dir in sorted(root.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith("qemu_logs"):
            continue
        for task_dir in sorted(app_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            ok = adapt_task(task_dir, out, app_dir.name, task_dir.name,
                            args.screen_width, args.screen_height)
            n_ok += int(ok); n_skip += int(not ok)
    manifest = {"collected_root": args.collected_root, "out_root": args.out_root,
                "n_adapted": n_ok, "n_skipped": n_skip}
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapt_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
