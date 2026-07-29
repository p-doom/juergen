"""Convert ABSOLUTE-format teacher rollouts -> native-RELATIVE training records.

The diff-of-absolute conversion at trajectory granularity: freeroll logs, per
step, ``cursor_before`` (the real VM cursor before the action) and
``intended_target`` (the absolute pixel the teacher's coordinate resolved to,
post-scale + post-clip). The RELATIVE delta that reproduces the *identical*
cursor motion is simply::

    rel[t] = intended_target[t] - cursor_before[t]

and because ``cursor_before[t] == cursor_after[t-1] == abs[t-1]`` this telescopes
exactly (same principle as videocua_diffabs). Behavior is preserved; only the
action ENCODING changes (absolute coordinate -> relative delta). Every other
field (type text / keys / scroll pixels / terminate status) passes through.

Output records match videocua_nativerel_v1 exactly: system = the native_rel_v1
SYSTEM_PROMPT, user turns = {"type":"image","image":<abs png path>} (+ instruction
on the first turn), assistant turns = one ``<tool_call>`` block, JSON separators
(", ", ": "). The same omegalax build_sft_records_from_chat.py tokenizer that
built videocua_nativerel_v1 then consumes _normalized/{train,val}/chat.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Byte-identical to osworld_system_prompts.SYSTEM_PROMPTS["native_rel_v1"] and
# native_rel_format.SYSTEM_PROMPT (a test in the repo asserts those two match).
NATIVE_REL_SYSTEM_PROMPT = (
    "You operate a desktop computer using the computer_use tool. The first user "
    "turn shows the initial screen and the user's goal; each subsequent user turn "
    "shows the current screen. Reply with one or more computer_use tool calls that "
    "advance toward the goal.\n"
    "\n"
    "Mouse movement is RELATIVE: `coordinate` is a [dx, dy] pixel offset from the "
    "CURRENT cursor position (positive dx = right, positive dy = down), NOT an "
    "absolute screen coordinate. Look at the visible cursor in the screenshot to "
    "judge how far and in which direction to move.\n"
    "\n"
    "Actions (computer_use `action` field):\n"
    "- mouse_move {coordinate:[dx,dy]}: move the cursor by (dx,dy).\n"
    "- left_click / right_click / middle_click {coordinate:[dx,dy]}: move by (dx,dy), "
    "then click at the new position.\n"
    "- double_click / triple_click {coordinate:[dx,dy]}: move, then double/triple click.\n"
    "- mouse_down {button, coordinate:[dx,dy]} / mouse_up {button}: press / release a "
    "mouse button (button = 'left','right','middle'). A drag is mouse_down, then one or "
    "more mouse_move, then mouse_up.\n"
    "- key {keys:[...]}: press a key or chord, e.g. ['ctrl','a'], ['enter'], ['tab'].\n"
    "- key_down {keys:[...]} / key_up {keys:[...]}: hold / release keys across steps.\n"
    "- type {text}: type a string of text.\n"
    "- scroll {pixels}: scroll the wheel (positive = up, negative = down).\n"
    "- wait {time}: do nothing this step.\n"
    "- terminate {status}: the goal is complete (status = 'success' or 'failure').\n"
    "\n"
    "For each action, return a JSON object within <tool_call></tool_call> tags:\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [12, -8]}}\n'
    "</tool_call>"
)

# Byte-identical to osworld_system_prompts.NATIVE_REL_THINK_PREAMBLE. Used as the
# system prompt for keep_prose (thinking+action) records so training MATCHES the
# native_rel_think eval prompt, and DISAMBIGUATES thinking on-policy turns from the
# tool-call-only retention set (which stays native_rel_v1) under the anneal mix.
NATIVE_REL_THINK_PREAMBLE = (
    "For each step, first reason in a single <think>...</think> block — your current "
    "sub-goal and what you observe on the screen — then a one-line `Action:` describing "
    "the move, then the computer_use tool call.\n\n"
)
NATIVE_REL_THINK_SYSTEM_PROMPT = NATIVE_REL_THINK_PREAMBLE + NATIVE_REL_SYSTEM_PROMPT

# actions that carry a positional coordinate we must rewrite abs -> rel
_COORD_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click",
    "double_click", "triple_click", "left_click_drag", "mouse_down",
}
# native_rel grammar action set (used to normalize / drop off-grammar actions)
_NATIVE_REL_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "mouse_down", "mouse_up", "key", "key_down", "key_up",
    "type", "scroll", "wait", "terminate", "left_click_drag",
}


_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def _extract_prose(raw: str) -> str:
    """Return the teacher's reasoning prose (everything OUTSIDE <tool_call> blocks).

    Preserves any <think>...</think> wrapper the teacher emitted; collapses blank
    runs. Empty string if the response was a bare tool call."""
    if not isinstance(raw, str):
        return ""
    prose = _TOOLCALL_RE.sub("", raw).strip()
    # drop code-fence leftovers / stray fragments
    return prose if len(prose) > 2 else ""


def _render_assistant_text(arg_dicts: list[dict]) -> str:
    """One <tool_call> block per arg dict (mirrors native_rel_format.render_assistant_text)."""
    blocks = []
    for a in arg_dicts:
        clean = {k: v for k, v in a.items() if not k.startswith("_")}
        payload = {"name": "computer_use", "arguments": clean}
        blocks.append("<tool_call>\n"
                      + json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
                      + "\n</tool_call>")
    return "\n".join(blocks)


def _normalize_abs_args_to_rel(abs_args: dict, cursor_before, intended_target,
                               *, coord_space: str = "normalized", screen=None) -> dict | None:
    """Rewrite ONE absolute computer_use arg dict to the native-relative grammar.

    coord_space:
      - "normalized" (DEFAULT, resolution-invariant — the fix for the fleet's
        14-24% magnitude miscalibration): keep deltas in the canonical 0-999
        NORMALIZED space end-to-end. rel_norm = target_norm - cursor_norm, where
        target_norm = the teacher's RAW emitted 0-999 coordinate and
        cursor_norm = cursor_before_px * 1000 / screen (per-axis; COORD_SCALE=1000). Training + eval
        dispatch must scale 0-999 -> screen (like upstream qwen3vl_agent). Do NOT
        down-convert to raw pixels here.
      - "pixel" (legacy): rel_px = intended_target - cursor_before (raw pixels).

    Returns None if the action is off-grammar and cannot be salvaged."""
    action = str(abs_args.get("action", "")).strip().lower()
    if action == "hscroll":
        action = "scroll"
    elif action == "answer":
        return {"action": "wait", "time": 1}
    if action not in _NATIVE_REL_ACTIONS:
        return None

    out = {k: v for k, v in abs_args.items() if not k.startswith("_")}
    out["action"] = action

    if action in _COORD_ACTIONS:
        if coord_space == "normalized":
            raw = abs_args.get("coordinate")
            if (raw is not None and len(raw) == 2 and cursor_before is not None
                    and len(cursor_before) == 2 and screen and screen[0] and screen[1]):
                sw, sh = float(screen[0]), float(screen[1])
                # px->norm inverse of the /1000 dispatch (COORD_SCALE=1000): norm = px*1000/dim.
                # (was 999 — fixed for internal consistency with collector coord_grid=1000 &
                # freeroll --rel_coord_grid 1000, and to match the RFT cold-start convention.)
                cur_nx = float(cursor_before[0]) * 1000.0 / sw
                cur_ny = float(cursor_before[1]) * 1000.0 / sh
                out["coordinate"] = [int(round(float(raw[0]) - cur_nx)),
                                     int(round(float(raw[1]) - cur_ny))]
            elif "coordinate" in out:
                if action == "mouse_move":
                    return None
                out.pop("coordinate", None)
        else:  # pixel (legacy)
            if (cursor_before is not None and intended_target is not None
                    and len(cursor_before) == 2 and len(intended_target) == 2):
                out["coordinate"] = [int(intended_target[0]) - int(cursor_before[0]),
                                     int(intended_target[1]) - int(cursor_before[1])]
            elif "coordinate" in out:
                if action == "mouse_move":
                    return None
                out.pop("coordinate", None)
    return out


def convert_rollout(run_dir: Path, *, min_valid_actions: int = 1,
                    max_parse_error_frac: float = 0.5, keep_prose: bool = False,
                    coord_space: str = "normalized", min_task_success: float | None = None) -> dict | None:
    """One rollout dir -> one native-relative chat record (or None if unusable)."""
    result_path = run_dir / "result.json"
    traj_path = run_dir / "trajectory.jsonl"
    if not result_path.is_file() or not traj_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    # deterministic-task-success filter (gold): keep only traces the OSWorld
    # evaluator scored >= threshold. None success (eval_error/skipped) is dropped.
    if min_task_success is not None:
        ts = result.get("task_success")
        if ts is None or float(ts) < min_task_success:
            return None
    instruction = result.get("instruction")
    screen = result.get("screen_size")
    steps_dir = run_dir / "steps"

    traj = []
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if line:
            traj.append(json.loads(line))

    turns = []          # list of (seen_frame_path, arg_dict, prose)
    n_steps = 0
    n_parse_err = 0
    for entry in traj:
        step_num = entry.get("step_num", 0)
        if step_num == 0:
            continue
        n_steps += 1
        info = entry.get("info", {}) or {}
        seen_frame = steps_dir / f"step_{step_num - 1:03d}.png"
        if not seen_frame.is_file():
            continue
        prose = _extract_prose(entry.get("action") or entry.get("response") or "") if keep_prose else ""
        parsed = info.get("parsed")
        if info.get("parse_error") or not parsed:
            n_parse_err += 1
            continue
        if parsed.get("terminate"):
            status = parsed.get("computer_use_status") or "success"
            turns.append((str(seen_frame), {"action": "terminate", "status": status}, prose))
            continue
        abs_args = parsed.get("computer_use")
        if not isinstance(abs_args, dict):
            n_parse_err += 1
            continue
        rel_args = _normalize_abs_args_to_rel(
            abs_args, info.get("cursor_before"), info.get("intended_target"),
            coord_space=coord_space, screen=screen)
        if rel_args is None:
            n_parse_err += 1
            continue
        turns.append((str(seen_frame), rel_args, prose))

    n_valid = len(turns)
    if n_valid < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    # thinking+action records get the native_rel_think system prompt (matches the
    # native_rel_think eval prompt + disambiguates from tool-call-only retention).
    system_text = NATIVE_REL_THINK_SYSTEM_PROMPT if keep_prose else NATIVE_REL_SYSTEM_PROMPT
    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for i, (frame_path, arg_dict, prose) in enumerate(turns):
        user_content = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        tool_text = _render_assistant_text([arg_dict])
        asst_text = f"{prose}\n{tool_text}" if (keep_prose and prose) else tool_text
        messages.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})

    slug = result.get("slug") or run_dir.name
    return {
        "sample_id": f"onpol_{slug}",
        "recording_id": slug,
        "app": "osworld_onpolicy",
        "platform": "UBUNTU",
        "instruction": instruction,
        "n_frames": n_valid,
        "subrecord_idx": 0,
        "n_subrecords": 1,
        "source_stop_reason": result.get("stop_reason"),
        "source_parse_errors": n_parse_err,
        "messages": messages,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts_dir", required=True,
                   help="collection output dir(s) with per-rollout subdirs. COMMA-separated to "
                        "combine multiple collect rounds (e.g. scale_v1,scale_v2).")
    p.add_argument("--min_task_success", type=float, default=None,
                   help="keep only traces with deterministic evaluate() task_success >= this "
                        "(e.g. 1.0). Requires collection ran with --evaluate.")
    p.add_argument("--out_dir", required=True, help="dataset root; writes _normalized/{train,val}/chat.jsonl")
    p.add_argument("--keep_slugs", default=None,
                   help="optional path to a text file of run slugs to KEEP (quality filter output). "
                        "One slug per line; if omitted, keep all convertible rollouts.")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--min_valid_actions", type=int, default=2)
    p.add_argument("--max_parse_error_frac", type=float, default=0.5)
    p.add_argument("--keep_prose", action="store_true",
                   help="retain the teacher's reasoning prose (<think>/Action:) before the "
                        "rewritten <tool_call> -> thinking+action records (the validated lever).")
    p.add_argument("--coord_space", choices=["normalized", "pixel"], default="normalized",
                   help="'normalized' (default): keep 0-999 normalized deltas end-to-end "
                        "(resolution-invariant; train+eval dispatch scale 0-999->screen). "
                        "'pixel': legacy raw-pixel deltas.")
    args = p.parse_args()

    keep = None
    if args.keep_slugs:
        keep = set(s.strip() for s in Path(args.keep_slugs).read_text().splitlines() if s.strip())

    run_dirs = []
    for rd in args.rollouts_dir.split(","):
        rd = Path(rd.strip())
        if rd.is_dir():
            run_dirs += [d for d in rd.iterdir() if d.is_dir() and (d / "result.json").is_file()]
    run_dirs = sorted(run_dirs)
    records = []
    n_seen = n_kept = n_dropped_filter = n_unusable = 0
    for d in run_dirs:
        n_seen += 1
        if keep is not None and d.name not in keep:
            n_dropped_filter += 1
            continue
        rec = convert_rollout(d, min_valid_actions=args.min_valid_actions,
                              max_parse_error_frac=args.max_parse_error_frac,
                              keep_prose=args.keep_prose, coord_space=args.coord_space,
                              min_task_success=args.min_task_success)
        if rec is None:
            n_unusable += 1
            continue
        records.append(rec)
        n_kept += 1

    # deterministic split by recording_id
    rng = random.Random(args.split_seed)
    ids = sorted({r["recording_id"] for r in records})
    rng.shuffle(ids)
    n_train = max(1, round(len(ids) * args.train_ratio)) if ids else 0
    train_ids = set(ids[:n_train])
    train = [r for r in records if r["recording_id"] in train_ids]
    val = [r for r in records if r["recording_id"] not in train_ids]

    out = Path(args.out_dir)
    for split, recs in (("train", train), ("val", val)):
        d = out / "_normalized" / split
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chat.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "rollouts_dir": args.rollouts_dir, "keep_slugs": args.keep_slugs,
        "n_seen": n_seen, "n_kept": n_kept, "n_dropped_by_filter": n_dropped_filter,
        "n_unusable": n_unusable, "n_train": len(train), "n_val": len(val),
        "train_ratio": args.train_ratio, "min_valid_actions": args.min_valid_actions,
        "keep_prose": args.keep_prose, "coord_space": args.coord_space,
        "min_task_success": args.min_task_success,
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
