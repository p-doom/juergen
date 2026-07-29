"""Convert ABSOLUTE teacher rollouts -> move_rel (native_rel_v2) training records.

v4 MOVE_REL analog of convert_abs_to_relative.py (v3 native_rel). Two-step reuse:

  (1) v3 diff-of-absolute NORMALIZED delta (coord_space="normalized"):
      rel_norm = teacher_raw_coord[0-999] - cursor_before_px*1000/dim  (per axis),
      via convert_abs_to_relative._normalize_abs_args_to_rel. Behaviour-preserving,
      resolution-invariant, deltas already in the canonical 0-999 space.

  (2) v2 GRAMMAR CORRECTION (native_rel_format_v2): a delta-carrying click/mouse_down
      is split into an EXPLICIT `move_rel` {coordinate:[dx,dy]} + the coordinate-less
      op ({"action":"left_click"}); a standalone relative `mouse_move` becomes an
      explicit `move_rel`. A zero-delta click emits ONLY the coordinate-less click.
      Coordinates are ALREADY 0-999 here (v3 normalized) so they are NOT re-normalized
      (unlike native_rel_format_v2.split_and_normalize, which takes raw px).

System prompt = native_rel_format_v2.SYSTEM_PROMPT (the canonical move_rel prompt).
Typing passes through as a SINGLE `type` {text} string (the teacher emits the whole
string in one action -- no fragmentation to fix, unlike the BC capture path).

Output chat.jsonl (_normalized/{train,val}) is byte-compatible with videocua_nativerel_v2
and consumed by the SAME omegalax build_sft_records_from_chat.py tokenizer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_V3_SCR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts"
_V2_DIR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_moverel"
_V1_DIR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_nativerel_v1"
for _p in (_V3_SCR, _V2_DIR, _V1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convert_abs_to_relative as v3          # noqa: E402  (v3 diff-of-abs core)
import move_rel_format as v2enc               # noqa: E402  (canonical move_rel system prompt; formerly native_rel_format_v2)

MOVEREL_SYSTEM_PROMPT = v2enc.SYSTEM_PROMPT

# v2 grammar: move-carrying ops whose relative delta becomes an explicit move_rel.
_CLICK_ACTIONS = {"left_click", "right_click", "double_click", "triple_click", "middle_click"}
_MOVE_CARRYING = _CLICK_ACTIONS | {"mouse_down"}


def split_already_normalized(arg_dicts):
    """v3 native_rel (already 0-999 normalized) arg dicts -> v2 move_rel arg dicts.

    Split a delta-carrying click/mouse_down into explicit `move_rel` + the
    coordinate-less op; a standalone `mouse_move` becomes an explicit `move_rel`.
    Deltas are ALREADY 0-999 (DO NOT re-normalize)."""
    out = []
    for a in arg_dicts:
        act = a.get("action")
        if act in _MOVE_CARRYING and "coordinate" in a:
            dx, dy = a["coordinate"]
            if dx != 0 or dy != 0:
                out.append({"action": "move_rel", "coordinate": [int(dx), int(dy)]})
            out.append({k: v for k, v in a.items() if k != "coordinate"})
        elif act == "mouse_move" and "coordinate" in a:
            dx, dy = a["coordinate"]
            out.append({"action": "move_rel", "coordinate": [int(dx), int(dy)]})
        else:
            out.append(dict(a))
    return out


def convert_rollout(run_dir: Path, *, min_valid_actions: int = 1,
                    max_parse_error_frac: float = 0.5, keep_prose: bool = False,
                    min_task_success: float | None = None) -> dict | None:
    """One rollout dir -> one move_rel chat record (or None if unusable).

    Mirrors v3.convert_rollout exactly, but (a) runs the abs->rel core in NORMALIZED
    coord space, (b) SPLITS each turn's arg dict into the v2 move_rel grammar, and
    (c) uses the v2 (move_rel) system prompt."""
    result_path = run_dir / "result.json"
    traj_path = run_dir / "trajectory.jsonl"
    if not result_path.is_file() or not traj_path.is_file():
        return None
    result = json.loads(result_path.read_text())
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

    turns = []          # list of (seen_frame_path, [arg_dict, ...], prose)
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
        prose = v3._extract_prose(entry.get("action") or entry.get("response") or "") if keep_prose else ""
        parsed = info.get("parsed")
        if info.get("parse_error") or not parsed:
            n_parse_err += 1
            continue
        if parsed.get("terminate"):
            status = parsed.get("computer_use_status") or "success"
            turns.append((str(seen_frame), [{"action": "terminate", "status": status}], prose))
            continue
        abs_args = parsed.get("computer_use")
        if not isinstance(abs_args, dict):
            n_parse_err += 1
            continue
        rel_args = v3._normalize_abs_args_to_rel(
            abs_args, info.get("cursor_before"), info.get("intended_target"),
            coord_space="normalized", screen=screen)
        if rel_args is None:
            n_parse_err += 1
            continue
        v2_args = split_already_normalized([rel_args])   # 1 native_rel arg -> 1-or-2 move_rel args
        turns.append((str(seen_frame), v2_args, prose))

    n_valid = len(turns)
    if n_valid < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    messages = [{"role": "system", "content": [{"type": "text", "text": MOVEREL_SYSTEM_PROMPT}]}]
    for i, (frame_path, arg_dicts, prose) in enumerate(turns):
        user_content = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        tool_text = v3._render_assistant_text(arg_dicts)   # one <tool_call> block per arg dict
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
    p.add_argument("--rollouts_dir", required=True)
    p.add_argument("--min_task_success", type=float, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--keep_slugs", default=None)
    p.add_argument("--exclude_slugs", default=None,
                   help="text file of slugs/recording_ids to DROP (OSWorld eval-leak filter).")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--min_valid_actions", type=int, default=2)
    p.add_argument("--max_parse_error_frac", type=float, default=0.5)
    p.add_argument("--keep_prose", action="store_true")
    args = p.parse_args()

    keep = None
    if args.keep_slugs:
        keep = set(s.strip() for s in Path(args.keep_slugs).read_text().splitlines() if s.strip())
    drop = set()
    if args.exclude_slugs and Path(args.exclude_slugs).is_file():
        drop = set(s.strip() for s in Path(args.exclude_slugs).read_text().splitlines() if s.strip())

    run_dirs = []
    for rd in args.rollouts_dir.split(","):
        rd = Path(rd.strip())
        if rd.is_dir():
            run_dirs += [d for d in rd.iterdir() if d.is_dir() and (d / "result.json").is_file()]
    run_dirs = sorted(run_dirs)
    records = []
    n_seen = n_kept = n_dropped_filter = n_dropped_leak = n_unusable = 0
    for d in run_dirs:
        n_seen += 1
        if keep is not None and d.name not in keep:
            n_dropped_filter += 1
            continue
        if d.name in drop:
            n_dropped_leak += 1
            continue
        rec = convert_rollout(d, min_valid_actions=args.min_valid_actions,
                              max_parse_error_frac=args.max_parse_error_frac,
                              keep_prose=args.keep_prose, min_task_success=args.min_task_success)
        if rec is None:
            n_unusable += 1
            continue
        if rec["recording_id"] in drop:
            n_dropped_leak += 1
            continue
        records.append(rec)
        n_kept += 1

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
        "grammar": "move_rel (native_rel_v2)", "rollouts_dir": args.rollouts_dir,
        "keep_slugs": args.keep_slugs, "exclude_slugs": args.exclude_slugs,
        "n_seen": n_seen, "n_kept": n_kept,
        "n_dropped_by_filter": n_dropped_filter, "n_dropped_leak": n_dropped_leak,
        "n_unusable": n_unusable,
        "n_train": len(train), "n_val": len(val), "train_ratio": args.train_ratio,
        "min_valid_actions": args.min_valid_actions, "keep_prose": args.keep_prose,
        "coord_space": "normalized", "min_task_success": args.min_task_success,
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
