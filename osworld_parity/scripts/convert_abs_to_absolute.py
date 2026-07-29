"""Convert ABSOLUTE teacher rollouts -> ABSOLUTE training records (POSITIVE CONTROL).

Format-shift ZERO: the student is trained on the teacher's OWN native computer_use grammar
(absolute 0-1000 `coordinate`), with the teacher's OWN off-the-shelf system prompt. If the
convert -> SFT -> closed-loop-eval pipeline is sound and the eval scores the absolute
convention correctly, this control MUST reach OSWorld parity with off-the-shelf. It is the
validator for the pipeline + eval; the move_rel / diffabs arms are the real format-shift tests.

Identical rollout parsing to convert_abs_to_relative.py / _moverel / _diffabs (same rollouts,
same parse, same keep_slugs filter, same split) -- the ONLY difference across the four is the
assistant action ENCODING. Here we render the parsed computer_use arg dict VERBATIM (abs
coordinate untouched) and use the rollout's own `system_prompt` from result.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_V3_SCR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts"
if _V3_SCR not in sys.path:
    sys.path.insert(0, _V3_SCR)
import convert_abs_to_relative as v3   # noqa: E402  (shared rollout-parsing helpers)

# actions the off-the-shelf computer_use grammar accepts (drop off-grammar turns)
_ABS_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "left_click_drag", "key", "type", "scroll", "hscroll",
    "wait", "terminate", "answer",
}


def _clean_abs_args(abs_args: dict) -> dict | None:
    """Verbatim off-the-shelf computer_use arg dict (abs coordinate untouched)."""
    action = str(abs_args.get("action", "")).strip().lower()
    if action not in _ABS_ACTIONS:
        return None
    return {k: v for k, v in abs_args.items() if not str(k).startswith("_")}


def convert_rollout(run_dir: Path, *, min_valid_actions: int = 1,
                    max_parse_error_frac: float = 0.5, keep_prose: bool = False,
                    min_task_success: float | None = None) -> dict | None:
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
    system_text = result.get("system_prompt")   # the teacher's OWN off-the-shelf abs prompt
    if not system_text:
        return None
    steps_dir = run_dir / "steps"

    traj = []
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if line:
            traj.append(json.loads(line))

    turns = []          # (frame_path, [arg_dict], prose)
    n_steps = n_parse_err = 0
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
        clean = _clean_abs_args(abs_args)
        if clean is None:
            n_parse_err += 1
            continue
        turns.append((str(seen_frame), [clean], prose))

    n_valid = len(turns)
    if n_valid < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for i, (frame_path, arg_dicts, prose) in enumerate(turns):
        user_content = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        tool_text = v3._render_assistant_text(arg_dicts)
        asst_text = f"{prose}\n{tool_text}" if (keep_prose and prose) else tool_text
        messages.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})

    slug = result.get("slug") or run_dir.name
    return {
        "sample_id": f"onpol_{slug}", "recording_id": slug, "app": "osworld_onpolicy",
        "platform": "UBUNTU", "instruction": instruction, "n_frames": n_valid,
        "subrecord_idx": 0, "n_subrecords": 1, "source_stop_reason": result.get("stop_reason"),
        "source_parse_errors": n_parse_err, "messages": messages,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--keep_slugs", default=None)
    p.add_argument("--exclude_slugs", default=None,
                   help="optional path to a text file of slugs / recording_ids to DROP (leak-check output).")
    p.add_argument("--min_task_success", type=float, default=None)
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
        # leak-drop by recording_id too (belt-and-suspenders vs dir name)
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
        "grammar": "absolute computer_use (POSITIVE CONTROL, format-shift 0)",
        "rollouts_dir": args.rollouts_dir, "keep_slugs": args.keep_slugs,
        "exclude_slugs": args.exclude_slugs, "n_seen": n_seen, "n_kept": n_kept,
        "n_dropped_by_filter": n_dropped_filter, "n_dropped_leak": n_dropped_leak,
        "n_unusable": n_unusable, "n_train": len(train), "n_val": len(val),
        "train_ratio": args.train_ratio, "min_valid_actions": args.min_valid_actions,
        "min_task_success": args.min_task_success,
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
