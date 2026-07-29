"""Convert ABSOLUTE teacher rollouts -> CUSTOM omegalax (diffabs/psai) grammar records.

The ORIGINAL custom grammar the psai/delta/diffabs crowd-cast & videocua BC runs used
(build_videocua_chat.render_labels). Each assistant turn is a PLAIN-TEXT label:

    "<dx> <dy> <scroll>"  optionally followed by  " ; +KEY -KEY ..."   (or "NO_OP", "TERMINATE")

  * dx,dy = diff-of-absolute cursor delta in RAW NATIVE PIXELS:
        dx,dy = round(intended_target - cursor_before)
    (the delta from the live VM cursor to the teacher's absolute target -- the same
    diff-of-absolute telescoping as videocua_diffabs, here at teacher-rollout granularity).
  * scroll = wheel amount.
  * +KEY / -KEY = press / release key events (mouse buttons LMB/RMB/MMB and JS-style
    keycodes ControlLeft/KeyA/...), produced by the SAME build_videocua_chat key machinery
    (normalize_button / normalize_chord / char_to_chord / chord_events) the diffabs BC used,
    so the token vocabulary is byte-identical to videocua_diffabs_v1.
  * NO_OP for an idle turn, TERMINATE for the terminal turn (the diffabs TERMINATE token).

System prompt = build_videocua_chat.SYSPROMPT (the psai/diffabs prompt baked into the
sol_diffabs BC). Output _normalized/{train,val}/chat.jsonl is consumed by the SAME omegalax
build_sft_records_from_chat.py tokenizer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_GOLDEN = "/home/franz.srambical/slurm/dev/franz/berlin/crowd-cast-bc/videocua_golden_v1"
if _GOLDEN not in sys.path:
    sys.path.insert(0, _GOLDEN)
import build_videocua_chat as bvc          # noqa: E402  (custom-grammar key machinery + SYSPROMPT)

_KEY_MAP = bvc.read_json(_GOLDEN + "/videocua_key_map.json")
SYSPROMPT = bvc.SYSPROMPT
TERMINATE_TOKEN = "TERMINATE"
# Failure/infeasible terminal token (fairness for the ~9% infeasible eval tasks; abs/move_rel can
# emit terminate{status:failure}/answer, diffabs previously could not). a4781827's diffabs eval
# parser recognizes bare-first-line uppercase `FAIL` -> terminate_failure. Map teacher
# terminate{status:failure} / answer / infeasible -> FAIL; terminate{status:success} -> TERMINATE.
FAIL_TOKEN = "FAIL"

_COORD_ACTIONS = {"left_click", "right_click", "middle_click", "double_click",
                  "triple_click", "mouse_move", "mouse_down", "left_click_drag"}
_CLICK_BUTTON = {"left_click": "left", "right_click": "right", "middle_click": "middle",
                 "double_click": "left", "triple_click": "left"}
_CLICK_N = {"double_click": 2, "triple_click": 3}


def action_to_label(a: dict, cursor_before, intended_target) -> str | None:
    """One absolute computer_use arg dict -> one custom-grammar label (or None if off-grammar)."""
    action = str(a.get("action", "")).strip().lower()
    if action == "hscroll":
        action = "scroll"
    elif action == "answer":
        return FAIL_TOKEN  # no text channel in diffabs; treat answer/infeasible as a failure declaration

    # diff-of-absolute cursor delta (RAW PIXELS) for coordinate-carrying actions only.
    dx = dy = 0
    if (action in _COORD_ACTIONS and cursor_before and intended_target
            and len(cursor_before) == 2 and len(intended_target) == 2):
        dx = int(round(float(intended_target[0]) - float(cursor_before[0])))
        dy = int(round(float(intended_target[1]) - float(cursor_before[1])))

    scroll = 0
    tokens: list[str] = []

    def add_events(evs):
        for kind, key in evs:
            tokens.append(("+" if kind == "press" else "-") + key)

    if action in {"left_click", "right_click", "middle_click", "double_click", "triple_click"}:
        button = bvc.normalize_button(_CLICK_BUTTON[action]) or "LMB"
        for _ in range(_CLICK_N.get(action, 1)):
            tokens.append("+" + button)
            tokens.append("-" + button)
    elif action == "left_click_drag":
        tokens.append("+LMB")
        tokens.append("-LMB")
    elif action == "mouse_move":
        pass  # movement delta only
    elif action == "mouse_down":
        tokens.append("+" + (bvc.normalize_button(a.get("button", "left")) or "LMB"))
    elif action == "mouse_up":
        tokens.append("-" + (bvc.normalize_button(a.get("button", "left")) or "LMB"))
    elif action in {"key", "key_down", "key_up"}:
        keys = a.get("keys")
        if keys is None:
            keys = a.get("key")
        raw = " + ".join(str(k) for k in keys) if isinstance(keys, list) else str(keys)
        chord = bvc.normalize_chord(raw, _KEY_MAP)
        if chord is None:
            return None
        mode = {"key": "tap", "key_down": "down", "key_up": "up"}[action]
        add_events(bvc.chord_events(*chord, mode=mode))
    elif action == "type":
        text = str(a.get("text") or "")
        for ch in text:
            chord = bvc.char_to_chord(ch, _KEY_MAP)
            if chord is None:
                continue
            add_events(bvc.chord_events(*chord, mode="tap"))
    elif action == "scroll":
        amt = a.get("pixels")
        if amt is None:
            amt = a.get("scroll_amount", a.get("amount", 0))
        scroll = int(round(bvc.number(amt, 0.0)))
        direction = str(a.get("scroll_direction") or "").strip().lower()
        if direction == "down" and scroll > 0:
            scroll = -scroll
    elif action == "wait":
        pass  # NO_OP
    else:
        return None  # off-grammar

    if dx == 0 and dy == 0 and scroll == 0 and not tokens:
        return "NO_OP"
    label = f"{dx} {dy} {scroll}"
    if tokens:
        label += " ; " + " ".join(tokens)
    return label


def convert_rollout(run_dir: Path, *, min_valid_actions: int = 1,
                    max_parse_error_frac: float = 0.5,
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
    steps_dir = run_dir / "steps"

    traj = []
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if line:
            traj.append(json.loads(line))

    turns = []          # list of (seen_frame_path, label)
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
        parsed = info.get("parsed")
        if info.get("parse_error") or not parsed:
            n_parse_err += 1
            continue
        if parsed.get("terminate"):
            status = str(parsed.get("computer_use_status") or "success").strip().lower()
            turns.append((str(seen_frame), FAIL_TOKEN if status == "failure" else TERMINATE_TOKEN))
            continue
        abs_args = parsed.get("computer_use")
        if not isinstance(abs_args, dict):
            n_parse_err += 1
            continue
        label = action_to_label(abs_args, info.get("cursor_before"), info.get("intended_target"))
        if label is None:
            n_parse_err += 1
            continue
        turns.append((str(seen_frame), label))

    n_valid = len(turns)
    if n_valid < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    messages = [{"role": "system", "content": [{"type": "text", "text": SYSPROMPT}]}]
    for i, (frame_path, label) in enumerate(turns):
        user_content = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

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
                              min_task_success=args.min_task_success)
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
        "grammar": "custom diffabs/psai (dx dy scroll ; +KEY -KEY)",
        "rollouts_dir": args.rollouts_dir, "keep_slugs": args.keep_slugs,
        "exclude_slugs": args.exclude_slugs,
        "n_seen": n_seen, "n_kept": n_kept, "n_dropped_by_filter": n_dropped_filter,
        "n_dropped_leak": n_dropped_leak,
        "n_unusable": n_unusable, "n_train": len(train), "n_val": len(val),
        "train_ratio": args.train_ratio, "min_valid_actions": args.min_valid_actions,
        "min_task_success": args.min_task_success, "coord_space": "pixel_diffabs",
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
