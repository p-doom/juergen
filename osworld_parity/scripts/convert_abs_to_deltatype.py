"""Convert ABSOLUTE teacher rollouts -> DELTATYPE grammar records.

deltatype = the crowd-cast-native bare-token diffabs grammar with TWO fixes over diffabs:
  (1) COALESCED typing: a teacher `type("...")` becomes a single `type("...")` element
      (JSON-escaped string) instead of the char-by-char `+KeyH -KeyH +KeyE -KeyE ...`
      expansion diffabs used (4.4-21x fewer tokens; uses the model's native text-entry ability).
  (2) DOCUMENTED terminate/fail: TERMINATE (success) / FAIL (infeasible) are first-class,
      described in the system prompt (diffabs's eval prompt omitted them -> 4/107 terminate
      vs move_rel's 17/107).

Everything else is byte-identical to diffabs (crowd-cast round-trippable, ScaleAugment-compatible):
  * bare-token `<dx> <dy> <scroll>` optionally ` ; ELEM ELEM ...`
  * coordinate-less click `+LMB -LMB` at the current cursor
  * chords / special keys stay as `+KEY -KEY` (Return, Tab, ControlLeft, ...)
  * NO_OP for an idle turn

Mouse delta ENCODING is parameterized (--coord_space):
  raw        : dx,dy = round(intended_target - cursor_before)  in RAW screen pixels (diffabs).
  normalized : the same delta rescaled to a 0-999-of-screen grid
               dx = round((tgt-cur) * 1000 / SW), dy = round((tgt-cur) * 1000 / SH), clamp [-999,999].
               (deterministic, invertible, ScaleAugment-compatible since it only scales dx,dy).
The chosen encoding is set by the isolation experiment (isonorm_pipeline / tf_decomp_iso).

Grammar element order within a turn is preserved (move first, then events in teacher order).
Output _normalized/{train,val}/chat.jsonl is consumed by the SAME omegalax
build_sft_records_from_chat.py tokenizer as the other formats.
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
import build_videocua_chat as bvc          # noqa: E402  (normalize_button/normalize_chord/chord_events key machinery)

_KEY_MAP = bvc.read_json(_GOLDEN + "/videocua_key_map.json")
TERMINATE_TOKEN = "TERMINATE"
FAIL_TOKEN = "FAIL"

SW_DEFAULT, SH_DEFAULT = 1920, 1080

_COORD_ACTIONS = {"left_click", "right_click", "middle_click", "double_click",
                  "triple_click", "mouse_move", "mouse_down", "left_click_drag"}
_CLICK_BUTTON = {"left_click": "left", "right_click": "right", "middle_click": "middle",
                 "double_click": "left", "triple_click": "left"}
_CLICK_N = {"double_click": 2, "triple_click": 3}


def _encode_delta(dx_raw: int, dy_raw: int, coord_space: str, sw: int, sh: int) -> tuple[int, int]:
    if coord_space == "raw":
        return dx_raw, dy_raw
    # normalized 0-999 grid
    def clamp(v): return max(-999, min(999, v))
    return clamp(round(dx_raw * 1000.0 / sw)), clamp(round(dy_raw * 1000.0 / sh))


def action_to_label(a: dict, cursor_before, intended_target, *,
                    coord_space: str, sw: int, sh: int) -> str | None:
    """One absolute computer_use arg dict -> one deltatype label (or None if off-grammar)."""
    action = str(a.get("action", "")).strip().lower()
    if action == "hscroll":
        action = "scroll"
    elif action == "answer":
        return FAIL_TOKEN  # no free-text answer channel; treat as a failure declaration

    dx_raw = dy_raw = 0
    if (action in _COORD_ACTIONS and cursor_before and intended_target
            and len(cursor_before) == 2 and len(intended_target) == 2):
        dx_raw = int(round(float(intended_target[0]) - float(cursor_before[0])))
        dy_raw = int(round(float(intended_target[1]) - float(cursor_before[1])))
    dx, dy = _encode_delta(dx_raw, dy_raw, coord_space, sw, sh)

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
        pass
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
        # COALESCED: one type("...") element, JSON-escaped so ';', '+', quotes, spaces are safe.
        text = str(a.get("text") or "")
        if text:
            tokens.append("type(" + json.dumps(text, ensure_ascii=False) + ")")
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
        return None

    if dx == 0 and dy == 0 and scroll == 0 and not tokens:
        return "NO_OP"
    label = f"{dx} {dy} {scroll}"
    if tokens:
        label += " ; " + " ".join(tokens)
    return label


def _sysprompt(coord_space: str) -> str:
    if coord_space == "normalized":
        unit = ("`dx dy scroll` are three integers: a RELATIVE mouse move plus scroll wheel "
                "ticks. dx,dy are NORMALIZED to thousandths of the screen (each axis in "
                "[-999, 999]; dx=1000 spans the full width, dy=1000 the full height; dx>0 "
                "right, dx<0 left; dy>0 down, dy<0 up), relative to the CURRENT cursor.")
    else:
        unit = ("`dx dy scroll` are three integers: a RELATIVE mouse move in screen PIXELS "
                "plus scroll wheel ticks (dx>0 right, dx<0 left; dy>0 down, dy<0 up), "
                "relative to the CURRENT cursor.")
    return (
        "You operate a desktop computer with a mouse and keyboard. The first user turn shows "
        "the initial screen and the user's goal; each later turn shows the current screen "
        "(the cursor is a small arrow). Reply with exactly ONE action line per turn.\n"
        "Action := NO_OP | TERMINATE | FAIL | `dx dy scroll` | `dx dy scroll ; EVENTS`.\n"
        + unit + " The move is applied first, then EVENTS (after `;`, in order): `+X` presses, "
        "`-X` releases; mouse buttons LMB/RMB/MMB; keyboard keys use rdev names (Return, Tab, "
        "Backspace, ControlLeft, ShiftLeft, ArrowUp, ...). To type literal text use "
        '`type("...")` (e.g. `0 0 0 ; type("hello world")`). To click, move then click: '
        "`dx dy 0 ; +LMB -LMB`. Say TERMINATE when the goal is complete, FAIL if it is "
        "impossible, NO_OP to wait for the screen to settle."
    )


def convert_rollout(run_dir: Path, *, coord_space: str, sw: int, sh: int,
                    min_valid_actions: int = 2, max_parse_error_frac: float = 0.5,
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

    traj = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    turns = []
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
        label = action_to_label(abs_args, info.get("cursor_before"), info.get("intended_target"),
                                coord_space=coord_space, sw=sw, sh=sh)
        if label is None:
            n_parse_err += 1
            continue
        turns.append((str(seen_frame), label))

    if len(turns) < min_valid_actions:
        return None
    if n_steps and (n_parse_err / n_steps) > max_parse_error_frac:
        return None

    messages = [{"role": "system", "content": [{"type": "text", "text": _sysprompt(coord_space)}]}]
    for i, (frame_path, label) in enumerate(turns):
        user_content = [{"type": "image", "image": frame_path}]
        if i == 0 and instruction:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

    slug = result.get("slug") or run_dir.name
    return {
        "sample_id": f"onpol_{slug}", "recording_id": slug, "app": "osworld_onpolicy",
        "platform": "UBUNTU", "instruction": instruction, "n_frames": len(turns),
        "subrecord_idx": 0, "n_subrecords": 1,
        "source_stop_reason": result.get("stop_reason"), "source_parse_errors": n_parse_err,
        "messages": messages,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--coord_space", choices=["raw", "normalized"], required=True)
    p.add_argument("--screen_w", type=int, default=SW_DEFAULT)
    p.add_argument("--screen_h", type=int, default=SH_DEFAULT)
    p.add_argument("--min_task_success", type=float, default=None)
    p.add_argument("--exclude_slugs", default=None, help="text file of slugs to DROP (eval-leak filter).")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--min_valid_actions", type=int, default=2)
    p.add_argument("--max_parse_error_frac", type=float, default=0.5)
    args = p.parse_args()

    drop = set()
    if args.exclude_slugs and Path(args.exclude_slugs).is_file():
        drop = set(s.strip() for s in Path(args.exclude_slugs).read_text().splitlines() if s.strip())

    run_dirs = []
    for rd in args.rollouts_dir.split(","):
        rd = Path(rd.strip())
        if rd.is_dir():
            run_dirs += [d for d in rd.rglob("*") if d.is_dir() and (d / "result.json").is_file()
                         and (d / "trajectory.jsonl").is_file()]
    run_dirs = sorted(set(run_dirs))
    records = []
    n_seen = n_kept = n_dropped_leak = n_unusable = 0
    for d in run_dirs:
        n_seen += 1
        if d.name in drop:
            n_dropped_leak += 1
            continue
        rec = convert_rollout(d, coord_space=args.coord_space, sw=args.screen_w, sh=args.screen_h,
                              min_valid_actions=args.min_valid_actions,
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
        "grammar": "deltatype (dx dy scroll ; +KEY -KEY / type(\"...\")) bare-token",
        "coord_space": args.coord_space, "screen": [args.screen_w, args.screen_h],
        "rollouts_dir": args.rollouts_dir, "exclude_slugs": args.exclude_slugs,
        "n_seen": n_seen, "n_kept": n_kept, "n_dropped_leak": n_dropped_leak,
        "n_unusable": n_unusable, "n_train": len(train), "n_val": len(val),
        "train_ratio": args.train_ratio, "min_valid_actions": args.min_valid_actions,
        "min_task_success": args.min_task_success,
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
