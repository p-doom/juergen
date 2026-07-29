"""Offline FORMAT-VALIDITY check for the Track-1 format-teaching SFTs.

Does an SFT'd model actually EMIT its target action grammar (absolute / move_rel / diffabs)
on held-out synthetic prompts? Self-contained: launches sglang on the exported merged-HF
checkpoint, then for N held-out val records feeds the FIRST-turn context (system prompt +
initial screenshot + instruction) and free-generates one response, then grades the grammar.

No VM / OSWorld needed -- this is an offline first-action generation probe (the "does SFT
teach the format at all" signal), NOT a closed-loop OSWorld measurement.
"""
from __future__ import annotations

import argparse
import atexit
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

_EVAL_DIR = Path("/fast/home/franz.srambical/juergen/eval")
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from osworld_runtime import _call_model, _wait_for  # noqa: E402
from action_parser import parse_computer_use_tool_calls  # noqa: E402


def _terminate_proc(proc) -> None:
    """Best-effort terminate the sglang subprocess."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

_DELTA_RE = re.compile(r"^\s*-?\d+\s+-?\d+\s+-?\d+\s*(;\s*[-+][^\s].*)?$")


def grade_absolute(text: str) -> tuple[bool, str]:
    if "move_rel" in text:
        return False, "contains move_rel (should be pure absolute)"
    try:
        calls = parse_computer_use_tool_calls(text)
    except Exception as e:  # noqa: BLE001
        return False, f"tool_call parse error: {e}"
    if not calls:
        return False, "no computer_use tool_call"
    for c in calls:
        a = c.arguments
        act = str(a.get("action", "")).lower()
        if act in {"left_click", "right_click", "middle_click", "double_click",
                   "triple_click", "mouse_move", "left_click_drag"}:
            co = a.get("coordinate")
            if not (isinstance(co, (list, tuple)) and len(co) == 2):
                return False, f"{act} missing absolute coordinate"
    return True, "ok"


def grade_moverel(text: str) -> tuple[bool, str]:
    try:
        calls = parse_computer_use_tool_calls(text)
    except Exception as e:  # noqa: BLE001
        return False, f"tool_call parse error: {e}"
    if not calls:
        return False, "no computer_use tool_call"
    acts = [str(c.arguments.get("action", "")).lower() for c in calls]
    has_move_rel = any(a == "move_rel" for a in acts)
    # a coordinate-less click is valid move_rel grammar (click in place); a move must be move_rel
    for c in calls:
        a = c.arguments
        act = str(a.get("action", "")).lower()
        if act == "mouse_move":
            return False, "used mouse_move (absolute) instead of move_rel"
        if act in {"left_click", "right_click", "middle_click", "double_click", "triple_click"} \
                and "coordinate" in a:
            return False, f"{act} carries a coordinate (should be coordinate-less; move via move_rel)"
    if not (has_move_rel or all(a in {"left_click", "right_click", "middle_click", "double_click",
                                       "triple_click", "type", "key", "scroll", "wait", "terminate"}
                                for a in acts)):
        return False, "no move_rel and not a valid coordinate-less action set"
    return True, "ok"


def grade_diffabs(text: str) -> tuple[bool, str]:
    line = text.strip().split("\n", 1)[0].strip()
    if line in ("NO_OP", "TERMINATE", "FAIL"):
        return True, "ok"
    if "<tool_call>" in text or "computer_use" in text:
        return False, "emitted tool_call/computer_use (should be bare-token delta)"
    if _DELTA_RE.match(line):
        return True, "ok"
    return False, f"not a valid '<dx> <dy> <scroll> ; +KEY' delta line: {line[:60]!r}"


_GRADERS = {"absolute": grade_absolute, "moverel": grade_moverel, "diffabs": grade_diffabs}


def first_turn(record: dict):
    """Return (system_text, instruction, first_image_path, gold_action)."""
    msgs = record["messages"]
    system = msgs[0]["content"][0]["text"]
    instruction = None
    first_img = None
    gold = None
    for m in msgs:
        if m["role"] == "user":
            for c in m["content"]:
                if c.get("type") == "image" and first_img is None:
                    first_img = c["image"]
                elif c.get("type") == "text" and instruction is None:
                    instruction = c["text"]
        elif m["role"] == "assistant" and gold is None:
            gold = m["content"][0]["text"]
        if first_img and gold:
            break
    return system, instruction, first_img, gold


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="exported merged-HF SFT checkpoint")
    p.add_argument("--val_jsonl", required=True, help="the format's _normalized/val/chat.jsonl")
    p.add_argument("--format", required=True, choices=list(_GRADERS))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out_json", required=True)
    p.add_argument("--sglang_port", type=int, default=31500)
    p.add_argument("--sglang_api_key", default="osworld")
    p.add_argument("--mem_fraction_static", type=float, default=0.80)
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.val_jsonl)][: args.n]
    grader = _GRADERS[args.format]

    log = Path(args.out_json).with_suffix(".sglang.log")
    proc = subprocess.Popen(
        ["uv", "run", "--project", str(_EVAL_DIR), "python", "-m", "sglang.launch_server",
         "--model-path", args.model_path, "--host", "0.0.0.0", "--port", str(args.sglang_port),
         "--api-key", args.sglang_api_key, "--mem-fraction-static", str(args.mem_fraction_static),
         "--chunked-prefill-size", "2048"],
        cwd=str(_EVAL_DIR), stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )
    atexit.register(lambda: _terminate_proc(proc))
    _wait_for(f"http://localhost:{args.sglang_port}/health_generate",
              headers={"Authorization": f"Bearer {args.sglang_api_key}"},
              proc=proc, poll_s=10, max_polls=180, label="sglang")
    sglang_url = f"http://localhost:{args.sglang_port}/v1"

    results = []
    n_valid = 0
    for i, rec in enumerate(records):
        system, instruction, img_path, gold = first_turn(rec)
        try:
            frame = Image.open(img_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            results.append({"i": i, "error": f"image load: {e}"}); continue
        pred = _call_model(sglang_url=sglang_url, api_key=args.sglang_api_key,
                           model=args.model_path, system_prompt=system, instruction=instruction,
                           recent_frames=[frame], recent_actions=[],
                           max_tokens=args.max_tokens, temperature=0.0)
        ok, why = grader(pred)
        n_valid += int(ok)
        results.append({"i": i, "recording_id": rec.get("recording_id"), "valid": ok,
                        "why": why, "pred": pred[:300], "gold": (gold or "")[:200]})

    summary = {"format": args.format, "model_path": args.model_path, "n": len(results),
               "n_valid": n_valid, "format_validity_rate": round(n_valid / max(1, len(results)), 4),
               "results": results}
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("format", "n", "n_valid", "format_validity_rate")}, indent=2))
    for r in results[:5]:
        print(f"  [{r.get('valid')}] {r.get('why','')} | pred={r.get('pred','')[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
