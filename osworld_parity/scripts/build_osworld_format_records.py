"""Build format-teaching SFT records from CAPTURED OSWorld teacher rollouts (Track 2).

Reads a4781827-collected rollouts at {app}/{task_id}/ with, per step:
  steps/messages_step_{NNN}.json  = the VERBATIM eval messages the off-shelf agent was sent
                                     (system + history_n=4 sliding window + current-frame user turn,
                                      images as data-URL of the process_image/smart_resize PNG).
  traj.jsonl                       = {step_num, response, action, reward, done}
  result.json                      = params.task_instruction, scores.reward, params.stop_reason

Per step k, one SFT example = the captured messages_step_{k}.json (the eval PROMPT, ≤4-turn window
PRESERVED VERBATIM -- never flattened/truncated) + the teacher's response to it appended as the
final assistant turn. So the training input byte-matches what off-shelf saw at eval time.

  * ABSOLUTE (positive control, format-shift 0): messages + teacher response VERBATIM; system =
    the captured eval system prompt. -> byte-identical to eval.
  * MOVE_REL / DIFFABS (treatment): SAME captured scaffold, but (a) swap the system prompt for the
    format's describing prompt (move_rel_format.SYSTEM_PROMPT / build_videocua_chat.SYSPROMPT --
    a4781827 saved these as moverel_system_prompt.txt / diffabs_system_prompt.txt; sha-verified),
    (b) convert every ASSISTANT turn (in-window history responses + the target) abs->format using
    the per-step telescoped cursor, (c) keep the NL user-turn text (Instruction/Previous actions).

Images: each data-URL frame is decoded to a PNG under --images_dir (dedup by content hash) and
referenced as {"type":"image","image":<path>}; the bytes are the eval-processed (smart_resize) PNG.
The SFT Qwen3-VL processor must be configured (min/max_pixels) so it does NOT re-resize -> train
image bytes == eval (pending a4781827's exact process_image params; flagged).
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

_SCR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts"
_MR_DIR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_moverel"
_V1_DIR = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_nativerel_v1"
_GOLDEN = "/home/franz.srambical/slurm/dev/franz/berlin/crowd-cast-bc/videocua_golden_v1"
for _p in (_SCR, _MR_DIR, _V1_DIR, _GOLDEN, "/fast/home/franz.srambical/juergen/eval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from action_parser import parse_computer_use_tool_calls  # noqa: E402
import convert_abs_to_diffabs as diffabs_conv            # noqa: E402  (action_to_label + FAIL/TERMINATE)
import convert_abs_to_deltatype as deltatype_conv         # noqa: E402  (coalesced type + terminate/fail)
import move_rel_format as mrfmt                            # noqa: E402  (move_rel SYSTEM_PROMPT + split)

_EVAL_PROMPT = Path("/fast/home/franz.srambical/osworld_parity_split/eval_system_prompt.txt").read_text()
_FMT_SYSTEM = {
    "absolute": _EVAL_PROMPT,
    "moverel": mrfmt.SYSTEM_PROMPT,
    "diffabs": diffabs_conv.SYSPROMPT,
    # deltatype = crowd-cast-native bare-token + coalesced type("...") + documented TERMINATE/FAIL.
    "deltatype_raw": deltatype_conv._sysprompt("raw"),
    "deltatype_normalized": deltatype_conv._sysprompt("normalized"),
}
_DELTATYPE_COORD = {"deltatype_raw": "raw", "deltatype_normalized": "normalized"}
_COORD_ACTIONS = {"mouse_move", "left_click", "right_click", "middle_click",
                  "double_click", "triple_click", "left_click_drag", "mouse_down"}


def _img_to_path(content_item, images_dir: Path) -> dict:
    """data-URL image_url -> {"type":"image","image":<png path>} (dedup by content hash)."""
    url = content_item.get("image_url", {}).get("url", "")
    m = re.match(r"data:image/\w+;base64,(.*)", url, re.DOTALL)
    if not m:
        return content_item
    raw = base64.b64decode(m.group(1))
    h = hashlib.sha1(raw).hexdigest()[:20]
    p = images_dir / f"{h}.png"
    if not p.exists():
        p.write_bytes(raw)
    return {"type": "image", "image": str(p)}


def _conv_image_msgs(messages, images_dir):
    """Rewrite image_url content items to file-path images (bytes unchanged)."""
    out = []
    for m in messages:
        nc = []
        for c in m["content"]:
            if c.get("type") == "image_url":
                nc.append(_img_to_path(c, images_dir))
            else:
                nc.append(dict(c))
        out.append({"role": m["role"], "content": nc})
    return out


def _telescope(traj_resps, sw, sh):
    """Per step -> (parsed_args_or_None, cursor_before_px, intended_target_px). Telescoped cursor."""
    per = {}
    cursor = [sw / 2.0, sh / 2.0]
    for k in sorted(traj_resps):
        resp = traj_resps[k]
        try:
            calls = parse_computer_use_tool_calls(resp)
        except Exception:
            calls = []
        if not calls:
            per[k] = (None, [int(cursor[0]), int(cursor[1])], [int(cursor[0]), int(cursor[1])])
            continue
        args = dict(calls[0].arguments)
        action = str(args.get("action", "")).lower()
        target = list(cursor)
        co = args.get("coordinate")
        if action in _COORD_ACTIONS and isinstance(co, (list, tuple)) and len(co) == 2:
            try:
                target = [float(co[0]) * sw / 1000.0, float(co[1]) * sh / 1000.0]
                cursor = list(target)
            except (TypeError, ValueError):
                pass
        per[k] = (args, [int(cursor[0]), int(cursor[1])], [int(round(target[0])), int(round(target[1]))])
        # note: cursor_before for THIS step is captured before advancing; store pre-advance below
    # recompute cursor_before correctly (pre-action) in a second pass
    per2 = {}
    cur = [sw / 2.0, sh / 2.0]
    for k in sorted(traj_resps):
        args, _, tgt = per[k]
        cb = [int(round(cur[0])), int(round(cur[1]))]
        if args is not None:
            action = str(args.get("action", "")).lower()
            co = args.get("coordinate")
            if action in _COORD_ACTIONS and isinstance(co, (list, tuple)) and len(co) == 2:
                cur = [float(tgt[0]), float(tgt[1])]
        per2[k] = (args, cb, tgt)
    return per2


def convert_response(resp_text, fmt, per_step, step_k):
    """abs response text -> the target-format assistant text for step_k."""
    if fmt == "absolute":
        return resp_text
    args, cb, tgt = per_step.get(step_k, (None, None, None))
    # terminate / no computer_use
    try:
        calls = parse_computer_use_tool_calls(resp_text)
    except Exception:
        calls = []
    _bare = fmt in ("diffabs", "deltatype_raw", "deltatype_normalized")
    if not calls or args is None:
        # fall back to a benign wait/no-op
        return "NO_OP" if _bare else '<tool_call>\n{"name": "computer_use", "arguments": {"action": "wait", "time": 1}}\n</tool_call>'
    a = dict(args)
    action = str(a.get("action", "")).lower()
    if fmt == "diffabs":
        if action == "terminate":
            status = str(a.get("computer_use_status") or a.get("status") or "success").lower()
            return diffabs_conv.FAIL_TOKEN if status == "failure" else diffabs_conv.TERMINATE_TOKEN
        lbl = diffabs_conv.action_to_label(a, cb, tgt)
        return lbl if lbl is not None else "NO_OP"
    if fmt in _DELTATYPE_COORD:
        if action == "terminate":
            status = str(a.get("computer_use_status") or a.get("status") or "success").lower()
            return deltatype_conv.FAIL_TOKEN if status == "failure" else deltatype_conv.TERMINATE_TOKEN
        lbl = deltatype_conv.action_to_label(a, cb, tgt, coord_space=_DELTATYPE_COORD[fmt],
                                             sw=_SW, sh=_SH)
        return lbl if lbl is not None else "NO_OP"
    # moverel: reuse the v3-normalized + split path via move_rel_format on a native_rel arg
    # build a native_rel arg (normalized 0-999 delta) then split into move_rel grammar
    if action == "terminate":
        payload = {"name": "computer_use", "arguments": {"action": "terminate",
                   "status": a.get("computer_use_status") or a.get("status") or "success"}}
        return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(", ", ": ")) + "\n</tool_call>"
    return _moverel_render(a, cb, tgt)


_SW, _SH = 1920, 1080  # set in main()


def _moverel_render(a, cursor_before, intended_target):
    """abs computer_use arg (0-999 coord) + telescoped cursor -> move_rel tool_call text.

    Reuses the EXACT synthetic-path move_rel conversion: v3 normalized diff-of-abs
    (rel_norm = raw_0_999 - cursor_px*1000/dim) then split into explicit move_rel +
    coordinate-less op. Byte-consistent with convert_abs_to_moverel / the move_rel BC suite."""
    import convert_abs_to_relative as v3
    import convert_abs_to_moverel as mrconv
    rel = v3._normalize_abs_args_to_rel(a, cursor_before, intended_target,
                                        coord_space="normalized", screen=[_SW, _SH])
    if rel is None:
        return '<tool_call>\n{"name": "computer_use", "arguments": {"action": "wait", "time": 1}}\n</tool_call>'
    v2 = mrconv.split_already_normalized([rel])
    return v3._render_assistant_text(v2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collected_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", required=True, choices=list(_FMT_SYSTEM))
    p.add_argument("--images_dir", default=None)
    p.add_argument("--screen_width", type=int, default=1920)
    p.add_argument("--screen_height", type=int, default=1080)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--exclude_slugs", default=None)
    args = p.parse_args()

    import random
    fmt = args.format
    out = Path(args.out_dir)
    images_dir = Path(args.images_dir or (out / "_images"))
    images_dir.mkdir(parents=True, exist_ok=True)
    sw, sh = args.screen_width, args.screen_height
    global _SW, _SH
    _SW, _SH = sw, sh
    sys_text = _FMT_SYSTEM[fmt]
    drop = set()
    if args.exclude_slugs and Path(args.exclude_slugs).is_file():
        drop = {s.strip() for s in Path(args.exclude_slugs).read_text().splitlines() if s.strip()}

    root = Path(args.collected_root)
    records = []
    n_tasks = n_steps = 0
    for app_dir in sorted(root.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith("qemu_logs"):
            continue
        for task_dir in sorted(app_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            slug = f"{app_dir.name}__{task_dir.name}"
            if slug in drop or f"{app_dir.name}/{task_dir.name}" in drop:
                continue
            traj_p = task_dir / "traj.jsonl"
            steps_dir = task_dir / "steps"
            if not (traj_p.is_file() and steps_dir.is_dir()):
                continue
            resp_by_step = {}
            for line in traj_p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                sn = e.get("step_num", 0)
                if sn >= 1 and sn not in resp_by_step:
                    resp_by_step[sn] = e.get("response") or ""
            per_step = _telescope({k - 1: resp_by_step[k] for k in resp_by_step}, sw, sh)  # 0-based
            msg_files = sorted(steps_dir.glob("messages_step_*.json"))
            if not msg_files:
                continue
            n_tasks += 1
            for mf in msg_files:
                k = int(re.search(r"messages_step_(\d+)", mf.name).group(1))  # 0-based
                target_abs = resp_by_step.get(k + 1)
                if not target_abs:
                    continue
                prompt_msgs = json.loads(mf.read_text())
                prompt_msgs = _conv_image_msgs(prompt_msgs, images_dir)
                if prompt_msgs and prompt_msgs[0]["role"] == "system":
                    prompt_msgs[0] = {"role": "system", "content": [{"type": "text", "text": sys_text}]}
                if fmt != "absolute":
                    # Convert in-window assistant history turns abs->format. The j-th assistant turn
                    # (in order) maps to action step (k - n_asst + j) [0-based per_step], i.e. the
                    # last n_asst actions before the current frame; use each step's telescoped cursor.
                    n_asst = sum(1 for m in prompt_msgs if m["role"] == "assistant")
                    j = 0
                    for m in prompt_msgs:
                        if m["role"] != "assistant":
                            continue
                        mapped = k - n_asst + j
                        m["content"] = [{"type": "text",
                                         "text": convert_response(m["content"][0]["text"], fmt, per_step, mapped)}]
                        j += 1
                target_text = convert_response(target_abs, fmt, per_step, k)
                msgs = prompt_msgs + [{"role": "assistant", "content": [{"type": "text", "text": target_text}]}]
                records.append({"sample_id": f"osw_{slug}_step{k:03d}", "recording_id": slug,
                                "app": app_dir.name, "task_id": task_dir.name, "step": k,
                                "format": fmt, "messages": msgs})
                n_steps += 1

    # split by task
    rng = random.Random(args.split_seed)
    ids = sorted({r["recording_id"] for r in records})
    rng.shuffle(ids)
    n_train = max(1, round(len(ids) * args.train_ratio)) if ids else 0
    train_ids = set(ids[:n_train])
    train = [r for r in records if r["recording_id"] in train_ids]
    val = [r for r in records if r["recording_id"] not in train_ids]
    for split, recs in (("train", train), ("val", val)):
        d = out / "_normalized" / split
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chat.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"format": fmt, "collected_root": args.collected_root, "system_prompt_len": len(sys_text),
                "n_tasks": n_tasks, "n_step_records": n_steps, "n_train": len(train), "n_val": len(val),
                "images_dir": str(images_dir)}
    (out).mkdir(parents=True, exist_ok=True)
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
