"""Re-normalize the videocua_nativerel retention set from raw-px deltas to the
0-999 / COORD_SCALE=1000 convention, so the anneal mix is coord-consistent with
the on-policy osworld_thinking_v1 data.

videocua_nativerel assistant tool_calls carry CURSOR-DELTA coordinates in the
per-recording SOURCE resolution's pixels (which VARIES: 1280x720, 1920x1008, ...).
Mixing those raw-px deltas with 0-999-normalized on-policy deltas silently corrupts
the anneal. Fix: rescale each tool_call coordinate delta [dx,dy] by (1000/w, 1000/h)
per-axis, where (w,h) = that recording's source resolution (videocua_frames_v1/
<recording_id>/extract_meta.json; PIL-fallback on the first frame image). Matches the
converter (cursor_norm = px*1000/dim) and freeroll --rel_coord_grid 1000.

Non-coordinate actions (type/key/scroll/wait/terminate) pass through unchanged.
Assistant turns are tool-call-only (no prose) in videocua_nativerel.
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from functools import lru_cache

FRAMES = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_frames_v1"
_TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@lru_cache(maxsize=4096)
def _resolution(recording_id: str, first_image: str | None) -> tuple[float, float] | None:
    meta = os.path.join(FRAMES, recording_id, "extract_meta.json")
    if os.path.isfile(meta):
        try:
            d = json.load(open(meta))
            w, h = float(d.get("width", 0)), float(d.get("height", 0))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    if first_image and os.path.isfile(first_image):
        try:
            from PIL import Image
            with Image.open(first_image) as im:
                return float(im.width), float(im.height)
        except Exception:
            pass
    return None


def _rescale_toolcall_block(block_json: str, sx: float, sy: float) -> str:
    """Rescale coordinate in one tool_call JSON string; re-render same format."""
    try:
        payload = json.loads(block_json)
    except json.JSONDecodeError:
        return block_json
    args = payload.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("coordinate"), (list, tuple)) and len(args["coordinate"]) == 2:
        c = args["coordinate"]
        args["coordinate"] = [int(round(float(c[0]) * sx)), int(round(float(c[1]) * sy))]
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


def _rescale_assistant(text: str, sx: float, sy: float) -> str:
    def repl(m):
        return "<tool_call>\n" + _rescale_toolcall_block(m.group(1), sx, sy) + "\n</tool_call>"
    return _TOOLCALL.sub(repl, text)


def _first_image(rec: dict) -> str | None:
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            for b in m.get("content", []):
                if isinstance(b, dict) and b.get("type") == "image":
                    return b.get("image")
    return None


def normalize_record(rec: dict) -> dict | None:
    rid = rec.get("recording_id")
    res = _resolution(rid, _first_image(rec)) if rid else None
    if res is None:
        return None  # can't determine source resolution -> drop (rare)
    w, h = res
    sx, sy = 1000.0 / w, 1000.0 / h
    out = dict(rec)
    new_msgs = []
    for m in rec["messages"]:
        if m["role"] == "assistant":
            c = [dict(b) for b in m["content"]]
            for b in c:
                if b.get("type") == "text":
                    b["text"] = _rescale_assistant(b["text"], sx, sy)
            new_msgs.append({"role": "assistant", "content": c})
        else:
            new_msgs.append(m)
    out["messages"] = new_msgs
    out["_coord_norm"] = "COORD_SCALE=1000"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_nativerel_v1/_normalized")
    ap.add_argument("--out", required=True, help="dataset root; writes _normalized/{train,val}/chat.jsonl")
    args = ap.parse_args()
    for split in ("train", "val"):
        srcf = Path(args.src) / split / "chat.jsonl"
        if not srcf.is_file():
            continue
        dst = Path(args.out) / "_normalized" / split
        dst.mkdir(parents=True, exist_ok=True)
        n = kept = dropped = 0
        with (dst / "chat.jsonl").open("w") as g:
            for line in srcf.open():
                line = line.strip()
                if not line:
                    continue
                n += 1
                rec = normalize_record(json.loads(line))
                if rec is None:
                    dropped += 1
                    continue
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{split}: {kept}/{n} kept ({dropped} dropped: no source resolution) -> {dst}/chat.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
