#!/usr/bin/env python3
"""Realign the actions of an ALREADY-BUILT canonical SFT dataset in place.

Takes the existing canonical SFT artifact (chat.jsonl + per-split chat.jsonl +
sample_manifest.jsonl) and writes a fixed copy whose assistant action turns are
recomputed from the realigned keylog -- WITHOUT re-running frames/annotate/
assemble/canonical. Annotations, instructions, images, frame order and splits are
untouched; only the action strings (and the derived n_non_noop) change, and each
sample is tagged with its keylog->video alignment status.

How it joins: a sample's assistant turn is the action for the frame shown in the
immediately-preceding user image; that image's ``ar://...#idx`` URI matches a row
in the clip's stage_01/frame_records.jsonl. For each clip with a non-trivial
time-map we recompute every surviving frame's OWN-bin action
(``patched_actions``), build {image_uri: action}, and rewrite the turns. `aligned`
clips are left unchanged (only tagged). The terminal ``<TERMINATED>`` assistant
turn is never touched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from annotation_pipeline import realign_lib as R
from annotation_pipeline.common import (
    ActionBin,
    ceil_frames,
    format_action,
    load_keylog_entries,
    read_jsonl,
    resolve_button_name,
    resolve_key_name,
)

# Old artifacts (ccast0618d) end with <TERMINATED>; plan-aware builds use
# <TERMINATE> (see prompts/desktop_action_plan.txt).
TERMINAL_TOKENS = {"<TERMINATED>", "<TERMINATE>"}


def _aggregate_actions_realigned(keylog_path, n_bins, target_fps, splices):
    """common.aggregate_actions, but each event is binned by its CORRECTED
    video-PTS time (``keylog_to_video``) instead of the raw keylog timestamp."""
    bins = [ActionBin() for _ in range(n_bins)]
    held: set[str] = set()
    for entry in load_keylog_entries(keylog_path):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        timestamp, event = entry[0], entry[1]
        if not isinstance(event, list) or not event:
            continue
        try:
            timestamp_us = int(timestamp)
        except (TypeError, ValueError):
            continue
        event_type = str(event[0])
        payload = event[1] if len(event) > 1 else None
        if event_type == "ContextChanged":
            continue
        bucket_idx = int(R.keylog_to_video(timestamp_us / 1_000_000, splices) * target_fps)
        if bucket_idx < 0 or bucket_idx >= n_bins:
            continue
        b = bins[bucket_idx]
        if event_type == "MouseMove":
            if isinstance(payload, list) and len(payload) >= 2:
                b.move_dx += float(payload[0]); b.move_dy += float(payload[1])
        elif event_type == "MouseScroll":
            if isinstance(payload, list) and len(payload) >= 2:
                b.scroll += float(payload[1] if payload[1] != 0 else payload[0])
        elif event_type in ("KeyPress", "MousePress"):
            name = resolve_key_name(payload) if event_type == "KeyPress" else resolve_button_name(payload)
            if name and name not in held:
                b.events.append(("+", name)); held.add(name)
        elif event_type in ("KeyRelease", "MouseRelease"):
            name = resolve_key_name(payload) if event_type == "KeyRelease" else resolve_button_name(payload)
            if name and name in held:
                b.events.append(("-", name)); held.remove(name)
    return bins


def patched_actions(frame_records, keylog_path, video_dur_s, target_fps, splices):
    """New action strings (in frame_records order) + (changed, recovered, lost).

    Each surviving frame is reassigned the action at its OWN video-time bin
    (``corrected_bin[local_bin_idx]``); dropped-frame activity is NOT relocated
    onto survivors (see module docstring)."""
    n_bins = ceil_frames(video_dur_s, target_fps)
    bins = _aggregate_actions_realigned(keylog_path, n_bins, target_fps, splices)
    new_actions, n_changed, n_recovered, n_lost = [], 0, 0, 0
    for fr in frame_records:
        b = int(fr["local_bin_idx"])
        new = format_action(bins[b] if 0 <= b < n_bins else ActionBin())
        old = fr.get("action", "NO_OP")
        if new != old:
            n_changed += 1
            if old == "NO_OP" and new != "NO_OP":
                n_recovered += 1
            elif old != "NO_OP" and new == "NO_OP":
                n_lost += 1
        new_actions.append(new)
    return new_actions, n_changed, n_recovered, n_lost


def _assistant_text(msg: dict) -> str | None:
    if msg.get("role") != "assistant":
        return None
    content = msg.get("content")
    if isinstance(content, list):
        for c in content:
            if c.get("type") == "text":
                return c.get("text")
    elif isinstance(content, str):
        return content
    return None


def _set_assistant_text(msg: dict, text: str) -> None:
    content = msg.get("content")
    if isinstance(content, list):
        for c in content:
            if c.get("type") == "text":
                c["text"] = text
                return
        content.append({"type": "text", "text": text})
    else:
        msg["content"] = text


def _user_image(msg: dict) -> str | None:
    if msg.get("role") != "user":
        return None
    content = msg.get("content")
    if isinstance(content, list):
        for c in content:
            if c.get("type") == "image":
                return c.get("image")
    return None


class ActionMapCache:
    """Lazily builds {image_uri: corrected_action} per clip (only for clips with
    a non-trivial map; `aligned` clips return None == leave unchanged)."""

    def __init__(self, frames_root: Path, align_by_sid: dict[str, dict]):
        self.frames_root = frames_root
        self.align = align_by_sid
        self.cache: dict[str, dict[str, str] | None] = {}

    def get(self, clip_id: str) -> dict[str, str] | None:
        if clip_id in self.cache:
            return self.cache[clip_id]
        row = self.align.get(clip_id)
        result: dict[str, str] | None = None
        if row and row.get("splices"):
            clip = self.frames_root / "clips" / clip_id
            frame_records = read_jsonl(clip / "stage_01" / "frame_records.jsonl")
            man = read_jsonl(clip / "stage_00" / "manifest.jsonl")
            summ = clip / "stage_01" / "frames_actions_summary.json"
            if frame_records and man and summ.exists():
                fps = json.loads(summ.read_text())["target_fps"]
                new_actions, *_ = patched_actions(
                    frame_records, Path(man[0]["keylog_path"]),
                    float(man[0]["video_duration_s"]), fps, row["splices"])
                result = {fr["image_path"]: act
                          for fr, act in zip(frame_records, new_actions, strict=True)}
        self.cache[clip_id] = result
        return result


def tags_for(row: dict | None) -> dict[str, Any]:
    if not row:
        return {"alignment_status": "aligned", "alignment_closed": True,
                "alignment_total_collapse_s": 0.0, "alignment_residual_s": 0.0}
    return {"alignment_status": row["status"], "alignment_closed": row["closed"],
            "alignment_total_collapse_s": row["total_collapse_s"],
            "alignment_residual_s": row["residual_s"]}


def _split_plan_prefix(text: str, plan: str) -> tuple[str, str]:
    """(prefix, action) for an assistant turn. A plan-bearing sample's FIRST
    assistant turn is `<plan>\\n<action>` (stage 02b/03); the prefix must
    survive action patching, so split it off and re-attach after."""
    if plan and text.startswith(plan):
        rest = text[len(plan):]
        if rest.startswith("\n"):
            return plan + "\n", rest[1:]
    return "", text


def patch_chat_row(row: dict, cache: ActionMapCache, align_by_sid: dict[str, dict],
                   stats: dict) -> int:
    """Patch a chat.jsonl record in place; return new n_non_noop."""
    clip_id = str(row.get("clip_id") or "")
    plan = str(row.get("plan") or "")
    amap = cache.get(clip_id)
    msgs = row.get("messages", [])
    last_img: str | None = None
    n_non_noop = 0
    for msg in msgs:
        img = _user_image(msg)
        if img is not None:
            last_img = img
            continue
        text = _assistant_text(msg)
        if text is None or text in TERMINAL_TOKENS:
            continue
        prefix, action = _split_plan_prefix(text, plan)
        if amap is not None and last_img is not None and last_img in amap:
            new_action = amap[last_img]
            if new_action != action:
                stats["turns_changed"] += 1
            action = new_action
            _set_assistant_text(msg, prefix + action)
        elif amap is not None and last_img is not None:
            stats["turns_unmatched"] += 1
        if action != "NO_OP":
            n_non_noop += 1
    row["n_non_noop"] = n_non_noop
    row.update(tags_for(align_by_sid.get(clip_id)))
    return n_non_noop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-dir", type=Path, required=True,
                   help="Existing canonical SFT artifact dir.")
    p.add_argument("--frames-root", type=Path, required=True,
                   help="Frames artifact _frames dir (for frame_records + manifests).")
    p.add_argument("--alignment", type=Path, required=True,
                   help="alignment.jsonl from stage_00_realign.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output dir for the realigned canonical SFT artifact.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    align_by_sid = {r["segment_id"]: r for r in read_jsonl(args.alignment)}
    cache = ActionMapCache(args.frames_root, align_by_sid)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    stats = {"turns_changed": 0, "turns_unmatched": 0}
    new_nnn: dict[str, int] = {}

    # chat.jsonl (top-level) + per-split chat.jsonl, streamed.
    chat_inputs = [args.canonical_dir / "chat.jsonl"]
    chat_inputs += sorted(args.canonical_dir.glob("*/chat.jsonl"))
    for src in chat_inputs:
        rel = src.relative_to(args.canonical_dir)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                nnn = patch_chat_row(row, cache, align_by_sid, stats)
                new_nnn[row["sample_id"]] = nnn
                fout.write(json.dumps(row, ensure_ascii=False,
                                      sort_keys=(rel.parent != Path("."))) + "\n")
                n += 1
        print(f"  wrote {n} rows -> {rel}", flush=True)

    # sample_manifest.jsonl: refresh n_non_noop + add alignment tags.
    sm_src = args.canonical_dir / "sample_manifest.jsonl"
    if sm_src.exists():
        rows = read_jsonl(sm_src)
        for r in rows:
            if r.get("sample_id") in new_nnn:
                r["n_non_noop"] = new_nnn[r["sample_id"]]
            r.update(tags_for(align_by_sid.get(str(r.get("clip_id") or ""))))
        with (out / "sample_manifest.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # carry split_manifest/rejected verbatim (frame set + splits unchanged).
    for name in ("split_manifest.jsonl", "rejected.jsonl"):
        src = args.canonical_dir / name
        if src.exists():
            shutil.copy2(src, out / name)

    # manifest.json + .meta.json: record realignment provenance rather than
    # claiming the original artifact's identity.
    man_src = args.canonical_dir / "manifest.json"
    if man_src.exists():
        man = json.loads(man_src.read_text())
        man["realignment"] = {
            "source_canonical": str(args.canonical_dir),
            "alignment": str(args.alignment),
            "method": "realign_patch_canonical (own-bin actions; annotations/frames unchanged)",
            "turns_changed": stats["turns_changed"],
        }
        man["artifact_type"] = man.get("artifact_type", "juergen_canonical_sft")
        (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    meta_src = args.canonical_dir / ".meta.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta.setdefault("metadata", {})["realigned_from"] = meta.get("alias", "")
        meta["alias"] = out.name
        meta["metadata"]["producer_recipe"] = "juergen_crowdcast_realign_canonical"
        (out / ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"turns_changed={stats['turns_changed']} turns_unmatched={stats['turns_unmatched']}")
    print(f"wrote realigned canonical -> {out}")


if __name__ == "__main__":
    main()
