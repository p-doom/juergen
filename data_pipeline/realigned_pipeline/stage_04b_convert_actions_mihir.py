#!/usr/bin/env python3
"""Stage 04b (action-format conversion): rewrite a stage-04 conversations
dataset's assistant action strings from the crowd-cast *canonical* grammar to
Mihir's IDM action format, leaving EVERYTHING ELSE (frames, ordering, goals,
system prompt, TERMINATE supervision, the train/val recording split applied at
stage 06) byte-for-byte identical.

This is the single-variable ablation lever: point stages 05/06 at this stage's
output instead of the canonical stage-04 output and the resulting run differs
from the canonical run ONLY in the action representation the model predicts.

Mihir normalizes mouse-move deltas by the ORIGINAL recording resolution
(dx/W*1000, dy/H*1000), and that varies ~2x across crowd-cast rigs, so this
stage joins each conversation's ``segment_id``/``recording_id`` to the stage-00
clip manifest's ``video_width``/``video_height`` — the exact source Mihir's
prepare_data.py probes with ffprobe. See lib/mihir_action_format for the full
mapping (and its documented lossy/adapted cases).

Output (--output_dir) mirrors stage 04 so stages 05/06 run untouched:
  chat.jsonl            one row per conversation, assistant actions in Mihir
                        format (split-agnostic drop-in source_path for 05/06).
  conversations.jsonl   identical rows (alias, matching stage 04).
  manifest.json         labctl dataset marker + conversion params/stats.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from absl import app, flags

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.manifest import write_manifest  # noqa: E402
from realigned_pipeline.lib.mihir_action_format import (  # noqa: E402
    ConversionCounters,
    convert_conversation,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Converted conversations output dir.", required=True)
flags.DEFINE_string(
    "source_path", None,
    "Stage-04 conversations dataset root (holds a single chat.jsonl).", required=True,
)
flags.DEFINE_string(
    "clips_manifest_path", None,
    "Stage-00 clip manifest: a clips_manifest.jsonl file or a dir containing it. "
    "Supplies per-recording video_width/video_height for Mihir mouse-delta "
    "normalization.", required=True,
)
flags.DEFINE_string(
    "terminal_token", "TERMINATE",
    "BC episode-end control token passed through verbatim (not a Mihir action).",
)
flags.DEFINE_integer("limit", 0, "Convert only the first N conversations (0 = all; debug).")
flags.DEFINE_string(
    "system_prompt_file", None,
    "Optional: replace each conversation's system message text with the contents "
    "of this file (used to align the system prompt with Mihir's action format). "
    "Only the leading system message is replaced; user/assistant turns untouched.",
)


def _load_resolution_maps(clips_manifest_path: Path) -> tuple[dict, dict]:
    """Build segment_id -> (w,h) and recording_id -> (w,h) from the clip manifest."""
    if clips_manifest_path.is_dir():
        clips_manifest_path = clips_manifest_path / "clips_manifest.jsonl"
    if not clips_manifest_path.is_file():
        raise FileNotFoundError(f"no clip manifest at {clips_manifest_path}")
    by_seg: dict[str, tuple[int, int]] = {}
    by_rec: dict[str, tuple[int, int]] = {}
    with clips_manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            w, h = row.get("video_width"), row.get("video_height")
            if not w or not h:
                continue
            seg = row.get("segment_id")
            rec = row.get("recording_id")
            if seg is not None:
                by_seg[str(seg)] = (int(w), int(h))
            if rec is not None:
                by_rec.setdefault(str(rec), (int(w), int(h)))
    if not by_seg and not by_rec:
        raise ValueError(f"no usable video_width/video_height rows in {clips_manifest_path}")
    return by_seg, by_rec


def _resolution_for(row: dict, by_seg: dict, by_rec: dict) -> tuple[int, int] | None:
    seg = row.get("segment_id")
    if seg is not None and str(seg) in by_seg:
        return by_seg[str(seg)]
    rec = row.get("recording_id")
    if rec is not None and str(rec) in by_rec:
        return by_rec[str(rec)]
    return None


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    src_chat = source_path / "chat.jsonl"
    if not src_chat.is_file():
        raise FileNotFoundError(f"no chat.jsonl under {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    by_seg, by_rec = _load_resolution_maps(Path(FLAGS.clips_manifest_path))
    terminal_tokens = (FLAGS.terminal_token,) if FLAGS.terminal_token else ()

    new_system_prompt = None
    if FLAGS.system_prompt_file:
        new_system_prompt = Path(FLAGS.system_prompt_file).read_text().rstrip("\n")

    counters = ConversionCounters()
    n_rows = 0
    n_missing_res = 0
    n_sysprompt_swapped = 0
    t0 = time.time()

    out_chat = output_dir / "chat.jsonl"
    out_convs = output_dir / "conversations.jsonl"
    with src_chat.open() as fin, out_chat.open("w") as fchat, out_convs.open("w") as fconv:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            res = _resolution_for(row, by_seg, by_rec)
            if res is None:
                n_missing_res += 1
                continue  # cannot normalize without resolution — drop + count
            w, h = res
            messages = row.get("messages", [])
            if new_system_prompt is not None and messages and messages[0].get("role") == "system":
                messages[0]["content"] = [{"type": "text", "text": new_system_prompt}]
                n_sysprompt_swapped += 1
            convert_conversation(
                messages,
                video_w=w,
                video_h=h,
                counters=counters,
                terminal_tokens=terminal_tokens,
            )
            out_line = json.dumps(row, ensure_ascii=False)
            fchat.write(out_line + "\n")
            fconv.write(out_line + "\n")
            n_rows += 1
            if FLAGS.limit and n_rows >= FLAGS.limit:
                break

    stats = {
        "n_conversations_written": n_rows,
        "n_conversations_dropped_no_resolution": n_missing_res,
        "n_system_prompt_swapped": n_sysprompt_swapped,
        "conversion_counters": counters.__dict__,
        "elapsed_s": int(time.time() - t0),
    }
    write_manifest(
        output_dir,
        stage="convert_actions_mihir",
        params={
            "source_path": str(source_path),
            "clips_manifest_path": str(FLAGS.clips_manifest_path),
            "terminal_token": FLAGS.terminal_token,
            "action_format": "mihir_idm_v1",
            "system_prompt_file": FLAGS.system_prompt_file,
        },
        inputs={"source": str(source_path), "clips_manifest": str(FLAGS.clips_manifest_path)},
        stats=stats,
    )
    print(f"[stage_04b] wrote {n_rows} conversations -> {out_chat}", flush=True)
    print(f"[stage_04b] dropped (no resolution): {n_missing_res}", flush=True)
    print(f"[stage_04b] system prompt swapped in {n_sysprompt_swapped} conversations", flush=True)
    print(f"[stage_04b] counters: {counters.__dict__}", flush=True)


if __name__ == "__main__":
    app.run(main)
