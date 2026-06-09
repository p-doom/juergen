"""Stage C (yll annotation pilot v2 — browseable):

Produce a per-segment, goal-conditioned dataset artifact that mirrors the
shape the labctl UI's dataset explorer knows how to walk
(``<split>/<seg>/{meta.json, chat_line.json, frames/}``), as a
counterpart to the opaque Stage-D chunk_index Grain shards.

Input is yll's existing per-segment frame layout under
``annotation_pilot/stage_a_2fps_540p/`` plus ``annotation_pilot/samples.jsonl``
(the sub_goal definitions that map ``(recording_id, seg_idx) -> instruction``).
The output is a fresh artifact under ``--output_dir`` with:

* the same per-segment frame directories, with ``frames/`` symlinked to
  the source so we don't duplicate JPEGs,
* a rewritten ``chat_line.json`` that injects the sub_goal text into
  the first user turn alongside the first image (matching the
  trainer-format produced by stage_d after its system-drop /
  instruction-into-first-user transform), with the last assistant turn
  replaced by ``TERMINATE``,
* a ``meta.json`` carrying the post-cap ``n_frames`` and ``n_no_op``
  counts so the UI's segments table renders correctly,
* a top-level ``manifest.json`` so labctl picks it up as a dataset
  artifact.

Optionally applies the same ``NO_OP`` run-length cap as
``stage_d_yll_annotation_pilot_v2``: contiguous NO_OP assistant turns
beyond ``k_frames = max(round(k_seconds * target_fps), 1)`` are
dropped along with their paired ``user(image)`` turns. The cap runs
per-segment (the natural unit for browsing) rather than per-bucketed-
sample, so the dropped counts are close to but not byte-identical to
stage_d's totals.

Train/val split mirrors stage_d's logic exactly: shuffle the segment
universe with ``split_seed`` and take the first
``round(N * train_ratio)`` for train, the rest for val. With v2's 28
segments at ``train_ratio=0.9`` / ``split_seed=0`` this reproduces
val = {12, 24, 27}.
"""

from __future__ import annotations

import collections
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from absl import app, flags

from _manifest import file_sha256_short, write_manifest

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Browseable artifact output dir.", required=True)
flags.DEFINE_string(
    "source_root",
    None,
    "Path to yll's per-segment stage_a_2fps_540p root "
    "(contains ``train/<recording_id>_seg<NNNN>/...``).",
    required=True,
)
flags.DEFINE_string(
    "samples_jsonl",
    None,
    "Path to yll's annotation_pilot/samples.jsonl with sub_goal definitions.",
    required=True,
)
flags.DEFINE_float(
    "k_seconds",
    None,
    "NO_OP cap budget. 0 disables the cap; otherwise contiguous NO_OP "
    "runs are capped at max(round(k_seconds * target_fps), 1) frames.",
    required=True,
)
flags.DEFINE_float(
    "target_fps",
    None,
    "Frame rate of the source trajectories. Combined with k_seconds to "
    "compute the per-run cap.",
    required=True,
)
flags.DEFINE_float("train_ratio", None, "Fraction of segments assigned to train.", required=True)
flags.DEFINE_integer("split_seed", None, "Seed for the segment shuffle.", required=True)
flags.DEFINE_string(
    "terminate_token",
    "TERMINATE",
    "Token written as the final assistant turn of every segment chat.",
)


_SEG_DIR_RE = re.compile(r"^recording_([0-9a-fA-F-]+)_seg(\d+)$")


def _load_sub_goal_index(samples_jsonl: Path) -> dict[tuple[str, int], str]:
    """Build ``(recording_id, seg_idx) -> instruction`` from samples.jsonl.

    Only ``level == "sub_goal"`` records carry segment ranges in v2.
    ``recording_id`` is recovered from the first image path in the
    record's messages (samples.jsonl entries don't have a top-level
    recording_id field).
    """
    image_re = re.compile(r"recording_([0-9a-fA-F-]+)_seg\d+")
    index: dict[tuple[str, int], str] = {}
    with samples_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("level") != "sub_goal":
                continue
            instr = s.get("instruction")
            segs = s.get("segments", [])
            if not isinstance(instr, str) or not instr or not segs:
                continue
            rec_id: str | None = None
            for m in s.get("messages", []):
                for c in m.get("content", []) or []:
                    if c.get("type") == "image":
                        mm = image_re.search(c.get("image", ""))
                        if mm:
                            rec_id = mm.group(1)
                            break
                if rec_id:
                    break
            assert rec_id, f"sub_goal {s.get('sample_id')!r} has no resolvable recording_id"
            for sg in segs:
                index[(rec_id, int(sg))] = instr
    return index


def _list_source_segments(source_root: Path) -> list[tuple[str, str, int, Path]]:
    """Return list of ``(seg_dir_name, recording_id, seg_idx, src_dir)``.

    Walks ``<source_root>/<split>/<seg_dir>`` for any splits present (yll
    currently has only ``train/`` upstream — the train/val assignment
    here is owned by this stage's own shuffle).
    """
    out: list[tuple[str, str, int, Path]] = []
    for src_split in ("train", "val", "test"):
        split_root = source_root / src_split
        if not split_root.is_dir():
            continue
        for ent in sorted(split_root.iterdir()):
            if not ent.is_dir():
                continue
            mm = _SEG_DIR_RE.match(ent.name)
            if not mm:
                continue
            out.append((ent.name, mm.group(1), int(mm.group(2)), ent))
    return out


def _cap_no_op_runs(
    messages: list[dict[str, Any]],
    *,
    k_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop NO_OP assistant turns past the per-run cap (and their paired
    user turns). Mirrors the logic in stage_d_yll_annotation_pilot_v2 but
    operates directly on a per-segment message list."""
    pairs: list[tuple[int, int, str]] = []
    last_user_idx: int | None = None
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "user":
            last_user_idx = i
        elif role == "assistant":
            assert last_user_idx is not None, (
                f"assistant turn at index {i} has no preceding user turn"
            )
            txt = ""
            for c in m.get("content", []) or []:
                if c.get("type") == "text":
                    txt = c.get("text", "")
                    break
            pairs.append((last_user_idx, i, txt))
            last_user_idx = None

    drop_idx: set[int] = set()
    run_pos = 0
    n_no_op_pre = 0
    if k_frames > 0:
        for u, a, text in pairs:
            if text == "NO_OP":
                n_no_op_pre += 1
                run_pos += 1
                if run_pos > k_frames:
                    drop_idx.add(u)
                    drop_idx.add(a)
            else:
                run_pos = 0
    else:
        n_no_op_pre = sum(1 for _, _, t in pairs if t == "NO_OP")

    n_dropped_asst = sum(1 for i in drop_idx if messages[i]["role"] == "assistant")
    n_dropped_user = sum(1 for i in drop_idx if messages[i]["role"] == "user")
    stats = {
        "n_assistant_pre": len(pairs),
        "n_assistant_post": len(pairs) - n_dropped_asst,
        "n_no_op_pre": n_no_op_pre,
        "n_no_op_post": n_no_op_pre - n_dropped_asst,
        "n_dropped_assistant": n_dropped_asst,
        "n_dropped_user": n_dropped_user,
    }
    if not drop_idx:
        return list(messages), stats
    return [m for i, m in enumerate(messages) if i not in drop_idx], stats


def _transform_segment(
    src_messages: list[dict[str, Any]],
    *,
    instruction: str,
    terminate_token: str,
    frame_path_for: callable,
) -> list[dict[str, Any]]:
    """Inject ``instruction`` as text in the first user turn, replace the
    last assistant turn's text with ``terminate_token``, and rewrite image
    paths via ``frame_path_for(filename)``."""
    new_messages: list[dict[str, Any]] = []
    first_user_emitted = False
    last_assistant_idx = -1
    for m in src_messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            new_content: list[dict[str, Any]] = []
            if not first_user_emitted:
                new_content.append({"type": "text", "text": instruction})
                first_user_emitted = True
            for c in m.get("content", []) or []:
                if c.get("type") == "image":
                    img_src = c.get("image", "")
                    fname = Path(img_src).name
                    new_content.append({"type": "image", "image": frame_path_for(fname)})
                else:
                    new_content.append(dict(c))
            new_messages.append({"role": "user", "content": new_content})
        else:
            new_messages.append({"role": role, "content": [dict(c) for c in m.get("content", []) or []]})
            if role == "assistant":
                last_assistant_idx = len(new_messages) - 1

    assert first_user_emitted, "segment has no user turn"
    assert last_assistant_idx >= 0, "segment has no assistant turn"
    new_messages[last_assistant_idx] = {
        "role": "assistant",
        "content": [{"type": "text", "text": terminate_token}],
    }
    return new_messages


def main(_) -> None:
    source_root = Path(FLAGS.source_root).resolve()
    samples_jsonl = Path(FLAGS.samples_jsonl).resolve()
    output_dir = Path(FLAGS.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k_frames = max(round(FLAGS.k_seconds * FLAGS.target_fps), 1) if FLAGS.k_seconds > 0 else 0

    sub_goal_index = _load_sub_goal_index(samples_jsonl)
    src_segments = _list_source_segments(source_root)
    assert src_segments, f"no segment dirs under {source_root}"

    seg_indices = sorted({seg_idx for _, _, seg_idx, _ in src_segments})
    rng = random.Random(FLAGS.split_seed)
    shuffled = list(seg_indices)
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * FLAGS.train_ratio))
    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train:])
    split_map = {
        "train_segments": sorted(train_set),
        "val_segments": sorted(val_set),
    }
    print(
        f"[stage_c_yll_browseable] {len(src_segments)} source segments, "
        f"split (train_ratio={FLAGS.train_ratio}, seed={FLAGS.split_seed}): "
        f"train={sorted(train_set)} val={sorted(val_set)}"
    )

    per_split_counts: dict[str, int] = {"train": 0, "val": 0}
    totals = collections.Counter()
    for seg_dir_name, rec_id, seg_idx, src_dir in src_segments:
        split = "train" if seg_idx in train_set else "val"
        instruction = sub_goal_index.get((rec_id, seg_idx))
        assert instruction, (
            f"no sub_goal instruction for ({rec_id}, seg{seg_idx}) in {samples_jsonl}"
        )

        src_chat = src_dir / "chat_line.json"
        src_meta = src_dir / "meta.json"
        src_frames = src_dir / "frames"
        assert src_chat.is_file(), f"missing chat_line.json in {src_dir}"
        assert src_meta.is_file(), f"missing meta.json in {src_dir}"
        assert src_frames.is_dir(), f"missing frames/ in {src_dir}"

        src_chat_obj = json.loads(src_chat.read_text())
        src_messages = src_chat_obj.get("messages", [])
        capped, cap_stats = _cap_no_op_runs(src_messages, k_frames=k_frames)
        for k, v in cap_stats.items():
            totals[k] += v

        out_seg_dir = output_dir / split / seg_dir_name
        out_seg_dir.mkdir(parents=True, exist_ok=True)
        out_frames = out_seg_dir / "frames"
        if out_frames.exists() or out_frames.is_symlink():
            out_frames.unlink()
        os.symlink(src_frames, out_frames)

        def _frame_path_for(fname: str) -> str:
            return str(out_frames / fname)

        new_messages = _transform_segment(
            capped,
            instruction=instruction,
            terminate_token=FLAGS.terminate_token,
            frame_path_for=_frame_path_for,
        )

        n_user_images = sum(
            1
            for m in new_messages
            if m["role"] == "user"
            for c in m.get("content", []) or []
            if c.get("type") == "image"
        )

        out_chat = {
            "segment_id": seg_dir_name,
            "recording_id": rec_id,
            "instruction": instruction,
            "messages": new_messages,
        }
        (out_seg_dir / "chat_line.json").write_text(json.dumps(out_chat))

        src_meta_obj = json.loads(src_meta.read_text())
        out_meta = dict(src_meta_obj)
        out_meta["segment_id"] = seg_dir_name
        out_meta["recording_id"] = rec_id
        out_meta["seg_idx"] = seg_idx
        out_meta["instruction"] = instruction
        out_meta["n_frames"] = n_user_images
        out_meta["n_no_op"] = cap_stats["n_no_op_post"]
        out_meta["stats"] = {
            "n_assistant_pre": cap_stats["n_assistant_pre"],
            "n_assistant_post": cap_stats["n_assistant_post"],
            "n_no_op_pre": cap_stats["n_no_op_pre"],
            "n_no_op_post": cap_stats["n_no_op_post"],
            "n_dropped_assistant": cap_stats["n_dropped_assistant"],
            "n_dropped_user": cap_stats["n_dropped_user"],
        }
        (out_seg_dir / "meta.json").write_text(json.dumps(out_meta))
        per_split_counts[split] += 1

    no_op_pct_pre = 100.0 * totals["n_no_op_pre"] / max(totals["n_assistant_pre"], 1)
    no_op_pct_post = 100.0 * totals["n_no_op_post"] / max(totals["n_assistant_post"], 1)
    print(
        f"[stage_c_yll_browseable] no_op cap k_frames={k_frames}: "
        f"assistant turns {totals['n_assistant_pre']}→{totals['n_assistant_post']}, "
        f"no_op {totals['n_no_op_pre']} ({no_op_pct_pre:.1f}%) → "
        f"{totals['n_no_op_post']} ({no_op_pct_post:.1f}%), "
        f"dropped {totals['n_dropped_assistant']} assistant + "
        f"{totals['n_dropped_user']} user turns; "
        f"per-split {per_split_counts}"
    )

    write_manifest(
        output_dir,
        stage="yll_annotation_pilot_v2_browseable_no_op_capped",
        params={
            "k_seconds": FLAGS.k_seconds,
            "target_fps": FLAGS.target_fps,
            "k_frames": k_frames,
            "train_ratio": FLAGS.train_ratio,
            "split_seed": FLAGS.split_seed,
            "terminate_token": FLAGS.terminate_token,
            "source_root": str(source_root),
            "samples_jsonl": str(samples_jsonl),
        },
        inputs={
            "samples_jsonl": str(samples_jsonl),
            "samples_jsonl_sha256_16": file_sha256_short(samples_jsonl),
        },
        stats={
            "per_split": per_split_counts,
            "split_map": split_map,
            "no_op_cap": {
                "n_assistant_pre": totals["n_assistant_pre"],
                "n_assistant_post": totals["n_assistant_post"],
                "n_no_op_pre": totals["n_no_op_pre"],
                "n_no_op_post": totals["n_no_op_post"],
                "no_op_pct_pre": round(no_op_pct_pre, 3),
                "no_op_pct_post": round(no_op_pct_post, 3),
                "n_dropped_assistant_turns": totals["n_dropped_assistant"],
                "n_dropped_user_turns": totals["n_dropped_user"],
            },
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
