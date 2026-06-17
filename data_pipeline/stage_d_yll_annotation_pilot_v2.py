"""Stage D (yll annotation pilot v2): like v1, with a NO_OP run-length cap.

Same transform as ``stage_d_yll_annotation_pilot_v1.py`` — drop the source
``system`` turn, move ``instruction`` into the first surviving user turn,
TERMINATE on the last assistant turn, resolve image paths, segment-based
train/val split — with one extra step in front: contiguous runs of
``NO_OP`` assistant labels in the source record are capped at
``k_frames = max(round(k_seconds * target_fps), 1)``. The first
``k_frames`` of each run are kept; surplus NO_OPs and their paired
preceding ``user(image)`` turns are dropped. The cap runs **before** the
v1 transform so the instruction is routed to whichever user turn ends up
first after dropping.

This mirrors the rule already used in ``stage_b_run_length_cap.py`` for
the 5fps event-stream pipeline. Source labels for the goal-conditioned
annotation_pilot corpus are ~53% ``NO_OP``; the cap brings that down to
single-digit percent, removing the constant-NO_OP attractor that
collapsed the v1-trained checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from absl import app, flags

# Importing v1 registers all of v1's flags on the shared absl FLAGS — we
# reuse every one of them unchanged and only define the new cap flags
# below. v1's helpers are reused verbatim to keep the transform/split/
# tokenize behavior bit-identical aside from the new pre-filter.
import stage_d_yll_annotation_pilot_v1 as v1
from _manifest import file_sha256_short, write_manifest


FLAGS = flags.FLAGS

flags.DEFINE_float(
    "k_seconds",
    None,
    "Agent response-time budget. Each contiguous NO_OP run in a record's "
    "assistant turns is capped at max(round(k_seconds * target_fps), 1) "
    "frames; surplus NO_OPs and their paired user(image) turns are dropped.",
    required=True,
)
flags.DEFINE_float(
    "target_fps",
    None,
    "Frame rate of the source trajectories (e.g. 2.0 for samples_*.jsonl "
    "captured at 2 fps). Combined with k_seconds to compute the cap.",
    required=True,
)


def _assistant_text(msg: dict[str, Any]) -> str:
    for c in msg["content"]:
        if c.get("type") == "text":
            return c["text"]
    return ""


def _cap_no_op_runs(
    rec: dict[str, Any],
    *,
    k_frames: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Cap NO_OP runs in one source record's assistant labels.

    Source records alternate user(image) and assistant(action) turns,
    optionally prefixed by a system turn. We walk the messages left to
    right, pairing each assistant turn with its immediately preceding
    user turn; an assistant ``NO_OP`` past the ``k_frames`` cap (and its
    paired user turn) is marked for removal. ``system`` turns are never
    touched here — the v1 transform downstream drops them.
    """
    messages = rec["messages"]
    pairs: list[tuple[int, int, str]] = []
    last_user_idx: int | None = None
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "user":
            last_user_idx = i
        elif role == "assistant":
            assert last_user_idx is not None, (
                f"assistant turn at index {i} has no preceding user turn "
                f"in record {rec.get('sample_id')!r}"
            )
            pairs.append((last_user_idx, i, _assistant_text(m)))
            last_user_idx = None

    drop_idx: set[int] = set()
    run_pos = 0
    n_no_op_pre = 0
    for u, a, text in pairs:
        if text == "NO_OP":
            n_no_op_pre += 1
            run_pos += 1
            if run_pos > k_frames:
                drop_idx.add(u)
                drop_idx.add(a)
        else:
            run_pos = 0

    n_dropped_asst = sum(1 for i in drop_idx if messages[i]["role"] == "assistant")
    n_dropped_user = sum(1 for i in drop_idx if messages[i]["role"] == "user")
    stats = {
        "n_no_op_pre": n_no_op_pre,
        "n_no_op_post": n_no_op_pre - n_dropped_asst,
        "n_dropped_assistant": n_dropped_asst,
        "n_dropped_user": n_dropped_user,
        "n_assistant_pre": len(pairs),
        "n_assistant_post": len(pairs) - n_dropped_asst,
    }
    if not drop_idx:
        return rec, stats

    out = dict(rec)
    out["messages"] = [m for i, m in enumerate(messages) if i not in drop_idx]
    return out, stats


def main(_) -> None:
    source_jsonl = Path(FLAGS.source_jsonl).resolve()
    image_base = Path(FLAGS.image_base).resolve()
    output_dir = Path(FLAGS.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k_frames = max(round(FLAGS.k_seconds * FLAGS.target_fps), 1)

    raw_records: list[dict[str, Any]] = []
    with source_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))

    capped: list[dict[str, Any]] = []
    total_no_op_pre = 0
    total_no_op_post = 0
    total_drop_asst = 0
    total_drop_user = 0
    total_asst_pre = 0
    total_asst_post = 0
    for r in raw_records:
        out, stats = _cap_no_op_runs(r, k_frames=k_frames)
        capped.append(out)
        total_no_op_pre += stats["n_no_op_pre"]
        total_no_op_post += stats["n_no_op_post"]
        total_drop_asst += stats["n_dropped_assistant"]
        total_drop_user += stats["n_dropped_user"]
        total_asst_pre += stats["n_assistant_pre"]
        total_asst_post += stats["n_assistant_post"]
    no_op_pct_pre = 100.0 * total_no_op_pre / max(total_asst_pre, 1)
    no_op_pct_post = 100.0 * total_no_op_post / max(total_asst_post, 1)
    print(
        f"[stage_d_yll v2] no_op cap k_frames={k_frames} "
        f"(k_seconds={FLAGS.k_seconds}, target_fps={FLAGS.target_fps}): "
        f"assistant turns {total_asst_pre}→{total_asst_post}, "
        f"no_op {total_no_op_pre} ({no_op_pct_pre:.1f}%) → "
        f"{total_no_op_post} ({no_op_pct_post:.1f}%), "
        f"dropped {total_drop_asst} assistant + {total_drop_user} user turns"
    )

    transformed = [
        v1._transform_record(r, image_base=image_base, terminate_token=FLAGS.terminate_token)
        for r in capped
    ]
    train_records, val_records, split_map = v1._split_records_by_segment(
        transformed, train_ratio=FLAGS.train_ratio, seed=FLAGS.split_seed
    )
    print(
        f"[stage_d_yll v2] {len(transformed)} records → "
        f"train={len(train_records)} val={len(val_records)} "
        f"(train_segs={len(split_map['train_segments'])}, "
        f"val_segs={len(split_map['val_segments'])})"
    )

    normalized_root = output_dir / "_normalized"
    payload_root = output_dir / "_payload"
    per_split: list[dict[str, Any]] = []
    for split, records in (("train", train_records), ("val", val_records)):
        if not records:
            print(f"[stage_d_yll v2] no records for split {split}, skipping")
            continue
        chat_path = normalized_root / split / "chat.jsonl"
        v1._write_jsonl(chat_path, records)
        payload_split = payload_root / split
        compile_s = v1._compile_split(chat_path, payload_split)
        chunk_split = output_dir / split
        chunk_s = v1._chunk_index_split(payload_split, chunk_split)
        n_shards = sum(1 for _ in chunk_split.glob("*.array_record"))
        per_split.append(
            {
                "split": split,
                "num_records": len(records),
                "n_shards": n_shards,
                "compile_elapsed_s": compile_s,
                "chunk_index_elapsed_s": chunk_s,
            }
        )

    write_manifest(
        output_dir,
        stage="chunk_index_yll_annotation_pilot_v2_no_op_capped",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "max_length": FLAGS.max_length,
            "records_per_shard": FLAGS.records_per_shard,
            "messages_per_record": FLAGS.messages_per_record,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
            "system_message_text": FLAGS.system_message_text,
            "terminate_token": FLAGS.terminate_token,
            "train_ratio": FLAGS.train_ratio,
            "split_seed": FLAGS.split_seed,
            "image_base": str(image_base),
            "k_seconds": FLAGS.k_seconds,
            "target_fps": FLAGS.target_fps,
            "k_frames": k_frames,
        },
        inputs={
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256_16": file_sha256_short(source_jsonl),
        },
        stats={
            "per_split": per_split,
            "split_map": split_map,
            "no_op_cap": {
                "n_assistant_pre": total_asst_pre,
                "n_assistant_post": total_asst_post,
                "n_no_op_pre": total_no_op_pre,
                "n_no_op_post": total_no_op_post,
                "no_op_pct_pre": round(no_op_pct_pre, 3),
                "no_op_pct_post": round(no_op_pct_post, 3),
                "n_dropped_assistant_turns": total_drop_asst,
                "n_dropped_user_turns": total_drop_user,
            },
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
