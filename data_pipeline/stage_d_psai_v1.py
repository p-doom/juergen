"""PSAI golden SFT ingest (anaisleila/computer-use-data-psai converted
to crowd-cast dense action format, goal-conditioned; NATIVE mouse deltas
from the DuckTrack move trace — no IDM; source jsonl built by
juergen/data_pipeline/prep_psai_build_chat.py + prep_psai_assemble_v1.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from absl import app, flags

import stage_d_videocua_v1 as videocua
from _manifest import file_sha256_short, write_manifest


FLAGS = flags.FLAGS


def main(_) -> None:
    source_jsonl = Path(FLAGS.source_jsonl).resolve()
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
    total_no_op_pre = total_no_op_post = 0
    total_drop_asst = total_drop_user = 0
    total_asst_pre = total_asst_post = 0
    for r in raw_records:
        out, stats = videocua.v2._cap_no_op_runs(r, k_frames=k_frames)
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
        f"[stage_d_psai v1] no_op cap k_frames={k_frames} "
        f"(k_seconds={FLAGS.k_seconds}, target_fps={FLAGS.target_fps}): "
        f"assistant turns {total_asst_pre}→{total_asst_post}, "
        f"no_op {total_no_op_pre} ({no_op_pct_pre:.1f}%) → "
        f"{total_no_op_post} ({no_op_pct_post:.1f}%), "
        f"dropped {total_drop_asst} assistant + {total_drop_user} user turns"
    )

    transformed = [
        videocua._transform_record(r, terminate_token=FLAGS.terminate_token)
        for r in capped
    ]
    train_records, val_records, split_map = videocua._split_records_by_task_stratified(
        transformed, train_ratio=FLAGS.train_ratio, seed=FLAGS.split_seed
    )
    print(
        f"[stage_d_psai v1] {len(transformed)} records → "
        f"train={len(train_records)} val={len(val_records)} "
        f"(train_recordings={len(split_map['train_recordings'])}, "
        f"val_recordings={len(split_map['val_recordings'])})"
    )

    normalized_root = output_dir / "_normalized"
    payload_root = output_dir / "_payload"
    per_split: list[dict[str, Any]] = []
    for split, records in (("train", train_records), ("val", val_records)):
        if not records:
            print(f"[stage_d_psai v1] no records for split {split}, skipping")
            continue
        chat_path = normalized_root / split / "chat.jsonl"
        videocua.v1._write_jsonl(chat_path, records)
        payload_split = payload_root / split
        compile_s = videocua.v1._compile_split(chat_path, payload_split)
        chunk_split = output_dir / split
        chunk_s = videocua.v1._chunk_index_split(payload_split, chunk_split)
        n_shards = sum(1 for _ in chunk_split.glob("*.array_record"))
        per_split.append(
            {
                "split": split,
                "num_records": len(records),
                "n_recordings": len(records),
                "n_distinct_recording_ids": len({r["recording_id"] for r in records}),
                "n_distinct_unique_data_ids": len(
                    {r["unique_data_id"] for r in records}
                ),
                "n_shards": n_shards,
                "compile_elapsed_s": compile_s,
                "chunk_index_elapsed_s": chunk_s,
            }
        )

    write_manifest(
        output_dir,
        stage="chunk_index_psai_v1_no_op_capped_k0p4",
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
