"""VideoCUA golden SFT ingest (ServiceNow/VideoCUA converted to crowd-cast
dense action format, goal-conditioned; IDM-densified MouseMove; source jsonl
built by slurm/dev/franz/berlin/crowd-cast-bc/videocua_golden_v1/
build_videocua_chat.py).
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

from absl import app, flags

# Reuse v1's chunk-tokenization helpers (run_omegalax_step, _compile_split,
# _chunk_index_split, _write_jsonl) and the NO_OP cap from v2 — only the
# per-record transform and segment-split logic are new in v3.
import stage_d_yll_annotation_pilot_v1 as v1
import stage_d_yll_annotation_pilot_v2 as v2
from _manifest import file_sha256_short, write_manifest


FLAGS = flags.FLAGS

# All flags v3 uses are already registered by the v1+v2 module imports
# above (output_dir, source_jsonl, omegalax_repo, train_ratio, split_seed,
# model_id, processor, max_length, records_per_shard, messages_per_record,
# num_workers, system_message_text, terminate_token, k_seconds, target_fps).
# v3 does not use ``image_base`` (paths are absolute) but the flag is
# still required by absl — pass a placeholder in the recipe.


def _transform_record(
    rec: dict[str, Any],
    *,
    terminate_token: str,
) -> dict[str, Any]:
    """Preserve source system turn (passthrough mode), preserve user turns
    as-is (already absolute paths, already have the instruction text),
    rewrite last assistant turn to ``terminate_token``.
    """
    instruction = rec.get("instruction")
    assert isinstance(instruction, str) and instruction, (
        f"record {rec.get('sample_id')!r} has no instruction string"
    )

    new_messages: list[dict[str, Any]] = []
    first_user_seen = False
    last_assistant_idx = -1
    for m in rec["messages"]:
        role = m["role"]
        if role == "system":
            new_messages.append({"role": "system", "content": [dict(c) for c in m["content"]]})
            continue
        if role == "user":
            content = m["content"]
            assert isinstance(content, list), (
                f"record {rec.get('sample_id')!r} user content is not a list"
            )
            if not first_user_seen:
                # Defensive: confirm the corpus's "instruction lives in
                # first user turn" invariant holds for this record so we
                # don't silently lose the goal.
                has_inst = any(
                    c.get("type") == "text" and c.get("text", "").strip() == instruction.strip()
                    for c in content
                )
                assert has_inst, (
                    f"record {rec.get('sample_id')!r}: first user turn does not contain "
                    f"the instruction text — v3 expected the corpus's "
                    f"goal-in-first-user-turn shape"
                )
                first_user_seen = True
            new_messages.append({"role": "user", "content": [dict(c) for c in content]})
        else:
            new_messages.append({"role": role, "content": [dict(c) for c in m["content"]]})
            if role == "assistant":
                last_assistant_idx = len(new_messages) - 1

    assert first_user_seen, f"record {rec.get('sample_id')!r} has no user turn"
    assert last_assistant_idx >= 0, f"record {rec.get('sample_id')!r} has no assistant turn"

    new_messages[last_assistant_idx] = {
        "role": "assistant",
        "content": [{"type": "text", "text": terminate_token}],
    }

    out = {k: val for k, val in rec.items() if k != "messages"}
    out["messages"] = new_messages
    return out


def _split_records_by_task_stratified(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assign whole tasks to train or val independently within each app."""
    recordings_by_app: dict[str, set[str]] = collections.defaultdict(set)
    rec_id_of: list[str] = []
    for rec in records:
        rid = rec.get("recording_id")
        assert isinstance(rid, str) and rid, (
            f"record {rec.get('sample_id')!r} has no recording_id"
        )
        app_name = rec.get("app")
        assert isinstance(app_name, str) and app_name, (
            f"record {rec.get('sample_id')!r} has no app"
        )
        rec_id_of.append(rid)
        recordings_by_app[app_name].add(rid)

    train_recs: set[str] = set()
    val_recs: set[str] = set()
    per_app: dict[str, dict[str, int]] = {}
    for app_name in sorted(recordings_by_app):
        recordings = sorted(recordings_by_app[app_name])
        rng = random.Random(f"{seed}:{app_name}")
        rng.shuffle(recordings)
        n_train = max(1, round(len(recordings) * train_ratio))
        app_train_recs = recordings[:n_train]
        app_val_recs = recordings[n_train:]
        train_recs.update(app_train_recs)
        val_recs.update(app_val_recs)
        per_app[app_name] = {
            "train": len(app_train_recs),
            "val": len(app_val_recs),
        }

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if rec_id_of[i] in train_recs:
            train_records.append(rec)
        else:
            val_records.append(rec)

    split_map = {
        "train_recordings": sorted(train_recs),
        "val_recordings": sorted(val_recs),
        "per_app": per_app,
    }
    return train_records, val_records, split_map


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
        out, stats = v2._cap_no_op_runs(r, k_frames=k_frames)
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
        f"[stage_d_videocua v1] no_op cap k_frames={k_frames} "
        f"(k_seconds={FLAGS.k_seconds}, target_fps={FLAGS.target_fps}): "
        f"assistant turns {total_asst_pre}→{total_asst_post}, "
        f"no_op {total_no_op_pre} ({no_op_pct_pre:.1f}%) → "
        f"{total_no_op_post} ({no_op_pct_post:.1f}%), "
        f"dropped {total_drop_asst} assistant + {total_drop_user} user turns"
    )

    transformed = [
        _transform_record(r, terminate_token=FLAGS.terminate_token) for r in capped
    ]
    train_records, val_records, split_map = _split_records_by_task_stratified(
        transformed, train_ratio=FLAGS.train_ratio, seed=FLAGS.split_seed
    )
    print(
        f"[stage_d_videocua v1] {len(transformed)} records → "
        f"train={len(train_records)} val={len(val_records)} "
        f"(train_recordings={len(split_map['train_recordings'])}, "
        f"val_recordings={len(split_map['val_recordings'])})"
    )

    normalized_root = output_dir / "_normalized"
    payload_root = output_dir / "_payload"
    per_split: list[dict[str, Any]] = []
    for split, records in (("train", train_records), ("val", val_records)):
        if not records:
            print(f"[stage_d_videocua v1] no records for split {split}, skipping")
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
        stage="chunk_index_videocua_v1_no_op_capped",
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
