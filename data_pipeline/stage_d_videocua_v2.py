"""Stage D (VideoCUA golden SFT, v2): terminate-gated variant of v1.

v1 inherited crowd-cast v3's convention of rewriting EVERY record's last
assistant turn to ``terminate_token``. For the VideoCUA corpus the records
are episode SUB-RECORDS (assembly splits long episodes into goal-carrying
windows, fields ``subrecord_idx`` / ``n_subrecords``), so v1 planted a
TERMINATE at arbitrary mid-episode points in 70.1% of train records, and
55% of records supervised TERMINATE after <=1 real (non-NO_OP) action.
Trained checkpoints over-fire terminate (typing freeroll: TERMINATE as the
first action on 2/4 tasks; offline terminate precision 0.472).

v2 gates the rewrite:

* only the EPISODE-FINAL sub-record (``subrecord_idx == n_subrecords - 1``;
  absent fields => treated as final, matching single-record episodes) gets
  the TERMINATE rewrite;
* and only when the post-cap record retains at least
  ``--min_preterminate_actions`` non-NO_OP assistant turns BEFORE the final
  turn — otherwise the record keeps its original action label (goal
  completion without evidence of work is not supervised as terminate);
* non-final sub-records keep their original last action (the model learns
  to keep acting mid-episode).

Everything else (NO_OP cap, passthrough sysprompt, instruction-in-first-
user-turn assert, per-app stratified task split, chunk-index) is v1
verbatim — v1 helpers are imported, not copied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from absl import app, flags

import stage_d_yll_annotation_pilot_v1 as v1
import stage_d_yll_annotation_pilot_v2 as v2
import stage_d_videocua_v1 as videocua_v1
from _manifest import file_sha256_short, write_manifest


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "min_preterminate_actions",
    2,
    "Minimum non-NO_OP assistant turns required before the final turn for "
    "the episode-final sub-record to receive the terminate_token rewrite.",
)


def _assistant_text(msg: dict[str, Any]) -> str:
    for c in msg["content"]:
        if c.get("type") == "text":
            return c["text"]
    return ""


def _transform_record_terminate_gated(
    rec: dict[str, Any],
    *,
    terminate_token: str,
    min_preterminate_actions: int,
) -> tuple[dict[str, Any], str]:
    """v3-style passthrough transform with a gated terminate rewrite.

    Returns (record, terminate_status) where terminate_status is one of
    "rewritten", "skipped_nonfinal", "skipped_low_context".
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
                has_inst = any(
                    c.get("type") == "text" and c.get("text", "").strip() == instruction.strip()
                    for c in content
                )
                assert has_inst, (
                    f"record {rec.get('sample_id')!r}: first user turn does not contain "
                    f"the instruction text"
                )
                first_user_seen = True
            new_messages.append({"role": "user", "content": [dict(c) for c in content]})
        else:
            new_messages.append({"role": role, "content": [dict(c) for c in m["content"]]})
            if role == "assistant":
                last_assistant_idx = len(new_messages) - 1

    assert first_user_seen, f"record {rec.get('sample_id')!r} has no user turn"
    assert last_assistant_idx >= 0, f"record {rec.get('sample_id')!r} has no assistant turn"

    n_sub = rec.get("n_subrecords", 1)
    sub_idx = rec.get("subrecord_idx", n_sub - 1)
    is_final = sub_idx == n_sub - 1

    out = {k: val for k, val in rec.items() if k != "messages"}
    out["messages"] = new_messages

    if not is_final:
        return out, "skipped_nonfinal"

    n_real_pre = sum(
        1
        for i, m in enumerate(new_messages)
        if m["role"] == "assistant" and i != last_assistant_idx
        and _assistant_text(m) != "NO_OP"
    )
    if n_real_pre < min_preterminate_actions:
        return out, "skipped_low_context"

    new_messages[last_assistant_idx] = {
        "role": "assistant",
        "content": [{"type": "text", "text": terminate_token}],
    }
    return out, "rewritten"


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
        f"[stage_d_videocua v2] no_op cap k_frames={k_frames}: "
        f"assistant turns {total_asst_pre}→{total_asst_post}, "
        f"no_op {total_no_op_pre} ({no_op_pct_pre:.1f}%) → "
        f"{total_no_op_post} ({no_op_pct_post:.1f}%)"
    )

    transformed: list[dict[str, Any]] = []
    term_stats = {"rewritten": 0, "skipped_nonfinal": 0, "skipped_low_context": 0}
    for r in capped:
        out, status = _transform_record_terminate_gated(
            r,
            terminate_token=FLAGS.terminate_token,
            min_preterminate_actions=FLAGS.min_preterminate_actions,
        )
        term_stats[status] += 1
        transformed.append(out)
    n = max(len(transformed), 1)
    print(
        f"[stage_d_videocua v2] terminate gating: rewritten {term_stats['rewritten']} "
        f"({100*term_stats['rewritten']/n:.1f}%), skipped_nonfinal "
        f"{term_stats['skipped_nonfinal']} ({100*term_stats['skipped_nonfinal']/n:.1f}%), "
        f"skipped_low_context {term_stats['skipped_low_context']} "
        f"({100*term_stats['skipped_low_context']/n:.1f}%)"
    )

    train_records, val_records, split_map = videocua_v1._split_records_by_task_stratified(
        transformed, train_ratio=FLAGS.train_ratio, seed=FLAGS.split_seed
    )
    print(
        f"[stage_d_videocua v2] {len(transformed)} records → "
        f"train={len(train_records)} val={len(val_records)}"
    )

    normalized_root = output_dir / "_normalized"
    payload_root = output_dir / "_payload"
    per_split: list[dict[str, Any]] = []
    for split, records in (("train", train_records), ("val", val_records)):
        if not records:
            print(f"[stage_d_videocua v2] no records for split {split}, skipping")
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
        stage="chunk_index_videocua_v2_no_op_capped_terminate_gated",
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
            "min_preterminate_actions": FLAGS.min_preterminate_actions,
        },
        inputs={
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256_16": file_sha256_short(source_jsonl),
        },
        stats={
            "per_split": per_split,
            "split_map": split_map,
            "terminate_gating": term_stats,
            "no_op_cap": {
                "n_assistant_pre": total_asst_pre,
                "n_assistant_post": total_asst_post,
                "n_no_op_pre": total_no_op_pre,
                "n_no_op_post": total_no_op_post,
                "no_op_pct_pre": round(no_op_pct_pre, 3),
                "no_op_pct_post": round(no_op_pct_post, 3),
            },
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
