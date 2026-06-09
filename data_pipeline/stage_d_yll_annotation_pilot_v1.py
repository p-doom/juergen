"""Stage D (yll annotation pilot v1): wrap colleague-built chat jsonl into
trainer-readable chunk-index shards.

Dataset-specific to ``/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/
annotation_pilot/bucket_samples/samples_*.jsonl``. Each input record is one
already-bounded chunk with ``system / user / assistant`` turns and a separate
``instruction`` field. We transform per record:

* drop the input ``system`` turn (a fresh sysprompt is injected by
  build_chunk_index via ``--system_message_text``).
* move ``instruction`` text into the first user turn alongside its image,
  matching the Qwen3-VL cookbook's "goal in user turn" convention.
* replace the final assistant turn's text with ``TERMINATE`` (goals are
  assumed achieved at end-of-sample; no terminal token in source data).
* resolve relative image paths to absolute under ``--image_base``.

After per-record transform we split records by segment id (lowest seg seen in
the record's images) so train/val don't share segments, write normalized
chat.jsonl per split, then run the existing omegalax
compile_sft_dataset / build_sft_chunk_index scripts to produce the artifact.
"""

from __future__ import annotations

import collections
import json
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from absl import app, flags

from _manifest import file_sha256_short, write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Chunk-index output dir.", required=True)
flags.DEFINE_string(
    "source_jsonl", None, "Colleague-built samples_*.jsonl file.", required=True
)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_string(
    "image_base",
    None,
    "Directory the jsonl's relative image paths are relative to "
    "(typically the colleague's data_pipeline root).",
    required=True,
)
flags.DEFINE_float("train_ratio", None, "Fraction of segments assigned to train.", required=True)
flags.DEFINE_integer("split_seed", None, "Seed for the segment-bucket shuffle.", required=True)
flags.DEFINE_string("model_id", None, "Model id (resolves the tokenizer).", required=True)
flags.DEFINE_string(
    "processor", None, "HF repo for image processor config (defaults to model_id).", required=True
)
flags.DEFINE_integer("max_length", None, "Max sequence length.", required=True)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2).",
    required=True,
    lower_bound=2,
)
flags.DEFINE_integer(
    "messages_per_record",
    None,
    "Maximum contiguous messages per payload block "
    "(forwarded to compile_sft_dataset).",
    required=True,
)
flags.DEFINE_string(
    "system_message_text",
    None,
    "System message prepended to every emitted chunk. Forwarded to "
    "build_sft_chunk_index.py --system_message_text.",
    required=True,
)
flags.DEFINE_string(
    "terminate_token",
    "TERMINATE",
    "Token written as the final assistant turn of every sample.",
)


_SEG_RE = re.compile(r"recording_[0-9a-fA-F-]+_seg(\d+)")


def _segments_in_record(rec: dict[str, Any]) -> set[int]:
    segs: set[int] = set()
    for m in rec.get("messages", []):
        for c in m.get("content", []):
            if c.get("type") != "image":
                continue
            mm = _SEG_RE.search(c.get("image", ""))
            if mm:
                segs.add(int(mm.group(1)))
    return segs


def _transform_record(
    rec: dict[str, Any],
    *,
    image_base: Path,
    terminate_token: str,
) -> dict[str, Any]:
    instruction = rec.get("instruction")
    assert isinstance(instruction, str) and instruction, (
        f"record {rec.get('sample_id')!r} has no instruction string"
    )

    new_messages: list[dict[str, Any]] = []
    first_user_emitted = False
    last_assistant_idx = -1
    src_messages = list(rec["messages"])
    for m in src_messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            new_content: list[dict[str, Any]] = []
            if not first_user_emitted:
                new_content.append({"type": "text", "text": instruction})
                first_user_emitted = True
            for c in m["content"]:
                if c.get("type") == "image":
                    img = c["image"]
                    if not img.startswith("/"):
                        img = str((image_base / img).resolve())
                    new_content.append({"type": "image", "image": img})
                else:
                    new_content.append(dict(c))
            new_messages.append({"role": "user", "content": new_content})
        else:
            new_messages.append({"role": m["role"], "content": [dict(c) for c in m["content"]]})
            if m["role"] == "assistant":
                last_assistant_idx = len(new_messages) - 1

    assert first_user_emitted, f"record {rec.get('sample_id')!r} has no user turn"
    assert last_assistant_idx >= 0, f"record {rec.get('sample_id')!r} has no assistant turn"

    new_messages[last_assistant_idx] = {
        "role": "assistant",
        "content": [{"type": "text", "text": terminate_token}],
    }

    out = {k: v for k, v in rec.items() if k != "messages"}
    out["messages"] = new_messages
    return out


def _split_records_by_segment(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    by_seg: dict[int, list[int]] = collections.defaultdict(list)
    primary_seg_of: list[int] = []
    for i, rec in enumerate(records):
        segs = _segments_in_record(rec)
        assert segs, f"record {rec.get('sample_id')!r} has no recoverable segment id"
        primary = min(segs)
        primary_seg_of.append(primary)
        by_seg[primary].append(i)

    segments = sorted(by_seg)
    rng = random.Random(seed)
    rng.shuffle(segments)
    n_train_segs = max(1, round(len(segments) * train_ratio))
    train_segs = set(segments[:n_train_segs])
    val_segs = set(segments[n_train_segs:])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if primary_seg_of[i] in train_segs:
            train_records.append(rec)
        else:
            val_records.append(rec)

    split_map = {
        "train_segments": sorted(train_segs),
        "val_segments": sorted(val_segs),
    }
    return train_records, val_records, split_map


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _run_omegalax_step(stage_tag: str, cmd: list[str]) -> int:
    print(f"[{stage_tag}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = int(time.time() - t0)
    if rc != 0:
        raise RuntimeError(f"{stage_tag} failed (rc={rc})")
    return elapsed


def _compile_split(src_jsonl: Path, payload_out: Path) -> int:
    return _run_omegalax_step(
        f"compile {src_jsonl.parent.name}",
        [
            "uv", "run", "--project", FLAGS.omegalax_repo,
            "python", "scripts/compile_sft_dataset.py",
            f"--data_path={src_jsonl}",
            f"--out_dir={payload_out}",
            f"--messages_per_record={FLAGS.messages_per_record}",
            f"--records_per_shard={FLAGS.records_per_shard}",
            "--overwrite",
        ],
    )


def _chunk_index_split(payload_dir: Path, out_dir: Path) -> int:
    return _run_omegalax_step(
        f"chunk_index {out_dir.name}",
        [
            "uv", "run", "--project", FLAGS.omegalax_repo,
            "python", "scripts/build_sft_chunk_index.py",
            f"--data_path={payload_dir}",
            f"--out_dir={out_dir}",
            f"--model_id={FLAGS.model_id}",
            f"--processor={FLAGS.processor}",
            f"--max_length={FLAGS.max_length}",
            f"--records_per_shard={FLAGS.records_per_shard}",
            f"--num_workers={FLAGS.num_workers}",
            f"--system_message_text={FLAGS.system_message_text}",
            "--overwrite",
        ],
    )


def main(_) -> None:
    source_jsonl = Path(FLAGS.source_jsonl).resolve()
    image_base = Path(FLAGS.image_base).resolve()
    output_dir = Path(FLAGS.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, Any]] = []
    with source_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))

    transformed = [
        _transform_record(r, image_base=image_base, terminate_token=FLAGS.terminate_token)
        for r in raw_records
    ]
    train_records, val_records, split_map = _split_records_by_segment(
        transformed, train_ratio=FLAGS.train_ratio, seed=FLAGS.split_seed
    )
    print(
        f"[stage_d_yll] {len(transformed)} records → "
        f"train={len(train_records)} val={len(val_records)} "
        f"(train_segs={len(split_map['train_segments'])}, "
        f"val_segs={len(split_map['val_segments'])})"
    )

    normalized_root = output_dir / "_normalized"
    payload_root = output_dir / "_payload"
    per_split: list[dict[str, Any]] = []
    for split, records in (("train", train_records), ("val", val_records)):
        if not records:
            print(f"[stage_d_yll] no records for split {split}, skipping")
            continue
        chat_path = normalized_root / split / "chat.jsonl"
        _write_jsonl(chat_path, records)
        payload_split = payload_root / split
        compile_s = _compile_split(chat_path, payload_split)
        chunk_split = output_dir / split
        chunk_s = _chunk_index_split(payload_split, chunk_split)
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
        stage="chunk_index_yll_annotation_pilot_v1",
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
        },
        inputs={
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256_16": file_sha256_short(source_jsonl),
        },
        stats={
            "per_split": per_split,
            "split_map": split_map,
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
