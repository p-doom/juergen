"""Training-record stage (payload-free): build inline SFT records from chat.jsonl.

Wrapper around omegalax/scripts/build_sft_records_from_chat.py. Payload-free
variant of stage_d_chunk_index.py: reads the stage-04 conversations dataset's
per-split <split>/chat.jsonl directly (NO grain payload / stage 05) and writes
self-contained inline records per split. Each record IS a training example
(message slice with ar:// image refs preserved), not a pointer into a shared
payload; the stage 01 master image store is unchanged.

Reuses the measure-stage cache (--message_lengths_path) so re-running at a
different max_length / overflow_mode never re-tokenizes.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# Filename of the per-message length cache written by the measure stage
# (mirrors omegalax grain_pipeline.MESSAGE_LENGTHS_FILENAME). The measure stage
# writes one per split under <message_lengths_path>/<split>/.
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Inline-records output dir.", required=True)
flags.DEFINE_string(
    "source_path", None, "Conversations dataset root (stage 04, with <split>/chat.jsonl).",
    required=True,
)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_string("model_id", None, "Model id (resolves the tokenizer).", required=True)
flags.DEFINE_string(
    "processor", None, "HF repo for image processor config (defaults to model_id).", required=True
)
flags.DEFINE_integer("max_length", None, "Max sequence length.", required=True)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2), used only when the "
    "measure cache is absent. Forwarded to build_sft_records_from_chat.py.",
    required=True,
    lower_bound=2,
)
flags.DEFINE_enum(
    "overflow_mode",
    "split",
    ["split", "truncate", "drop"],
    "Behaviour for conversations longer than max_length. 'split' (default): "
    "pack into multiple consecutive chunks at turn boundaries (no turns "
    "dropped). 'truncate': keep only the first fitting chunk and drop the "
    "overflowing turn plus the rest of the conversation. 'drop': discard the "
    "whole conversation if it does not fit in a single chunk. Forwarded to "
    "omegalax/scripts/build_sft_records_from_chat.py --overflow_mode; per-split "
    "truncation stats land in each split's truncation_stats.json.",
)
flags.DEFINE_string(
    "message_lengths_path",
    None,
    "Root of a measure-stage artifact holding per-split "
    f"<split>/{MESSAGE_LENGTHS_FILENAME} caches. When set, each split forwards "
    "its cache to build_sft_records_from_chat.py so the tokenizer pass is "
    "skipped (per-message lengths are independent of max_length / overflow_mode, "
    "so one cache serves every sequence length). Optional: omit to tokenize "
    "in-line.",
)


def _run_split(split: str, src_chat: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/build_sft_records_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--max_length={FLAGS.max_length}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        f"--num_workers={FLAGS.num_workers}",
        f"--overflow_mode={FLAGS.overflow_mode}",
        "--overwrite",
    ]
    if FLAGS.message_lengths_path:
        split_cache = Path(FLAGS.message_lengths_path) / split / MESSAGE_LENGTHS_FILENAME
        cmd.append(f"--message_lengths_path={split_cache}")
    print(f"[stage_records] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"build_sft_records_from_chat.py failed (rc={rc}) for {split}")
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    return {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src_chat = source_path / split / "chat.jsonl"
        if not src_chat.is_file():
            print(f"[stage_records] no chat.jsonl for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src_chat, out_split_dir))

    write_manifest(
        output_dir,
        stage="inline_records",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "max_length": FLAGS.max_length,
            "records_per_shard": FLAGS.records_per_shard,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
            "overflow_mode": FLAGS.overflow_mode,
            "message_lengths_path": FLAGS.message_lengths_path,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
