"""Measure stage (payload-free): tokenize chat.jsonl into a per-message cache.

Wrapper around omegalax/scripts/measure_message_lengths_from_chat.py. Payload-free
variant of stage_measure_lengths.py: reads the stage-04 conversations dataset's
per-split <split>/chat.jsonl directly (NO grain payload / stage 05) and writes
<output_dir>/<split>/message_lengths.jsonl.

Per-message token lengths are the only tokenizer/processor-bound product of
record building and are independent of max_length / overflow_mode /
system_message. Running this once lets every records build over the same chat
reuse the cache (via --message_lengths_path) instead of re-tokenizing per
sequence length.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from absl import app, flags

# Make the ``realigned_pipeline`` package importable when this stage is run
# directly as a script (mirrors the other stages' PYTHONPATH setup).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.manifest import write_manifest  # noqa: E402

FLAGS = flags.FLAGS

MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Message-length cache output dir.", required=True)
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
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2). "
    "Forwarded to omegalax/scripts/measure_message_lengths_from_chat.py.",
    required=True,
    lower_bound=2,
)


def _run_split(split: str, src_chat: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/measure_message_lengths_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--num_workers={FLAGS.num_workers}",
    ]
    print(f"[stage_measure_chat] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"measure_message_lengths_from_chat.py failed (rc={rc}) for {split}")
    cache = out_split_dir / MESSAGE_LENGTHS_FILENAME
    n_messages = sum(1 for _ in cache.open()) if cache.is_file() else 0
    return {"split": split, "n_messages": n_messages, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src_chat = source_path / split / "chat.jsonl"
        if not src_chat.is_file():
            print(f"[stage_measure_chat] no chat.jsonl for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src_chat, out_split_dir))

    write_manifest(
        output_dir,
        stage="message_lengths",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
