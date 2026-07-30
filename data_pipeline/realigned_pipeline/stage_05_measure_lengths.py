"""Measure stage (payload-free): tokenize chat.jsonl into a per-message cache.

Wrapper around omegalax/scripts/measure_message_lengths_from_chat.py. Payload-free
variant of stage_measure_lengths.py: reads the stage-04 conversations dataset's
single <source>/chat.jsonl directly (NO grain payload) and measures it ONCE ->
<output_dir>/message_lengths.jsonl.

The train/val split is applied downstream at the records stage (stage 06), so
this cache is split-agnostic and is reused across every split / val_fraction --
changing the split never re-runs this stage.

Per-message token lengths are the only tokenizer/processor-bound product of
record building and are independent of max_length / overflow_mode /
system_message / split. Running this once lets every records build over the same
chat reuse the cache (via --message_lengths_path) instead of re-tokenizing per
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
    "source_path", None, "Conversations dataset root (stage 04, with a single chat.jsonl).",
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
flags.DEFINE_string(
    "chat_relpath",
    "chat.jsonl",
    "Chat JSONL path relative to source_path. Replay prep artifacts commonly "
    "use train/chat.jsonl; stage 04 uses chat.jsonl.",
)


def _run_measure(src_chat: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/measure_message_lengths_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--num_workers={FLAGS.num_workers}",
    ]
    print(f"[stage_measure_chat] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"measure_message_lengths_from_chat.py failed (rc={rc})")
    cache = out_dir / MESSAGE_LENGTHS_FILENAME
    n_messages = sum(1 for _ in cache.open()) if cache.is_file() else 0
    return {"n_messages": n_messages, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 04 writes a single split-agnostic <source>/chat.jsonl; measure it ONCE
    # -> <output_dir>/message_lengths.jsonl. The train/val split is applied at the
    # records stage, so this cache is reused across every split / val_fraction and
    # never needs re-running when the split changes.
    src_chat = source_path / FLAGS.chat_relpath
    if not src_chat.is_file():
        raise FileNotFoundError(
            f"no {FLAGS.chat_relpath} under {source_path}"
        )
    per_unit = [_run_measure(src_chat, output_dir)]

    write_manifest(
        output_dir,
        stage="message_lengths",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "num_workers": FLAGS.num_workers,
            "chat_relpath": FLAGS.chat_relpath,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_unit},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
