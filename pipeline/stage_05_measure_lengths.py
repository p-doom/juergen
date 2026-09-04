"""Measure the shared chat artifact into a tokenizer-bound length cache."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from absl import app, flags

# Make the ``pipeline`` package importable when this stage is run
# directly as a script (mirrors the other stages' PYTHONPATH setup).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib.manifest import (
    file_sha256_short,
    make_artifact_id,
    resolve_chat_artifact,
    write_manifest,
)

FLAGS = flags.FLAGS

MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string(
    "output_dir", None, "Message-length cache output dir.", required=True
)
flags.DEFINE_string(
    "source_path",
    None,
    "Conversations dataset root (stage 04, with a single chat.jsonl).",
    required=True,
)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo",
    None,
    "Path to omegalax repo root (used as uv --project).",
    required=True,
)
flags.DEFINE_string(
    "model_id", None, "Model id (resolves the tokenizer).", required=True
)
flags.DEFINE_string(
    "processor",
    None,
    "Exact HF image processor identity.",
    required=True,
)
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2). "
    "Forwarded to omegalax/scripts/measure_message_lengths_from_chat.py.",
    required=True,
    lower_bound=2,
)


def _run_measure(src_chat: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / MESSAGE_LENGTHS_FILENAME
    cache.unlink(missing_ok=True)
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
    if not cache.is_file():
        raise RuntimeError(f"measurement produced no cache: {cache}")
    n_messages = 0
    with cache.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank cache row at {cache}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"cache row must be an object at {cache}:{line_number}")
            n_messages += 1
    if n_messages == 0:
        raise RuntimeError(f"measurement produced an empty cache: {cache}")
    return {
        "file": MESSAGE_LENGTHS_FILENAME,
        "sha256": file_sha256_short(cache, n=64),
        "n_messages": n_messages,
        "elapsed_s": int(elapsed),
    }


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_chat = resolve_chat_artifact(source_path)
    source_id = make_artifact_id(source_path)
    cache = _run_measure(src_chat, output_dir)

    write_manifest(
        output_dir,
        stage="message_lengths",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"source": str(source_path), "source_id": source_id},
        stats={"cache": cache},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
