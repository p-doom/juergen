"""Measure stage: tokenize the grain payload once into a per-message length cache.

Wrapper around omegalax/scripts/measure_message_lengths.py. Same per-split loop
pattern as stage D; each call runs inside omegalax's uv venv and writes
<output_dir>/<split>/message_lengths.jsonl.

Per-message token lengths are the only tokenizer/processor-bound product of
chunk-index building and are independent of max_length / overflow_mode /
system_message. Running this once lets every stage D build over the same
payload reuse the cache (via --message_lengths_path) instead of re-tokenizing
per sequence length.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Message-length cache output dir.", required=True)
flags.DEFINE_string("payload_path", None, "Grain payload root (stage C).", required=True)
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
    "Forwarded to omegalax/scripts/measure_message_lengths.py.",
    required=True,
    lower_bound=2,
)


def _run_split(split: str, src_payload: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/measure_message_lengths.py",
        f"--data_path={src_payload}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--num_workers={FLAGS.num_workers}",
    ]
    print(f"[stage_measure] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"measure_message_lengths.py failed (rc={rc}) for {split}")
    cache = out_split_dir / MESSAGE_LENGTHS_FILENAME
    n_messages = sum(1 for _ in cache.open()) if cache.is_file() else 0
    return {"split": split, "n_messages": n_messages, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    payload_path = Path(FLAGS.payload_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src = payload_path / split
        if not src.is_dir():
            print(f"[stage_measure] no payload for split {split}, skipping")
            continue
        if not (src / "metadata.json").is_file():
            print(f"[stage_measure] incomplete/empty payload for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src, out_split_dir))

    write_manifest(
        output_dir,
        stage="message_lengths",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"payload": str(payload_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
