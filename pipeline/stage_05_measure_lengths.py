"""Measure the shared chat artifact into a tokenizer-bound length cache."""

from __future__ import annotations

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
from pipeline.lib.omegalax import (
    attest_omegalax,
    attest_processor_snapshot,
    isolated_subprocess_environment,
    omegalax_python,
    validate_message_lengths,
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
    "processor_snapshot",
    None,
    "Immutable local Hugging Face snapshot used for tokenizer and processor.",
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


def _run_measure(
    src_chat: Path,
    out_dir: Path,
    processor_snapshot: dict,
    omegalax: dict,
) -> dict:
    cache = out_dir / MESSAGE_LENGTHS_FILENAME
    cmd = omegalax_python(
        Path(omegalax["path"]),
        "scripts/measure_message_lengths_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_dir}",
        f"--model_id={processor_snapshot['path']}",
        f"--processor={processor_snapshot['path']}",
        f"--num_workers={FLAGS.num_workers}",
    )
    print(f"[stage_measure_chat] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(
        cmd,
        cwd=omegalax["path"],
        env=isolated_subprocess_environment(),
        check=False,
    ).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"measure_message_lengths_from_chat.py failed (rc={rc})")
    entries = {path.name: path for path in out_dir.iterdir()}
    if set(entries) != {MESSAGE_LENGTHS_FILENAME} or not cache.is_file():
        raise RuntimeError(f"measurement produced no cache: {cache}")
    n_messages = validate_message_lengths(
        cache, src_chat, merge_size=processor_snapshot["merge_size"]
    )
    return {
        "file": MESSAGE_LENGTHS_FILENAME,
        "sha256": file_sha256_short(cache, n=64),
        "n_messages": n_messages,
        "elapsed_s": int(elapsed),
    }


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir).resolve()
    source_path = Path(FLAGS.source_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").unlink(missing_ok=True)
    (output_dir / "manifest.json.tmp").unlink(missing_ok=True)
    allowed = {MESSAGE_LENGTHS_FILENAME}
    unexpected = [path for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise ValueError(f"unexpected Stage05 output entries: {unexpected}")
    (output_dir / MESSAGE_LENGTHS_FILENAME).unlink(missing_ok=True)

    src_chat = resolve_chat_artifact(source_path)
    source_id = make_artifact_id(source_path)
    processor_snapshot = attest_processor_snapshot(Path(FLAGS.processor_snapshot))
    omegalax = attest_omegalax(Path(FLAGS.omegalax_repo), processor_snapshot)
    cache = _run_measure(src_chat, output_dir, processor_snapshot, omegalax)
    post_processor = attest_processor_snapshot(Path(FLAGS.processor_snapshot))
    post_omegalax = attest_omegalax(Path(FLAGS.omegalax_repo), post_processor)
    if post_processor != processor_snapshot or post_omegalax != omegalax:
        raise RuntimeError("Stage05 compiler identity changed during execution")
    if (
        resolve_chat_artifact(source_path) != src_chat
        or make_artifact_id(source_path) != source_id
    ):
        raise RuntimeError("Stage05 source changed during execution")
    cache_path = output_dir / MESSAGE_LENGTHS_FILENAME
    if (
        validate_message_lengths(
            cache_path, src_chat, merge_size=processor_snapshot["merge_size"]
        )
        != cache["n_messages"]
    ):
        raise RuntimeError("Stage05 cache changed during validation")
    if file_sha256_short(cache_path, n=64) != cache["sha256"]:
        raise RuntimeError("Stage05 cache digest changed during validation")

    write_manifest(
        output_dir,
        stage="message_lengths",
        params={
            "processor_snapshot": processor_snapshot,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": omegalax["path"],
            "omegalax": omegalax,
        },
        inputs={"source": str(source_path), "source_id": source_id},
        stats={"cache": cache},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
