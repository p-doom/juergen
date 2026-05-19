"""Stage G: build the offline chunk index from compiled grain payload.

Wrapper around omegalax/scripts/build_sft_chunk_index.py. Same per-split
loop pattern as Stage F; each call runs inside omegalax's uv venv.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Chunk-index output dir.", required=True)
flags.DEFINE_string("payload_path", None, "Grain payload root (stage C).", required=True)
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
# omegalax's build_sft_chunk_index.py is single-threaded (build_chunk_index
# iterates messages serially); no num_workers flag is exposed there. We
# accept the flag here for recipe-side compatibility but never forward it.
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2). "
    "Forwarded to omegalax/scripts/build_sft_chunk_index.py.",
    required=True,
    lower_bound=2,
)
flags.DEFINE_string(
    "system_message_text",
    "",
    "If non-empty, prepend a text-only system message with this content to "
    "every emitted chunk. Forwarded verbatim to "
    "omegalax/scripts/build_sft_chunk_index.py --system_message_text.",
)


def _run_split(split: str, src_payload: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/build_sft_chunk_index.py",
        f"--data_path={src_payload}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--max_length={FLAGS.max_length}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        f"--num_workers={FLAGS.num_workers}",
        "--overwrite",
    ]
    if FLAGS.system_message_text:
        cmd.append(f"--system_message_text={FLAGS.system_message_text}")
    print(f"[stage_g] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"build_sft_chunk_index.py failed (rc={rc}) for {split}")
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    return {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    payload_path = Path(FLAGS.payload_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src = payload_path / split
        if not src.is_dir():
            print(f"[stage_g] no payload for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src, out_split_dir))

    write_manifest(
        output_dir,
        stage="chunk_index",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "max_length": FLAGS.max_length,
            "records_per_shard": FLAGS.records_per_shard,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
            "system_message_text": FLAGS.system_message_text,
        },
        inputs={"payload": str(payload_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
