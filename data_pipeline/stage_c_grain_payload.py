"""Stage C: compile per-split chat.jsonl into Grain ArrayRecord shards.

Wrapper around omegalax/scripts/compile_sft_dataset.py. omegalax's script
processes one split per invocation; this entrypoint loops over the three
splits in the source dataset and orchestrates the calls. Each subprocess
runs inside omegalax's own uv-managed venv via ``uv run --project``.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Grain payload output dir.", required=True)
flags.DEFINE_string("source_path", None, "Filtered dataset root (stage B).", required=True)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_integer(
    "messages_per_record", None, "Maximum contiguous messages per payload block.", required=True
)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)


def _run_split(split: str, src_chat: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/compile_sft_dataset.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--messages_per_record={FLAGS.messages_per_record}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        "--overwrite",
    ]
    print(f"[stage_c] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"compile_sft_dataset.py failed (rc={rc}) for {split}")
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
            print(f"[stage_c] no chat.jsonl for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src_chat, out_split_dir))

    write_manifest(
        output_dir,
        stage="grain_payload",
        params={
            "messages_per_record": FLAGS.messages_per_record,
            "records_per_shard": FLAGS.records_per_shard,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
