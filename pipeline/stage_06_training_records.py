"""Build split SFT records from chat and its verified length cache."""

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
flags.DEFINE_string("output_dir", None, "Inline-records output dir.", required=True)
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
    "max_length", None, "Max sequence length.", required=True, lower_bound=1
)
flags.DEFINE_integer(
    "records_per_shard",
    None,
    "Records per output shard.",
    required=True,
    lower_bound=1,
)
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel record-building workers (>=2).",
    required=True,
    lower_bound=2,
)
flags.DEFINE_string(
    "message_lengths_path",
    None,
    "Verified stage-05 artifact root.",
    required=True,
)
flags.DEFINE_float(
    "val_fraction",
    0.0,
    "Recording-level val fraction, applied here (records stage) over the single "
    "<source>/chat.jsonl: > 0 writes <out>/train/ and <out>/val/ (split by "
    "recording_id), 0 writes <out>/train/ only. Because the split is applied here, "
    "the measure cache stays split-agnostic and is reused when you change this value.",
    lower_bound=0.0,
    upper_bound=1.0,
)


def _run_split(
    split: str, src_chat: Path, out_split_dir: Path, cache_path: Path
) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    for shard in out_split_dir.glob("*.array_record"):
        shard.unlink()
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
        "--overflow_mode=split",
        f"--val_fraction={FLAGS.val_fraction}",
        f"--split={split}",
        "--overwrite",
        f"--message_lengths_path={cache_path}",
    ]
    print(f"[stage_records] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(
            f"build_sft_records_from_chat.py failed (rc={rc}) for {split}"
        )
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    if n_shards == 0:
        raise RuntimeError(f"record builder produced no shards for {split}")
    return {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}


def _resolve_cache(root: Path, source_id: str) -> tuple[Path, str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "message_lengths"
    ):
        raise ValueError(f"invalid message-length manifest: {manifest_path}")
    inputs = manifest.get("inputs")
    params = manifest.get("params")
    stats = manifest.get("stats")
    if not isinstance(inputs, dict) or inputs.get("source_id") != source_id:
        raise ValueError("message-length cache source mismatch")
    if not isinstance(params, dict) or params.get("model_id") != FLAGS.model_id:
        raise ValueError("message-length cache model_id mismatch")
    if params.get("processor") != FLAGS.processor:
        raise ValueError("message-length cache processor mismatch")
    cache = stats.get("cache") if isinstance(stats, dict) else None
    if not isinstance(cache, dict) or cache.get("file") != MESSAGE_LENGTHS_FILENAME:
        raise ValueError(f"invalid message-length cache contract: {manifest_path}")
    expected = cache.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid message-length cache digest: {manifest_path}")
    if not isinstance(cache.get("n_messages"), int) or cache["n_messages"] <= 0:
        raise ValueError(f"invalid message-length cache count: {manifest_path}")
    path = root / MESSAGE_LENGTHS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"message-length cache is missing: {path}")
    observed = file_sha256_short(path, n=64)
    if observed != expected:
        raise ValueError(
            f"message-length cache digest mismatch: expected {expected}, got {observed}"
        )
    return path, make_artifact_id(root)


def main(_) -> None:
    if FLAGS.val_fraction >= 1.0:
        raise ValueError("val_fraction must be less than 1")
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    lengths_root = Path(FLAGS.message_lengths_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_chat = resolve_chat_artifact(source_path)
    source_id = make_artifact_id(source_path)
    cache_path, cache_id = _resolve_cache(lengths_root, source_id)
    splits = ("train", "val") if FLAGS.val_fraction > 0.0 else ("train",)
    per_split = [_run_split(s, src_chat, output_dir / s, cache_path) for s in splits]

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
            "message_lengths_path": FLAGS.message_lengths_path,
            "val_fraction": FLAGS.val_fraction,
        },
        inputs={
            "source": str(source_path),
            "source_id": source_id,
            "message_lengths_id": cache_id,
        },
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
