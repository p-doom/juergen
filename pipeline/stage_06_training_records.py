"""Build split SFT records from chat and its verified length cache."""

from __future__ import annotations

import json
import shutil
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
    validate_record_dataset,
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
    "processor_snapshot",
    None,
    "Immutable local Hugging Face snapshot used for tokenizer and processor.",
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
    split: str,
    src_chat: Path,
    out_split_dir: Path,
    cache_path: Path,
    processor_snapshot: dict,
) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = omegalax_python(
        Path(FLAGS.omegalax_repo),
        "scripts/build_sft_records_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--model_id={processor_snapshot['path']}",
        f"--processor={processor_snapshot['path']}",
        f"--max_length={FLAGS.max_length}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        f"--num_workers={FLAGS.num_workers}",
        "--overflow_mode=split",
        f"--val_fraction={FLAGS.val_fraction}",
        f"--split={split}",
        "--overwrite",
        f"--message_lengths_path={cache_path}",
    )
    print(f"[stage_records] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(
        cmd,
        cwd=Path(FLAGS.omegalax_repo).resolve(),
        env=isolated_subprocess_environment(),
        check=False,
    ).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(
            f"build_sft_records_from_chat.py failed (rc={rc}) for {split}"
        )
    validated = validate_record_dataset(
        out_split_dir,
        source_chat=src_chat,
        processor_snapshot=processor_snapshot,
        max_length=FLAGS.max_length,
        split=split,
        val_fraction=FLAGS.val_fraction,
    )
    return {"split": split, **validated, "elapsed_s": int(elapsed)}


def _resolve_cache(
    root: Path,
    source_id: str,
    source_chat: Path,
    processor_snapshot: dict,
    omegalax: dict,
) -> tuple[Path, str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "built_at",
            "inputs",
            "params",
            "pmanager_parent_run_id",
            "pmanager_run_id",
            "schema_version",
            "stage",
            "stats",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("stage") != "message_lengths"
        or isinstance(manifest.get("built_at"), bool)
        or not isinstance(manifest.get("built_at"), int)
        or manifest["built_at"] <= 0
        or not isinstance(manifest.get("pmanager_run_id"), str)
        or not isinstance(manifest.get("pmanager_parent_run_id"), str)
    ):
        raise ValueError(f"invalid message-length manifest: {manifest_path}")
    inputs = manifest.get("inputs")
    params = manifest.get("params")
    stats = manifest.get("stats")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"source", "source_id"}
        or inputs.get("source") != str(source_chat.parent.resolve())
        or inputs.get("source_id") != source_id
    ):
        raise ValueError("message-length cache source mismatch")
    if (
        not isinstance(params, dict)
        or set(params)
        != {"processor_snapshot", "num_workers", "omegalax_repo", "omegalax"}
        or params.get("processor_snapshot") != processor_snapshot
        or params.get("omegalax_repo") != omegalax["path"]
        or isinstance(params.get("num_workers"), bool)
        or not isinstance(params.get("num_workers"), int)
        or params["num_workers"] < 2
    ):
        raise ValueError("message-length cache processor snapshot mismatch")
    if params.get("omegalax") != omegalax:
        raise ValueError("message-length cache Omegalax identity mismatch")
    cache = (
        stats.get("cache")
        if isinstance(stats, dict) and set(stats) == {"cache"}
        else None
    )
    if (
        not isinstance(cache, dict)
        or set(cache) != {"elapsed_s", "file", "n_messages", "sha256"}
        or cache.get("file") != MESSAGE_LENGTHS_FILENAME
        or isinstance(cache.get("elapsed_s"), bool)
        or not isinstance(cache.get("elapsed_s"), int)
        or cache["elapsed_s"] < 0
    ):
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
    if (
        validate_message_lengths(
            path,
            source_chat,
            merge_size=processor_snapshot["merge_size"],
        )
        != cache["n_messages"]
    ):
        raise ValueError("message-length cache count mismatch")
    return path, make_artifact_id(root)


def main(_) -> None:
    if FLAGS.val_fraction >= 1.0:
        raise ValueError("val_fraction must be less than 1")
    output_dir = Path(FLAGS.output_dir).resolve()
    source_path = Path(FLAGS.source_path).resolve()
    lengths_root = Path(FLAGS.message_lengths_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").unlink(missing_ok=True)
    unexpected = [
        path for path in output_dir.iterdir() if path.name not in {"train", "val"}
    ]
    if unexpected:
        raise ValueError(f"unexpected Stage06 output entries: {unexpected}")
    for split in ("train", "val"):
        path = output_dir / split
        if path.exists():
            shutil.rmtree(path)

    src_chat = resolve_chat_artifact(source_path)
    source_id = make_artifact_id(source_path)
    processor_snapshot = attest_processor_snapshot(Path(FLAGS.processor_snapshot))
    omegalax = attest_omegalax(Path(FLAGS.omegalax_repo), processor_snapshot)
    cache_path, cache_id = _resolve_cache(
        lengths_root,
        source_id,
        src_chat,
        processor_snapshot,
        omegalax,
    )
    splits = ("train", "val") if FLAGS.val_fraction > 0.0 else ("train",)
    per_split = []
    for split in splits:
        result = _run_split(
            split,
            src_chat,
            output_dir / split,
            cache_path,
            processor_snapshot,
        )
        post_processor = attest_processor_snapshot(Path(FLAGS.processor_snapshot))
        post_omegalax = attest_omegalax(Path(FLAGS.omegalax_repo), post_processor)
        if post_processor != processor_snapshot or post_omegalax != omegalax:
            raise RuntimeError("Stage06 compiler identity changed during execution")
        if (
            resolve_chat_artifact(source_path) != src_chat
            or make_artifact_id(source_path) != source_id
        ):
            raise RuntimeError("Stage06 source changed during execution")
        observed_cache_path, observed_cache_id = _resolve_cache(
            lengths_root,
            source_id,
            src_chat,
            processor_snapshot,
            omegalax,
        )
        if observed_cache_path != cache_path or observed_cache_id != cache_id:
            raise RuntimeError("Stage06 cache changed during execution")
        per_split.append(result)

    write_manifest(
        output_dir,
        stage="inline_records",
        params={
            "processor_snapshot": processor_snapshot,
            "max_length": FLAGS.max_length,
            "records_per_shard": FLAGS.records_per_shard,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": omegalax["path"],
            "omegalax": omegalax,
            "message_lengths_path": str(lengths_root),
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
