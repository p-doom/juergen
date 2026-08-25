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

SHARDING (SLURM job array, mirrors stage_01_master_frames):
Measurement is the slowest stage in the chain -- every message with an ar://
image ref pays a full image-processor preprocess just to size its vision tokens
-- and a single node caps out at its own core count. ``--num_shards N`` /
``--shard_index I`` split the work across N jobs:

  shard task  slices chat.jsonl into the disjoint round-robin subset
              {conv_idx : conv_idx % N == I}, measures ONLY that slice, and
              writes message_lengths.shard<I>_of_<N>.jsonl into the SHARED
              --output_dir. It deliberately does NOT write manifest.json.
  --merge     folds every shard file into the canonical message_lengths.jsonl
              and writes the manifest.json marker.

The cache is keyed by ``(conv_idx, msg_offset)`` where conv_idx is the
0-based index of the conversation in the FULL chat.jsonl, so each shard remaps
its slice-local conv_idx ``j`` back to the global ``I + j*N`` before writing --
downstream (stage 06) validates the cache against the full chat and rejects any
gap. The merge re-checks that invariant itself (exact key-for-key equality with
the source chat) so a missing/stale shard fails here, loudly, instead of at
record-build time.
"""

from __future__ import annotations

import heapq
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from absl import app, flags

# Make the ``pipeline`` package importable when this stage is run
# directly as a script (mirrors the other stages' PYTHONPATH setup).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.crowdcast.lib.manifest import (  # noqa: E402
    make_artifact_id,
    write_manifest,
)

FLAGS = flags.FLAGS

MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Message-length cache output dir.", required=True)
flags.DEFINE_string(
    "source_path", None, "Conversations dataset root (stage 04, with a single chat.jsonl).",
    required=True,
)
# Stage-specific (required for measuring; unused in --merge mode):
flags.DEFINE_string("omegalax_repo", None, "Path to omegalax repo root (used as uv --project).")
flags.DEFINE_string("model_id", None, "Model id (resolves the tokenizer).")
flags.DEFINE_string("processor", None, "HF repo for image processor config (defaults to model_id).")
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2). "
    "Forwarded to omegalax/scripts/measure_message_lengths_from_chat.py.",
    lower_bound=2,
)
# Job-array sharding:
flags.DEFINE_integer(
    "num_shards",
    1,
    "Split chat.jsonl into N disjoint round-robin slices for parallel jobs; this "
    "shard measures the conversations whose conv_idx modulo N equals shard_index. "
    "All shards MUST share ONE --output_dir. With N>1 each shard writes "
    "message_lengths.shard<I>_of_<N>.jsonl (NOT the top-level manifest.json); run "
    "--merge afterwards to fold them into the canonical message_lengths.jsonl.",
    lower_bound=1,
)
flags.DEFINE_integer(
    "shard_index",
    0,
    "This job's shard, in [0, num_shards). Injected from $SLURM_ARRAY_TASK_ID by "
    "the labctl [sweep].",
    lower_bound=0,
)
flags.DEFINE_bool(
    "merge",
    False,
    "Merge mode: fold every message_lengths.shard*_of_<num_shards>.jsonl under "
    "--output_dir into the canonical message_lengths.jsonl and write the "
    "manifest.json marker. Measures nothing; only --output_dir, --source_path and "
    "--num_shards are read.",
)
flags.DEFINE_string(
    "work_dir",
    None,
    "Scratch dir for a shard's chat slice + raw (slice-local) cache; deleted on "
    "success. Defaults to <output_dir>/_shard_work. Point at node-local scratch "
    "(e.g. $TMPDIR) to keep the intermediate off NFS.",
)
flags.DEFINE_bool(
    "force", False, "Re-measure a shard that already has its message_lengths.shard*.jsonl."
)


def _shard_tag(shard_index: int, num_shards: int) -> str:
    return f"shard{shard_index:04d}_of_{num_shards:04d}"


def _iter_chat_lines(chat_path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(conv_idx, raw_line)`` for every conversation in a chat.jsonl.

    conv_idx numbering MUST match omegalax's ``_iter_chat_conversations`` (0-based
    over NON-BLANK lines, file order) -- it is the key half of the cache that
    stage 06 validates against, so any divergence silently mis-keys the cache.
    """
    with chat_path.open() as f:
        conv_idx = 0
        for line in f:
            if not line.strip():
                continue
            yield conv_idx, line
            conv_idx += 1


def _write_chat_slice(src_chat: Path, dst_chat: Path, shard_index: int, num_shards: int) -> int:
    """Write this shard's round-robin slice of ``src_chat`` and return its size.

    Stride slicing (``conv_idx % N == I``) is disjoint AND exhaustive across
    shards, and load-balances better than contiguous blocks when long
    conversations cluster together. Order is preserved, so slice-local conv_idx
    ``j`` is exactly global ``shard_index + j*num_shards`` (see
    :func:`_remap_shard_cache`).
    """
    n = 0
    with dst_chat.open("w") as out:
        for conv_idx, line in _iter_chat_lines(src_chat):
            if conv_idx % num_shards != shard_index:
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            n += 1
    return n


def _remap_shard_cache(src: Path, dst: Path, shard_index: int, num_shards: int) -> int:
    """Rewrite a slice-local cache with GLOBAL conv_idx, returning the row count.

    ``j -> shard_index + j*num_shards`` is strictly monotone, so the input's
    ``(conv_idx, msg_offset)`` sort order carries over unchanged -- the merge
    relies on each shard file being sorted. Written via .tmp + rename so an
    interrupted shard never leaves a short file that resume would accept.
    """
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    n = 0
    with src.open() as fin, tmp.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row["conv_idx"] = shard_index + int(row["conv_idx"]) * num_shards
            fout.write(json.dumps(row) + "\n")
            n += 1
    tmp.replace(dst)
    return n


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


def _measure_params() -> dict[str, Any]:
    return {
        "model_id": FLAGS.model_id,
        "processor": FLAGS.processor,
        "num_workers": FLAGS.num_workers,
        "omegalax_repo": FLAGS.omegalax_repo,
    }


def _require_measure_flags() -> None:
    missing = [
        name
        for name in ("omegalax_repo", "model_id", "processor", "num_workers")
        if getattr(FLAGS, name) is None
    ]
    if missing:
        raise SystemExit(
            f"missing required flag(s) for measuring: {', '.join('--' + m for m in missing)} "
            f"(only --merge may omit them)"
        )


def _resolve_src_chat(source_path: Path) -> Path:
    src_chat = source_path / "chat.jsonl"
    if not src_chat.is_file():
        raise FileNotFoundError(
            f"no chat.jsonl under {source_path} (stage 04 writes a single "
            f"<source>/chat.jsonl)"
        )
    return src_chat


def run_shard(src_chat: Path, out_dir: Path, shard_index: int, num_shards: int) -> None:
    """Measure this shard's slice into ``message_lengths.shard<I>_of_<N>.jsonl``.

    Writes a per-shard summary alongside it and deliberately SKIPS manifest.json:
    the merge step writes the marker once, after every shard has succeeded.
    """
    tag = _shard_tag(shard_index, num_shards)
    shard_cache = out_dir / f"message_lengths.{tag}.jsonl"

    if shard_cache.is_file() and not FLAGS.force:
        n_messages = sum(1 for _ in shard_cache.open())
        print(
            f"[stage_measure_chat] {tag} cached: {n_messages} messages already at "
            f"{shard_cache} (--force to re-measure)",
            flush=True,
        )
        return

    work_root = Path(FLAGS.work_dir) if FLAGS.work_dir else out_dir / "_shard_work"
    work_dir = work_root / tag
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    slice_chat = work_dir / "chat.jsonl"
    n_convs = _write_chat_slice(src_chat, slice_chat, shard_index, num_shards)
    print(
        f"[stage_measure_chat] {tag}: {n_convs} conversations "
        f"(conv_idx % {num_shards} == {shard_index}) -> {slice_chat}",
        flush=True,
    )

    stats = _run_measure(slice_chat, work_dir)
    n_messages = _remap_shard_cache(
        work_dir / MESSAGE_LENGTHS_FILENAME, shard_cache, shard_index, num_shards
    )
    if n_messages != stats["n_messages"]:
        raise RuntimeError(
            f"{tag}: remapped {n_messages} rows but the measure wrote "
            f"{stats['n_messages']} -- refusing to publish a truncated shard"
        )

    summary = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_conversations": n_convs,
        "n_messages": n_messages,
        "elapsed_s": stats["elapsed_s"],
        "source_chat": str(src_chat),
        **_measure_params(),
    }
    (out_dir / f"measure_summary.{tag}.json").write_text(json.dumps(summary, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)
    print(
        f"[stage_measure_chat] {tag} done: {n_messages} messages in "
        f"{stats['elapsed_s']}s -> {shard_cache} "
        f"(run --merge --num_shards={num_shards} to finalize)",
        flush=True,
    )


def _iter_keyed_rows(path: Path) -> Iterator[tuple[tuple[int, int], str]]:
    """Yield ``((conv_idx, msg_offset), raw_line)`` from an already-sorted shard."""
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            row = json.loads(line)
            yield (int(row["conv_idx"]), int(row["msg_offset"])), line


def _closing_iter(path: Path):
    """``_iter_keyed_rows`` as a context manager, so ExitStack closes the file
    handle even if the merge aborts part-way through the heap merge."""
    return closing(_iter_keyed_rows(path))


def _iter_expected_keys(chat_path: Path) -> Iterator[tuple[int, int]]:
    """Yield every ``(conv_idx, msg_offset)`` the source chat.jsonl implies, in
    sorted order -- the exact key set stage 06's cache validation demands."""
    for conv_idx, line in _iter_chat_lines(chat_path):
        for msg_offset in range(len(json.loads(line)["messages"])):
            yield conv_idx, msg_offset


def run_merge(src_chat: Path, out_dir: Path, num_shards: int) -> None:
    """Fold the per-shard caches into the canonical message_lengths.jsonl.

    Scoped to ``_of_<num_shards>`` so stale files from a run with a different
    shard count are ignored. Each shard file is sorted by (conv_idx, msg_offset),
    so a k-way heap merge streams them into one sorted cache in O(num_shards)
    memory -- the full corpus never has to fit in RAM. The merged stream is
    checked key-for-key against the source chat: a missing shard, a stale shard
    from an older chat.jsonl, or a duplicate row all fail HERE rather than
    surfacing as a confusing "cache is stale" abort in stage 06.
    """
    if num_shards < 2:
        raise SystemExit("--merge requires --num_shards > 1")
    if not out_dir.is_dir():
        raise SystemExit(f"--merge: --output_dir does not exist: {out_dir}")

    suffix = f"_of_{num_shards:04d}"
    shard_files = sorted(out_dir.glob(f"message_lengths.shard*{suffix}.jsonl"))
    if not shard_files:
        raise SystemExit(
            f"[merge] no message_lengths.shard*{suffix}.jsonl under {out_dir} "
            f"(did the shard tasks run with --num_shards={num_shards}?)"
        )
    present = sorted(int(p.name.split(".shard")[1].split("_of_")[0]) for p in shard_files)
    missing = sorted(set(range(num_shards)) - set(present))
    if missing:
        # Unlike the frames merge, a gap here is fatal: a cache missing any
        # (conv_idx, msg_offset) is rejected wholesale by the records build.
        raise SystemExit(f"[merge] no shard cache for shards {missing} -- re-run them first")

    out_path = out_dir / MESSAGE_LENGTHS_FILENAME
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_messages = 0
    try:
        with ExitStack() as stack:
            streams = [stack.enter_context(_closing_iter(p)) for p in shard_files]
            merged = heapq.merge(*streams, key=lambda kv: kv[0])
            expected = _iter_expected_keys(src_chat)
            fout = stack.enter_context(tmp_path.open("w"))
            for got_key, line in merged:
                want_key = next(expected, None)
                if want_key is None:
                    raise SystemExit(
                        f"[merge] shard caches hold more messages than {src_chat} "
                        f"(first extra key: {got_key}) -- the shards were measured "
                        f"against a different chat.jsonl"
                    )
                if got_key != want_key:
                    raise SystemExit(
                        f"[merge] key mismatch at message {n_messages}: shards have "
                        f"{got_key}, chat expects {want_key} (duplicate or missing "
                        f"row; shards may be stale for this chat.jsonl)"
                    )
                fout.write(line + "\n")
                n_messages += 1
            leftover = next(expected, None)
            if leftover is not None:
                raise SystemExit(
                    f"[merge] shard caches are incomplete: {src_chat} has messages "
                    f"the shards never measured (first missing key: {leftover})"
                )
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    shard_summaries = sorted(out_dir.glob(f"measure_summary.shard*{suffix}.json"))
    per_shard = [json.loads(p.read_text()) for p in shard_summaries]
    scalars = per_shard[0] if per_shard else {}
    write_manifest(
        out_dir,
        stage="message_lengths",
        params={
            "model_id": scalars.get("model_id", FLAGS.model_id),
            "processor": scalars.get("processor", FLAGS.processor),
            "num_workers": scalars.get("num_workers", FLAGS.num_workers),
            "omegalax_repo": scalars.get("omegalax_repo", FLAGS.omegalax_repo),
            "num_shards": num_shards,
        },
        # Identity, not just a path: stage 06 reuses this cache and must be able
        # to refuse one measured from a different chat.jsonl, which would
        # silently pack that dataset at another one's lengths.
        inputs={
            "source": str(src_chat.parent),
            "source_id": make_artifact_id(src_chat.parent),
        },
        stats={
            "per_split": [{"n_messages": n_messages}],
            "n_messages": n_messages,
            "merged_shards": present,
            "per_shard": per_shard,
        },
    )
    print(
        f"[merge] {len(shard_files)} shards -> {n_messages} messages -> {out_path}",
        flush=True,
    )
    print(f"Wrote {out_dir / 'manifest.json'}")


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_chat = _resolve_src_chat(source_path)

    if FLAGS.merge:
        run_merge(src_chat, output_dir, FLAGS.num_shards)
        return

    _require_measure_flags()
    if not 0 <= FLAGS.shard_index < FLAGS.num_shards:
        raise SystemExit(
            f"--shard_index must be in [0, {FLAGS.num_shards}); got {FLAGS.shard_index}"
        )

    if FLAGS.num_shards > 1:
        run_shard(src_chat, output_dir, FLAGS.shard_index, FLAGS.num_shards)
        return

    # Single-job path (unsharded): stage 04 writes a single split-agnostic
    # <source>/chat.jsonl; measure it ONCE -> <output_dir>/message_lengths.jsonl.
    # The train/val split is applied at the records stage, so this cache is reused
    # across every split / val_fraction and never needs re-running when the split
    # changes.
    per_unit = [_run_measure(src_chat, output_dir)]

    write_manifest(
        output_dir,
        stage="message_lengths",
        params=_measure_params(),
        inputs={"source": str(source_path), "source_id": make_artifact_id(source_path)},
        stats={"per_split": per_unit},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
