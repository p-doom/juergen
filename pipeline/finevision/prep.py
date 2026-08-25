"""Replay-source prep: HuggingFaceM4/FineVision → canonical chat.jsonl.

Run as a file path, the way labctl dispatches every stage::

    uv run python pipeline/finevision/prep.py --configs ... --output_dir ...

The emitted `chat.jsonl` is the same shape `pipeline.crowdcast` stage 04
writes, so `stage_06_training_records` consumes either.

FineVision is structured as 100+ HF configs, each one a single-source
sub-corpus (e.g. ``DoclingMatrix`` for document OCR, ``SynthChartNet`` for
charts, ``GroundUI`` for GUI grounding). There is no single ``category``
column; we instead select configs explicitly via ``--configs`` and
stratified-sample up to ``--per_config_max`` rows from each.

Each FineVision row carries
  - ``images``: list of PIL images
  - ``texts``: list of ``{user, assistant}`` turn dicts
  - ``source``: origin sub-corpus name
  - assorted rating fields (e.g. ``relevance_min``)

We materialize the PIL images to disk under ``<output_dir>/train/images/``
and rewrite the conversation as canonical chat-format messages with
``{"type": "image", "url": "<abs_path>"}`` blocks inline in the first user
turn. This matches what omegalax's ``extract_images`` (qwen3_encoding.py)
expects.

PARALLELISM. Per-row cost is dominated by the arrow image decode plus the JPEG
re-encode, so rows are farmed out to a ``--num_workers`` process pool. Workers do
NOT receive rows over the pipe (that would ship every image twice) -- they are
handed row INDICES into a dataset view they already have, write their own JPEGs,
and return one small metadata dict per row.

The pool is ``fork``-based, so each worker inherits the parent's memory-mapped
arrow tables and its already-imported modules. That matters: a ``spawn`` pool was
measured at ~100 s of startup for 32 workers (every child re-importing
``datasets``), which dwarfs the work for anything but the largest configs. This is
the same idiom a PyTorch DataLoader uses over an HF dataset; the arrow data is
read-only and offset-addressed, so concurrent reads through the inherited mapping
are safe. Nothing native and stateful (e.g. an ArrayRecord reader) is open at fork
time.

Results come back through ``imap`` (ORDERED), so chat.jsonl line order and the
``<config>_<row_idx>_<img_idx>.jpg`` filenames are byte-identical to a serial run
-- verified by diffing a 1-worker against a 16-worker build.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

from absl import app, flags
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm  # progress bar; arrives transitively via datasets

# Make the ``pipeline`` package importable when run directly as a file path,
# which is how labctl dispatches this. Previously the manifest writer sat beside
# this file as a bare ``_manifest`` module, so no bootstrap was needed; it is now
# ``pipeline.manifest``, shared with the crowd-cast corpus.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.manifest import write_manifest  # noqa: E402



def _load_dataset_with_retries(*args, attempts: int = 5, backoff_s: float = 30.0, **kwargs):
    """Wrap ``load_dataset`` with bounded exponential retries; flaky-network safe."""
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(*args, **kwargs)
        except (ConnectionError, OSError, TimeoutError) as e:
            if attempt == attempts:
                raise
            delay = backoff_s * (2 ** (attempt - 1))
            print(
                f"[prep] load_dataset attempt {attempt}/{attempts} failed: {e!r}; "
                f"sleeping {delay:.0f}s before retry",
                flush=True,
            )
            time.sleep(delay)
    return None


FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output dataset root.", required=True)
flags.DEFINE_string("hf_dataset_id", "HuggingFaceM4/FineVision", "HF dataset id.")
flags.DEFINE_list(
    "configs",
    None,
    "Comma-separated list of FineVision configs to load and stratify over.",
)
flags.DEFINE_integer(
    "per_config_max",
    0,
    "Max samples per config (0 = unlimited). Subsampled uniformly at random.",
)
flags.DEFINE_integer(
    "min_relevance",
    0,
    "Drop rows whose 'relevance_min' is below this threshold (0 = keep all). "
    "FineVision rates each turn 1-5 by multiple annotators; 'relevance_min' is "
    "the minimum across annotators.",
)
flags.DEFINE_string("hf_split", "train", "HF split.")
flags.DEFINE_integer("seed", 0, "Shuffle seed when subsampling.")
flags.DEFINE_integer("jpeg_quality", 90, "JPEG quality for materialized images.")
flags.DEFINE_integer(
    "num_workers",
    0,
    "Parallel row workers (0 = os.cpu_count()). Workers fork from the parent, "
    "so they share its memory-mapped arrow tables; each decodes its own rows "
    "and writes its own JPEGs, and the parent only serialises the returned "
    "metadata into chat.jsonl. Set 1 for a single-process run with tracebacks "
    "in the foreground.",
    lower_bound=0,
)
flags.DEFINE_integer(
    "max_chars_per_message",
    10000,
    "Drop the conversation if any single user/assistant turn has more characters "
    "than this. Char-count proxy for Qwen3 token-count; tightened to 10000 to "
    "absorb dense/structured text where the typical English ratio doesn't hold.",
)


def _turn_too_long(turn: dict, limit: int) -> bool:
    """Char-budget check for a FineVision (user, assistant) turn dict."""
    return len(turn.get("user", "") or "") > limit or len(turn.get("assistant", "") or "") > limit


def _save_image(img: Image.Image, path: Path, *, quality: int) -> None:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=quality, optimize=False)


def _row_to_messages(
    row: dict[str, Any],
    image_paths: list[str],
) -> list[dict[str, Any]]:
    """Convert FineVision (``images``, ``texts``) into chat-format messages.

    Images are inlined as ``{"type": "image", "url": ...}`` blocks at the
    start of the first user turn. Subsequent turns are plain-text only.
    """
    turns = row["texts"]
    messages: list[dict[str, Any]] = []
    for turn_idx, turn in enumerate(turns):
        if turn_idx == 0 and image_paths:
            user_content: list[dict[str, Any]] = [{"type": "image", "url": p} for p in image_paths]
            user_content.append({"type": "text", "text": turn["user"]})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def _build_view(
    hf_dataset_id: str,
    config_name: str,
    hf_split: str,
    per_config_max: int,
    seed: int,
):
    """Load one FineVision config and apply the deterministic subsample.

    Built once in the parent (for the row count) and once per worker. The view is
    fully determined by ``(config_name, per_config_max, seed)``, so every process
    lands on the identical selection -- that is what lets workers address rows by
    index instead of having them shipped over the pipe.

    ``keep_in_memory=True`` keeps only the INDICES MAPPING in memory (the arrow
    data stays memory-mapped, so this is safe for the 250 GB configs). It also
    keeps the mapping out of the datasets cache: N processes racing the same
    ``indices-<hash>.arrow`` temp->rename is exactly the "Text file busy" class of
    failure the stage-05 recipe already works around for uv.
    """
    ds = _load_dataset_with_retries(hf_dataset_id, config_name, split=hf_split)
    n_loaded = len(ds)
    if per_config_max > 0 and per_config_max < n_loaded:
        ds = ds.shuffle(seed=seed, keep_in_memory=True).select(
            range(per_config_max), keep_in_memory=True
        )
    return ds, n_loaded


# Per-config worker state. Populated in the PARENT before the pool forks, so every
# child inherits it -- including the dataset view, which is why no worker pays a
# rebuild. Kept in a module global (rather than closed over) because a bound method
# or closure could not be dispatched by Pool.imap.
_W: dict[str, Any] = {}


def _process_row(row_idx: int) -> tuple[str, dict[str, Any] | None]:
    """Filter one row and materialize its images. Runs in a worker.

    Returns ``(verdict, record | None)`` where verdict is ``"kept"`` or the drop
    reason. The JPEG bytes are written here and never cross the pipe -- only the
    record dict (messages + provenance, with paths as strings) goes back.
    """
    row = _W["ds"][row_idx]
    config_name = _W["config_name"]

    if _W["min_relevance"] > 0:
        r = row.get("relevance_min")
        if isinstance(r, (int, float)) and r < _W["min_relevance"]:
            return "relevance", None

    turns = row.get("texts") or []
    if any(_turn_too_long(t, _W["char_limit"]) for t in turns):
        return "too_long", None

    images = row.get("images") or []
    if not images:
        return "no_images", None

    image_paths: list[str] = []
    for img_idx, img in enumerate(images):
        stem = f"{config_name}_{row_idx:08d}_{img_idx:02d}.jpg"
        path = _W["images_dir"] / stem
        _save_image(img, path, quality=_W["jpeg_quality"])
        image_paths.append(str(path))

    return "kept", {
        "messages": _row_to_messages(row, image_paths),
        "_source": row.get("source", config_name),
        "_finevision_config": config_name,
    }


def _process_config(
    config_name: str,
    *,
    images_dir: Path,
    out_fh,
    per_config_max: int,
    min_relevance: int,
    seed: int,
    jpeg_quality: int,
    num_workers: int,
) -> dict[str, int]:
    print(f"[finevision] loading config={config_name}", flush=True)
    ds, n_loaded = _build_view(
        FLAGS.hf_dataset_id, config_name, FLAGS.hf_split, per_config_max, seed
    )
    n_rows = len(ds)
    n_workers = num_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_rows or 1))
    char_limit = FLAGS.max_chars_per_message
    print(
        f"[finevision] {config_name}: {n_rows} row(s) after subsample "
        f"(loaded {n_loaded}), {n_workers} worker(s)",
        flush=True,
    )

    counts = {"kept": 0, "relevance": 0, "too_long": 0, "no_images": 0}
    # Publish this config's state BEFORE forking; the children inherit it as-is.
    _W.update(
        ds=ds,
        config_name=config_name,
        images_dir=images_dir,
        min_relevance=min_relevance,
        char_limit=char_limit,
        jpeg_quality=jpeg_quality,
    )
    # `fork`, so workers inherit the view and the imported modules instead of
    # rebuilding both (see the module docstring for the measured spawn penalty).
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        # imap (ORDERED) -- chat.jsonl comes out in row order, byte-identical to a
        # serial run. chunksize=1 because per-row cost varies by orders of
        # magnitude across configs (a chart thumbnail vs a full page scan), and
        # coarse chunks would leave workers idle at the tail.
        # mininterval throttles redraws so a slurm log isn't spammed.
        bar = tqdm(
            pool.imap(_process_row, range(n_rows), chunksize=1),
            total=n_rows, unit="row", desc=f"[finevision] {config_name}",
            smoothing=0.05, mininterval=2.0, dynamic_ncols=True,
        )
        for verdict, record in bar:
            counts[verdict] += 1
            if record is not None:
                out_fh.write(json.dumps(record, ensure_ascii=False))
                out_fh.write("\n")
            bar.set_postfix(
                kept=counts["kept"],
                rel=counts["relevance"],
                long=counts["too_long"],
                noimg=counts["no_images"],
                refresh=False,
            )
        bar.close()

    return {
        "config": config_name,
        "n_loaded": n_loaded,
        "n_rows_after_subsample": n_rows,
        "n_workers": n_workers,
        "n_kept": counts["kept"],
        "n_skipped_relevance": counts["relevance"],
        "n_skipped_no_images": counts["no_images"],
        "n_skipped_too_long": counts["too_long"],
    }


def main(_) -> None:
    if not FLAGS.configs:
        raise ValueError("--configs is required (comma-separated list).")

    output_dir = Path(FLAGS.output_dir)
    train_dir = output_dir / "train"
    images_dir = train_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "chat.jsonl"

    t0 = time.time()
    per_config_stats: list[dict[str, int]] = []
    with out_path.open("w") as out_fh:
        for config_name in FLAGS.configs:
            stats = _process_config(
                config_name.strip(),
                images_dir=images_dir,
                out_fh=out_fh,
                per_config_max=FLAGS.per_config_max,
                min_relevance=FLAGS.min_relevance,
                seed=FLAGS.seed,
                jpeg_quality=FLAGS.jpeg_quality,
                num_workers=FLAGS.num_workers,
            )
            per_config_stats.append(stats)

    n_written = sum(s["n_kept"] for s in per_config_stats)
    write_manifest(
        output_dir,
        stage="replay_prep_finevision",
        params={
            "hf_dataset_id": FLAGS.hf_dataset_id,
            "configs": list(FLAGS.configs),
            "per_config_max": FLAGS.per_config_max,
            "min_relevance": FLAGS.min_relevance,
            "hf_split": FLAGS.hf_split,
            "seed": FLAGS.seed,
            "jpeg_quality": FLAGS.jpeg_quality,
            "num_workers": FLAGS.num_workers,
        },
        inputs={},
        stats={
            "per_config": per_config_stats,
            "n_written": n_written,
            "elapsed_s": int(time.time() - t0),
        },
    )
    print(f"Wrote {out_path} ({n_written} lines, {len(per_config_stats)} configs)")


if __name__ == "__main__":
    app.run(main)
