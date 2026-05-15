"""Replay-source prep: HuggingFaceM4/FineVision → canonical chat.jsonl.

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
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from absl import app, flags
from datasets import load_dataset
from PIL import Image

from _manifest import write_manifest


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


def _process_config(
    config_name: str,
    *,
    images_dir: Path,
    out_fh,
    per_config_max: int,
    min_relevance: int,
    seed: int,
    jpeg_quality: int,
) -> dict[str, int]:
    print(f"[finevision] loading config={config_name}", flush=True)
    ds = _load_dataset_with_retries(FLAGS.hf_dataset_id, config_name, split=FLAGS.hf_split)
    n_loaded = len(ds)

    if per_config_max > 0 and per_config_max < n_loaded:
        ds = ds.shuffle(seed=seed).select(range(per_config_max))

    n_kept = 0
    n_skipped_relevance = 0
    n_skipped_no_images = 0
    n_skipped_too_long = 0
    char_limit = FLAGS.max_chars_per_message
    for row_idx, row in enumerate(ds):
        if min_relevance > 0:
            r = row.get("relevance_min")
            if isinstance(r, (int, float)) and r < min_relevance:
                n_skipped_relevance += 1
                continue

        turns = row.get("texts") or []
        if any(_turn_too_long(t, char_limit) for t in turns):
            n_skipped_too_long += 1
            continue

        images = row.get("images") or []
        if not images:
            n_skipped_no_images += 1
            continue

        image_paths: list[str] = []
        for img_idx, img in enumerate(images):
            stem = f"{config_name}_{row_idx:08d}_{img_idx:02d}.jpg"
            path = images_dir / stem
            _save_image(img, path, quality=jpeg_quality)
            image_paths.append(str(path))

        messages = _row_to_messages(row, image_paths)
        record = {
            "messages": messages,
            "_source": row.get("source", config_name),
            "_finevision_config": config_name,
        }
        out_fh.write(json.dumps(record, ensure_ascii=False))
        out_fh.write("\n")
        n_kept += 1

    return {
        "config": config_name,
        "n_loaded": n_loaded,
        "n_kept": n_kept,
        "n_skipped_relevance": n_skipped_relevance,
        "n_skipped_no_images": n_skipped_no_images,
        "n_skipped_too_long": n_skipped_too_long,
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
