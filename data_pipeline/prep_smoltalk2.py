"""Replay-source prep: SmolTalk2 SFT subset → canonical chat.jsonl.

Downloads HuggingFaceTB/smoltalk2 (SFT config) from the HF cache (or live),
optionally subsamples uniformly at random, and writes one chat.jsonl line per
example under ``<output_dir>/train/``. Output schema matches what stage_c
(omegalax/scripts/compile_sft_dataset.py) consumes: each line is
``{"messages": [...], "_source": "<sub_source>"}``.

SmolTalk2's SFT subset stores chat-formatted ``messages`` (list of
``{role, content}`` dicts) and a ``source`` column tagging which sub-corpus
the row came from. We pass ``messages`` through verbatim and stash ``source``
under ``_source`` for later analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from absl import app, flags
from datasets import concatenate_datasets, load_dataset

from _manifest import write_manifest


def _content_char_count(message: dict) -> int:
    """Approximate char-length of a message's content (string or block-list)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(block.get("text", "") or "")
        return total
    return 0


def _any_message_too_long(messages, limit: int) -> bool:
    return any(_content_char_count(m) > limit for m in messages)


def _load_dataset_with_retries(*args, attempts: int = 5, backoff_s: float = 30.0, **kwargs):
    """Wrap ``load_dataset`` with bounded exponential retries.

    HuggingFace's load_dataset routinely times out the metadata fetch on
    flaky cluster networks; the library itself does not retry the
    initial dataset_module_factory call. We do.
    """
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
flags.DEFINE_string("hf_dataset_id", "HuggingFaceTB/smoltalk2", "HF dataset id.")
flags.DEFINE_string("hf_config", "SFT", "HF dataset config name.")
flags.DEFINE_list(
    "hf_splits",
    None,
    "Comma-separated split names. SmolTalk2's SFT config has no plain 'train' "
    "split; instead each sub-corpus is its own named split (e.g. "
    "'smoltalk_smollm3_smol_magpie_ultra_no_think', 'OpenHermes_2.5_no_think', "
    "'OpenThoughts3_1.2M_no_think_no_think'). Multiple splits are concatenated.",
)
flags.DEFINE_integer(
    "max_samples",
    0,
    "Maximum samples to write (0 = full concatenated splits). Subsampled "
    "uniformly at random across the union.",
)
flags.DEFINE_integer("seed", 0, "Shuffle seed when subsampling.")
flags.DEFINE_integer(
    "max_chars_per_message",
    10000,
    "Drop the entire conversation if any single message has more characters than "
    "this. Char-count proxy for Qwen3 token-count; tightened to 10000 from the "
    "naive 14000 because dense/structured text tokenizes denser than the typical "
    "English ratio. Stage_d (chunk_index) hard-errors on any message exceeding "
    "max_length tokens; we filter at prep to keep that error from killing runs.",
)


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "chat.jsonl"

    if not FLAGS.hf_splits:
        raise ValueError("--hf_splits must list at least one named split.")
    t0 = time.time()
    per_split: list = []
    for raw_split in FLAGS.hf_splits:
        split_name = raw_split.strip()
        ds_part = _load_dataset_with_retries(
            FLAGS.hf_dataset_id,
            FLAGS.hf_config,
            split=split_name,
        )
        per_split.append((split_name, len(ds_part), ds_part))
    ds = concatenate_datasets([d for _, _, d in per_split])
    n_loaded = len(ds)
    if FLAGS.max_samples > 0 and FLAGS.max_samples < n_loaded:
        ds = ds.shuffle(seed=FLAGS.seed).select(range(FLAGS.max_samples))

    n_written = 0
    n_skipped_too_long = 0
    with out_path.open("w") as f:
        for row in ds:
            messages = row.get("messages")
            if not messages:
                raise ValueError(
                    f"Row missing 'messages' field; got keys {list(row)}. "
                    f"This script expects SmolTalk2 SFT schema."
                )
            if _any_message_too_long(messages, FLAGS.max_chars_per_message):
                n_skipped_too_long += 1
                continue
            record = {
                "messages": messages,
                "_source": row.get("source", FLAGS.hf_dataset_id),
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    write_manifest(
        output_dir,
        stage="replay_prep_smoltalk2",
        params={
            "hf_dataset_id": FLAGS.hf_dataset_id,
            "hf_config": FLAGS.hf_config,
            "hf_splits": list(FLAGS.hf_splits),
            "max_samples": FLAGS.max_samples,
            "seed": FLAGS.seed,
            "max_chars_per_message": FLAGS.max_chars_per_message,
        },
        inputs={},
        stats={
            "n_loaded": n_loaded,
            "per_split": [{"name": n, "rows": rows} for n, rows, _ in per_split],
            "n_written": n_written,
            "n_skipped_too_long": n_skipped_too_long,
            "elapsed_s": int(time.time() - t0),
        },
    )
    print(f"Wrote {out_path} ({n_written} lines, dropped {n_skipped_too_long} too-long)")


if __name__ == "__main__":
    app.run(main)
