"""Replay-source prep: Tulu3-Persona-IF → canonical chat.jsonl.

Downloads ``allenai/tulu-3-sft-personas-instruction-following`` (~30k rows,
single ``train`` split, ODC-BY-1.0) and writes one chat.jsonl line per
example. The dataset's ``messages`` column is already in chat format
(``[{role, content}, ...]``); we copy it through verbatim and stash the
``constraints`` list under ``_constraints`` for later analysis (potentially
useful as an IFEval-style signal proxy at training time).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from absl import app, flags
from datasets import load_dataset

from _manifest import write_manifest


def _content_char_count(message: dict) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "") or "") for b in content if isinstance(b, dict))
    return 0


def _any_message_too_long(messages, limit: int) -> bool:
    return any(_content_char_count(m) > limit for m in messages)


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
flags.DEFINE_string(
    "hf_dataset_id",
    "allenai/tulu-3-sft-personas-instruction-following",
    "HF dataset id.",
)
flags.DEFINE_string("hf_split", "train", "HF split.")
flags.DEFINE_integer(
    "max_samples",
    0,
    "Maximum samples to write (0 = full split). Subsampled uniformly at random.",
)
flags.DEFINE_integer("seed", 0, "Shuffle seed when subsampling.")
flags.DEFINE_integer(
    "max_chars_per_message",
    10000,
    "Drop the conversation if any single message has more characters than this. "
    "Tightened to 10000 because Tulu3-Persona-IF has dense persona+constraint "
    "messages that tokenize denser than the naive ~3.5 chars/token English ratio.",
)


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "chat.jsonl"

    t0 = time.time()
    ds = _load_dataset_with_retries(FLAGS.hf_dataset_id, split=FLAGS.hf_split)
    n_loaded = len(ds)
    if FLAGS.max_samples > 0 and FLAGS.max_samples < n_loaded:
        ds = ds.shuffle(seed=FLAGS.seed).select(range(FLAGS.max_samples))

    n_written = 0
    n_skipped_too_long = 0
    with out_path.open("w") as f:
        for row in ds:
            messages = row.get("messages")
            if not messages:
                raise ValueError(f"Row missing 'messages' field; got keys {list(row)}.")
            if _any_message_too_long(messages, FLAGS.max_chars_per_message):
                n_skipped_too_long += 1
                continue
            record = {
                "messages": messages,
                "_source": FLAGS.hf_dataset_id,
                "_constraints": row.get("constraints", []),
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    write_manifest(
        output_dir,
        stage="replay_prep_tulu3_persona_if",
        params={
            "hf_dataset_id": FLAGS.hf_dataset_id,
            "hf_split": FLAGS.hf_split,
            "max_samples": FLAGS.max_samples,
            "seed": FLAGS.seed,
        },
        inputs={},
        stats={
            "n_loaded": n_loaded,
            "n_written": n_written,
            "n_skipped_too_long": n_skipped_too_long,
            "elapsed_s": int(time.time() - t0),
        },
    )
    print(f"Wrote {out_path} ({n_written} lines, dropped {n_skipped_too_long} too-long)")


if __name__ == "__main__":
    app.run(main)
