"""Drop chat.jsonl rows whose assistant turn was truncated by the
generation-time ``max_tokens`` cap.

Heuristic recovery of the OpenAI-style ``finish_reason == "length"`` flag,
which the on-policy prep stage did not persist. Re-tokenises each row's
last (assistant) turn with the *same* model's tokenizer and drops rows
whose content tokenises to >= ``max_tokens`` tokens.

False-positive rate (natural-EOS responses landing at exactly the cap)
is empirically negligible; false-negative rate (model emitted EOS
mid-thought below the cap) exists in principle but is dominated by the
cap-truncation case for our data.

Runs under the omegalax uv venv so we get its already-pinned
``transformers`` (and the matching Qwen3-VL fast tokenizer) without
adding a new dependency to the data_pipeline venv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from absl import app, flags

sys.path.insert(0, "/fast/home/franz.srambical/data_pipeline")
from transformers import AutoTokenizer

from _manifest import write_manifest  # type: ignore[import-not-found]

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output dataset root.", required=True)
flags.DEFINE_string(
    "source_chat_jsonl",
    None,
    "Path to the source chat.jsonl whose truncated rows we want to drop.",
    required=True,
)
flags.DEFINE_string(
    "tokenizer_model",
    "Qwen/Qwen3-VL-2B-Instruct",
    "HF model_id whose tokenizer matches the generation-time tokenizer. "
    "MUST equal the ``teacher_model`` used in prep — that's how the "
    "max-tokens equivalence holds.",
)
flags.DEFINE_integer(
    "max_tokens",
    1280,
    "Generation-time max_tokens cap from the prep stage. Rows whose "
    "assistant content re-tokenises to >= this length are dropped.",
)
flags.DEFINE_integer(
    "tokenize_batch_size",
    1024,
    "Batch size for fast-tokenizer calls. Tokenization is CPU-bound; "
    "batching trades a small constant memory bump for ~10x throughput.",
)


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "chat.jsonl"

    src_path = Path(FLAGS.source_chat_jsonl)
    if not src_path.is_file():
        raise FileNotFoundError(f"source_chat_jsonl not found: {src_path}")

    print(f"[filter] loading tokenizer {FLAGS.tokenizer_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(FLAGS.tokenizer_model, use_fast=True)

    t0 = time.time()
    n_loaded = 0
    n_kept = 0
    n_dropped_truncated = 0
    n_dropped_no_assistant = 0

    batch_lines: list[str] = []
    batch_contents: list[str] = []

    def _flush_batch(out_f) -> None:
        nonlocal n_kept, n_dropped_truncated
        if not batch_lines:
            return
        # add_special_tokens=False so we count only the content tokens,
        # matching what sglang's max_tokens cap counted at generation time.
        enc = tok(
            batch_contents,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for raw_line, ids in zip(batch_lines, enc["input_ids"], strict=False):
            if len(ids) >= FLAGS.max_tokens:
                n_dropped_truncated += 1
                continue
            out_f.write(raw_line)
            n_kept += 1
        batch_lines.clear()
        batch_contents.clear()

    with src_path.open() as in_f, out_path.open("w") as out_f:
        for line in in_f:
            stripped = line.strip()
            if not stripped:
                continue
            n_loaded += 1
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"bad jsonl at row {n_loaded}: {e}") from e
            messages = rec.get("messages") or []
            last = messages[-1] if messages else None
            if not last or last.get("role") != "assistant":
                # Source rows that don't end in an assistant turn are
                # neither useful for replay SFT nor producible by our
                # prep stage; treat as dropped for visibility.
                n_dropped_no_assistant += 1
                continue
            content = _content_to_str(last.get("content", ""))
            batch_lines.append(line)
            batch_contents.append(content)
            if len(batch_lines) >= FLAGS.tokenize_batch_size:
                _flush_batch(out_f)
                if (n_kept + n_dropped_truncated) % (FLAGS.tokenize_batch_size * 20) == 0:
                    rate = (n_kept + n_dropped_truncated) / max(time.time() - t0, 1e-6)
                    print(
                        f"[filter] processed={n_kept + n_dropped_truncated} "
                        f"kept={n_kept} dropped_truncated={n_dropped_truncated} "
                        f"rate={rate:.0f}/s",
                        flush=True,
                    )
        _flush_batch(out_f)

    write_manifest(
        output_dir,
        stage="replay_filter_truncated",
        params={
            "source_chat_jsonl": str(src_path),
            "tokenizer_model": FLAGS.tokenizer_model,
            "max_tokens": FLAGS.max_tokens,
        },
        inputs={"source_chat_jsonl": str(src_path)},
        stats={
            "n_loaded": n_loaded,
            "n_kept": n_kept,
            "n_dropped_truncated": n_dropped_truncated,
            "n_dropped_no_assistant": n_dropped_no_assistant,
            "elapsed_s": int(time.time() - t0),
        },
    )
    print(
        f"[filter] wrote {out_path} "
        f"(loaded={n_loaded}, kept={n_kept}, "
        f"dropped_truncated={n_dropped_truncated}, "
        f"dropped_no_assistant={n_dropped_no_assistant})",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
