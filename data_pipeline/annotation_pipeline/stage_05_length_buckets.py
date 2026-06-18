#!/usr/bin/env python3
"""Stage 05: optional local token-count and length-bucket distribution inspector.

Two counting modes:

- Exact (default): --tokenizer/--processor load the trainee model's real
  tokenizer and image processor and measure every message through the vendored
  `qwen3_encoding.make_message_length_fn` - the same code the training chunk
  indexer uses, so bucket boundaries match training exactly. Defaults to
  Qwen3-VL-2B-Instruct. The data_pipeline uv environment pins transformers 5.3
  to match the juergen workspace; exact counting preprocesses every frame, so
  run on a compute node, not a login node.

- Estimated: --tokens-per-image with a value you calibrated
  yourself, used only if --tokenizer is set to "" to disable exact mode.

With neither, stage 05 writes chat.jsonl and the manifest but skips length
bucketing instead of guessing.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from annotation_pipeline import config
from annotation_pipeline.common import (
    ensure_dir,
    read_jsonl,
    text_token_estimate,
    write_json,
    write_jsonl,
)


def message_text(message: dict[str, Any]) -> str:
    text_parts: list[str] = []
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
    return "\n".join(text_parts)


def count_images(message: dict[str, Any]) -> int:
    content = message.get("content", [])
    if not isinstance(content, list):
        return 0
    return sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")


def make_exact_measurer(
    tokenizer_name: str,
    processor_name: str | None,
    preprocessor_config_path: Path | None,
) -> Callable[[dict[str, Any]], Any]:
    """Build the per-message measurer from the trainee model's actual stack.

    Uses the vendored qwen3_encoding (identical to the training chunk indexer),
    so bucket boundaries match training.
    """
    from transformers import AutoImageProcessor, AutoTokenizer

    from annotation_pipeline.qwen3_encoding import make_message_length_fn

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    image_processor = None
    if processor_name:
        ip_kwargs: dict[str, Any] = {}
        if preprocessor_config_path:
            ip_kwargs = json.loads(Path(preprocessor_config_path).read_text())
        image_processor = AutoImageProcessor.from_pretrained(
            processor_name, use_fast=False, **ip_kwargs
        )
    return make_message_length_fn(tokenizer, image_processor)


def measure_sample(
    sample: dict[str, Any], measure: Callable[[dict[str, Any]], Any]
) -> dict[str, Any]:
    total = 0
    vision_tokens = 0
    num_images = 0
    for message in sample["messages"]:
        measured = measure(message)
        if isinstance(measured, int):
            total += measured
        else:
            total += int(measured["length"])
            vision_tokens += int(measured["vision_tokens"])
            num_images += int(measured["num_images"])
    return {
        "token_count": total,
        "vision_tokens": vision_tokens,
        "num_images": num_images,
        "num_messages": len(sample["messages"]),
        "token_count_mode": "measured",
    }


def estimate_sample(
    sample: dict[str, Any], tokens_per_image: int | None, overhead_tokens: int
) -> dict[str, Any]:
    text_tokens = 0
    num_images = 0
    num_messages = len(sample["messages"])
    for message in sample["messages"]:
        text_tokens += text_token_estimate(message_text(message))
        num_images += count_images(message)
    counts: dict[str, Any] = {
        "estimated_text_tokens": int(text_tokens),
        "num_images": int(num_images),
        "num_messages": int(num_messages),
    }
    if tokens_per_image is not None:
        counts["token_count"] = int(
            overhead_tokens + text_tokens + num_images * tokens_per_image + num_messages * 6
        )
        counts["token_count_mode"] = "estimated"
    else:
        counts["token_count"] = None
        counts["token_count_mode"] = "none"
    return counts


def choose_bucket(token_count: int) -> str | None:
    for label, limit in config.BUCKET_LIMITS.items():
        if token_count <= limit:
            return label
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Exact mode (preferred): trainee model's real tokenizer + image processor.
    parser.add_argument(
        "--tokenizer",
        default=config.DEFAULT_TRAINEE_MODEL,
        help="HF tokenizer name/path of the trainee model; enables exact counting.",
    )
    parser.add_argument(
        "--processor",
        default=config.DEFAULT_TRAINEE_MODEL,
        help="HF repo for the image processor (required for exact mode with images).",
    )
    parser.add_argument(
        "--preprocessor-config",
        type=Path,
        default=None,
        help="JSON file whose keys override the image processor config.",
    )
    # Estimated mode: only with an explicitly calibrated value.
    parser.add_argument(
        "--tokens-per-image",
        type=int,
        default=None,
        help=(
            "Calibrated image token cost per frame. Ignored when --tokenizer is "
            "given. Without either, bucketing is skipped instead of guessed."
        ),
    )
    parser.add_argument("--overhead-tokens", type=int, default=config.DEFAULT_TOKEN_OVERHEAD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    samples = read_jsonl(args.samples)

    measure = None
    if args.tokenizer:
        measure = make_exact_measurer(args.tokenizer, args.processor, args.preprocessor_config)

    bucketed: dict[str, list[dict[str, Any]]] = {label: [] for label in config.BUCKET_LIMITS}
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    oversize: list[dict[str, Any]] = []

    for sample in samples:
        if measure is not None:
            counts = measure_sample(sample, measure)
        else:
            counts = estimate_sample(sample, args.tokens_per_image, args.overhead_tokens)
        token_count = counts["token_count"]
        bucket = choose_bucket(token_count) if token_count is not None else None

        row = copy.deepcopy(sample)
        row.update(counts)
        row["bucket"] = bucket
        if token_count is not None and bucket is None:
            oversize.append(row)
            continue
        if bucket is not None:
            bucketed[bucket].append(row)
        all_rows.append(row)
        manifest.append(
            {
                "sample_id": row["sample_id"],
                "bucket": bucket,
                "token_count": token_count,
                "token_count_mode": counts["token_count_mode"],
                "n_frames": row["n_frames"],
                "duration_s": row["duration_s"],
                "instruction": row["instruction"],
                "start_time_s": row["start_time_s"],
                "end_time_s": row["end_time_s"],
            }
        )

    bucketing_enabled = measure is not None or args.tokens_per_image is not None
    write_jsonl(output_dir / "chat.jsonl", all_rows)
    write_jsonl(output_dir / "trajectory_manifest.jsonl", manifest)
    write_jsonl(output_dir / "rejected_oversize.jsonl", oversize)
    if bucketing_enabled:
        for label, rows in bucketed.items():
            write_jsonl(output_dir / f"chat_{label}.jsonl", rows)

    summary = {
        "n_input_samples": len(samples),
        "n_emitted": len(all_rows),
        "n_oversize": len(oversize),
        "token_count_mode": "measured" if measure is not None else (
            "estimated" if args.tokens_per_image is not None else "none"
        ),
        "tokenizer": args.tokenizer,
        "processor": args.processor,
        "tokens_per_image": args.tokens_per_image,
        "overhead_tokens": args.overhead_tokens,
        "buckets": {
            label: {
                "count": len(rows),
                "min_tokens": min((row["token_count"] for row in rows), default=0),
                "max_tokens": max((row["token_count"] for row in rows), default=0),
            }
            for label, rows in bucketed.items()
        }
        if bucketing_enabled
        else None,
    }
    write_json(output_dir / "bucket_summary.json", summary)
    if bucketing_enabled:
        print(f"Wrote {len(all_rows)} bucketed rows to {output_dir} ({summary['token_count_mode']})")
    else:
        print(
            f"Wrote {len(all_rows)} rows to {output_dir} (no length buckets: "
            "pass --tokenizer/--processor for exact counts, or a calibrated "
            "--tokens-per-image)"
        )


if __name__ == "__main__":
    main()
