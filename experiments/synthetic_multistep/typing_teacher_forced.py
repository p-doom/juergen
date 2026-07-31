#!/usr/bin/env python3
"""Teacher-forced assistant/action NLL for one typing-factorial cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from .typing_evaluate import TypingEvalError, validate
except ImportError:
    from typing_evaluate import TypingEvalError, validate


def assistant_text(row: dict[str, Any]) -> str:
    return row["messages"][-1]["content"][0]["text"]


def image_path(row: dict[str, Any]) -> Path:
    values = [item["image"] for item in row["messages"][1]["content"] if item.get("type") == "image"]
    if len(values) != 1:
        raise TypingEvalError(f"record must have one image: {row.get('sample_id')}")
    return Path(values[0])


def find_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int]:
    starts = [index for index in range(len(sequence) - len(target) + 1)
              if sequence[index:index + len(target)] == target]
    if len(starts) != 1:
        raise TypingEvalError(f"expected unique token span, found {len(starts)}")
    return starts[0], starts[0] + len(target)


def content_ids(tokenizer: Any, full_text: str, content: str) -> list[int]:
    start = full_text.rfind(content)
    if start < 0:
        raise TypingEvalError("assistant content absent from chat template")
    end = start + len(content)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    indices = [index for index, (left, right) in enumerate(encoded["offset_mapping"])
               if right > start and left < end]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise TypingEvalError("assistant content token span is absent/noncontiguous")
    return encoded["input_ids"][indices[0]:indices[-1] + 1]


def atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_root = args.model_root.resolve()
    dataset = args.dataset.resolve()
    manifest, rows, _ = validate(
        model_root, dataset, args.lineage, args.format, args.parser_dir.resolve()
    )
    from PIL import Image
    import torch
    import torch.nn.functional as functional
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_dir = model_root / "hf"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).to(torch.device("cuda:0")).eval()
    results = []
    for index, row in enumerate(rows):
        assistant = assistant_text(row)
        action = assistant.splitlines()[-1]
        full_text = processor.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        assistant_ids = content_ids(processor.tokenizer, full_text, assistant)
        action_ids = content_ids(processor.tokenizer, full_text, action)
        with Image.open(image_path(row)) as source:
            image = source.convert("RGB")
        inputs = processor(text=[full_text], images=[image], padding=False, return_tensors="pt").to(model.device)
        ids = inputs["input_ids"][0].tolist()
        assistant_start, assistant_end = find_subsequence(ids, assistant_ids)
        action_start, action_end = find_subsequence(ids, action_ids)
        if not (assistant_start <= action_start < action_end <= assistant_end):
            raise TypingEvalError(f"action span outside assistant span: {row['sample_id']}")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, return_dict=True).logits[0]

        def score(start: int, end: int) -> tuple[float, int]:
            if start < 1:
                raise TypingEvalError("cannot score token at sequence start")
            losses = functional.cross_entropy(
                logits[start - 1:end - 1].float(), inputs["input_ids"][0, start:end], reduction="none"
            )
            return float(losses.sum().item()), int(losses.numel())

        assistant_sum, assistant_count = score(assistant_start, assistant_end)
        action_sum, action_count = score(action_start, action_end)
        results.append({
            "sample_id": row["sample_id"], "lineage": args.lineage, "format": args.format,
            "assistant_nll_sum": assistant_sum, "assistant_tokens": assistant_count,
            "assistant_token_nll": assistant_sum / assistant_count,
            "action_nll_sum": action_sum, "action_tokens": action_count,
            "action_line_token_nll": action_sum / action_count,
        })
        del logits, inputs
        if (index + 1) % 20 == 0:
            print(f"[typing-teacher] {args.lineage}/{args.format} {index + 1}/200", flush=True)
    summary = {
        "n_examples": len(results),
        "assistant_token_nll": sum(row["assistant_nll_sum"] for row in results)
        / sum(row["assistant_tokens"] for row in results),
        "action_line_token_nll": sum(row["action_nll_sum"] for row in results)
        / sum(row["action_tokens"] for row in results),
        "assistant_example_mean_nll": sum(row["assistant_token_nll"] for row in results) / len(results),
        "action_example_mean_nll": sum(row["action_line_token_nll"] for row in results) / len(results),
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "typing_teacher_forced_rows.jsonl"
    atomic(rows_path, "".join(json.dumps(row) + "\n" for row in results))
    report = {
        "schema_version": 1, "artifact_type": "synthetic_typing_teacher_forced_eval",
        "status": "complete", "lineage": args.lineage, "target_format": args.format,
        "summary": summary, "model_manifest": manifest,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    atomic(out / "typing_teacher_forced_report.json", json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--parser-dir", required=True, type=Path)
    parser.add_argument("--lineage", required=True, choices=("A", "B"))
    parser.add_argument("--format", required=True, choices=("coalesced", "perkey"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args), indent=2))
    except TypingEvalError as exc:
        print(f"FATAL typing teacher-forced eval: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
