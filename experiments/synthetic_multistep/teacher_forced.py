#!/usr/bin/env python3
"""Teacher-forced B-format token NLL on the frozen fresh validation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class EvalError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assistant_text(row: dict[str, Any]) -> str:
    return row["messages"][-1]["content"][0]["text"]


def _image_path(row: dict[str, Any]) -> Path:
    images = [item["image"] for item in row["messages"][1]["content"]
              if item.get("type") == "image"]
    if len(images) != 1:
        raise EvalError(f"record must contain one image: {row.get('sample_id')}")
    return Path(images[0])


def _find_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int]:
    starts = [index for index in range(len(sequence) - len(target) + 1)
              if sequence[index:index + len(target)] == target]
    if len(starts) != 1:
        raise EvalError(f"expected unique token span, found {len(starts)} matches")
    return starts[0], starts[0] + len(target)


def _content_token_ids(tokenizer, full_text: str, content: str) -> list[int]:
    start = full_text.rfind(content)
    if start < 0:
        raise EvalError("assistant content absent from rendered chat template")
    end = start + len(content)
    encoded = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    indices = [index for index, (left, right) in enumerate(encoded["offset_mapping"])
               if right > start and left < end]
    if not indices:
        raise EvalError("assistant content produced no tokens")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise EvalError("assistant content token span is not contiguous")
    return encoded["input_ids"][indices[0]:indices[-1] + 1]


def _validate(model_root: Path, dataset: Path, expected_branch: str) -> dict[str, Any]:
    manifest_path = model_root / "curriculum_train_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
        "schema_version": 1, "status": "complete", "branch": expected_branch,
        "target_format": "deltatype_raw_pre", "step": 750,
        "fresh_optimizer": True, "lora_rank": 256, "lora_alpha": 256,
        "hf_subdir": "hf",
    }
    bad = {key: (manifest.get(key), value) for key, value in expected.items()
           if manifest.get(key) != value}
    if bad:
        raise EvalError(f"wrong curriculum model: {bad}")
    dataset_manifest_path = dataset / "curriculum_dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    if (dataset_manifest.get("status") != "complete"
            or dataset_manifest.get("validation_records") != 200):
        raise EvalError(f"wrong curriculum dataset: {dataset_manifest}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if Path(manifest.get("dataset", "")).resolve() != dataset.resolve():
        raise EvalError("model and evaluation dataset paths do not match")
    if manifest.get("dataset_manifest_sha256") != digest(dataset_manifest_path):
        raise EvalError("model and evaluation dataset manifest hashes do not match")
    source = Path(manifest.get("source_model", ""))
    source_manifest = source / "train_export_manifest.json"
    if (not source_manifest.is_file()
            or manifest.get("source_manifest_sha256") != digest(source_manifest)):
        raise EvalError("stage-1 source manifest hash mismatch")
    endpoint_paths = {
        "checkpoint_metadata_sha256": Path(manifest.get("source_checkpoint", ""))
        / "_CHECKPOINT_METADATA",
        "lora_metadata_sha256": Path(manifest.get("source_checkpoint", "")).parent
        / "lora_metadata.json",
    }
    for key, path in endpoint_paths.items():
        if not path.is_file() or manifest.get("endpoint_hashes", {}).get(key) != digest(path):
            raise EvalError(f"recovered endpoint hash mismatch: {key}")
    hf = model_root / manifest["hf_subdir"]
    weights = sorted((path.name, path.stat().st_size) for path in hf.glob("*.safetensors"))
    if not weights:
        raise EvalError("model export contains no safetensors weights")
    manifest["evaluation_input_hashes"] = {
        "model_manifest_sha256": digest(manifest_path),
        "dataset_manifest_sha256": digest(dataset_manifest_path),
        "config_sha256": digest(hf / "config.json"),
        "weights": [{"name": name, "size": size} for name, size in weights],
    }
    return manifest


def evaluate(model_root: Path, dataset: Path, branch: str, out: Path) -> dict[str, Any]:
    manifest = _validate(model_root.resolve(), dataset.resolve(), branch)
    from PIL import Image
    import torch
    import torch.nn.functional as functional
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_dir = model_root / manifest["hf_subdir"]
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to(torch.device("cuda:0")).eval()
    rows = _jsonl(dataset / "deltatype_raw_pre" / "_normalized" / "val" / "chat.jsonl")
    if len(rows) != 200:
        raise EvalError(f"expected 200 validation records, found {len(rows)}")
    results = []
    for index, row in enumerate(rows):
        assistant = _assistant_text(row)
        action = assistant.splitlines()[-1]
        messages = row["messages"]
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        assistant_ids = _content_token_ids(processor.tokenizer, full_text, assistant)
        action_ids = _content_token_ids(processor.tokenizer, full_text, action)
        with Image.open(_image_path(row)) as source:
            image = source.convert("RGB")
        inputs = processor(
            text=[full_text], images=[image], padding=False, return_tensors="pt"
        ).to(model.device)
        input_ids = inputs["input_ids"][0].tolist()
        assistant_start, assistant_end = _find_subsequence(input_ids, assistant_ids)
        action_start, action_end = _find_subsequence(input_ids, action_ids)
        if not (assistant_start <= action_start < action_end <= assistant_end):
            raise EvalError(f"action span is outside assistant span: {row['sample_id']}")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, return_dict=True).logits[0]

        def score(start: int, end: int) -> tuple[float, int]:
            if start < 1:
                raise EvalError("cannot score token at sequence start")
            token_logits = logits[start - 1:end - 1].float()
            targets = inputs["input_ids"][0, start:end]
            losses = functional.cross_entropy(token_logits, targets, reduction="none")
            return float(losses.sum().item()), int(losses.numel())

        assistant_sum, assistant_count = score(assistant_start, assistant_end)
        action_sum, action_count = score(action_start, action_end)
        results.append({
            "sample_id": row["sample_id"], "scene_id": row["scene_id"],
            "assistant_nll_sum": assistant_sum, "assistant_tokens": assistant_count,
            "assistant_token_nll": assistant_sum / assistant_count,
            "action_nll_sum": action_sum, "action_tokens": action_count,
            "action_line_token_nll": action_sum / action_count,
        })
        del logits, inputs
        if (index + 1) % 20 == 0:
            print(f"[teacher-forced] {branch} {index + 1}/200", flush=True)
    summary = {
        "n_examples": len(results),
        "assistant_token_nll": sum(x["assistant_nll_sum"] for x in results)
        / sum(x["assistant_tokens"] for x in results),
        "action_line_token_nll": sum(x["action_nll_sum"] for x in results)
        / sum(x["action_tokens"] for x in results),
        "assistant_example_mean_nll": sum(x["assistant_token_nll"] for x in results) / len(results),
        "action_example_mean_nll": sum(x["action_line_token_nll"] for x in results) / len(results),
    }
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "teacher_forced_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
    report = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_curriculum_teacher_forced_eval",
        "status": "complete", "branch": branch,
        "target_format": "deltatype_raw_pre", "summary": summary,
        "model_manifest": manifest,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    path = out / "teacher_forced_report.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=("A_to_B", "B_to_B"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_root, args.dataset, args.branch, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
