#!/usr/bin/env python3
"""Audit a Phase-B raw-v2 dataset and seal its tokenized records.

Dataset *identity* is the recipe's job (`[inputs.dataset]` pins the build run)
and the test suite's job (`test_full_source.py` asserts the sealed split
digests). This stage asserts only the invariants nothing else guards: the
manifest's self-seal, the record schema, and that every action label is an exact
byte suffix of its assistant message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MAX_LENGTH = 16384
VISION_BUDGET = {"max_images": 29, "max_patches": 64000}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def text(message: dict[str, Any]) -> str:
    parts = message.get("content")
    if not isinstance(parts, list):
        raise AuditError("message content is not a list")
    values = [part.get("text") for part in parts
              if isinstance(part, dict) and part.get("type") == "text"]
    if len(values) != 1 or not isinstance(values[0], str):
        raise AuditError("message does not have exactly one text part")
    return values[0]


def expect(observed: dict[str, Any], required: dict[str, Any], what: str) -> None:
    bad = {key: (observed.get(key), value) for key, value in required.items()
           if observed.get(key) != value}
    if bad:
        raise AuditError(f"{what}: {bad}")


def audit_split(path: Path, split: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    order: list[Any] = []
    images: list[Any] = []
    users: list[Any] = []
    prose: list[Any] = []
    actions: list[Any] = []
    for row in rows:
        sample_id = row.get("sample_id")
        if row.get("format") != "deltatype_raw_v2" or not isinstance(sample_id, str):
            raise AuditError(f"{split}: invalid record identity/format")
        order.append([row.get(key) for key in
                      ("sample_id", "recording_id", "app", "task_id", "step")])
        messages = row.get("messages")
        audits = row.get("raw_deltatype_v2_audit")
        if not isinstance(messages, list) or not isinstance(audits, list):
            raise AuditError(f"{split}: missing messages/action audit")
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if len(assistants) != len(audits):
            raise AuditError(f"{split}: assistant/audit count mismatch")
        for message in messages:
            if message.get("role") == "user":
                users.append([sample_id, message.get("content")])
                for part in message.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "image":
                        images.append([sample_id, part.get("image")])
        for message, action in zip(assistants, audits, strict=True):
            label = action.get("label")
            body = text(message)
            if not isinstance(label, str) or not body.endswith(label):
                raise AuditError(f"{split}: action span is not an exact assistant suffix")
            prose.append([sample_id, action.get("mapped_step"), body[:-len(label)]])
            actions.append([sample_id, action.get("mapped_step"), label,
                            action.get("source_sequence"), action.get("command_plan")])
    return {
        "records": len(rows),
        "file_sha256": sha256(path),
        "order_sha256": canonical_hash(order),
        "image_refs_sha256": canonical_hash(images),
        "user_content_sha256": canonical_hash(users),
        "natural_prose_sha256": canonical_hash(prose),
        "action_spans_sha256": canonical_hash(actions),
        "assistant_spans": len(actions),
        "image_references": len(images),
    }


def validate_inputs(dataset: Path, source: Path) -> dict[str, Any]:
    dataset_manifest_path = dataset / "dataset_manifest.json"
    source_manifest_path = source / "curriculum_train_export_manifest.json"
    dataset_manifest = load(dataset_manifest_path)
    payload = dataset_manifest.pop("payload_sha256", None)
    if payload != canonical_hash(dataset_manifest):
        raise AuditError("dataset payload seal failed")
    expect(
        dataset_manifest,
        {
            "artifact_type": "phaseb_raw_deltatype_v2_dataset",
            "status": "complete",
            "format": "deltatype_raw_v2",
            "production_gpu_training_authorized": False,
        },
        "dataset contract changed",
    )
    expect(
        load(source_manifest_path),
        {
            "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
            "status": "complete", "branch": "A_to_B",
            "target_format": "deltatype_raw_pre", "hf_subdir": "hf",
            "fresh_optimizer": True,
        },
        "warm-start lineage changed",
    )
    hf = source / "hf"
    for name in ("config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.json",
                 "preprocessor_config.json"):
        if not (hf / name).is_file():
            raise AuditError(f"warm-start runtime file missing: {name}")
    return {
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "warmstart_manifest_sha256": sha256(source_manifest_path),
        "warmstart_config_sha256": sha256(hf / "config.json"),
        "warmstart_weight_bytes": (hf / "model.safetensors").stat().st_size,
        "splits": {split: audit_split(dataset / split / "chat.jsonl", split)
                   for split in ("train", "val")},
    }


def finalize(dataset: Path, source: Path, output: Path, audit: dict[str, Any]) -> None:
    tokenized: dict[str, Any] = {}
    for split, split_audit in audit["splits"].items():
        metadata_path = output / "raw_v2" / split / "metadata.json"
        metadata = load(metadata_path)
        if (metadata.get("num_records") != split_audit["records"]
                or metadata.get("max_length") != MAX_LENGTH):
            raise AuditError(f"{split}: tokenized metadata mismatch: {metadata}")
        tokenized[split] = {
            "metadata_sha256": sha256(metadata_path),
            "num_records": split_audit["records"],
            "max_length": MAX_LENGTH,
        }
    vision_path = output / "vision_budget_preflight.json"
    vision = load(vision_path)
    records = sum(split["records"] for split in audit["splits"].values())
    if (vision.get("status") != "pass"
            or vision.get("records_scanned") != records
            or vision.get("configured") != VISION_BUDGET):
        raise AuditError(f"full-scan vision preflight failed: {vision}")
    manifest = {
        "artifact_type": "phaseb_raw_deltatype_v2_tokenized_authorized",
        "schema_version": 1, "status": "complete", "format": "deltatype_raw_v2",
        "source_dataset": str(dataset.resolve()), "source_model": str(source.resolve()),
        "source_dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "source_model_manifest_sha256": audit["warmstart_manifest_sha256"],
        "raw_audit": audit["splits"], "tokenized": tokenized,
        "vision_budget_preflight_sha256": sha256(vision_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path = output / "tokenization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "finalize"), required=True)
    args = parser.parse_args()
    try:
        audit = validate_inputs(args.dataset, args.source_model)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.stage == "preflight":
            (args.output / "raw_content_audit.json").write_text(
                json.dumps({"status": "pass", **audit}, indent=2, sort_keys=True) + "\n")
        else:
            finalize(args.dataset, args.source_model, args.output, audit)
    except (AuditError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL raw-v2 tokenization audit: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
