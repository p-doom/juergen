#!/usr/bin/env python3
"""Preflight and finalizer for the Phase-B raw-v2 LoRA SFT.

``preflight`` asserts the tokenized dataset and warm-start endpoint are still
byte-identical to what ``tokenize_authorize.py`` sealed; ``finalize`` asserts
every declared save step produced a complete orbax checkpoint at the declared
LoRA shape and writes the labctl artifact manifest.

Shapes and LoRA geometry come from ``train_production_r256.sh``, which is the
single source of truth for the hyperparameters; the manifest pins that script's
SHA-256 so the exact invocation stays recoverable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tokenize_authorize import load, sha256


class ContractError(RuntimeError):
    pass


def expect(observed: dict[str, Any], expected: dict[str, Any], what: str) -> None:
    bad = {
        key: (observed.get(key), value)
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if bad:
        raise ContractError(f"{what}: {bad}")


def validate(
    dataset: Path,
    source: Path,
    *,
    records: dict[str, int],
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
) -> dict[str, Any]:
    token_path = dataset / "tokenization_manifest.json"
    token = load(token_path)
    expect(
        token,
        {
            "artifact_type": "phaseb_raw_deltatype_v2_tokenized_authorized",
            "status": "complete",
            "format": "deltatype_raw_v2",
        },
        "tokenization manifest",
    )
    for split, count in records.items():
        metadata_path = dataset / "raw_v2" / split / "metadata.json"
        metadata = load(metadata_path)
        if metadata.get("num_records") != count or metadata.get("max_length") != max_length:
            raise ContractError(
                f"tokenized {split} shape {metadata} != {count} records / {max_length}"
            )
        sealed = token.get("tokenized", {}).get(split, {}).get("metadata_sha256")
        if sealed != sha256(metadata_path):
            raise ContractError(f"tokenized {split} metadata changed after sealing")
    vision_path = dataset / "vision_budget_preflight.json"
    vision = load(vision_path)
    if vision.get("status") != "pass":
        raise ContractError(f"vision budget preflight did not pass: {vision}")
    if token.get("vision_budget_preflight_sha256") != sha256(vision_path):
        raise ContractError("vision budget preflight changed after sealing")
    source_manifest_path = source / "curriculum_train_export_manifest.json"
    source_manifest = load(source_manifest_path)
    expect(
        source_manifest,
        {
            "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
            "status": "complete",
            "hf_subdir": "hf",
        },
        "warm-start manifest",
    )
    if token.get("source_model_manifest_sha256") != sha256(source_manifest_path):
        raise ContractError("warm start is not the endpoint the tokenizer sealed")
    hf = source / "hf"
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.json", "preprocessor_config.json"):
        if not (hf / name).is_file():
            raise ContractError(f"warm-start runtime file missing: {name}")
    weights = sorted(hf.glob("*.safetensors"))
    if not weights:
        raise ContractError(f"warm start has no safetensors: {hf}")
    return {
        "status": "pass",
        "tokenization_manifest_sha256": sha256(token_path),
        "vision_budget_preflight_sha256": sha256(vision_path),
        "warmstart_manifest_sha256": sha256(source_manifest_path),
        "warmstart_config_sha256": sha256(hf / "config.json"),
        "warmstart_weight_bytes": sum(path.stat().st_size for path in weights),
        "records": records,
        "max_length": max_length,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }


def finalize(
    dataset: Path,
    source: Path,
    output: Path,
    preflight: dict[str, Any],
    *,
    num_steps: int,
    save_steps: list[int],
    training_script: Path,
) -> None:
    orbax = output / "orbax"
    checkpoint_hashes: dict[str, str] = {}
    for step in save_steps:
        metadata = orbax / f"{step:06d}" / "_CHECKPOINT_METADATA"
        if not metadata.is_file():
            raise ContractError(f"incomplete checkpoint: {metadata.parent}")
        checkpoint_hashes[str(step)] = sha256(metadata)
    surviving = sorted(str(path) for path in orbax.glob("*.orbax-checkpoint-tmp*"))
    if surviving:
        raise ContractError(f"temporary checkpoints remain: {surviving}")
    lora_path = orbax / "lora_metadata.json"
    lora = load(lora_path)
    if (int(lora.get("lora_rank", -1)) != preflight["lora_rank"]
            or float(lora.get("lora_alpha", -1)) != float(preflight["lora_alpha"])):
        raise ContractError(f"LoRA metadata is not the declared geometry: {lora}")
    manifest = {
        "artifact_type": "phaseb_raw_deltatype_v2_production_orbax",
        "schema_version": 1,
        "status": "complete",
        "format": "deltatype_raw_v2",
        "source_model": str(source.resolve()),
        "dataset": str(dataset.resolve()),
        "step": num_steps,
        "save_steps": save_steps,
        "checkpoint_metadata_sha256": checkpoint_hashes,
        "lora_metadata_sha256": sha256(lora_path),
        "lora_rank": preflight["lora_rank"],
        "lora_alpha": preflight["lora_alpha"],
        "max_length": preflight["max_length"],
        "training_script_sha256": sha256(training_script),
        "preflight": preflight,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    (output / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "finalize"), required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--train-records", type=int, required=True)
    parser.add_argument("--val-records", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument(
        "--save-steps", required=True,
        help="comma-separated orbax save steps that must exist at finalize",
    )
    args = parser.parse_args()
    try:
        save_steps = [int(part) for part in args.save_steps.split(",") if part]
        if not save_steps or args.num_steps not in save_steps:
            raise ContractError(
                f"--save-steps {save_steps} must be non-empty and contain "
                f"--num-steps {args.num_steps}"
            )
        preflight = validate(
            args.dataset,
            args.source_model,
            records={"train": args.train_records, "val": args.val_records},
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        preflight_path = args.output / "training_preflight.json"
        if args.stage == "preflight":
            preflight_path.write_text(
                json.dumps(preflight, indent=2, sort_keys=True) + "\n")
        else:
            if load(preflight_path) != preflight:
                raise ContractError("saved training preflight changed before finalization")
            finalize(
                args.dataset,
                args.source_model,
                args.output,
                preflight,
                num_steps=args.num_steps,
                save_steps=save_steps,
                training_script=args.training_script,
            )
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL raw-v2 production training contract: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
