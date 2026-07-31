#!/usr/bin/env python3
"""Fail-loud lineage gate/finalizer for the normalized-v2 production control."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


WARMSTART_MANIFEST_SHA256 = "d37db583163fddf85c40815417b77cafaba42bafaa1ab31382a1d427ab054e71"
WARMSTART_CONFIG_SHA256 = "306e36b825faad2e26a884add51223b67c8ef109521c69c4bb72ddd490e73efa"
WARMSTART_WEIGHT_BYTES = 35_068_587_488
EXPECTED_OUTPUT_SHA256 = {
    "train": "4cc72eb35c845ecd1aad5412ee0872f9be784675452677a88face644236c97aa",
    "val": "b51221df5f044f21092fec6a973c6d8164a7119f2b4971841fc5926df6e9ef7c",
}
LATEST_START = "2026-07-31T04:35:00+02:00"
HARD_DEADLINE = "2026-07-31T09:50:00+02:00"
ALLOWED_NODES = ["hai003", "hai007", "hai008"]
ESTIMAND = {
    "comparison_estimand": "end_to_end_best_pipeline_control",
    "causal_action_semantics_only_claimed": False,
    "warmstart_exposure_matched_to_raw_v2": False,
    "warmstart_exposure_note": (
        "normalized A-to-A starts directly from Phase-A reltool_pre r256 step 750; "
        "raw-v2 A-to-B includes Phase-A step 750 plus a separate 750-step curriculum "
        "warm-start, so this comparison does not isolate action semantics"
    ),
}
PREREGISTRATION = {
    "status": "preregistered",
    "branch": "A_to_A",
    "format": "move_rel_full_v2",
    "warmstart": "Phase-A reltool_pre r256 step 750 merged HF",
    "fresh_lora": True,
    "fresh_optimizer": True,
    "lora_rank": 256,
    "lora_alpha": 256,
    "learning_rate": 0.0001,
    "num_steps": 900,
    "max_length": 16384,
    "save_steps": [300, 600, 900],
    "in_loop_validation": False,
    "external_eval_only": True,
    "allowed_nodes": ALLOWED_NODES,
    "time_limit": "05:15:00",
    "hard_completion_deadline": HARD_DEADLINE,
    "required_actual_start_by": LATEST_START,
    "deadline_amendment": (
        "operational deadline amended from 09:35 to 09:50 before launch; "
        "science contract unchanged; preserves endpoint/report buffer before 10:00"
    ),
    **ESTIMAND,
}


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected object: {path}")
    return value


def validate(
    dataset: Path,
    source: Path,
    *,
    attested_slurm_start: str | None = None,
    attested_slurm_node: str | None = None,
) -> dict[str, Any]:
    token_path = dataset / "tokenization_manifest.json"
    token = load(token_path)
    seal = token.pop("payload_sha256", None)
    if seal != canonical_hash(token):
        raise ContractError("tokenization manifest self-seal failed")
    token["payload_sha256"] = seal
    required = {
        "artifact_type": "phaseb_normalized_move_rel_v2_tokenized",
        "status": "complete",
        "format": "move_rel_full_v2",
        "warmstart_manifest_sha256": WARMSTART_MANIFEST_SHA256,
    }
    bad = {key: (token.get(key), value) for key, value in required.items()
           if token.get(key) != value}
    if bad:
        raise ContractError(f"tokenization contract mismatch: {bad}")
    build_path = dataset / "build_manifest.json"
    build = load(build_path)
    if (token.get("build_manifest_sha256") != sha256(build_path)
            or build.get("output_file_sha256") != EXPECTED_OUTPUT_SHA256
            or build.get("output_tool_calls") != 18483
            or build.get("source_tool_calls") != 11471):
        raise ContractError("full-call build lineage changed")
    for split, count in (("train", 2383), ("val", 233)):
        root = dataset / "normalized_v2" / split
        metadata_path = root / "metadata.json"
        metadata = load(metadata_path)
        sealed = token.get("tokenized", {}).get(split, {})
        if (metadata.get("num_records") != count
                or metadata.get("max_length") != 16384
                or sealed.get("metadata_sha256") != sha256(metadata_path)):
            raise ContractError(f"tokenized {split} contract changed")
        shard = root / str(sealed.get("compiled_shard", ""))
        if not shard.is_file() or sealed.get("compiled_shard_sha256") != sha256(shard):
            raise ContractError(f"tokenized {split} shard seal failed")
    vision_path = dataset / "vision_budget_preflight.json"
    vision = load(vision_path)
    if (vision.get("status") != "pass" or vision.get("records_scanned") != 2616
            or vision.get("configured") != {"max_images": 29, "max_patches": 64000}
            or token.get("vision_budget_preflight_sha256") != sha256(vision_path)):
        raise ContractError("vision budget preflight changed")
    manifest_path = source / "train_export_manifest.json"
    hf = source / "hf"
    if (sha256(manifest_path) != WARMSTART_MANIFEST_SHA256
            or sha256(hf / "config.json") != WARMSTART_CONFIG_SHA256
            or (hf / "model.safetensors").stat().st_size != WARMSTART_WEIGHT_BYTES):
        raise ContractError("Phase-A normalized warm-start changed")
    warm = load(manifest_path)
    if not (warm.get("status") == "complete" and warm.get("arm") == "reltool_pre"
            and warm.get("step") == 750 and warm.get("lora_rank") == 256
            and warm.get("lora_alpha") == 256):
        raise ContractError("wrong normalized warm-start lineage")
    node = attested_slurm_node or os.environ.get("SLURMD_NODENAME")
    slurm_context = bool(os.environ.get("SLURM_JOB_ID") or attested_slurm_start)
    if slurm_context and node not in ALLOWED_NODES:
        raise ContractError(f"job landed on unauthorized node: {node}")
    actual_start = (
        dt.datetime.fromisoformat(attested_slurm_start)
        if attested_slurm_start
        else dt.datetime.now(dt.timezone(dt.timedelta(hours=2)))
    )
    if slurm_context and actual_start > dt.datetime.fromisoformat(LATEST_START):
        raise ContractError(
            f"actual start {actual_start.isoformat()} is later than {LATEST_START}"
        )
    return {
        "status": "pass",
        "tokenization_manifest_sha256": sha256(token_path),
        "tokenization_manifest_self_seal": seal,
        "build_manifest_sha256": sha256(build_path),
        "warmstart_manifest_sha256": sha256(manifest_path),
        "warmstart_config_sha256": sha256(hf / "config.json"),
        "warmstart_weight_bytes": (hf / "model.safetensors").stat().st_size,
        "node": node,
    }


def finalize(dataset: Path, source: Path, output: Path, preflight: dict[str, Any]) -> None:
    orbax = output / "orbax"
    checkpoint_hashes: dict[str, str] = {}
    for step in (300, 600, 900):
        metadata = orbax / f"{step:06d}" / "_CHECKPOINT_METADATA"
        if not metadata.is_file():
            raise ContractError(f"complete checkpoint missing: {metadata.parent}")
        checkpoint_hashes[str(step)] = sha256(metadata)
    tmp_dirs = sorted(str(path) for path in orbax.glob("*.orbax-checkpoint-tmp"))
    if tmp_dirs:
        raise ContractError(f"temporary checkpoints remain: {tmp_dirs}")
    lora_path = orbax / "lora_metadata.json"
    lora = load(lora_path)
    if int(lora.get("lora_rank", -1)) != 256 or float(lora.get("lora_alpha", -1)) != 256:
        raise ContractError(f"LoRA metadata mismatch: {lora}")
    manifest = {
        "artifact_type": "phaseb_normalized_move_rel_v2_A_to_A_production_control_orbax",
        "schema_version": 1,
        "status": "complete",
        "format": "move_rel_full_v2",
        "source_model": str(source.resolve()),
        "dataset": str(dataset.resolve()),
        "step": 900,
        "save_steps": [300, 600, 900],
        "checkpoint_metadata_sha256": checkpoint_hashes,
        "lora_metadata_sha256": sha256(lora_path),
        "fresh_lora": True,
        "fresh_optimizer": True,
        "lora_rank": 256,
        "lora_alpha": 256,
        "learning_rate": 0.0001,
        "lr_schedule": "wsd",
        "lr_stable_fraction": 0.7,
        "warmup_steps": 30,
        "max_length": 16384,
        "num_loss_tiles": 16,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "seed": 0,
        "in_loop_validation": False,
        "external_validation_contract": "unchanged held-out 233-row own-val evaluation",
        "preregistration": PREREGISTRATION,
        **ESTIMAND,
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
    args = parser.parse_args()
    try:
        preflight = validate(args.dataset, args.source_model)
        args.output.mkdir(parents=True, exist_ok=True)
        prereg_path = args.output / "preregistration.json"
        preflight_path = args.output / "training_preflight.json"
        if args.stage == "preflight":
            prereg_path.write_text(json.dumps(PREREGISTRATION, indent=2, sort_keys=True) + "\n")
            preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
        else:
            if load(prereg_path) != PREREGISTRATION:
                raise ContractError("preregistration changed during training")
            if load(preflight_path) != preflight:
                raise ContractError("saved preflight changed during training")
            finalize(args.dataset, args.source_model, args.output, preflight)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL normalized-v2 production-control contract: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
