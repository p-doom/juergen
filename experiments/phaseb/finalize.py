#!/usr/bin/env python3
"""Fail-loud validation and provenance sealing for a Phase-B evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARM_CONTRACT = {
    "prose_keep": {"source_job_id": "135312", "checkpoint": "pb_prose_keep_r32"},
    "prose_strip": {"source_job_id": "135313", "checkpoint": "pb_prose_strip_r32"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_CONTRACT), required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-chat", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract = ARM_CONTRACT[args.arm]
    checkpoint_root = Path(args.source_checkpoint_root).resolve()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    model_dir = Path(args.model_dir).resolve()
    val_chat = Path(args.val_chat).resolve()
    out = Path(args.out).resolve()
    report_path = out / "report.json"
    rows_path = out / "rows.jsonl"

    if args.source_job_id != contract["source_job_id"]:
        raise SystemExit("source job/arm contract mismatch")
    if checkpoint_root.name != contract["checkpoint"]:
        raise SystemExit("source checkpoint/arm contract mismatch")
    if source_checkpoint != checkpoint_root / "000900":
        raise SystemExit("source checkpoint is not the fixed step900 endpoint")
    if model_dir != Path(str(checkpoint_root) + "_hf"):
        raise SystemExit("model is not the exporter-defined sibling of the source checkpoint")
    own_val_fragment = f"/phaseb/{args.arm}/_normalized/val/chat.jsonl"
    if not str(val_chat).endswith(own_val_fragment):
        raise SystemExit("cross-arm validation prompt reuse detected")

    report = load_json(report_path)
    summary = report.get("summary", {})
    meta = report.get("meta", {})
    if meta.get("valid") is not True:
        raise SystemExit("report is not marked valid")
    if Path(meta.get("val_chat", "")).resolve() != val_chat:
        raise SystemExit("report validation path does not match the own-arm path")
    if meta.get("n") != 233 or summary.get("n_rows") != 233:
        raise SystemExit("Phase-B row count is not 233")
    if summary.get("n_coord_records") != 178:
        raise SystemExit("Phase-B coordinate record count is not 178")
    if summary.get("n_request_errors") != 0 or summary.get("request_error_rate") != 0:
        raise SystemExit("request errors invalidate the Phase-B report")

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len(rows) != 233:
        raise SystemExit("rows.jsonl does not contain 233 rows")
    if any(row.get("request_error") is not False for row in rows):
        raise SystemExit("one or more Phase-B rows lacks an explicit zero-error status")
    if any(row.get("teacher_action") is None for row in rows):
        raise SystemExit("one or more Phase-B teacher actions failed to parse")

    config_path = model_dir / "config.json"
    config = load_json(config_path)
    if not config.get("architectures"):
        raise SystemExit("model config lacks architectures")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise SystemExit("no model weights found")

    inventory = []
    for path in weights:
        inventory.append({
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })

    context = {}
    context_path = os.environ.get("LABCTL_CONTEXT")
    if context_path and Path(context_path).is_file():
        raw_context = load_json(Path(context_path))
        context = {
            "run_id": raw_context.get("run_id"),
            "recipe_name": raw_context.get("recipe_name"),
            "source_hash": raw_context.get("source_hash"),
            "recipe_hash": raw_context.get("recipe_hash"),
        }

    manifest = {
        "schema_version": 1,
        "valid": True,
        "arm": args.arm,
        "own_val_contract": {
            "path": str(val_chat),
            "sha256": sha256(val_chat),
            "cross_arm_prompt_reuse": False,
            "n_rows": 233,
            "n_coordinate_rows": 178,
        },
        "source_training": {
            "slurm_job_id": args.source_job_id,
            "checkpoint": str(source_checkpoint),
            "checkpoint_metadata_sha256": sha256(source_checkpoint / "_CHECKPOINT_METADATA"),
            "lora_metadata_sha256": sha256(checkpoint_root / "lora_metadata.json"),
            "training_log": str(Path(args.training_log).resolve()),
            "training_log_sha256": sha256(Path(args.training_log)),
            "training_script": str(Path(args.training_script).resolve()),
            "training_script_sha256": sha256(Path(args.training_script)),
        },
        "model": {
            "path": str(model_dir),
            "config_sha256": sha256(config_path),
            "architectures": config["architectures"],
            "weights": inventory,
        },
        "evaluation": {
            "sampling": {"temperature": 0.0},
            "request_errors": 0,
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "rows": str(rows_path),
            "rows_sha256": sha256(rows_path),
            "evaluator": str(Path(args.evaluator).resolve()),
            "evaluator_sha256": sha256(Path(args.evaluator)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "labctl": context,
    }
    tmp = out / ".eval_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(out / "eval_manifest.json")
    print(f"wrote trusted Phase-B manifest: {out / 'eval_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
