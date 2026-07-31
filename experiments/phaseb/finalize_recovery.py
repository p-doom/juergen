#!/usr/bin/env python3
"""Seal an own-val eval whose HF model came from the audited CPU recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CONTRACT = {
    "prose_keep": ("135312", "pb_prose_keep_r32"),
    "prose_strip": ("135313", "pb_prose_strip_r32"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(CONTRACT), required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--val-chat", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    expected_job, expected_root = CONTRACT[args.arm]
    source = args.source_checkpoint_root.resolve()
    checkpoint = source / "000900"
    model_artifact = args.model_artifact.resolve()
    model = model_artifact / "hf"
    export_path = model_artifact / "export_manifest.json"
    val = args.val_chat.resolve()
    out = args.out.resolve()
    if (args.source_job_id != expected_job or source.name != expected_root
            or not str(val).endswith(f"/phaseb/{args.arm}/_normalized/val/chat.jsonl")):
        raise SystemExit("FATAL arm/source/own-val contract mismatch")
    export = json.loads(export_path.read_text())
    if (export.get("artifact_type") != "phaseb_absolute_hf_checkpoint"
            or export.get("status") != "complete" or export.get("arm") != args.arm
            or export.get("source_training_job_id") != expected_job
            or export.get("source_training_state")
            != "FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT"
            or Path(export.get("source_checkpoint", "")).resolve() != checkpoint
            or export.get("step") != 900):
        raise SystemExit("FATAL recovery export provenance mismatch")
    report_path, rows_path = out / "report.json", out / "rows.jsonl"
    report = json.loads(report_path.read_text())
    summary, meta = report.get("summary", {}), report.get("meta", {})
    if (meta.get("valid") is not True or Path(meta.get("val_chat", "")).resolve() != val
            or meta.get("n") != 233 or summary.get("n_rows") != 233
            or summary.get("n_coord_records") != 178
            or summary.get("n_request_errors") != 0
            or summary.get("request_error_rate") != 0):
        raise SystemExit("FATAL evaluation report contract mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if (len(rows) != 233 or any(row.get("request_error") is not False for row in rows)
            or any(row.get("teacher_action") is None for row in rows)):
        raise SystemExit("FATAL evaluation rows are incomplete or invalid")
    config = json.loads((model / "config.json").read_text())
    if not config.get("architectures"):
        raise SystemExit("FATAL model config lacks architectures")
    weights = sorted(model.glob("*.safetensors"))
    expected_weights = {item["name"]: item for item in export.get("weights", [])}
    inventory = []
    for path in weights:
        item = {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        if expected_weights.get(path.name) != item:
            raise SystemExit(f"FATAL recovery weight disagrees with export manifest: {path}")
        inventory.append(item)
    if not inventory or set(expected_weights) != {item["name"] for item in inventory}:
        raise SystemExit("FATAL recovery weight inventory mismatch")
    manifest = {
        "schema_version": 2, "valid": True, "arm": args.arm,
        "own_val_contract": {"path": str(val), "sha256": sha256(val),
                             "cross_arm_prompt_reuse": False, "n_rows": 233,
                             "n_coordinate_rows": 178},
        "source_training": {
            "slurm_job_id": expected_job,
            "state": "FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT",
            "checkpoint": str(checkpoint),
            "checkpoint_metadata_sha256": sha256(checkpoint / "_CHECKPOINT_METADATA"),
            "lora_metadata_sha256": sha256(source / "lora_metadata.json"),
            "training_log": str(args.training_log.resolve()),
            "training_log_sha256": sha256(args.training_log),
            "training_script": str(args.training_script.resolve()),
            "training_script_sha256": sha256(args.training_script),
        },
        "model": {"path": str(model), "artifact_root": str(model_artifact),
                  "export_manifest": str(export_path),
                  "export_manifest_sha256": sha256(export_path),
                  "export_slurm_job_id": export.get("export_slurm_job_id"),
                  "config_sha256": sha256(model / "config.json"),
                  "architectures": config["architectures"], "weights": inventory},
        "evaluation": {"sampling": {"temperature": 0.0}, "request_errors": 0,
                       "report": str(report_path), "report_sha256": sha256(report_path),
                       "rows": str(rows_path), "rows_sha256": sha256(rows_path),
                       "evaluator": str(args.evaluator.resolve()),
                       "evaluator_sha256": sha256(args.evaluator),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    }
    tmp = out / ".eval_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(out / "eval_manifest.json")
    print(f"wrote trusted recovered Phase-B eval manifest: {out / 'eval_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
