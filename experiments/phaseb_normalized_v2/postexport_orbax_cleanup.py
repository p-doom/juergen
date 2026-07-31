#!/usr/bin/env python3
"""Delete only normalized-v2 Orbax steps after exact HF/eval validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


EXPECTED = {
    "original": (
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
        "franz.srambical/phaseb_normalized_v2_A_to_A_r256_s900_"
        "production_control_v1_run_019fb5faf966715194bd16bfeee051cd"
    ),
    "recovery": (
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
        "franz.srambical/phaseb_normalized_v2_A_to_A_r256_s900_"
        "manifest_recovery_v1"
    ),
    "export": (
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
        "franz.srambical/phaseb_normalized_v2_A_to_A_r256_s900_hf_v1"
    ),
    "evaluation": (
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/eval_logs/"
        "franz.srambical/phaseb_normalized_v2_eval_A_to_A_r256_s900_v1_"
        "run_019fb71df66776f383d630f5d5763095"
    ),
}
STEPS = (300, 600, 900)


class CleanupError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CleanupError(f"expected JSON object: {path}")
    return value


def exact_root(label: str, supplied: Path) -> Path:
    root = supplied.resolve()
    if root != Path(EXPECTED[label]).resolve() or supplied.is_symlink():
        raise CleanupError(f"unsafe or unexpected {label} root: {supplied}")
    return root


def tree_inventory(root: Path) -> tuple[list[list[Any]], dict[tuple[int, int], int]]:
    rows = []
    physical = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append([
            str(path.relative_to(root)), stat.st_size, stat.st_dev, stat.st_ino,
            stat.st_nlink,
        ])
        physical[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
    if not rows:
        raise CleanupError(f"empty cleanup target: {root}")
    return rows, physical


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    original = exact_root("original", args.original)
    recovery = exact_root("recovery", args.recovery)
    export = exact_root("export", args.export)
    evaluation = exact_root("evaluation", args.evaluation)

    recovery_manifest_path = recovery / "train_manifest.json"
    export_manifest_path = export / "export_manifest.json"
    eval_manifest_path = evaluation / "eval_manifest.json"
    report_path = evaluation / "report.json"
    rows_path = evaluation / "rows.jsonl"
    recovery_manifest = load(recovery_manifest_path)
    export_manifest = load(export_manifest_path)
    eval_manifest = load(eval_manifest_path)
    report = load(report_path)
    if (
        recovery_manifest.get("status") != "complete"
        or recovery_manifest.get("step") != 900
        or recovery_manifest.get("manifest_recovery", {}).get("status") != "pass"
        or export_manifest.get("status") != "complete"
        or export_manifest.get("step") != 900
        or export_manifest.get("train_manifest_sha256") != sha256(recovery_manifest_path)
        or eval_manifest.get("status") != "complete"
        or eval_manifest.get("valid") is not True
        or eval_manifest.get("slurm_job_id") != "135683"
        or eval_manifest.get("request_errors") != 0
        or eval_manifest.get("model_manifest_sha256") != sha256(export_manifest_path)
        or eval_manifest.get("report_sha256") != sha256(report_path)
        or eval_manifest.get("rows_sha256") != sha256(rows_path)
        or report.get("valid") is not True
        or report.get("summary", {}).get("n_rows") != 233
        or report.get("summary", {}).get("n_request_errors") != 0
    ):
        raise CleanupError("sealed normalized HF/eval lineage validation failed")
    hf = export / "hf"
    config = hf / "config.json"
    weights = hf / "model.safetensors"
    if (
        sha256(config) != export_manifest.get("config_sha256")
        or weights.stat().st_size != 35_068_587_488
        or export_manifest.get("weights")
        != [{"name": "model.safetensors", "size": 35_068_587_488}]
        or sum(1 for _ in rows_path.open("rb")) != 233
    ):
        raise CleanupError("retained HF model or row-count seal failed")

    retained = {
        "recovery_train_manifest_sha256": sha256(recovery_manifest_path),
        "export_manifest_sha256": sha256(export_manifest_path),
        "hf_config_sha256": sha256(config),
        "eval_manifest_sha256": sha256(eval_manifest_path),
        "eval_report_sha256": sha256(report_path),
        "eval_rows_sha256": sha256(rows_path),
    }
    targets = []
    unique_physical: dict[tuple[int, int], int] = {}
    for step in STEPS:
        left = original / "orbax" / f"{step:06d}"
        right = recovery / "orbax" / f"{step:06d}"
        for target in (left, right):
            if target.is_symlink() or target.resolve().parent != target.parent.resolve():
                raise CleanupError(f"unsafe checkpoint target: {target}")
            metadata = target / "_CHECKPOINT_METADATA"
            expected_hash = recovery_manifest.get("checkpoint_metadata_sha256", {}).get(
                str(step)
            )
            if not metadata.is_file() or sha256(metadata) != expected_hash:
                raise CleanupError(f"checkpoint seal failed: {target}")
        left_rows, left_physical = tree_inventory(left)
        right_rows, right_physical = tree_inventory(right)
        if len(left_rows) != len(right_rows):
            raise CleanupError(f"hardlink inventory count differs at step {step}")
        for left_row, right_row in zip(left_rows, right_rows, strict=True):
            if left_row[:2] != right_row[:2] or left_row[2:4] != right_row[2:4]:
                raise CleanupError(f"original/recovery hardlink parity failed at step {step}")
        unique_physical.update(left_physical)
        unique_physical.update(right_physical)
        targets.extend((left, right))

    inventory = {
        "target_count": len(targets),
        "targets": [str(path) for path in targets],
        "logical_bytes": sum(
            sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
            for target in targets
        ),
        "unique_allocated_bytes": sum(unique_physical.values()),
        "hardlink_parity": True,
    }
    # All validation is complete before the first destructive operation.
    for target in targets:
        shutil.rmtree(target)
    if any(target.exists() for target in targets):
        raise CleanupError("checkpoint deletion postcondition failed")
    for root in (original, recovery):
        remaining = sorted(path.name for path in (root / "orbax").iterdir())
        if remaining != ["config.json", "lora_metadata.json"]:
            raise CleanupError(f"unexpected retained Orbax entries at {root}: {remaining}")
    observed_retained = {
        "recovery_train_manifest_sha256": sha256(recovery_manifest_path),
        "export_manifest_sha256": sha256(export_manifest_path),
        "hf_config_sha256": sha256(config),
        "eval_manifest_sha256": sha256(eval_manifest_path),
        "eval_report_sha256": sha256(report_path),
        "eval_rows_sha256": sha256(rows_path),
    }
    if observed_retained != retained or weights.stat().st_size != 35_068_587_488:
        raise CleanupError("retained artifact changed during cleanup")
    return {
        "artifact_type": "phaseb_normalized_v2_postexport_orbax_cleanup",
        "schema_version": 1,
        "status": "complete",
        "validation": "HF export and final eval sealed before any deletion",
        "deleted": inventory,
        "retained": retained,
        "hf_weight_bytes_retained": weights.stat().st_size,
        "raw_inflight_root_touched": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = cleanup(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    except (CleanupError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL normalized-v2 Orbax cleanup: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
