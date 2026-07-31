#!/usr/bin/env python3
"""Delete only the Orbax clones from six exact quarantined TP2 attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


CHECKPOINT_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"
)
SOURCE_ALIAS = (
    "synthetic_typing_factorial_A_coalesced_r256_lr5e5_v1_"
    "run_019fb58d8adb7620aa4909251740e263"
)
SOURCE_METADATA_SHA256 = (
    "7214eab7f13bf3556be18ee25b3ec5368fe62ce46e1150c88ec26bba9d6c00ea"
)
SOURCE_TREE_SHA256 = (
    "ae55a1544200a1af692f5dcf2e597ec8a8ab0f3dc17830f45ea1b4d68d602833"
)
RESTORED_TREE_SHA256 = (
    "8db25150cf8ed8b93abda3fe1b1fece97fbae596e0e50f84cc18f257fc58d2fe"
)
EXPECTED = {
    135590: (
        "run_019fb624f46c7d51b13554ca5d96f877",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v4_"
        "run_019fb624f46c7d51b13554ca5d96f877",
        "device_preflight_pass",
    ),
    135593: (
        "run_019fb632d92976c1baaf40444ef3ddb3",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v5_"
        "run_019fb632d92976c1baaf40444ef3ddb3",
        "pre_restore_pass",
    ),
    135595: (
        "run_019fb63f393b7ab3a4e3d03a80d8c7ac",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v6_"
        "run_019fb63f393b7ab3a4e3d03a80d8c7ac",
        "pre_restore_pass",
    ),
    135602: (
        "run_019fb655d20c7922ba34fe8a62be800f",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v7_"
        "run_019fb655d20c7922ba34fe8a62be800f",
        "restore_pass",
    ),
    135606: (
        "run_019fb66b74757d129f97679f7b5cedd9",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v8_"
        "run_019fb66b74757d129f97679f7b5cedd9",
        "restore_pass",
    ),
    135617: (
        "run_019fb6824a3b70c1869361278ca4894b",
        "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v9_"
        "run_019fb6824a3b70c1869361278ca4894b",
        "restore_pass",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def tree_size(root: Path) -> dict[str, int]:
    logical = allocated = files = directories = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink inside deletion target: {path}")
            directories += 1
            allocated += path.stat().st_blocks * 512
        for name in filenames:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink inside deletion target: {path}")
            stat = path.stat()
            files += 1
            logical += stat.st_size
            allocated += stat.st_blocks * 512
    allocated += root.stat().st_blocks * 512
    return {
        "file_count": files,
        "directory_count": directories + 1,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def retained_hashes(root: Path, orbax: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if not path.is_file() or path == orbax or orbax in path.parents:
            continue
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in quarantine evidence: {path}")
        result[str(path.relative_to(root))] = sha256(path)
    return result


def validate_marker(marker: dict, job_id: int, run_id: str) -> None:
    if job_id <= 135595:
        required = {
            "failed_job_id": job_id,
            "failed_run_id": run_id,
            "must_not_register_or_use": True,
            "parent_source_mutated": False,
            "training_updates_after_250": 0,
            "trusted_hf_export": False,
        }
    else:
        required = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "quarantined_non_result",
            "post_250_training_steps_completed": 0,
            "first_finite_logged_step": None,
            "source_parent_preserved": True,
            "usable_as_factorial_result": False,
        }
    if any(marker.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"quarantine marker mismatch for job {job_id}: {marker}")


def validate_audit(audit: dict, job_id: int, expected_status: str) -> None:
    if audit.get("status") != expected_status:
        raise RuntimeError(f"TP2 audit status mismatch for job {job_id}: {audit.get('status')}")
    if job_id >= 135602:
        required = {
            "bitwise_leaf_count": 2772,
            "all_train_state_leaves_bitwise_equal_to_cpu_source_restore": True,
            "source_tree_sha256": RESTORED_TREE_SHA256,
            "restored_tree_sha256": RESTORED_TREE_SHA256,
            "restored_counters": {
                "global_gradient_step": 250,
                "optimizer_micro_step": 2000,
                "gradient_accumulation_remainder": 0,
            },
        }
        if any(audit.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"exact restore audit mismatch for job {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = args.checkpoint_root.resolve()
    if base != CHECKPOINT_ROOT:
        raise SystemExit(f"refusing unexpected checkpoint root: {base}")
    source_orbax = base / SOURCE_ALIAS / "orbax"
    source_metadata = source_orbax / "000250/_CHECKPOINT_METADATA"
    source_hash_before = sha256(source_metadata)
    if source_hash_before != SOURCE_METADATA_SHA256:
        raise RuntimeError(f"source checkpoint metadata hash mismatch: {source_hash_before}")

    records = []
    retained_before: dict[str, dict[str, str]] = {}
    targets = []
    for job_id, (run_id, alias, expected_audit_status) in EXPECTED.items():
        root = base / alias
        if root.parent.resolve() != base or not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"invalid exact quarantine root for job {job_id}: {root}")
        marker_path = root / "UNTRUSTED_PARTIAL.json"
        audit_path = root / "tp2_sharding_audit.json"
        clone_path = root / "orbax_clone_manifest.json"
        marker = json.loads(marker_path.read_text())
        audit = json.loads(audit_path.read_text())
        clone = json.loads(clone_path.read_text())
        validate_marker(marker, job_id, run_id)
        validate_audit(audit, job_id, expected_audit_status)

        orbax = root / "orbax"
        expected_clone = {
            "status": "pass",
            "byte_compared_every_file": True,
            "destination_root": str(orbax),
            "source_root": str(source_orbax),
            "parent_source_is_not_training_save_root": True,
            "file_count": 32,
            "tree_sha256": SOURCE_TREE_SHA256,
        }
        if any(clone.get(key) != value for key, value in expected_clone.items()):
            raise RuntimeError(f"clone proof mismatch for job {job_id}: {clone}")
        if not orbax.is_dir() or orbax.is_symlink() or orbax.parent != root:
            raise RuntimeError(f"unsafe Orbax deletion target for job {job_id}: {orbax}")
        top_entries = {path.name for path in orbax.iterdir()}
        if top_entries != {"000250", "config.json", "lora_metadata.json"}:
            raise RuntimeError(f"unexpected Orbax entries for job {job_id}: {top_entries}")
        forbidden = {
            "typing_train_export_manifest.json",
            "typing_eval_manifest.json",
            "final_checkpoint_manifest.json",
            "hf",
            "hf_model",
            "huggingface",
        }
        present_forbidden = forbidden & {path.name for path in root.iterdir()}
        if present_forbidden:
            raise RuntimeError(f"usable output unexpectedly present for job {job_id}: {present_forbidden}")

        retained_before[alias] = retained_hashes(root, orbax)
        size = tree_size(orbax)
        records.append(
            {
                "job_id": job_id,
                "run_id": run_id,
                "quarantine_root": str(root),
                "orbax_target": str(orbax),
                "quarantine_marker_sha256": sha256(marker_path),
                "tp2_audit_sha256": sha256(audit_path),
                "clone_manifest_sha256": sha256(clone_path),
                "clone_proof_status": "pass",
                "no_post_250_checkpoint_or_usable_export": True,
                **size,
            }
        )
        targets.append((alias, root, orbax))

    before = available(base)
    for _, _, orbax in targets:
        shutil.rmtree(orbax)
        orbax.mkdir(mode=0o755)
    after = available(base)

    for alias, root, orbax in targets:
        if not orbax.is_dir() or any(orbax.iterdir()):
            raise RuntimeError(f"Orbax target was not replaced by an empty directory: {orbax}")
        if retained_hashes(root, orbax) != retained_before[alias]:
            raise RuntimeError(f"retained quarantine evidence changed: {root}")
    source_hash_after = sha256(source_metadata)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source step-250 checkpoint metadata changed during cleanup")

    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_typing_tp2_quarantined_orbax_cleanup",
        "status": "complete",
        "cpu_only": True,
        "exact_failed_job_ids": sorted(EXPECTED),
        "removed_tree_count": len(records),
        "removed_logical_bytes": sum(record["logical_bytes"] for record in records),
        "removed_allocated_bytes": sum(record["allocated_bytes"] for record in records),
        "filesystem_available_bytes_before": before,
        "filesystem_available_bytes_after": after,
        "filesystem_available_delta_bytes": after - before,
        "source_step_250_metadata_sha256_before": source_hash_before,
        "source_step_250_metadata_sha256_after": source_hash_after,
        "source_step_250_preserved": True,
        "quarantine_records": records,
        "preserved": (
            "all run-root logs, audits, quarantine and clone-proof markers, the source step-250 "
            "checkpoint, and every non-allowlisted path; each deleted Orbax path was recreated empty"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "cleanup_manifest.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
