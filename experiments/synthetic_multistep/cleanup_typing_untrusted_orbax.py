#!/usr/bin/env python3
"""Delete only quarantined typing-recovery Orbax clones from exact failed jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED = {
    135567: "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_v1_run_019fb5cc42da7a919c8909a700fa2abd",
    135568: "synthetic_typing_factorial_A_perkey_r256_lr5e5_recovered_deadline0900_v1_run_019fb5cc619575838c583f1e68a65423",
    135569: "synthetic_typing_factorial_B_coalesced_r256_lr5e5_recovered_deadline0900_v1_run_019fb5cc7ea97de0b09ca952e85993f5",
    135570: "synthetic_typing_factorial_B_perkey_r256_lr5e5_recovered_deadline0900_v1_run_019fb5cc9c8072d28d4f8237441fcae3",
    135571: "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_v1_run_019fb5cdf1c47e61b740976e674278c9",
    135572: "synthetic_typing_factorial_A_perkey_r256_lr5e5_recovered_deadline0900_v1_run_019fb5ce10417d71b874e912951c2d1d",
    135573: "synthetic_typing_factorial_B_coalesced_r256_lr5e5_recovered_deadline0900_v1_run_019fb5ce2d8474d3b0ae975735090418",
    135574: "synthetic_typing_factorial_B_perkey_r256_lr5e5_recovered_deadline0900_v1_run_019fb5ce4a8e7533b408a6fd47971508",
    135579: "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_gate_v1_run_019fb5e271f5766280da8e9378f6dcb6",
    135582: "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_gate_v2_run_019fb5eb3f1f7390aabbbddaa6d4bbe6",
    135584: "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_gate_v3_run_019fb5f11c237e939d76067437a0b5d5",
}
PARENTS = {
    "synthetic_typing_factorial_A_coalesced_r256_lr5e5_v1_run_019fb58d8adb7620aa4909251740e263":
        "7214eab7f13bf3556be18ee25b3ec5368fe62ce46e1150c88ec26bba9d6c00ea",
    "synthetic_typing_factorial_A_perkey_r256_lr5e5_v1_run_019fb58da1da7293b72bed4d289ca203":
        "a20f01fe82c02b4cc1c91292cd2a2256340749b231f7a763e9f71ab104694dd8",
    "synthetic_typing_factorial_B_coalesced_r256_lr5e5_v1_run_019fb58db83479f08c78f3a7c1ca3a66":
        "d71b1af18f0fb50f691c7d69ed2a7c5847b6464438b89b2b6df2fe4002867675",
    "synthetic_typing_factorial_B_perkey_r256_lr5e5_v1_run_019fb58dcdf278d19980dd043c67f77a":
        "ff41e60139ae1a060e9e326459c14fe38ae607c264701168806037fca0bc3bb3",
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
    logical = allocated = files = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink inside deletion target: {path}")
            stat = path.stat()
            allocated += stat.st_blocks * 512
            if path.is_file():
                files += 1
                logical += stat.st_size
    root_stat = root.stat()
    allocated += root_stat.st_blocks * 512
    return {"file_count": files, "logical_bytes": logical, "allocated_bytes": allocated}


def retained_hashes(root: Path, deletion_targets: set[Path]) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if not path.is_file() or any(target in path.parents for target in deletion_targets):
            continue
        result[str(path.relative_to(root))] = sha256(path)
    return result


def parent_hashes(base: Path) -> dict[str, str]:
    result = {}
    for alias, expected in PARENTS.items():
        metadata = base / alias / "orbax/000250/_CHECKPOINT_METADATA"
        actual = sha256(metadata)
        if actual != expected:
            raise RuntimeError(f"original parent hash mismatch: {metadata}: {actual}")
        result[alias] = actual
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.checkpoint_root.resolve()
    if base != Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"):
        raise SystemExit(f"refusing unexpected checkpoint root: {base}")

    parents_before = parent_hashes(base)
    records = []
    all_targets: set[Path] = set()
    retained_before = {}
    for job_id, alias in EXPECTED.items():
        root = base / alias
        if root.parent.resolve() != base or not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"invalid quarantined root for {job_id}: {root}")
        marker_path = root / "UNTRUSTED_PARTIAL.json"
        marker = json.loads(marker_path.read_text())
        marker_job = marker.get("failed_job_id", marker.get("job_id"))
        if marker_job != job_id or marker.get("must_not_register_or_use") is not True:
            raise RuntimeError(f"untrusted marker mismatch for {job_id}: {marker}")
        if (root / "typing_train_export_manifest.json").exists():
            raise RuntimeError(f"trusted export manifest unexpectedly exists: {root}")
        candidates = [root / "orbax", root / f".orbax_clone_{job_id}.tmp"]
        existing = [path for path in candidates if path.exists()]
        if len(existing) > 1:
            raise RuntimeError(f"multiple clone trees for {job_id}: {existing}")
        target_records = []
        for target in existing:
            if not target.is_dir() or target.is_symlink() or target.parent != root:
                raise RuntimeError(f"unsafe deletion target: {target}")
            top_entries = {path.name for path in target.iterdir()}
            numeric = {name for name in top_entries if name.isdigit()}
            if numeric - {"000250"}:
                raise RuntimeError(f"later checkpoint found in {target}: {numeric}")
            if top_entries - {"000250", "config.json", "lora_metadata.json"}:
                raise RuntimeError(f"unexpected top-level clone entries in {target}: {top_entries}")
            record = {"path": str(target), **tree_size(target)}
            target_records.append(record)
            all_targets.add(target)
        records.append({
            "job_id": job_id,
            "quarantine_root": str(root),
            "marker_sha256": sha256(marker_path),
            "trusted_export_manifest_absent": True,
            "targets": target_records,
        })

    for job_id, alias in EXPECTED.items():
        root = base / alias
        retained_before[alias] = retained_hashes(root, all_targets)
    before = available(base)
    for target in sorted(all_targets, key=str):
        shutil.rmtree(target)
    after = available(base)

    for record in records:
        root = Path(record["quarantine_root"])
        if not (root / "UNTRUSTED_PARTIAL.json").is_file():
            raise RuntimeError(f"quarantine marker disappeared: {root}")
        for target in record["targets"]:
            if Path(target["path"]).exists():
                raise RuntimeError(f"clone target remained after cleanup: {target['path']}")
        alias = root.name
        if retained_hashes(root, all_targets) != retained_before[alias]:
            raise RuntimeError(f"non-clone quarantine evidence changed: {root}")
    if parent_hashes(base) != parents_before:
        raise RuntimeError("original parent checkpoint changed during cleanup")

    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_typing_untrusted_orbax_cleanup",
        "status": "complete",
        "cpu_only": True,
        "exact_failed_job_ids": sorted(EXPECTED),
        "quarantine_records": records,
        "removed_tree_count": len(all_targets),
        "removed_logical_bytes": sum(
            target["logical_bytes"] for record in records for target in record["targets"]
        ),
        "removed_allocated_bytes": sum(
            target["allocated_bytes"] for record in records for target in record["targets"]
        ),
        "filesystem_available_bytes_before": before,
        "filesystem_available_bytes_after": after,
        "filesystem_available_delta_bytes": after - before,
        "original_parent_checkpoint_hashes_unchanged": parents_before,
        "preserved": "UNTRUSTED_PARTIAL markers, preflights, audits, logs, and all original parent step-250 roots",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "cleanup_manifest.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
