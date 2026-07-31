#!/usr/bin/env python3
"""Delete only the 12 sealed TP1-v10 typing Orbax step directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from postexport_orbax_cleanup import _validate_safetensors


class CleanupError(RuntimeError):
    pass


EXPECTED_STEPS = ("000250", "000500", "000750")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CleanupError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def du(path: Path, apparent: bool) -> int:
    command = ["du", "-sx", "--block-size=1"]
    if apparent:
        command.append("--apparent-size")
    command.append(str(path))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(result.stdout.split(maxsplit=1)[0])


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise CleanupError(f"hash mismatch for {path}: {actual} != {expected}")


def producer_complete(job_id: int) -> None:
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", str(job_id), "-o", "JobIDRaw,State,ExitCode"],
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line]
    exact = [row for row in rows if row[0] == str(job_id)]
    if exact != [[str(job_id), "COMPLETED", "0:0"]]:
        raise CleanupError(f"producer {job_id} is not uniquely COMPLETED 0:0: {exact}")


def active_job_check(targets: list[Path], producer_jobs: set[int]) -> list[dict[str, str]]:
    user = os.environ.get("USER")
    if not user:
        raise CleanupError("USER is unset")
    result = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%A|%j|%T|%o|%Z"],
        check=True, capture_output=True, text=True,
    )
    snapshot = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 4)
        if len(fields) != 5:
            raise CleanupError(f"cannot parse squeue row: {line}")
        job_id, name, state, command, workdir = fields
        if job_id == os.environ.get("SLURM_JOB_ID"):
            continue
        if job_id.isdigit() and int(job_id) in producer_jobs:
            raise CleanupError(f"producer job is active: {job_id}")
        detail = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=True, capture_output=True, text=True,
        ).stdout
        combined = " ".join((name, command, workdir, detail))
        if any(str(target) in combined for target in targets):
            raise CleanupError(f"active job {job_id} references an exact deletion target")
        snapshot.append({"job_id": job_id, "name": name, "state": state})
    return snapshot


def validate_entry(entry: dict[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    root = Path(entry["root"])
    if root.parent != checkpoint_root or root.name != entry["artifact"]:
        raise CleanupError(f"root is outside exact checkpoint namespace: {root}")
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise CleanupError(f"unsafe artifact root: {root}")
    meta_path = root / ".meta.json"
    manifest_path = root / "typing_train_export_manifest.json"
    require_hash(meta_path, entry["meta_sha256"])
    require_hash(manifest_path, entry["manifest_sha256"])
    meta = load_json(meta_path)
    if (meta.get("id") != entry["artifact_id"]
            or meta.get("producer_run_id") != entry["run_id"]
            or meta.get("alias") != entry["artifact"]):
        raise CleanupError(f"artifact identity changed: {root}")
    producer_complete(entry["job_id"])

    manifest = load_json(manifest_path)
    fixed = {
        "artifact_type": "synthetic_typing_factorial_hf_checkpoint",
        "schema_version": 2, "status": "complete", "lineage": entry["lineage"],
        "target_format": entry["format"], "step": 750, "fresh_optimizer": False,
        "exact_resume_from_step": 250,
        "recovery_change": "in_loop_validation_disabled_only",
        "sealed_parent_orbax_unchanged": True, "hf_subdir": "hf",
    }
    bad = {key: (manifest.get(key), value) for key, value in fixed.items()
           if manifest.get(key) != value}
    if bad:
        raise CleanupError(f"incomplete export manifest {root}: {bad}")
    if (manifest.get("training_topology") != {
            "tp_size": 1, "fsdp_size": 1, "dp_size": 1,
            "global_batch_size": 1, "gradient_accumulation_steps": 8,
            "unchanged_from_parent": True,
            }
            or manifest.get("logged_finite_recovery_steps") != list(range(260, 751, 10))
            or manifest.get("production_restore_gate", {}).get("status") != "pass"
            or manifest.get("lora_base_frozen_gate", {}).get("status") != "pass"):
        raise CleanupError(f"export proof gates failed: {root}")

    hf = root / "hf"
    model = hf / "model.safetensors"
    config = hf / "config.json"
    for path in (hf, model, config, hf / "tokenizer_config.json", hf / "chat_template.json"):
        if path.is_symlink() or not path.exists():
            raise CleanupError(f"missing or unsafe retained HF component: {path}")
    require_hash(config, entry["config_sha256"])
    if model.stat().st_size != entry["model_size"]:
        raise CleanupError(f"HF model size changed: {model}")
    if manifest.get("weights") != [{"name": "model.safetensors", "size": entry["model_size"]}]:
        raise CleanupError(f"manifest weight inventory changed: {root}")
    safetensors = _validate_safetensors(model)

    orbax = root / "orbax"
    if orbax.is_symlink() or orbax.resolve(strict=True) != orbax:
        raise CleanupError(f"unsafe Orbax root: {orbax}")
    actual_steps = sorted(path.name for path in orbax.iterdir() if path.is_dir())
    if actual_steps != list(EXPECTED_STEPS):
        raise CleanupError(f"unexpected Orbax step inventory at {orbax}: {actual_steps}")
    audited_steps = entry.get("steps")
    if not isinstance(audited_steps, list) or [item.get("name") for item in audited_steps] != list(EXPECTED_STEPS):
        raise CleanupError(f"invalid allowlist step inventory: {root}")
    targets = []
    for item in audited_steps:
        target = orbax / item["name"]
        if target.is_symlink() or target.resolve(strict=True) != target or target.parent != orbax:
            raise CleanupError(f"unsafe target: {target}")
        for required in (target / "_CHECKPOINT_METADATA", target / "train_state", target / "input_iter"):
            if not required.exists():
                raise CleanupError(f"incomplete Orbax checkpoint: {required}")
        require_hash(target / "_CHECKPOINT_METADATA", item["metadata_sha256"])
        allocated, logical = du(target, False), du(target, True)
        if (allocated, logical) != (item["allocated_bytes"], item["logical_bytes"]):
            raise CleanupError(f"Orbax byte inventory changed: {target}")
        targets.append({"path": str(target), "allocated_bytes": allocated,
                        "logical_bytes": logical, "metadata_sha256": item["metadata_sha256"]})
    return {
        "artifact": entry["artifact"], "artifact_id": entry["artifact_id"],
        "run_id": entry["run_id"], "job_id": entry["job_id"],
        "root": str(root), "manifest_sha256": entry["manifest_sha256"],
        "meta_sha256": entry["meta_sha256"], "config_sha256": entry["config_sha256"],
        "model_size": entry["model_size"], "safetensors": safetensors,
        "targets": targets,
    }


def cleanup(allowlist_path: Path, expected_sha: str, out: Path,
            *, validate_only: bool = False) -> dict[str, Any]:
    require_hash(allowlist_path, expected_sha)
    allowlist = load_json(allowlist_path)
    if (allowlist.get("schema_version") != 1
            or allowlist.get("artifact_type") != "typing_tp1_v10_orbax_cleanup_allowlist"):
        raise CleanupError("wrong allowlist schema")
    entries = allowlist.get("entries")
    if not isinstance(entries, list) or len(entries) != 4:
        raise CleanupError("allowlist must contain exactly four streams")
    checkpoint_root = Path(allowlist["checkpoint_root"])
    if checkpoint_root.resolve(strict=True) != checkpoint_root:
        raise CleanupError("unsafe checkpoint root")
    validated = [validate_entry(entry, checkpoint_root) for entry in entries]
    targets = [Path(item["path"]) for stream in validated for item in stream["targets"]]
    if len(targets) != 12 or len(set(targets)) != 12:
        raise CleanupError("cleanup must resolve to exactly 12 unique targets")
    allocated = sum(item["allocated_bytes"] for stream in validated for item in stream["targets"])
    logical = sum(item["logical_bytes"] for stream in validated for item in stream["targets"])
    if (allocated != allowlist.get("expected_allocated_bytes")
            or logical != allowlist.get("expected_logical_bytes")):
        raise CleanupError("allowlist byte totals changed")
    active = active_job_check(targets, {entry["job_id"] for entry in entries})
    free_before = shutil.disk_usage(checkpoint_root).free
    result = {
        "schema_version": 1, "artifact_type": "typing_tp1_v10_orbax_cleanup",
        "status": "validated_pre_delete", "authorization": allowlist.get("authorization"),
        "allowlist_sha256": expected_sha, "target_count": 12, "stream_count": 4,
        "allocated_bytes_validated": allocated, "logical_bytes_validated": logical,
        "filesystem_free_bytes_before": free_before, "active_slurm_jobs_checked": active,
        "streams": validated, "deleted_paths": [],
        "retained_hf_roots": [str(Path(stream["root"]) / "hf") for stream in validated],
    }
    atomic_json(out, result)
    if validate_only:
        return result
    for target in targets:
        shutil.rmtree(target)
        if target.exists() or target.is_symlink():
            raise CleanupError(f"deletion failed: {target}")
        result["status"] = "deleting"
        result["deleted_paths"].append(str(target))
        result["filesystem_free_bytes_current"] = shutil.disk_usage(checkpoint_root).free
        atomic_json(out, result)
    for stream in validated:
        root = Path(stream["root"])
        require_hash(root / "typing_train_export_manifest.json", stream["manifest_sha256"])
        if (root / "hf/model.safetensors").stat().st_size != stream["model_size"]:
            raise CleanupError(f"retained HF changed after deletion: {root}")
    result["status"] = "complete"
    result["completed_unix"] = int(time.time())
    result["deleted_allocated_bytes"] = allocated
    result["deleted_logical_bytes"] = logical
    result["filesystem_free_bytes_after"] = shutil.disk_usage(checkpoint_root).free
    result["filesystem_free_bytes_delta"] = result["filesystem_free_bytes_after"] - free_before
    result["all_hf_exports_and_manifests_retained"] = True
    atomic_json(out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--expected-allowlist-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(cleanup(args.allowlist.resolve(strict=True), args.expected_allowlist_sha256,
                                 args.out, validate_only=args.validate_only),
                         indent=2, sort_keys=True))
    except (CleanupError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FATAL typing TP1-v10 Orbax cleanup: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
