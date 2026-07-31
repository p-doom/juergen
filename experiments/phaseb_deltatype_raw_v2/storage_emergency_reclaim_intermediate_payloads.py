#!/usr/bin/env python3
"""Emergency reclaim of restored intermediate payloads while retaining seals."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
    "franz.srambical"
)
ROOT_NAMES = (
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_production_v1_"
    "run_019fb5c2d5b770719d8aec010bbb7891",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_v1",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_recovery_v2",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_recovery_v3",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_recovery_"
    "v4_memory_safe",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_recovery_"
    "v5_no_cast",
    "phaseb_raw_deltatype_v2_A_to_B_r256_s900_exact_resume_recovery_"
    "v7_no_signal",
)
EXPECTED_METADATA = {
    300: "b36d893969864c858d5d1bde943e39093c46c5942ec6ba168ba2bbcdc8d04417",
    600: "475de36487c52708f4c27091ca2a6d25b2e3789c3c5573a3f559f1b68f18a316",
}
RUN_ROOT = BASE / ROOT_NAMES[-1]
LOG = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs/"
    "franz.srambical/run_019fb71771ee7e30b5259c3735b00587/.lab/"
    "phaseb_raw_deltatype_v2_resume_contingency_step600_recovery_v7_n_135676.log"
)
OUT = RUN_ROOT / "storage_emergency_reclamation.json"
DOWNSTREAM_CONTEXTS = (
    Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs/"
        "franz.srambical/run_019fb74552257ae2ba62f0f420a2dc02/.lab/context.json"
    ),
)
ACTIVE_JOB_STATES = {"RUNNING", "SUSPENDED"}


class ReclamationSafetyError(RuntimeError):
    pass


def validate_reclamation_safety(
    *,
    active_job_state: str,
    durable_resumable_payloads_after: int,
    downstream_references: list[str],
) -> None:
    """Block any cleanup that makes live optimizer memory the sole continuation."""
    if durable_resumable_payloads_after < 0:
        raise ReclamationSafetyError("negative durable checkpoint count")
    if (active_job_state in ACTIVE_JOB_STATES
            and durable_resumable_payloads_after < 1):
        raise ReclamationSafetyError(
            "active job would have no durable resumable checkpoint after cleanup"
        )
    if downstream_references and durable_resumable_payloads_after < 1:
        raise ReclamationSafetyError(
            "downstream export references checkpoint payloads removed by cleanup: "
            + ", ".join(downstream_references)
        )


def active_job_state(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ReclamationSafetyError(
            f"cannot prove active job state for {job_id}: {result.stderr.strip()}"
        )
    states = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(states) != 1:
        raise ReclamationSafetyError(
            f"expected one scheduler state for active job {job_id}, got {sorted(states)}"
        )
    return states.pop()


def downstream_references(roots: tuple[Path, ...]) -> list[str]:
    resolved_roots = {str(root.resolve()) for root in roots}
    references: list[str] = []
    for context_path in DOWNSTREAM_CONTEXTS:
        if not context_path.is_file():
            raise ReclamationSafetyError(
                f"cannot prove downstream-reference safety; context missing: {context_path}"
            )
        context = json.loads(context_path.read_text(encoding="utf-8"))
        for item in context.get("inputs", []):
            resolved_path = str(Path(str(item.get("resolved_path", ""))).resolve())
            if resolved_path in resolved_roots:
                references.append(f"{context_path}:{item.get('role')}")
    return references


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes() -> int:
    stat = os.statvfs(RUN_ROOT)
    return stat.f_bavail * stat.f_frsize


def main() -> int:
    roots = tuple(BASE / name for name in ROOT_NAMES)
    if OUT.exists() or (RUN_ROOT / "train_manifest.json").exists():
        raise SystemExit("FATAL emergency cleanup precondition changed")
    log_text = LOG.read_text(encoding="utf-8")
    if any(token in log_text for token in ("FATAL", "Traceback", "OutOfMemory")):
        raise SystemExit("FATAL active training log contains a fatal marker")
    steps = [int(value) for value in re.findall(r"\bstep=(\d+)\b", log_text)]
    if not steps or max(steps) < 770:
        raise SystemExit("FATAL active exact resume has not passed step 770")
    if list((RUN_ROOT / "orbax").glob("*.orbax-checkpoint-tmp")):
        raise SystemExit("FATAL checkpoint save already in progress")

    inventories: dict[int, list[tuple[str, int, int, int]]] = {}
    unique_payload_blocks: dict[tuple[int, int], int] = {}
    targets: list[Path] = []
    for step, expected_sha in EXPECTED_METADATA.items():
        for root in roots:
            if root.is_symlink() or root.resolve() != root:
                raise SystemExit(f"FATAL unsafe exact root: {root}")
            checkpoint = root / "orbax" / f"{step:06d}"
            metadata = checkpoint / "_CHECKPOINT_METADATA"
            if sha256(metadata) != expected_sha:
                raise SystemExit(f"FATAL metadata seal mismatch: {checkpoint}")
            rows = []
            for path in sorted(checkpoint.rglob("*")):
                if path.is_symlink():
                    raise SystemExit(f"FATAL symlink in checkpoint: {path}")
                if not path.is_file():
                    continue
                stat = path.stat()
                row = (
                    str(path.relative_to(checkpoint)), stat.st_size,
                    stat.st_dev, stat.st_ino,
                )
                rows.append(row)
                if path.name != "_CHECKPOINT_METADATA":
                    unique_payload_blocks[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
            if step not in inventories:
                inventories[step] = rows
            elif inventories[step] != rows:
                raise SystemExit(f"FATAL hardlink inventory mismatch: {checkpoint}")
            targets.extend(path for path in checkpoint.iterdir()
                           if path.name != "_CHECKPOINT_METADATA")

    # Every payload copy above is a hard link to the same durable checkpoint
    # state and every one is targeted. Metadata alone is not restorable. An
    # active process's in-memory optimizer is never counted as durable state.
    validate_reclamation_safety(
        active_job_state=active_job_state("135676"),
        durable_resumable_payloads_after=0,
        downstream_references=downstream_references(roots),
    )

    before = free_bytes()
    payload = {
        "artifact_type": "phaseb_raw_v2_storage_emergency_reclamation",
        "schema_version": 1,
        "status": "validated_pre_delete",
        "reason": "project free space below final checkpoint allocation",
        "science_change": False,
        "active_job_id": "135676",
        "latest_logged_step_before_cleanup": max(steps),
        "preserved_steps": sorted(EXPECTED_METADATA),
        "preserved_metadata_sha256": {
            str(step): value for step, value in EXPECTED_METADATA.items()
        },
        "deleted_content": "checkpoint payloads only; _CHECKPOINT_METADATA retained",
        "root_count": len(roots),
        "roots": [str(root) for root in roots],
        "unique_payload_allocated_bytes": sum(unique_payload_blocks.values()),
        "free_bytes_before": before,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    for step, expected_sha in EXPECTED_METADATA.items():
        for root in roots:
            checkpoint = root / "orbax" / f"{step:06d}"
            if sorted(path.name for path in checkpoint.iterdir()) != ["_CHECKPOINT_METADATA"]:
                raise SystemExit(f"FATAL reclamation postcondition failed: {checkpoint}")
            if sha256(checkpoint / "_CHECKPOINT_METADATA") != expected_sha:
                raise SystemExit(f"FATAL retained metadata changed: {checkpoint}")

    payload["status"] = "complete"
    payload["free_bytes_after"] = free_bytes()
    payload["free_bytes_delta"] = payload["free_bytes_after"] - before
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
