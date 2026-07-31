#!/usr/bin/env python3
"""Delete Phase-B source Orbax trees only after exports and evals are sealed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class CleanupError(RuntimeError):
    pass


ABSOLUTE_CONTRACT = {
    "absolute_keep": (
        "prose_keep", "135312", "pb_prose_keep_r32",
        "phaseb_absolute_prose_keep_r32_s900_recovery_135312_hf_v1",
    ),
    "absolute_strip": (
        "prose_strip", "135313", "pb_prose_strip_r32",
        "phaseb_absolute_prose_strip_r32_s900_recovery_135313_hf_v1",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CleanupError(f"manifest is not an object: {path}")
    return value


def logical_bytes(root: Path) -> int:
    return sum((Path(directory) / name).stat().st_size
               for directory, _subdirs, files in os.walk(root)
               for name in files)


def completed(job_id: str) -> None:
    if not job_id.isdigit():
        raise CleanupError(f"invalid Slurm job id: {job_id!r}")
    proc = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "JobIDRaw,State"],
        text=True, capture_output=True, check=False,
    )
    if proc.returncode:
        raise CleanupError(f"sacct failed for {job_id}: {proc.stderr.strip()}")
    states = {}
    for line in proc.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2:
            states[fields[0]] = fields[1]
    state = states.get(job_id, "")
    if not state.startswith("COMPLETED"):
        raise CleanupError(f"refusing in-flight/failed job {job_id}: {state or 'unknown'}")


def safe_absolute_root(root: Path, expected_name: str) -> Path:
    root = root.resolve()
    expected_parent = Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/audit_operand/ckpt"
    ).resolve()
    if root.is_symlink() or root.parent != expected_parent or root.name != expected_name:
        raise CleanupError(f"unsafe absolute Orbax root: {root}")
    return root


def safe_relative_root(source: Path) -> Path:
    source = source.resolve()
    expected_parent = Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"
    ).resolve()
    expected_name = (
        "phaseb_relative_prose_keep_r32_s900_visionfixed_v2_"
        "run_019fb4cbe728789086b8a63931c9979e"
    )
    if source.is_symlink() or source.parent != expected_parent or source.name != expected_name:
        raise CleanupError(f"unsafe relative source artifact: {source}")
    target = source / "orbax"
    if target.is_symlink() or target.resolve().parent != source:
        raise CleanupError(f"unsafe relative Orbax root: {target}")
    return target


def validate_absolute(*, label: str, source: Path, evaluation: Path) -> dict[str, Any]:
    arm, source_job, expected_name, expected_model_name = ABSOLUTE_CONTRACT[label]
    source = safe_absolute_root(source, expected_name)
    manifest_path = evaluation.resolve() / "eval_manifest.json"
    manifest = load(manifest_path)
    if manifest.get("valid") is not True or manifest.get("arm") != arm:
        raise CleanupError(f"{label}: eval manifest is not valid {arm}")
    training = manifest.get("source_training", {})
    model = manifest.get("model", {})
    eval_info = manifest.get("evaluation", {})
    model_root = Path(model.get("artifact_root", "")).resolve()
    expected_model_parent = Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"
    ).resolve()
    export_path = model_root / "export_manifest.json"
    export = load(export_path)
    if (training.get("slurm_job_id") != source_job
            or Path(training.get("checkpoint", "")).resolve() != source / "000900"
            or training.get("state") != "FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT"
            or model_root.parent != expected_model_parent
            or model_root.name != expected_model_name
            or Path(model.get("path", "")).resolve() != model_root / "hf"
            or eval_info.get("request_errors") != 0):
        raise CleanupError(f"{label}: source/export/eval provenance mismatch")
    if (export.get("artifact_type") != "phaseb_absolute_hf_checkpoint"
            or export.get("status") != "complete" or export.get("arm") != arm
            or export.get("source_training_job_id") != source_job
            or export.get("source_training_state")
               != "FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT"
            or Path(export.get("source_checkpoint", "")).resolve() != source / "000900"
            or export.get("checkpoint_metadata_sha256")
               != sha256(source / "000900/_CHECKPOINT_METADATA")
            or model.get("export_manifest_sha256") != sha256(export_path)):
        raise CleanupError(f"{label}: recovered export manifest mismatch")
    export_job = str(export.get("export_slurm_job_id") or "")
    if str(model.get("export_slurm_job_id") or "") != export_job:
        raise CleanupError(f"{label}: eval/export job linkage mismatch")
    eval_job = str(eval_info.get("slurm_job_id") or "")
    # The source jobs are the two audited special case: training and step-900
    # save completed, then their inline CPU export failed.  The recovered export
    # and own-val jobs, rather than the failed source wrapper, must be COMPLETED.
    completed(export_job)
    completed(eval_job)
    if not (source / "000900/_CHECKPOINT_METADATA").is_file():
        raise CleanupError(f"{label}: intact step900 checkpoint missing")
    hf = model_root / "hf"
    if not (hf / "config.json").is_file() or not list(hf.glob("*.safetensors")):
        raise CleanupError(f"{label}: retained HF export is incomplete")
    return {
        "label": label,
        "arm": arm,
        "source_job_id": source_job,
        "source_job_state": "FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT",
        "export_job_id": export_job,
        "eval_job_id": eval_job,
        "target": str(source),
        "logical_bytes_before": logical_bytes(source),
        "retained_hf": str(hf),
        "retained_eval": str(evaluation.resolve()),
        "export_manifest": str(export_path),
        "export_manifest_sha256": sha256(export_path),
        "eval_manifest": str(manifest_path),
        "eval_manifest_sha256": sha256(manifest_path),
    }


def validate_relative(*, source: Path, model: Path, evaluation: Path) -> dict[str, Any]:
    target = safe_relative_root(source)
    train_path = source.resolve() / "train_manifest.json"
    export_path = model.resolve() / "export_manifest.json"
    eval_path = evaluation.resolve() / "eval_manifest.json"
    train, export, eval_manifest = load(train_path), load(export_path), load(eval_path)
    if (train.get("artifact_type") != "phaseb_relative_orbax"
            or train.get("status") != "complete" or train.get("step") != 900
            or train.get("arm") != "prose_keep"):
        raise CleanupError("relative training manifest is not the complete step900 endpoint")
    source_job = str(train.get("slurm_job_id") or "")
    if source_job != "135403":
        raise CleanupError(f"relative source job mismatch: {source_job}")
    if (export.get("artifact_type") != "phaseb_relative_hf_checkpoint"
            or export.get("status") != "complete" or export.get("step") != 900
            or Path(export.get("source_checkpoint", "")).resolve() != target / "000900"
            or export.get("train_manifest_sha256") != sha256(train_path)):
        raise CleanupError("relative export does not seal the requested step900 source")
    export_job = str(export.get("slurm_job_id") or "")
    if (eval_manifest.get("artifact_type") != "phaseb_relative_eval"
            or eval_manifest.get("status") != "complete"
            or eval_manifest.get("valid") is not True
            or eval_manifest.get("request_errors") != 0
            or Path(eval_manifest.get("model_dir", "")).resolve() != model.resolve() / "hf"
            or eval_manifest.get("export_manifest_sha256") != sha256(export_path)):
        raise CleanupError("relative eval does not seal the retained HF export")
    eval_job = str(eval_manifest.get("slurm_job_id") or "")
    for job_id in (source_job, export_job, eval_job):
        completed(job_id)
    if not (target / "000900/_CHECKPOINT_METADATA").is_file():
        raise CleanupError("relative intact step900 checkpoint missing")
    if not (model.resolve() / "hf/config.json").is_file():
        raise CleanupError("relative retained HF export is incomplete")
    return {
        "label": "relative_keep",
        "arm": "prose_keep",
        "source_job_id": source_job,
        "export_job_id": export_job,
        "eval_job_id": eval_job,
        "target": str(target),
        "logical_bytes_before": logical_bytes(target),
        "retained_hf": str(model.resolve()),
        "retained_eval": str(evaluation.resolve()),
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256(train_path),
        "export_manifest": str(export_path),
        "export_manifest_sha256": sha256(export_path),
        "eval_manifest": str(eval_path),
        "eval_manifest_sha256": sha256(eval_path),
    }


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    # Every target and every retained artifact is validated before any deletion.
    entries = [
        validate_absolute(label="absolute_keep", source=args.absolute_keep_source,
                          evaluation=args.absolute_keep_eval),
        validate_absolute(label="absolute_strip", source=args.absolute_strip_source,
                          evaluation=args.absolute_strip_eval),
        validate_relative(source=args.relative_source, model=args.relative_model,
                          evaluation=args.relative_eval),
    ]
    retained_hashes = {
        item["eval_manifest"]: item["eval_manifest_sha256"] for item in entries
    }
    retained_hashes.update({item["export_manifest"]: item["export_manifest_sha256"]
                            for item in entries if "export_manifest" in item})
    targets = [Path(item["target"]) for item in entries]
    if len({str(path.resolve()) for path in targets}) != len(targets):
        raise CleanupError("cleanup target collision")

    for target in targets:
        shutil.rmtree(target)
        target.mkdir()
    for target in targets:
        if not target.is_dir() or any(target.iterdir()):
            raise CleanupError(f"postcondition failed: {target} is not an empty tombstone")
    for raw_path, expected_hash in retained_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise CleanupError(f"retained artifact changed during cleanup: {path}")
    return {
        "artifact_type": "phaseb_source_orbax_cleanup",
        "schema_version": 1,
        "status": "complete",
        "execution": "CPU-only Slurm job",
        "validation": "all source/export/eval jobs COMPLETED and manifests sealed before deletion",
        "deletion": "three source Orbax roots replaced by empty tombstone directories",
        "retention": "all HF exports, eval artifacts, and their manifests retained",
        "logical_bytes_removed": sum(item["logical_bytes_before"] for item in entries),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute-keep-source", type=Path, required=True)
    parser.add_argument("--absolute-strip-source", type=Path, required=True)
    parser.add_argument("--relative-source", type=Path, required=True)
    parser.add_argument("--absolute-keep-eval", type=Path, required=True)
    parser.add_argument("--absolute-strip-eval", type=Path, required=True)
    parser.add_argument("--relative-model", type=Path, required=True)
    parser.add_argument("--relative-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = cleanup(args)
    except CleanupError as exc:
        print(f"FATAL Phase-B cleanup: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "cleanup_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
