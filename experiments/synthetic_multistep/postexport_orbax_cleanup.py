#!/usr/bin/env python3
"""Delete only hash-pinned Orbax steps with a present, valid HF export.

This is intentionally a one-way maintenance utility, not a general garbage
collector.  Every deletion target and its successful export lineage must be
listed in a reviewed allowlist.  The utility validates the complete allowlist
before removing anything and checkpoints its output manifest after each exact
step-directory deletion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


class CleanupError(RuntimeError):
    """Raised when a safety or provenance condition is not met."""


_RUN_ID_RE = re.compile(r"run_[0-9a-f]{32}")
_STEP_RE = re.compile(r"[0-9]{6}")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "C64": 8,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C128": 16,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CleanupError(f"expected a JSON object at {path}")
    return value


def _sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise CleanupError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise CleanupError(
            f"{label} hash changed for {path}: expected {expected}, got {actual}"
        )
    return actual


def _du_bytes(path: Path, *, apparent: bool) -> int:
    command = ["du", "-sx", "--block-size=1"]
    if apparent:
        command.append("--apparent-size")
    command.append(str(path))
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return int(result.stdout.split(maxsplit=1)[0])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise CleanupError(f"cannot measure {path} with du: {exc}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_plain_contained_step(path: Path, checkpoint_root: Path) -> None:
    if not path.is_absolute() or not checkpoint_root.is_absolute():
        raise CleanupError("checkpoint root and source paths must be absolute")
    if not _STEP_RE.fullmatch(path.name):
        raise CleanupError(f"source is not an exact six-digit step directory: {path}")
    try:
        resolved_root = checkpoint_root.resolve(strict=True)
        resolved_parent = path.parent.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise CleanupError(f"cannot resolve cleanup target {path}: {exc}") from exc
    if resolved_root != checkpoint_root:
        raise CleanupError(f"checkpoint root contains a symlink: {checkpoint_root}")
    if resolved_parent != path.parent or resolved_path != path:
        raise CleanupError(f"cleanup target or its stream parent is a symlink: {path}")
    if not _is_relative_to(path, checkpoint_root) or path.parent == checkpoint_root:
        raise CleanupError(f"cleanup target is outside an individual stream: {path}")
    if path.is_symlink() or not path.is_dir():
        raise CleanupError(f"cleanup target is not a plain directory: {path}")


def _validate_safetensors(path: Path) -> dict[str, Any]:
    """Validate safetensors header, tensor byte sizes, and exact data coverage."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CleanupError(f"truncated safetensors prefix: {path}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes == 0 or header_bytes > 512 * 1024 * 1024:
                raise CleanupError(
                    f"implausible safetensors header length {header_bytes}: {path}"
                )
            raw_header = handle.read(header_bytes)
    except OSError as exc:
        raise CleanupError(f"cannot inspect safetensors {path}: {exc}") from exc
    if len(raw_header) != header_bytes:
        raise CleanupError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanupError(f"invalid safetensors JSON header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise CleanupError(f"safetensors header is not an object: {path}")
    payload_bytes = file_size - 8 - header_bytes
    if payload_bytes < 0:
        raise CleanupError(f"safetensors header exceeds file size: {path}")

    intervals: list[tuple[int, int, str]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise CleanupError(f"invalid __metadata__ in {path}")
            continue
        if not isinstance(name, str) or not isinstance(tensor, dict):
            raise CleanupError(f"invalid tensor entry in {path}")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise CleanupError(f"unsupported dtype {dtype!r} for {name} in {path}")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in shape
        ):
            raise CleanupError(f"invalid shape for {name} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise CleanupError(f"invalid data offsets for {name} in {path}")
        start, end = offsets
        if start < 0 or start > end or end > payload_bytes:
            raise CleanupError(f"out-of-range data offsets for {name} in {path}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_bytes = elements * _DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise CleanupError(
                f"tensor byte length mismatch for {name} in {path}: "
                f"expected {expected_bytes}, got {end - start}"
            )
        intervals.append((start, end, name))
    if not intervals:
        raise CleanupError(f"safetensors export contains no tensors: {path}")

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise CleanupError(
                f"safetensors data is overlapping or non-contiguous before {name}: {path}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise CleanupError(
            f"safetensors tensors cover {cursor} of {payload_bytes} payload bytes: {path}"
        )
    return {
        "file_size": file_size,
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "payload_bytes": payload_bytes,
        "tensor_count": len(intervals),
    }


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_relative_to(left, right) or _is_relative_to(right, left)


def _active_job_snapshot(
    *,
    run_root: Path,
    targets: list[Path],
    producer_run_ids: set[str],
) -> list[dict[str, Any]]:
    """Reject active labctl jobs whose contexts overlap a target stream."""
    user = os.environ.get("USER")
    if not user:
        raise CleanupError("USER is unset; cannot perform active-job revalidation")
    try:
        queued = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%A|%j|%T|%o|%Z"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CleanupError(f"cannot inventory active Slurm jobs: {exc}") from exc
    current_job = os.environ.get("SLURM_JOB_ID")
    streams = sorted({target.parent for target in targets})
    snapshot: list[dict[str, Any]] = []
    for line in queued.stdout.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            raise CleanupError(f"cannot parse squeue row: {line!r}")
        job_id, job_name, state, command, workdir = parts
        if job_id == current_job:
            continue
        try:
            detail = subprocess.run(
                ["scontrol", "show", "job", "-o", job_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise CleanupError(f"cannot inspect active Slurm job {job_id}: {exc}") from exc
        combined = " ".join((job_name, command, workdir, detail))
        run_match = _RUN_ID_RE.search(combined)
        record: dict[str, Any] = {
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "run_id": run_match.group(0) if run_match else None,
        }
        if any(str(stream) in combined or stream.name in combined for stream in streams):
            raise CleanupError(
                f"active Slurm job {job_id} command mentions an allowlisted stream"
            )
        if run_match:
            run_id = run_match.group(0)
            if run_id in producer_run_ids:
                raise CleanupError(f"source producer {run_id} is still active as job {job_id}")
            context_path = run_root / run_id / ".lab" / "context.json"
            if not context_path.is_file():
                raise CleanupError(
                    f"active labctl-looking job {job_id} has no readable context: {context_path}"
                )
            context = _load_json(context_path)
            for value in _walk_strings(context):
                if not value.startswith("/"):
                    continue
                candidate = Path(value)
                if any(_paths_overlap(candidate, stream) for stream in streams):
                    raise CleanupError(
                        f"active job {job_id}/{run_id} context overlaps allowlisted stream: "
                        f"{candidate}"
                    )
            record["context_sha256"] = _sha256(context_path)
        snapshot.append(record)
    return snapshot


def _validate_entry(
    entry: dict[str, Any],
    *,
    checkpoint_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    required_strings = (
        "source_path",
        "source_artifact_id",
        "source_producer_run_id",
        "source_meta_sha256",
        "source_checkpoint_metadata_sha256",
        "export_path",
        "export_artifact_id",
        "export_run_id",
        "export_context_path",
        "export_context_sha256",
        "export_meta_sha256",
        "export_config_sha256",
    )
    for field in required_strings:
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise CleanupError(f"allowlist entry has invalid {field}")
    for field in ("source_step", "expected_allocated_bytes", "expected_logical_bytes"):
        if not isinstance(entry.get(field), int) or isinstance(entry[field], bool):
            raise CleanupError(f"allowlist entry has invalid {field}")
    if entry.get("export_run_status") != "succeeded":
        raise CleanupError("allowlist export status proof is not succeeded")
    if entry.get("source_producer_status") not in {"succeeded", "failed"}:
        raise CleanupError("source producer is active, ambiguous, or has no audited terminal status")

    source = Path(entry["source_path"])
    export = Path(entry["export_path"])
    context_path = Path(entry["export_context_path"])
    _require_plain_contained_step(source, checkpoint_root)
    if source.name != f"{entry['source_step']:06d}":
        raise CleanupError(f"allowlisted step does not match source basename: {source}")
    if not _is_relative_to(export, checkpoint_root) or export.parent == checkpoint_root:
        raise CleanupError(f"export is not an individual checkpoint under the root: {export}")
    if export.is_symlink() or export.resolve(strict=True) != export or not export.is_dir():
        raise CleanupError(f"export is not a present plain directory: {export}")
    expected_context = run_root / entry["export_run_id"] / ".lab" / "context.json"
    if context_path != expected_context or context_path.resolve(strict=True) != context_path:
        raise CleanupError(f"export context is outside the exact producer run: {context_path}")

    source_meta_path = source / ".meta.json"
    source_marker = source / "_CHECKPOINT_METADATA"
    for required in (source_meta_path, source_marker, source / "train_state", source / "input_iter"):
        if not required.exists():
            raise CleanupError(f"required Orbax source component is absent: {required}")
    _require_hash(source_meta_path, entry["source_meta_sha256"], "source metadata")
    _require_hash(
        source_marker,
        entry["source_checkpoint_metadata_sha256"],
        "source checkpoint marker",
    )
    source_meta = _load_json(source_meta_path)
    source_metadata = source_meta.get("metadata")
    if not isinstance(source_metadata, dict):
        raise CleanupError(f"source metadata payload is absent: {source_meta_path}")
    if (
        source_meta.get("id") != entry["source_artifact_id"]
        or source_meta.get("producer_run_id") != entry["source_producer_run_id"]
        or source_metadata.get("step") != entry["source_step"]
        or source_meta.get("alias") != f"{source.parent.name}/{source.name}"
        or source_metadata.get("marker") != "_CHECKPOINT_METADATA"
    ):
        raise CleanupError(f"source artifact identity changed: {source}")

    context = _load_json(context_path)
    _require_hash(context_path, entry["export_context_sha256"], "export context")
    inputs = context.get("inputs")
    outputs = context.get("outputs")
    if (
        context.get("run_id") != entry["export_run_id"]
        or not isinstance(context.get("recipe_name"), str)
        or not context["recipe_name"].startswith("bc_export_hf_per_checkpoint")
        or not isinstance(inputs, list)
        or len(inputs) != 1
        or inputs[0].get("artifact_id") != entry["source_artifact_id"]
        or inputs[0].get("resolved_path") != str(source)
        or inputs[0].get("role") != "checkpoint"
        or not isinstance(outputs, dict)
    ):
        raise CleanupError(f"export context lineage does not match source: {context_path}")
    hf_output = outputs.get("hf_checkpoint")
    if (
        not isinstance(hf_output, dict)
        or hf_output.get("path") != str(export.parent)
        or hf_output.get("marker") != "model.safetensors"
        or hf_output.get("role") != "hf_checkpoint"
    ):
        raise CleanupError(f"export context output does not match HF export: {context_path}")

    export_meta_path = export / ".meta.json"
    export_config = export / "config.json"
    export_model = export / "model.safetensors"
    for required in (export_meta_path, export_config, export_model):
        if not required.is_file() or required.is_symlink():
            raise CleanupError(f"required HF export file is absent or unsafe: {required}")
    _require_hash(export_meta_path, entry["export_meta_sha256"], "export metadata")
    _require_hash(export_config, entry["export_config_sha256"], "export config")
    config = _load_json(export_config)
    if not config:
        raise CleanupError(f"HF config is empty: {export_config}")
    export_meta = _load_json(export_meta_path)
    export_metadata = export_meta.get("metadata")
    if not isinstance(export_metadata, dict):
        raise CleanupError(f"HF export metadata payload is absent: {export_meta_path}")
    if (
        export_meta.get("id") != entry["export_artifact_id"]
        or export_meta.get("producer_run_id") != entry["export_run_id"]
        or export_metadata.get("step") != entry["source_step"]
        or export_metadata.get("marker") != "model.safetensors"
        or export_meta.get("alias") != f"{export.parent.name}/{export.name}"
    ):
        raise CleanupError(f"HF export artifact identity changed: {export}")
    safetensors = _validate_safetensors(export_model)
    if safetensors["file_size"] != entry.get("export_model_size"):
        raise CleanupError(f"HF model size changed: {export_model}")

    allocated = _du_bytes(source, apparent=False)
    logical = _du_bytes(source, apparent=True)
    if allocated != entry["expected_allocated_bytes"]:
        raise CleanupError(
            f"allocated byte count changed for {source}: "
            f"expected {entry['expected_allocated_bytes']}, got {allocated}"
        )
    if logical != entry["expected_logical_bytes"]:
        raise CleanupError(
            f"logical byte count changed for {source}: "
            f"expected {entry['expected_logical_bytes']}, got {logical}"
        )
    return {
        "source_path": str(source),
        "source_artifact_id": entry["source_artifact_id"],
        "source_producer_run_id": entry["source_producer_run_id"],
        "source_producer_status": entry["source_producer_status"],
        "source_step": entry["source_step"],
        "allocated_bytes_before": allocated,
        "logical_bytes_before": logical,
        "source_meta_sha256": entry["source_meta_sha256"],
        "source_checkpoint_metadata_sha256": entry[
            "source_checkpoint_metadata_sha256"
        ],
        "export_path": str(export),
        "export_artifact_id": entry["export_artifact_id"],
        "export_run_id": entry["export_run_id"],
        "export_run_status": entry["export_run_status"],
        "export_context_path": str(context_path),
        "export_context_sha256": entry["export_context_sha256"],
        "export_meta_sha256": entry["export_meta_sha256"],
        "export_config_sha256": entry["export_config_sha256"],
        "export_model_sha256": _sha256(export_model),
        "safetensors": safetensors,
    }


def cleanup(
    *,
    allowlist_path: Path,
    expected_allowlist_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    allowlist_hash = _require_hash(
        allowlist_path, expected_allowlist_sha256, "cleanup allowlist"
    )
    allowlist = _load_json(allowlist_path)
    if allowlist.get("schema_version") != 1:
        raise CleanupError("unsupported cleanup allowlist schema")
    entries = allowlist.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CleanupError("cleanup allowlist must contain at least one entry")
    if allowlist.get("target_count") != len(entries):
        raise CleanupError("cleanup allowlist target count is inconsistent")
    checkpoint_root = Path(str(allowlist.get("checkpoint_root", "")))
    run_root = Path(str(allowlist.get("run_root", "")))
    targets = [Path(str(entry.get("source_path", ""))) for entry in entries]
    if len(set(targets)) != len(targets):
        raise CleanupError("cleanup allowlist contains duplicate source paths")

    active_jobs = _active_job_snapshot(
        run_root=run_root,
        targets=targets,
        producer_run_ids={str(entry.get("source_producer_run_id")) for entry in entries},
    )
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        print(f"validating {index}/{len(entries)}: {entry.get('source_path')}", flush=True)
        validated.append(
            _validate_entry(entry, checkpoint_root=checkpoint_root, run_root=run_root)
        )

    allocated_total = sum(item["allocated_bytes_before"] for item in validated)
    logical_total = sum(item["logical_bytes_before"] for item in validated)
    if allocated_total != allowlist.get("expected_allocated_bytes"):
        raise CleanupError("allowlist allocated-byte total is inconsistent")
    if logical_total != allowlist.get("expected_logical_bytes"):
        raise CleanupError("allowlist logical-byte total is inconsistent")

    filesystem_free_before = shutil.disk_usage(checkpoint_root).free
    manifest: dict[str, Any] = {
        "artifact_type": "franz_postexport_orbax_step_cleanup",
        "schema_version": 1,
        "status": "validated_pre_delete",
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": allowlist_hash,
        "authorization": allowlist.get("authorization"),
        "validation_completed_unix": int(time.time()),
        "checkpoint_root": str(checkpoint_root),
        "target_count": len(validated),
        "allocated_bytes_validated": allocated_total,
        "logical_bytes_validated": logical_total,
        "filesystem_free_bytes_before": filesystem_free_before,
        "active_slurm_jobs_checked": active_jobs,
        "entries": validated,
        "deleted_source_paths": [],
        "retained_hf_export_paths": [item["export_path"] for item in validated],
    }
    _atomic_json(output_path, manifest)

    for index, item in enumerate(validated, start=1):
        source = Path(item["source_path"])
        # Close the validation/deletion race as much as a path-based maintenance
        # utility can: recheck containment and artifact identity immediately.
        _require_plain_contained_step(source, checkpoint_root)
        source_meta = _load_json(source / ".meta.json")
        if source_meta.get("id") != item["source_artifact_id"]:
            raise CleanupError(f"source identity changed immediately before deletion: {source}")
        _require_hash(source / ".meta.json", item["source_meta_sha256"], "source metadata")
        print(f"deleting {index}/{len(validated)}: {source}", flush=True)
        shutil.rmtree(source)
        if source.exists() or source.is_symlink():
            raise CleanupError(f"deletion postcondition failed: {source}")
        if not Path(item["export_path"]).is_dir():
            raise CleanupError(f"HF export disappeared during cleanup: {item['export_path']}")
        manifest["status"] = "deleting"
        manifest["deleted_source_paths"].append(str(source))
        manifest["deleted_allocated_bytes"] = sum(
            row["allocated_bytes_before"]
            for row in validated[: len(manifest["deleted_source_paths"])]
        )
        manifest["deleted_logical_bytes"] = sum(
            row["logical_bytes_before"]
            for row in validated[: len(manifest["deleted_source_paths"])]
        )
        manifest["filesystem_free_bytes_current"] = shutil.disk_usage(checkpoint_root).free
        _atomic_json(output_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_unix"] = int(time.time())
    manifest["filesystem_free_bytes_after"] = shutil.disk_usage(checkpoint_root).free
    manifest["filesystem_free_bytes_delta"] = (
        manifest["filesystem_free_bytes_after"] - filesystem_free_before
    )
    manifest["all_hf_exports_retained"] = all(
        Path(path).is_dir() for path in manifest["retained_hf_export_paths"]
    )
    if not manifest["all_hf_exports_retained"]:
        raise CleanupError("an HF export was not retained")
    _atomic_json(output_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--expected-allowlist-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = cleanup(
            allowlist_path=args.allowlist.resolve(strict=True),
            expected_allowlist_sha256=args.expected_allowlist_sha256,
            output_path=args.out,
        )
    except CleanupError as exc:
        print(f"FATAL post-export Orbax cleanup: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
