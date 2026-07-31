#!/usr/bin/env python3
"""Strictly assemble four preregistered fresh-VM chunks into one arm artifact."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    from .aggregate import _load_object, _load_rows, _validate_arm
    from .gate import PROTOCOL_PATH, GateError, load_cells, load_protocol, sha256_file, validate_protocol
    from .run_arm import ARM_NAMES, CHUNK_BOUNDS
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aggregate import _load_object, _load_rows, _validate_arm  # type: ignore
    from gate import (  # type: ignore
        PROTOCOL_PATH,
        GateError,
        load_cells,
        load_protocol,
        sha256_file,
        validate_protocol,
    )
    from run_arm import ARM_NAMES, CHUNK_BOUNDS  # type: ignore

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Contract, sha256_bytes  # type: ignore  # noqa: E402


def assemble(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol, require_launch_authorized=True)
    validate_protocol(protocol)
    cells = load_cells(protocol, Contract())
    protocol_hash = sha256_file(args.protocol)
    arm = protocol["arms"][args.arm]
    if len(args.chunks) != len(CHUNK_BOUNDS):
        raise GateError("exactly four chunk roots are required")
    all_rows = []
    chunk_hashes = []
    seen = set()
    prior_protocol_scopes = protocol["execution_recovery"].get(
        "accepted_prior_chunk_protocols", []
    )
    for index, (root, (start, stop)) in enumerate(zip(args.chunks, CHUNK_BOUNDS, strict=True)):
        if (root / "rows.partial.jsonl").exists():
            raise GateError(f"chunk {index}: partial rows coexist with output")
        manifest_path = root / "chunk_manifest.json"
        rows_path = root / "rows.jsonl"
        manifest = _load_object(manifest_path)
        expected_cells = cells[start:stop]
        required = {
            "schema_version": 1,
            "artifact_type": "synthetic_proper_vm_stage1_5_endpoint_actuation_chunk",
            "status": "complete",
            "scope_classification": protocol["scope_classification"],
            "arm": args.arm,
            "semantic": arm["semantic"],
            "preamble": arm["preamble"],
            "chunk_index": index,
            "chunk_start": start,
            "chunk_stop": stop,
            "cell_ids_sha256": sha256_bytes(
                json.dumps(
                    [cell.cell_id for cell in expected_cells], separators=(",", ":")
                ).encode()
            ),
            "checkpoint_alias": arm["checkpoint_alias"],
            "checkpoint_manifest_sha256": arm["checkpoint_manifest_sha256"],
            "model_weights_sha256": arm["model_weights_sha256"],
            "model_dir": str((Path(arm["checkpoint_root"]) / "hf").resolve()),
            "live_smoke_manifest_sha256": protocol["live_smoke_evidence"]["manifest_sha256"],
            "provider_sha256": protocol["vm"]["provider_sha256"],
            "n_cells": stop - start,
            "sampling": protocol["sampling"],
            "context": protocol["context"],
            "request_errors": 0,
            "infrastructure_mismatches": 0,
        }
        mismatch = {
            key: (manifest.get(key), value)
            for key, value in required.items()
            if manifest.get(key) != value
        }
        if mismatch:
            raise GateError(f"chunk {index}: manifest mismatch: {mismatch}")
        allowed_protocol_hashes = {protocol_hash}
        for record in prior_protocol_scopes:
            if index in record.get("scopes", {}).get(args.arm, []):
                allowed_protocol_hashes.add(record["protocol_sha256"])
        if manifest.get("protocol_sha256") not in allowed_protocol_hashes:
            raise GateError(
                f"chunk {index}: unregistered protocol lineage: "
                f"{manifest.get('protocol_sha256')} not in {sorted(allowed_protocol_hashes)}"
            )
        if sha256_file(rows_path) != manifest.get("rows_sha256"):
            raise GateError(f"chunk {index}: rows hash drift")
        rows = _load_rows(rows_path)
        observed_ids = [row.get("cell_id") for row in rows]
        expected_ids = [cell.cell_id for cell in expected_cells]
        if observed_ids != expected_ids or len(rows) != stop - start:
            raise GateError(f"chunk {index}: ordering/coverage drift")
        overlap = seen.intersection(observed_ids)
        if overlap:
            raise GateError(f"chunk {index}: overlapping cells: {sorted(overlap)[:3]}")
        seen.update(observed_ids)
        all_rows.extend(rows)
        chunk_hashes.append(sha256_file(manifest_path))
    expected_all = [cell.cell_id for cell in cells]
    if [row.get("cell_id") for row in all_rows] != expected_all or seen != set(expected_all):
        raise GateError("assembled chunks do not exactly cover the frozen 320-cell order")
    if args.out.exists() and any(args.out.iterdir()):
        raise GateError(f"refusing to overwrite nonempty arm output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pvm_chunks_", dir=args.out.parent))
    try:
        rows_path = staging / "rows.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
            encoding="utf-8",
        )
        counters = {
            key: sum(int(bool(row[column])) for row in all_rows)
            for key, column in {
                "n_compound_success": "compound_success",
                "n_parse_ok": "parse_ok",
                "n_schema_ok": "schema_ok",
                "n_unit_range_ok": "unit_range_ok",
                "n_endpoint_in_bbox": "endpoint_in_bbox",
            }.items()
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": "synthetic_proper_vm_stage1_5_endpoint_actuation_arm",
            "status": "complete",
            "scope_classification": protocol["scope_classification"],
            "arm": args.arm,
            "semantic": arm["semantic"],
            "preamble": arm["preamble"],
            "checkpoint_alias": arm["checkpoint_alias"],
            "checkpoint_manifest_sha256": arm["checkpoint_manifest_sha256"],
            "model_weights_sha256": arm["model_weights_sha256"],
            "model_dir": str((Path(arm["checkpoint_root"]) / "hf").resolve()),
            "protocol_sha256": protocol_hash,
            "live_smoke_manifest_sha256": protocol["live_smoke_evidence"]["manifest_sha256"],
            "provider_sha256": protocol["vm"]["provider_sha256"],
            "n_cells": len(all_rows),
            "sampling": protocol["sampling"],
            "context": protocol["context"],
            "rows_sha256": sha256_file(rows_path),
            "request_errors": 0,
            "infrastructure_mismatches": 0,
            "fresh_vm_chunk_manifest_sha256": chunk_hashes,
            **counters,
        }
        (staging / "arm_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _validate_arm(
            staging, args.arm, protocol, protocol_hash, cells, Contract()
        )
        args.out.mkdir(parents=True, exist_ok=True)
        os.replace(rows_path, args.out / "rows.jsonl")
        os.replace(staging / "arm_manifest.json", args.out / "arm_manifest.json")
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--chunks", type=Path, nargs=4, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(assemble(parse_args()), indent=2, sort_keys=True))
    except BaseException as exc:
        print(f"FATAL proper-VM chunk assembly: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
