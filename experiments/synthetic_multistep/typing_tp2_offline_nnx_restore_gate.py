#!/usr/bin/env python3
"""CPU-only real-NNX restore gate for the exact quarantined TP2 pilot clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from flax import nnx
import jax
import orbax.checkpoint as ocp

from experiments.synthetic_multistep import typing_tp2_metadata_diff as metadata_diff
from experiments.synthetic_multistep import typing_tp2_train_entrypoint as restore


EXPECTED_ALIAS = (
    "synthetic_typing_factorial_A_coalesced_r256_lr5e5_recovered_tp2_exact_v6_"
    "run_019fb63f393b7ab3a4e3d03a80d8c7ac"
)
EXPECTED_HASHES = {
    "_CHECKPOINT_METADATA": "7214eab7f13bf3556be18ee25b3ec5368fe62ce46e1150c88ec26bba9d6c00ea",
    "train_state/_METADATA": "729381d3b8f4ed8d96ba1733dba2a42adfb70117e6d8c5775cee73bec6e8298c",
    "input_iter/process_0-of-1.json": "7c62175f1690f6699c5e358147258c40f6d4f9814850ffc9797fd3048b3f6ba5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = root / "orbax/000250"
    if root.name != EXPECTED_ALIAS or checkpoint.name != "000250":
        raise SystemExit(f"FATAL unexpected quarantined clone: {root}")
    marker = json.loads((root / "UNTRUSTED_PARTIAL.json").read_text())
    if marker.get("failed_job_id") != 135595 or not marker.get("must_not_register_or_use"):
        raise SystemExit("FATAL missing v6 quarantine marker")
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(checkpoint / relative)
        if actual != expected:
            raise SystemExit(f"FATAL cloned endpoint hash mismatch {relative}: {actual}")
    clone = json.loads((root / "orbax_clone_manifest.json").read_text())
    if (
        clone.get("status") != "pass"
        or not clone.get("byte_compared_every_file")
        or Path(clone["destination_root"]).resolve() != (root / "orbax").resolve()
    ):
        raise SystemExit("FATAL clone manifest is not exact")
    source_reference = json.loads((root / "tp2_source_bitwise_reference.json").read_text())
    if (
        source_reference.get("status") != "pass"
        or source_reference.get("leaf_count") != 2772
        or Path(source_reference["checkpoint"]).resolve() != checkpoint
    ):
        raise SystemExit("FATAL source bitwise reference is not exact")
    if len(jax.devices()) != 2 or jax.default_backend() != "cpu":
        raise SystemExit(f"FATAL offline gate requires exactly two CPU devices: {jax.devices()}")

    source_tree = restore._metadata_tree(checkpoint)
    logical_target, optimizer = metadata_diff._target_state(checkpoint)
    exact_target = restore._checkpoint_dtype_tp2_target(source_tree, logical_target)
    per_leaf_restore_args = jax.tree.map(
        lambda value: ocp.ArrayRestoreArgs(
            sharding=value.sharding, global_shape=value.shape, dtype=value.dtype
        ),
        exact_target,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            checkpoint / "train_state",
            args=ocp.args.PyTreeRestore(exact_target, restore_args=per_leaf_restore_args),
        )
    restored_state = restored["optimizer"]
    restore._assert_optimizer_contract(restored_state, exact_target["optimizer"])
    target_bitwise = restore._bitwise_tree_records(restored)
    if source_reference["physical_leaf_records"] != target_bitwise["physical_leaf_records"]:
        raise SystemExit("FATAL offline real-NNX restore differs at a canonical leaf")
    if source_reference["tree_sha256"] != target_bitwise["tree_sha256"]:
        raise SystemExit("FATAL offline real-NNX canonical tree hash differs")
    counters = restore._optimizer_counters(restored_state)
    expected_counters = {
        "global_gradient_step": 250,
        "optimizer_micro_step": 2000,
        "gradient_accumulation_remainder": 0,
    }
    if counters != expected_counters:
        raise SystemExit(f"FATAL offline real-NNX counters differ: {counters}")
    nnx.update(optimizer, restored_state)
    restore._assert_optimizer_contract(nnx.state(optimizer), exact_target["optimizer"])

    result = {
        "schema_version": 1,
        "artifact_type": "typing_tp2_offline_real_nnx_restore_gate",
        "status": "pass",
        "cpu_only": True,
        "checkpoint": str(checkpoint),
        "quarantined_job_id": 135595,
        "clone_manifest_sha256": sha256(root / "orbax_clone_manifest.json"),
        "quarantine_marker_sha256": sha256(root / "UNTRUSTED_PARTIAL.json"),
        "source_bitwise_reference_sha256": sha256(root / "tp2_source_bitwise_reference.json"),
        "optimizer_state_python_type": f"{type(restored_state).__module__}.{type(restored_state).__qualname__}",
        "leaf_count": target_bitwise["leaf_count"],
        "canonical_tree_sha256": target_bitwise["tree_sha256"],
        "all_source_target_leaf_records_equal": True,
        "optimizer_contract_matches_checkpoint_dtypes_and_tp2_shardings": True,
        "nnx_update_preserves_exact_contract": True,
        "restored_counters": counters,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "offline_nnx_restore_gate.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
