#!/usr/bin/env python3
"""Run the frozen VLM trainer with an explicit TP2 Orbax restore contract."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import grain
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from omegalax.trainers import checkpoint_utils
from omegalax.trainers import vlm as vlm_trainer


_JAX_DISTRIBUTED_INITIALIZE = jax.distributed.initialize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape_counts(shapes: list[tuple[int, ...]]) -> dict[str, int]:
    counts = collections.Counter("x".join(map(str, shape)) or "scalar" for shape in shapes)
    return dict(sorted(counts.items()))


def _path_tuple(path: tuple[Any, ...]) -> tuple[str, ...]:
    result = []
    for key in path:
        if hasattr(key, "key"):
            result.append(str(key.key))
        elif hasattr(key, "idx"):
            result.append(str(key.idx))
        elif hasattr(key, "name"):
            result.append(str(key.name))
        else:
            result.append(str(key))
    return tuple(result)


def _metadata_tree(checkpoint: Path) -> Any:
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.metadata(checkpoint / "train_state").item_metadata


def _shape_dtype_records(tree: Any) -> dict[tuple[str, ...], Any]:
    records = {}
    for path, value in jax.tree_util.tree_leaves_with_path(
        tree, is_leaf=lambda item: isinstance(item, jax.ShapeDtypeStruct)
    ):
        # Orbax metadata() yields ArrayMetadata leaves while fresh abstract
        # optimizer trees yield ShapeDtypeStruct leaves. Both are deliberately
        # accepted here and only their immutable global shape/dtype contract is
        # consumed.
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise RuntimeError(f"non-array shape/dtype leaf at {_path_tuple(path)}: {value!r}")
        records[_path_tuple(path)] = value
    return records


def _source_checkpoint_summary(checkpoint: Path, metadata_tree: Any) -> dict[str, Any]:
    metadata_path = checkpoint / "train_state/_METADATA"
    sharding_path = checkpoint / "train_state/_sharding"
    metadata = json.loads(metadata_path.read_text())
    sharding = [json.loads(value) for value in json.loads(sharding_path.read_text()).values()]
    source_mesh_shapes = sorted({tuple(value.get("shape", [])) for value in sharding})
    if source_mesh_shapes != [(1, 1, 1)]:
        raise RuntimeError(f"source checkpoint is not the frozen TP1 topology: {source_mesh_shapes}")
    records = _shape_dtype_records(metadata_tree)
    shapes = [tuple(value.shape) for value in records.values()]
    dtype_counts = collections.Counter(str(value.dtype) for value in records.values())
    if len(records) != 2772:
        raise RuntimeError(f"expected 2772 exact source leaves, got {len(records)}")
    rng = records.get(("rng",))
    if rng is None or tuple(rng.shape) != (2,) or rng.dtype != jnp.uint32:
        raise RuntimeError(f"source RNG is not physical uint32[2]: {rng}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "sharding_sha256": _sha256(sharding_path),
        "array_count": len(shapes),
        "global_shape_counts": _shape_counts(shapes),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "mesh_shapes": [list(shape) for shape in source_mesh_shapes],
        "sharding_types": dict(sorted(collections.Counter(
            value["sharding_type"] for value in sharding
        ).items())),
    }


def _target_summary(abstract_state: Any, *, require_explicit: bool = True) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(
        abstract_state, is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct)
    )
    if not leaves or not all(isinstance(value, jax.ShapeDtypeStruct) for value in leaves):
        raise RuntimeError("target restore state contains non-ShapeDtypeStruct leaves")
    shardings = [value.sharding for value in leaves]
    named_shardings = [sharding for sharding in shardings if isinstance(sharding, NamedSharding)]
    if require_explicit and len(named_shardings) != len(shardings):
        raise RuntimeError("target restore state contains non-NamedSharding leaves")
    mesh_shapes = sorted({
        tuple(int(sharding.mesh.shape[axis]) for axis in ("tp", "fsdp", "dp"))
        for sharding in named_shardings
    })
    if mesh_shapes != [(2, 1, 1)]:
        raise RuntimeError(f"target restore state is not TP2/FSDP1/DP1: {mesh_shapes}")
    specs = collections.Counter(
        str(sharding.spec) if isinstance(sharding, NamedSharding) else "None"
        for sharding in shardings
    )
    if not any("tp" in spec and count for spec, count in specs.items()):
        raise RuntimeError("target restore state has no tensor-parallel leaves")
    return {
        "array_count": len(leaves),
        "global_shape_counts": _shape_counts([tuple(value.shape) for value in leaves]),
        "mesh_shapes": [list(shape) for shape in mesh_shapes],
        "partition_spec_counts": dict(sorted(specs.items())),
        "explicit_named_sharding_leaf_count": len(named_shardings),
        "all_leaves_have_explicit_named_sharding": len(named_shardings) == len(shardings),
    }


def _checkpoint_dtype_tp2_target(source_tree: Any, logical_target: Any) -> Any:
    """Combine checkpoint shape/dtype with each fresh TP2 leaf's sharding."""
    source = _shape_dtype_records(source_tree)
    seen: set[tuple[str, ...]] = set()
    logical_leaves = jax.tree_util.tree_leaves(
        logical_target, is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct)
    )
    meshes = {
        value.sharding.mesh
        for value in logical_leaves
        if isinstance(value.sharding, NamedSharding)
        and isinstance(value.sharding.mesh, Mesh)
    }
    if len(meshes) != 1:
        raise RuntimeError(f"fresh logical target does not identify one TP2 mesh: {meshes}")
    target_mesh = next(iter(meshes))

    def convert(path, target):
        normalized = _path_tuple(path)
        if normalized not in source:
            raise RuntimeError(f"target path absent from checkpoint metadata: {normalized}")
        checkpoint_leaf = source[normalized]
        seen.add(normalized)
        if normalized == ("rng",):
            if (
                tuple(checkpoint_leaf.shape) != (2,)
                or checkpoint_leaf.dtype != jnp.uint32
                or tuple(target.shape) != ()
                or not jax.dtypes.issubdtype(target.dtype, jax.dtypes.prng_key)
            ):
                raise RuntimeError(
                    f"unexpected physical/logical RNG contract: {checkpoint_leaf} -> {target}"
                )
            # Orbax reconstructs the typed scalar key from its extended dtype
            # metadata even though the explicit request uses physical storage.
            sharding = NamedSharding(target_mesh, PartitionSpec(None))
            return jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=sharding)
        if tuple(checkpoint_leaf.shape) != tuple(target.shape):
            raise RuntimeError(
                f"checkpoint/TP2 shape mismatch at {normalized}: "
                f"{checkpoint_leaf.shape} != {target.shape}"
            )
        sharding = NamedSharding(
            target_mesh,
            target.sharding.spec
            if isinstance(target.sharding, NamedSharding)
            else PartitionSpec(),
        )
        return jax.ShapeDtypeStruct(
            checkpoint_leaf.shape,
            checkpoint_leaf.dtype,
            sharding=sharding,
        )

    result = jax.tree_util.tree_map_with_path(
        convert,
        logical_target,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    missing = set(source) - seen
    if missing:
        raise RuntimeError(f"checkpoint paths absent from TP2 target: {sorted(missing)[:10]}")
    return result


def _dtype_transition_summary(source_tree: Any, logical_target: Any) -> dict[str, Any]:
    source = _shape_dtype_records(source_tree)
    target = _shape_dtype_records(logical_target)
    if set(source) != set(target):
        raise RuntimeError("checkpoint/logical target paths differ")
    groups: collections.Counter[str] = collections.Counter()
    differences = 0
    shape_differences = []
    for path in sorted(source):
        left, right = source[path], target[path]
        if tuple(left.shape) != tuple(right.shape):
            shape_differences.append(path)
        if left.dtype == right.dtype:
            continue
        differences += 1
        if path[:3] == ("optimizer", "opt_state", "acc_grads"):
            groups["gradient_accumulator_fp32_to_bf16"] += 1
        elif "mu" in path:
            groups["adam_first_moment_fp32_to_bf16"] += 1
        elif "nu" in path:
            groups["adam_second_moment_fp32_to_bf16"] += 1
        elif path == ("rng",):
            groups["rng_uint32_physical_to_typed_logical"] += 1
        else:
            groups[f"unexpected:{path}"] += 1
    expected = {
        "gradient_accumulator_fp32_to_bf16": 504,
        "adam_first_moment_fp32_to_bf16": 504,
        "adam_second_moment_fp32_to_bf16": 504,
        "rng_uint32_physical_to_typed_logical": 1,
    }
    if dict(groups) != expected or differences != 1513 or shape_differences != [("rng",)]:
        raise RuntimeError(
            f"unexpected checkpoint/fresh dtype phase contract: "
            f"groups={groups}, differences={differences}, shapes={shape_differences}"
        )
    return {
        "difference_count": differences,
        "groups": dict(sorted(groups.items())),
        "only_shape_difference_is_physical_rng": True,
        "checkpoint_fp32_optimizer_leaves_preserved": 1512,
    }


def _array_physical_form(value: Any) -> tuple[np.ndarray, str, tuple[int, ...]]:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    return array, str(array.dtype), tuple(int(size) for size in array.shape)


def _bitwise_tree_records(tree: Any) -> dict[str, Any]:
    leaves: dict[str, dict[str, Any]] = {}
    for path, value in jax.tree_util.tree_leaves_with_path(tree):
        normalized = _path_tuple(path)
        path_key = json.dumps(normalized, separators=(",", ":"))
        array, dtype, shape = _array_physical_form(value)
        digest = hashlib.sha256()
        # ml_dtypes.bfloat16 exposes PEP-3118 format code ``E``, which Python's
        # memoryview cannot cast. View the already C-contiguous array through
        # uint8 so hashing is dtype-agnostic while dtype/shape remain explicit
        # in the leaf record.
        digest.update(array.view(np.uint8).reshape(-1))
        record = {"shape": list(shape), "dtype": dtype, "sha256": digest.hexdigest()}
        leaves[path_key] = record
    return {
        "leaf_count": len(leaves),
        "physical_leaf_records": leaves,
        "tree_sha256": _aggregate_leaf_hash(leaves),
        "typed_rng_hashed_as_physical_key_data": True,
    }


def _aggregate_leaf_hash(leaves: dict[str, dict[str, Any]]) -> str:
    """Hash canonical leaf records independently of PyTree traversal order."""
    overall = hashlib.sha256()
    for path_key in sorted(leaves):
        overall.update(path_key.encode())
        overall.update(
            json.dumps(leaves[path_key], sort_keys=True, separators=(",", ":")).encode()
        )
    return overall.hexdigest()


def _source_bitwise_reference(checkpoint: Path, output: Path) -> None:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"source reference subprocess must be CPU-only, got {jax.default_backend()}")
    source_tree = _metadata_tree(checkpoint)
    cpu = jax.local_devices(backend="cpu")
    if len(cpu) != 1:
        raise RuntimeError(f"source reference requires exactly one local CPU device: {cpu}")
    sharding = SingleDeviceSharding(cpu[0])
    cpu_target = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding),
        source_tree,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    per_leaf_restore_args = jax.tree.map(
        lambda value: ocp.ArrayRestoreArgs(
            sharding=value.sharding, global_shape=value.shape, dtype=value.dtype
        ),
        cpu_target,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            checkpoint / "train_state",
            args=ocp.args.PyTreeRestore(cpu_target, restore_args=per_leaf_restore_args),
        )
    result = {
        "schema_version": 1,
        "status": "pass",
        "checkpoint": str(checkpoint.resolve()),
        "backend": "cpu",
        **_bitwise_tree_records(restored),
    }
    if result["leaf_count"] != 2772:
        raise RuntimeError(f"unexpected CPU source reference leaf count: {result['leaf_count']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _make_source_bitwise_reference(checkpoint: Path) -> dict[str, Any]:
    output = Path(os.environ["TYPING_TP2_SOURCE_BITWISE_REFERENCE"])
    if output.exists():
        raise RuntimeError(f"refusing to reuse source bitwise reference: {output}")
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--source-bitwise-reference", str(checkpoint), str(output)],
        check=True,
        env=environment,
    )
    result = json.loads(output.read_text())
    if (
        result.get("status") != "pass"
        or result.get("checkpoint") != str(checkpoint.resolve())
        or result.get("leaf_count") != 2772
    ):
        raise RuntimeError(f"invalid source bitwise reference: {result}")
    return result


def _assert_optimizer_contract(actual: Any, expected: Any) -> None:
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(
        expected, is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct)
    )
    if len(actual_leaves) != len(expected_leaves):
        raise RuntimeError("restored optimizer leaf count differs from exact TP2 target")
    for index, (got, wanted) in enumerate(zip(actual_leaves, expected_leaves)):
        if (
            tuple(got.shape) != tuple(wanted.shape)
            or got.dtype != wanted.dtype
            or got.sharding != wanted.sharding
        ):
            raise RuntimeError(
                f"restored optimizer contract mismatch at leaf {index}: "
                f"{got.shape}/{got.dtype}/{got.sharding} != "
                f"{wanted.shape}/{wanted.dtype}/{wanted.sharding}"
            )


def _extract_optimizer_counters_from_pairs(
    pairs: list[tuple[tuple[str, ...], Any]],
) -> dict[str, int]:
    expected = {
        ("opt_state", "gradient_step", "value"): "global_gradient_step",
        ("step", "value"): "optimizer_micro_step",
        ("opt_state", "mini_step", "value"): "gradient_accumulation_remainder",
    }
    found: dict[str, int] = {}
    for path, value in pairs:
        name = expected.get(path)
        if name is None:
            continue
        if name in found:
            raise RuntimeError(f"duplicate optimizer counter path: {path}")
        array = np.asarray(jax.device_get(value))
        if array.shape != () or array.dtype.kind not in "iu":
            raise RuntimeError(
                f"optimizer counter {path} is not an integer scalar: {array.shape}/{array.dtype}"
            )
        found[name] = int(array)
    missing = set(expected.values()) - set(found)
    if missing:
        raise RuntimeError(f"missing optimizer counter paths: {sorted(missing)}")
    return found


def _optimizer_counters(optimizer_state: Any) -> dict[str, int]:
    pairs = [
        (_path_tuple(path), value)
        for path, value in jax.tree_util.tree_leaves_with_path(optimizer_state)
    ]
    return _extract_optimizer_counters_from_pairs(pairs)


def _write_audit(value: dict[str, Any]) -> None:
    path = Path(os.environ["TYPING_TP2_SHARDING_AUDIT"])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _initialize_and_verify_two_local_devices(*args, **kwargs) -> None:
    """Initialize JAX, then fail before model/Orbax work unless this step owns two GPUs."""
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [value.strip() for value in visible_raw.split(",") if value.strip()]
    device_preflight = {
        "cuda_visible_devices": visible,
        "cuda_visible_devices_raw": visible_raw,
        "expected_local_device_count": 2,
        "explicit_jax_local_device_ids": [0, 1],
        "status": "fail",
    }
    audit = {
        "schema_version": 3,
        "status": "device_preflight_fail",
        "device_preflight": device_preflight,
    }
    if len(visible) != 2 or len(set(visible)) != 2:
        _write_audit(audit)
        raise RuntimeError(
            f"TP2 step requires exactly two distinct CUDA_VISIBLE_DEVICES, got {visible_raw!r}"
        )
    if args or "local_device_ids" in kwargs:
        _write_audit(audit)
        raise RuntimeError("unexpected explicit JAX distributed initialization arguments")
    # JAX's Slurm autodetection maps one local device per task. This recovery
    # deliberately uses one task owning both allocated/visible GPUs, so supply
    # both local ordinals explicitly instead of accepting the detected [0].
    _JAX_DISTRIBUTED_INITIALIZE(local_device_ids=[0, 1], **kwargs)
    local_device_count = int(jax.local_device_count())
    global_device_count = int(jax.device_count())
    process_count = int(jax.process_count())
    device_preflight.update({
        "global_device_count": global_device_count,
        "local_device_count": local_device_count,
        "process_count": process_count,
    })
    if local_device_count != 2 or global_device_count != 2 or process_count != 1:
        _write_audit(audit)
        raise RuntimeError(
            "TP2 step requires local_device_count=global_device_count=2 and process_count=1, "
            f"got local={local_device_count}, global={global_device_count}, processes={process_count}"
        )
    device_preflight["status"] = "pass"
    audit["status"] = "device_preflight_pass"
    _write_audit(audit)


def _explicit_tp2_restore(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer,
    rng: jax.Array,
    input_iter: checkpoint_utils.GrainIterator,
):
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError("No checkpoint found to restore.")
    if int(latest_step) != 250:
        raise ValueError(f"TP2 recovery requires exact step 250, got {latest_step}")

    checkpoint = Path(os.environ["TYPING_TP2_CHECKPOINT"])
    device_audit = json.loads(Path(os.environ["TYPING_TP2_SHARDING_AUDIT"]).read_text())
    if device_audit.get("status") != "device_preflight_pass":
        raise RuntimeError("TP2 device preflight did not pass before restore")
    logical_target = vlm_trainer._abstract_train_state(optimizer, rng)
    source_tree = _metadata_tree(checkpoint)
    source = _source_checkpoint_summary(checkpoint, source_tree)
    logical_target_summary = _target_summary(logical_target, require_explicit=False)
    exact_target = _checkpoint_dtype_tp2_target(source_tree, logical_target)
    target = _target_summary(exact_target)
    if source["array_count"] != target["array_count"]:
        raise RuntimeError(f"source/target array count mismatch: {source} vs {target}")
    if source["global_shape_counts"] != target["global_shape_counts"]:
        raise RuntimeError("source/target global shape multiset mismatch")
    dtype_transition = _dtype_transition_summary(source_tree, logical_target)
    source_bitwise = _make_source_bitwise_reference(checkpoint)
    audit = {
        "schema_version": 4,
        "status": "pre_restore_pass",
        "device_preflight": device_audit["device_preflight"],
        "restore_step": 250,
        "source": source,
        "target": target,
        "fresh_logical_target": logical_target_summary,
        "checkpoint_to_fresh_dtype_transition": dtype_transition,
        "global_shapes_match": True,
        "physical_rng_shape_exception_resolved": True,
        "topology_change": "TP1/FSDP1/DP1 to TP2/FSDP1/DP1",
        "global_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "data_parallel_size": 1,
        "explicit_array_restore_args": True,
        "restore_dtype_source": "checkpoint metadata",
        "restore_sharding_source": "fresh TP2 logical target",
        "fresh_optimizer_dtype_canonicalization_applied": False,
        "source_bitwise_reference": {
            "path": os.environ["TYPING_TP2_SOURCE_BITWISE_REFERENCE"],
            "leaf_count": source_bitwise["leaf_count"],
            "tree_sha256": source_bitwise["tree_sha256"],
        },
    }
    _write_audit(audit)

    def to_restore_args(value):
        if not isinstance(value, jax.ShapeDtypeStruct):
            return value
        return ocp.ArrayRestoreArgs(
            sharding=value.sharding,
            global_shape=value.shape,
            dtype=value.dtype,
        )

    per_leaf_restore_args = jax.tree.map(
        to_restore_args,
        exact_target,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    restore_args = ocp.args.Composite(
        train_state=ocp.args.PyTreeRestore(
            exact_target,
            restore_args=per_leaf_restore_args,
        ),
        input_iter=grain.checkpoint.CheckpointRestore(input_iter),
    )
    restored = checkpoint_manager.restore(latest_step, args=restore_args)
    train_state = restored["train_state"]
    restored_state = train_state["optimizer"]
    _assert_optimizer_contract(restored_state, exact_target["optimizer"])

    target_bitwise = _bitwise_tree_records(train_state)
    source_leaves = source_bitwise["physical_leaf_records"]
    target_leaves = target_bitwise["physical_leaf_records"]
    if source_leaves != target_leaves:
        differing = [
            key for key in sorted(set(source_leaves) | set(target_leaves))
            if source_leaves.get(key) != target_leaves.get(key)
        ]
        raise RuntimeError(f"TP1-to-TP2 restore is not bit-exact at {differing[:10]}")
    if source_bitwise["tree_sha256"] != target_bitwise["tree_sha256"]:
        raise RuntimeError("TP1-to-TP2 restored tree hash differs despite matching leaves")

    counters = _optimizer_counters(restored_state)
    if counters != {
        "global_gradient_step": 250,
        "optimizer_micro_step": 2000,
        "gradient_accumulation_remainder": 0,
    }:
        raise RuntimeError(f"restored optimizer counters mismatch: {counters}")
    restored_iterator = checkpoint_utils.restored_input_iter(restored)
    iterator_state = restored_iterator.get_state()
    expected_iterator_state = {
        "next_index_in_cycle": 0,
        "next_index_in_datasets": 2,
        "iterators_in_use_indices": [0, 1],
        "iterators_in_use_states": [{"next_index": 1000}, {"next_index": 1000}],
        "exhausted": [0, 0],
    }
    if iterator_state != expected_iterator_state:
        raise RuntimeError(f"live restored Grain iterator state mismatch: {iterator_state}")

    nnx.update(optimizer, restored_state)
    _assert_optimizer_contract(nnx.state(optimizer), exact_target["optimizer"])
    audit["status"] = "restore_pass"
    audit["restored_target_shardings_match"] = True
    audit["checkpoint_dtypes_preserved"] = True
    audit["all_train_state_leaves_bitwise_equal_to_cpu_source_restore"] = True
    audit["source_tree_sha256"] = source_bitwise["tree_sha256"]
    audit["restored_tree_sha256"] = target_bitwise["tree_sha256"]
    audit["bitwise_leaf_count"] = target_bitwise["leaf_count"]
    audit["restored_counters"] = counters
    audit["restored_iterator_state"] = iterator_state
    audit["restored_iterator_state_exact"] = True
    _write_audit(audit)
    return (
        optimizer,
        int(latest_step),
        train_state["rng"],
        restored_iterator,
    )


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--source-bitwise-reference":
        _source_bitwise_reference(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
        return
    if not os.environ.get("TYPING_TP2_SHARDING_AUDIT"):
        raise SystemExit("FATAL missing TYPING_TP2_SHARDING_AUDIT")
    if not os.environ.get("TYPING_TP2_CHECKPOINT"):
        raise SystemExit("FATAL missing TYPING_TP2_CHECKPOINT")
    if not os.environ.get("TYPING_TP2_SOURCE_BITWISE_REFERENCE"):
        raise SystemExit("FATAL missing TYPING_TP2_SOURCE_BITWISE_REFERENCE")
    entrypoint = Path(os.environ["OMEGALAX_TRAIN_ENTRYPOINT"])
    if not entrypoint.is_file():
        raise SystemExit(f"FATAL missing Omega entrypoint: {entrypoint}")
    jax.distributed.initialize = _initialize_and_verify_two_local_devices
    vlm_trainer._restore_sft_checkpoint = _explicit_tp2_restore
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
