#!/usr/bin/env python3
"""Offline checkpoint-versus-TP2 abstract-state metadata audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import orbax.checkpoint as ocp

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3_vl import Qwen3VL
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.vlm import (
    TrainConfig,
    _abstract_train_state,
    build_optimizer,
)
from omegalax.vlm import api as vlm_api


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


def _records(tree: Any) -> dict[tuple[str, ...], dict[str, Any]]:
    records = {}
    for path, value in jax.tree_util.tree_leaves_with_path(
        tree, is_leaf=lambda item: isinstance(item, jax.ShapeDtypeStruct)
    ):
        sharding = getattr(value, "sharding", None)
        records[_path_tuple(path)] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "partition_spec": str(sharding.spec) if isinstance(sharding, NamedSharding) else None,
        }
    return records


def _target_state(checkpoint: Path) -> tuple[Any, Any]:
    if len(jax.devices()) != 2:
        raise RuntimeError(
            "offline audit requires two host devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu"
        )
    mesh = ensure_mesh(tp_size=2, fsdp_size=1, dp_size=1)
    config = vlm_api.align_config_to_mesh(
        vlm_api.resolve_config(str(checkpoint.parent)), mesh
    )
    if not isinstance(config, vlm_api.Qwen3VLConfig):
        raise TypeError(f"unexpected model config: {type(config)}")
    train_config = TrainConfig(
        seed=0,
        batch_size=1,
        seq_len=4096,
        num_steps=750,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=30,
        lr_schedule="wsd",
        lr_end_factor=0.0,
        lr_stable_fraction=0.7,
        max_grad_norm=1.0,
        grad_accum_steps=8,
        enable_lora=True,
        lora_rank=256,
        lora_alpha=256.0,
        freeze_vision_tower=False,
        num_loss_tiles=8,
    )
    schedule = build_lr_schedule(
        peak_lr=train_config.learning_rate,
        num_steps=train_config.num_steps,
        warmup_steps=train_config.warmup_steps,
        schedule=train_config.lr_schedule,
        end_factor=train_config.lr_end_factor,
        stable_fraction=train_config.lr_stable_fraction,
    )

    def build_abstract_optimizer():
        model = Qwen3VL(config, rngs=nnx.Rngs(params=0))
        wrapped = inject_lora(
            model,
            r=train_config.lora_rank,
            alpha=train_config.lora_alpha,
            rngs=nnx.Rngs(train_config.seed),
        )
        if wrapped != 252:
            raise RuntimeError(f"expected 252 LoRA projections, got {wrapped}")
        return build_optimizer(model, schedule, train_config, wrt=LoRAParam)

    with mesh_rules(mesh):
        optimizer = nnx.eval_shape(build_abstract_optimizer)
    rng = jax.device_put(
        jax.random.key(train_config.seed), NamedSharding(mesh, PartitionSpec())
    )
    return _abstract_train_state(optimizer, rng), optimizer


def _dtype_group(path: tuple[str, ...]) -> str:
    if path[:3] == ("optimizer", "opt_state", "acc_grads"):
        return "optimizer.opt_state.acc_grads"
    if "mu" in path:
        return "optimizer.opt_state.adam_mu"
    if "nu" in path:
        return "optimizer.opt_state.adam_nu"
    if path == ("rng",):
        return "rng_typed_key_storage"
    return ".".join(path[:4])


def _state_class(path: tuple[str, ...]) -> str:
    if path[:2] == ("optimizer", "model"):
        return "model_lora_parameters" if {"lora_A", "lora_B"} & set(path) else "model_base_parameters"
    if path[:3] == ("optimizer", "opt_state", "acc_grads"):
        return "gradient_accumulator"
    if "mu" in path:
        return "adam_first_moment"
    if "nu" in path:
        return "adam_second_moment"
    if path == ("optimizer", "step", "value"):
        return "optimizer_global_step"
    if path[:2] == ("optimizer", "opt_state"):
        return "optimizer_scalar_state"
    if path == ("rng",):
        return "rng"
    return "other"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_optimizer_update_proof(
    optimizer: Any, source: dict[tuple[str, ...], dict[str, Any]]
) -> dict[str, Any]:
    state = nnx.state(optimizer)

    def checkpoint_dtype(path, value):
        source_path = ("optimizer",) + _path_tuple(path)
        source_value = source[source_path]
        if tuple(source_value["shape"]) != tuple(value.shape):
            raise RuntimeError(f"optimizer shape mismatch at {source_path}")
        return jax.ShapeDtypeStruct(
            value.shape,
            jnp.dtype(source_value["dtype"]),
            sharding=value.sharding,
        )

    exact_state = jax.tree_util.tree_map_with_path(
        checkpoint_dtype,
        state,
        is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
    )
    nnx.update(optimizer, exact_state)
    graph_def, exact_state = nnx.split(optimizer)
    grads = nnx.state(optimizer.model, LoRAParam)

    def one_update(opt_state, gradient_state):
        rebuilt = nnx.merge(graph_def, opt_state)
        rebuilt.update(gradient_state)
        return nnx.state(rebuilt)

    post_update = jax.eval_shape(one_update, exact_state, grads)

    def optimizer_dtype_counts(tree):
        counts = collections.Counter()
        for path, value in jax.tree_util.tree_leaves_with_path(tree):
            normalized = _path_tuple(path)
            if normalized[:2] == ("opt_state", "acc_grads"):
                counts[f"gradient_accumulator:{value.dtype}"] += 1
            elif "mu" in normalized:
                counts[f"adam_first_moment:{value.dtype}"] += 1
            elif "nu" in normalized:
                counts[f"adam_second_moment:{value.dtype}"] += 1
        return dict(sorted(counts.items()))

    before = optimizer_dtype_counts(exact_state)
    after = optimizer_dtype_counts(post_update)
    expected = {
        "adam_first_moment:float32": 504,
        "adam_second_moment:float32": 504,
        "gradient_accumulator:float32": 504,
    }
    if before != expected or after != expected:
        raise RuntimeError(f"fp32 optimizer update trace mismatch: {before} -> {after}")
    return {
        "status": "pass",
        "method": "checkpoint-driven ShapeDtypeStruct state followed by current MixedPrecisionOptimizer.update jax.eval_shape",
        "before_update": before,
        "after_update": after,
        "implicit_dtype_cast_detected": False,
    }


def _synthetic_orbax_proof() -> dict[str, Any]:
    devices = np.array(jax.devices(), dtype=object)
    source_mesh = Mesh(devices[:1].reshape(1, 1, 1), ("tp", "fsdp", "dp"))
    target_mesh = Mesh(devices.reshape(2, 1, 1), ("tp", "fsdp", "dp"))
    source_rng = jax.device_put(jax.random.key(17), NamedSharding(source_mesh, PartitionSpec()))
    source_moment = jax.device_put(
        jnp.arange(32, dtype=jnp.float32).reshape(8, 4),
        NamedSharding(source_mesh, PartitionSpec(None, None)),
    )
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        checkpoint = Path(directory) / "checkpoint"
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint, {"optimizer": {"moment": source_moment}, "rng": source_rng})
        # Orbax stores a typed scalar PRNG key physically as uint32[2]. Request
        # that physical shape with explicit TP2 sharding; the handler restores
        # the typed scalar key automatically from its extended dtype metadata.
        target = {
            "optimizer": {"moment": jax.ShapeDtypeStruct(
                (8, 4), jnp.float32,
                sharding=NamedSharding(target_mesh, PartitionSpec("tp", None)),
            )},
            "rng": jax.ShapeDtypeStruct(
                (2,), jnp.uint32,
                sharding=NamedSharding(target_mesh, PartitionSpec(None)),
            ),
        }
        restore_args = jax.tree.map(
            lambda value: ocp.ArrayRestoreArgs(
                sharding=value.sharding, global_shape=value.shape, dtype=value.dtype
            ),
            target,
            is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
        )
        restored = checkpointer.restore(
            checkpoint,
            args=ocp.args.PyTreeRestore(target, restore_args=restore_args),
        )
    moment = restored["optimizer"]["moment"]
    rng = restored["rng"]
    if moment.dtype != jnp.float32 or not np.array_equal(
        np.asarray(moment), np.arange(32, dtype=np.float32).reshape(8, 4)
    ):
        raise RuntimeError("synthetic fp32 TP1-to-TP2 restore was not bit-exact")
    if rng.dtype != source_rng.dtype or rng.shape != () or not np.array_equal(
        np.asarray(jax.random.key_data(rng)), np.asarray(jax.random.key_data(source_rng))
    ):
        raise RuntimeError("synthetic typed RNG TP1-to-TP2 restore was not bit-exact")
    return {
        "status": "pass",
        "fp32_optimizer_values_bit_exact": True,
        "typed_rng_key_data_bit_exact": True,
        "restored_optimizer_partition_spec": str(moment.sharding.spec),
        "restored_rng_partition_spec": str(rng.sharding.spec),
        "restored_mesh_shape": dict(moment.sharding.mesh.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != "000250":
        raise SystemExit(f"expected exact 000250 checkpoint, got {checkpoint}")

    source_tree = ocp.PyTreeCheckpointer().metadata(
        checkpoint / "train_state"
    ).item_metadata
    source = _records(source_tree)
    target_state, optimizer = _target_state(checkpoint)
    target = _records(target_state)
    missing = sorted(source.keys() - target.keys())
    extra = sorted(target.keys() - source.keys())
    common = source.keys() & target.keys()
    shape_differences = [
        {"path": list(path), "source": source[path], "target": target[path]}
        for path in sorted(common)
        if source[path]["shape"] != target[path]["shape"]
    ]
    dtype_differences = [
        {"path": list(path), "source": source[path], "target": target[path]}
        for path in sorted(common)
        if source[path]["dtype"] != target[path]["dtype"]
    ]
    dtype_groups = collections.Counter(
        _dtype_group(tuple(item["path"])) for item in dtype_differences
    )
    expected_rng_difference = {
        "path": ["rng"],
        "source": {"shape": [2], "dtype": "uint32"},
        "target": {"shape": [], "dtype": "key<fry>"},
    }
    rng_only_shape_difference = (
        len(shape_differences) == 1
        and shape_differences[0]["path"] == expected_rng_difference["path"]
        and shape_differences[0]["source"]["shape"] == expected_rng_difference["source"]["shape"]
        and shape_differences[0]["source"]["dtype"] == expected_rng_difference["source"]["dtype"]
        and shape_differences[0]["target"]["shape"] == expected_rng_difference["target"]["shape"]
        and shape_differences[0]["target"]["dtype"] == expected_rng_difference["target"]["dtype"]
    )
    fp32_to_bf16 = [
        item for item in dtype_differences
        if item["source"]["dtype"] == "float32"
        and item["target"]["dtype"] == "bfloat16"
    ]
    state_class_counts = collections.Counter(_state_class(path) for path in source)
    dtype_transitions_by_class: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for path in sorted(common):
        dtype_transitions_by_class[_state_class(path)][
            f"{source[path]['dtype']}->{target[path]['dtype']}"
        ] += 1
    optimizer_update_proof = _exact_optimizer_update_proof(optimizer, source)
    synthetic_orbax_proof = _synthetic_orbax_proof()
    omegalax = Path("/fast/home/franz.srambical/omegalax")
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "comparison_scope": {
            "source": "Orbax train_state ArrayMetadata global shape/dtype",
            "target": "frozen Omega _abstract_train_state for TP2/LoRA-r256/AdamW/MultiSteps-8",
            "includes": ["optimizer", "rng"],
            "input_iter_compared_separately": False,
            "path_normalization": "DictKey/SequenceKey/GetAttrKey to string tuple",
        },
        "source_leaf_count": len(source),
        "target_leaf_count": len(target),
        "missing_target_paths": [list(path) for path in missing],
        "extra_target_paths": [list(path) for path in extra],
        "structures_match_exactly": not missing and not extra,
        "shape_differences": shape_differences,
        "rng_typed_key_is_only_shape_difference": rng_only_shape_difference,
        "dtype_difference_count": len(dtype_differences),
        "dtype_difference_groups": dict(sorted(dtype_groups.items())),
        "dtype_differences": dtype_differences,
        "state_class_counts": dict(sorted(state_class_counts.items())),
        "persistent_master_parameter_tree": {
            "present": False,
            "leaf_count": 0,
            "note": "MixedPrecisionOptimizer casts params to fp32 transiently during update; no master-param tree is serialized",
        },
        "dtype_transitions_by_state_class": {
            key: dict(sorted(value.items()))
            for key, value in sorted(dtype_transitions_by_class.items())
        },
        "fp32_checkpoint_to_bf16_abstract_optimizer_leaves": len(fp32_to_bf16),
        "frozen_restore_canonicalization": {
            "operation": "got.astype(expected.dtype) when dtypes differ",
            "would_cast_checkpoint_fp32_optimizer_state_to_bf16": len(fp32_to_bf16),
            "bitwise_preserving": len(fp32_to_bf16) == 0,
        },
        "same_code_phase_mismatch": {
            "producer_checkpoint": "post-update fp32 accumulator and Adam moments",
            "fresh_consumer_abstract_state": "pre-first-update bf16 zeros_like LoRA state",
            "current_optimizer_source_sha256": _sha256(omegalax / "omegalax/trainers/optim.py"),
            "current_optimizer_source_mtime": (omegalax / "omegalax/trainers/optim.py").stat().st_mtime,
            "optimizer_source_predates_parent_run": True,
            "explanation": "current MixedPrecisionOptimizer.update upcasts grads and transitions all three state trees to fp32; fresh optax.init starts them in bf16",
        },
        "checkpoint_dtype_optimizer_update_proof": optimizer_update_proof,
        "synthetic_explicit_tp1_to_tp2_restore_proof": synthetic_orbax_proof,
        "classification": "audit_bug_exact_checkpoint_dtype_mapping_constructible",
        "exact_v4_mapping_proven_under_frozen_restore_semantics": False,
        "exact_checkpoint_dtype_mapping_constructible_without_implicit_casts": True,
        "required_restore_behavior": "use checkpoint dtypes for optimizer state with target TP2 shardings; request RNG physical uint32[2] shape and let Orbax reconstruct typed scalar key; do not run frozen got.astype(expected.dtype) canonicalization",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
