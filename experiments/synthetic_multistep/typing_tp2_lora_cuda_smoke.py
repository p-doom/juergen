#!/usr/bin/env python3
"""Tiny explicit-TP LoRA compile/forward/optimizer backend smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from omegalax.trainers.lora import LoRAParam, LoRALinear
from omegalax.trainers.optim import MixedPrecisionOptimizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omegalax", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-backend", "--expected_backend", choices=("cpu", "gpu"), default="gpu"
    )
    parser.add_argument("--expected-commit", "--expected_commit", required=True)
    parser.add_argument("--expected-diff-sha", "--expected_diff_sha", required=True)
    parser.add_argument("--expected-lora-sha", "--expected_lora_sha", required=True)
    parser.add_argument("--expected-test-sha", "--expected_test_sha", required=True)
    args = parser.parse_args()

    omegalax = args.omegalax.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(omegalax), "rev-parse", "HEAD"], text=True
    ).strip()
    diff_sha = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(omegalax), "diff", "--binary"])
    ).hexdigest()
    lora_sha = _sha256(omegalax / "omegalax/trainers/lora.py")
    test_sha = _sha256(omegalax / "tests/test_lora.py")
    if (commit, diff_sha, lora_sha, test_sha) != (
        args.expected_commit,
        args.expected_diff_sha,
        args.expected_lora_sha,
        args.expected_test_sha,
    ):
        raise SystemExit("FATAL OmegaLAX smoke source snapshot mismatch")

    devices = jax.devices()
    if (
        jax.default_backend() != args.expected_backend
        or jax.process_count() != 1
        or jax.local_device_count() != 2
        or len(devices) != 2
    ):
        raise SystemExit(
            "FATAL smoke requires one process and exactly two devices on "
            f"{args.expected_backend}: backend={jax.default_backend()} "
            f"processes={jax.process_count()} devices={devices}"
        )

    mesh = Mesh(
        np.asarray(devices).reshape(2, 1, 1),
        ("tp", "fsdp", "dp"),
        (AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_spec = P(("dp", "fsdp"), None, None)
    output_spec = P(("dp", "fsdp"), None, "tp")
    input_sharding = NamedSharding(mesh, input_spec)
    with jax.set_mesh(mesh):
        layer = LoRALinear(
            nnx.Linear(4, 4, use_bias=False, rngs=nnx.Rngs(0)),
            r=2,
            alpha=2,
            rngs=nnx.Rngs(1),
            dtype=jnp.float32,
        )
        # Nonzero B prevents a vacuous zero-delta forward pass.
        layer.lora_B[...] = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        inputs = jax.device_put(
            jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4), input_sharding
        )
        expected = np.asarray(layer(inputs, out_sharding=None))
        forward = nnx.jit(
            lambda module, x: module(x, out_sharding=output_spec)
        )
        actual = forward(layer, inputs)
        actual.block_until_ready()

        native_delta = jax.jit(
            lambda x, a, b: jnp.matmul(
                jnp.matmul(x, a), b, out_sharding=output_spec
            )
        )
        stablehlo = str(
            native_delta.lower(
                inputs, layer.lora_A[...], layer.lora_B[...]
            ).compiler_ir(dialect="stablehlo")
        )
        if "custom_call @Sharding" in stablehlo:
            raise RuntimeError("native dot output unexpectedly lowered a Sharding custom call")

        base_before = np.asarray(layer.base.kernel[...]).copy()
        lora_before = np.asarray(layer.lora_B[...]).copy()
        optimizer = MixedPrecisionOptimizer(
            layer, optax.adamw(1e-3), wrt=LoRAParam
        )

        @nnx.jit
        def train_step(opt, x):
            def loss_fn(module):
                output = module(x, out_sharding=output_spec)
                return jnp.sum(output**2)

            loss, grads = nnx.value_and_grad(
                loss_fn, argnums=nnx.DiffState(0, LoRAParam)
            )(opt.model)
            opt.update(grads)
            return loss

        loss = train_step(optimizer, inputs)
        loss.block_until_ready()

    if actual.sharding.spec != output_spec:
        raise RuntimeError(f"wrong output sharding: {actual.sharding}")
    shard_shapes = [tuple(shard.data.shape) for shard in actual.addressable_shards]
    if shard_shapes != [(1, 2, 2), (1, 2, 2)]:
        raise RuntimeError(f"wrong physical shard shapes: {shard_shapes}")
    np.testing.assert_array_equal(np.asarray(actual), expected)
    np.testing.assert_array_equal(np.asarray(layer.base.kernel[...]), base_before)
    if np.array_equal(np.asarray(layer.lora_B[...]), lora_before):
        raise RuntimeError("LoRA optimizer smoke did not update adapter B")
    if not np.isfinite(np.asarray(loss)).item():
        raise RuntimeError(f"nonfinite optimizer smoke loss: {loss}")

    result = {
        "schema_version": 1,
        "status": "pass",
        "backend": jax.default_backend(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": len(devices),
        "mesh_shape": [2, 1, 1],
        "output_partition_spec": str(output_spec),
        "physical_shard_shapes": [list(shape) for shape in shard_shapes],
        "forward_exact_to_unsharded": True,
        "optimizer_loss": float(np.asarray(loss)),
        "optimizer_loss_finite": True,
        "base_kernel_bit_exact_after_step": True,
        "lora_adapter_updated": True,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_has_custom_call_sharding": False,
        "omegalax_commit": commit,
        "omegalax_diff_sha256": diff_sha,
        "lora_source_sha256": lora_sha,
        "lora_test_sha256": test_sha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
