#!/usr/bin/env python3
"""One-GPU legacy-dtype save/release/restore/update smoke for VLM resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from flax import nnx
import grain
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.trainers import vlm


class TinyModel(nnx.Module):
    def __init__(self):
        self.linear = nnx.Linear(
            2048,
            2048,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(7),
        )

    def __call__(self, value):
        return self.linear(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(tree) -> str:
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_leaves_with_path(tree):
        array = np.asarray(jax.device_get(value))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def make_iterator():
    dataset = grain.MapDataset.source([0, 1, 2, 3]).repeat(None)
    return iter(dataset.to_iter_dataset())


@nnx.jit
def train_step(opt, value):
    def loss_fn(module):
        return jnp.mean(module(value).astype(jnp.float32) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
    opt.update(grads)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omegalax", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", "--expected_commit", required=True)
    parser.add_argument("--expected-vlm-sha", "--expected_vlm_sha", required=True)
    parser.add_argument("--expected-test-sha", "--expected_test_sha", required=True)
    args = parser.parse_args()

    # Regression for optimizer counters, which are zero-dimensional arrays.
    scalar_hash = tree_hash({"counter": jnp.asarray(0, dtype=jnp.int32)})
    if len(scalar_hash) != 64:
        raise RuntimeError(f"invalid scalar tree hash: {scalar_hash}")

    omegalax = args.omegalax.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(omegalax), "rev-parse", "HEAD"], text=True
    ).strip()
    vlm_sha = sha256(omegalax / "omegalax/trainers/vlm.py")
    test_sha = sha256(omegalax / "tests/test_vlm_memory_safe_restore.py")
    if (commit, vlm_sha, test_sha) != (
        args.expected_commit,
        args.expected_vlm_sha,
        args.expected_test_sha,
    ):
        raise SystemExit("FATAL memory-safe restore smoke source snapshot mismatch")
    if (
        jax.default_backend() != "gpu"
        or jax.process_count() != 1
        or jax.local_device_count() != 1
        or jax.device_count() != 1
    ):
        raise SystemExit(
            "FATAL smoke requires exactly one process and one GPU: "
            f"backend={jax.default_backend()} processes={jax.process_count()} "
            f"local={jax.local_device_count()} global={jax.device_count()}"
        )

    work = args.out.parent / "smoke_checkpoint"
    manager = vlm._make_checkpoint_manager(work, save_interval=1, keep_latest=1)
    model = TinyModel()
    config = vlm.TrainConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        grad_accum_steps=2,
    )
    optimizer = vlm.build_optimizer(model, 1e-3, config)
    rng = jax.random.key(123)
    batch = jnp.ones((2, 2048), dtype=jnp.bfloat16)
    source_losses = [train_step(optimizer, batch), train_step(optimizer, batch)]
    jax.block_until_ready(source_losses)
    if not all(np.isfinite(np.asarray(loss)).item() for loss in source_losses):
        raise RuntimeError(f"nonfinite source-step losses: {source_losses}")

    input_iter = make_iterator()
    next(input_iter)
    next(input_iter)
    expected_iterator_state = input_iter.get_state()
    expected_iterator_sha = hashlib.sha256(
        json.dumps(expected_iterator_state, indent=4).encode()
    ).hexdigest()
    expected_counters = vlm._restored_optimizer_counters(nnx.state(optimizer))
    expected_rng = [int(value) for value in jax.random.key_data(rng)]
    source_hash = tree_hash(nnx.state(optimizer))

    vlm._save_sft_checkpoint(manager, optimizer, rng, 1, input_iter)
    manager.wait_until_finished()
    del optimizer
    del model
    jax.clear_caches()

    fresh_model = TinyModel()
    fresh_optimizer = vlm.build_optimizer(fresh_model, 1e-3, config)
    blueprint = vlm._prepare_memory_safe_restore(fresh_optimizer, rng)
    expected_optimizer = blueprint.abstract_train_state["optimizer"]
    del fresh_optimizer
    del fresh_model
    release = vlm._verify_initialized_state_released(blueprint)
    vlm._write_restore_release_audit(work, release)

    os.environ.update(
        {
            "OMEGALAX_REQUIRE_EXACT_RESTORE_ATTESTATION": "1",
            "OMEGALAX_EXPECT_RESUME_STEP": "1",
            "OMEGALAX_EXPECT_OPTIMIZER_COUNTERS_JSON": json.dumps(expected_counters),
            "OMEGALAX_EXPECT_RNG_KEY_DATA_JSON": json.dumps(expected_rng),
            "OMEGALAX_EXPECT_ITERATOR_STATE_JSON": json.dumps(expected_iterator_state),
            "OMEGALAX_EXPECT_ITERATOR_SHA256": expected_iterator_sha,
            "OMEGALAX_EXPECT_PROMOTED_OPTIMIZER_STATE_JSON": json.dumps(
                {
                    "promoted_leaf_count": 6,
                    "promoted_source_bytes": 50_356_224,
                    "fresh_zero_state_bytes": 25_178_112,
                }
            ),
        }
    )
    optimizer, step, restored_rng, restored_iter = vlm._restore_sft_checkpoint(
        manager, blueprint, make_iterator(), {"tp": 1, "fsdp": 1, "dp": 1}
    )
    restored_hash = tree_hash(nnx.state(optimizer))
    if source_hash != restored_hash:
        raise RuntimeError(f"restored optimizer/model hash mismatch: {source_hash} != {restored_hash}")
    if [int(value) for value in jax.random.key_data(restored_rng)] != expected_rng:
        raise RuntimeError("restored RNG differs")
    if restored_iter.get_state() != expected_iterator_state:
        raise RuntimeError("restored iterator differs")

    losses = [train_step(optimizer, batch), train_step(optimizer, batch)]
    jax.block_until_ready(losses)
    if not all(np.isfinite(np.asarray(loss)).item() for loss in losses):
        raise RuntimeError(f"nonfinite resumed-step losses: {losses}")
    post_update_contract = vlm._assert_restored_optimizer_contract(
        expected_optimizer, nnx.state(optimizer)
    )
    if (
        post_update_contract["promoted_leaf_count"] != 6
        or post_update_contract["converted_leaf_count"] != 0
        or set(post_update_contract["groups"]) != {"acc_grads", "mu", "nu"}
    ):
        raise RuntimeError(
            f"unexpected post-update optimizer contract: {post_update_contract}"
        )
    del blueprint
    manager.close()

    exact = json.loads((work / "restore_exact_state.json").read_text())
    result = {
        "schema_version": 2,
        "artifact_type": "typing_tp1_legacy_dtype_memory_safe_resume_cuda_smoke",
        "status": "pass",
        "backend": jax.default_backend(),
        "local_device_count": jax.local_device_count(),
        "target_topology": {"tp": 1, "fsdp": 1, "dp": 1},
        "restore_step": step,
        "source_optimizer_model_tree_sha256": source_hash,
        "restored_optimizer_model_tree_sha256": restored_hash,
        "optimizer_model_bit_exact": True,
        "rng_exact": True,
        "iterator_exact": True,
        "release_attestation": release,
        "exact_restore_attestation": exact,
        "source_full_accumulation_losses": [
            float(np.asarray(loss)) for loss in source_losses
        ],
        "resumed_full_accumulation_losses": [
            float(np.asarray(loss)) for loss in losses
        ],
        "post_update_optimizer_contract": post_update_contract,
        "omegalax_commit": commit,
        "omegalax_vlm_sha256": vlm_sha,
        "omegalax_test_sha256": test_sha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
