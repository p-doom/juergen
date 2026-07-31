"""Regression tests for the tiny one-GPU resume smoke instrument."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import jax.numpy as jnp

from experiments.synthetic_multistep.typing_tp1_memory_safe_resume_cuda_smoke import tree_hash


class TreeHashTest(absltest.TestCase):
    def test_zero_dimensional_optimizer_counter(self):
        first = tree_hash({"counter": jnp.asarray(0, dtype=jnp.int32)})
        second = tree_hash({"counter": jnp.asarray(0, dtype=jnp.int32)})
        changed = tree_hash({"counter": jnp.asarray(1, dtype=jnp.int32)})
        self.assertLen(first, 64)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    absltest.main()
