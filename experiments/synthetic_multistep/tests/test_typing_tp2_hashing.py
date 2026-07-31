from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from experiments.synthetic_multistep.typing_tp2_train_entrypoint import (
    _aggregate_leaf_hash,
    _bitwise_tree_records,
    _extract_optimizer_counters_from_pairs,
)


def _leaf(tree):
    records = _bitwise_tree_records(tree)
    assert records["leaf_count"] == 1
    return next(iter(records["physical_leaf_records"].values())), records["tree_sha256"]


def test_bfloat16_hash_is_c_order_value_exact_for_contiguous_and_noncontiguous_arrays():
    values = jnp.asarray(
        [[1.0, -2.5, 3.25], [4.5, 5.75, -6.0]],
        dtype=jnp.bfloat16,
    )
    contiguous = np.ascontiguousarray(np.asarray(values))
    # Transpose twice to retain the same values/shape through a non-contiguous
    # input view. The hashing contract canonicalizes both to C order.
    noncontiguous = contiguous.T.copy().T
    assert not noncontiguous.flags.c_contiguous

    contiguous_leaf, contiguous_tree = _leaf({"value": contiguous})
    noncontiguous_leaf, noncontiguous_tree = _leaf({"value": noncontiguous})
    assert contiguous_leaf == noncontiguous_leaf
    assert contiguous_tree == noncontiguous_tree
    assert contiguous_leaf["dtype"] == "bfloat16"
    assert contiguous_leaf["shape"] == [2, 3]

    changed = contiguous.copy()
    changed[0, 0] = 1.5
    changed_leaf, changed_tree = _leaf({"value": changed})
    assert changed_leaf["sha256"] != contiguous_leaf["sha256"]
    assert changed_tree != contiguous_tree

    one_bit_changed = contiguous.copy()
    one_bit_changed.view(np.uint16)[0, 0] ^= np.uint16(1)
    one_bit_leaf, one_bit_tree = _leaf({"value": one_bit_changed})
    assert one_bit_leaf["sha256"] != contiguous_leaf["sha256"]
    assert one_bit_tree != contiguous_tree


def test_aggregate_leaf_hash_is_order_independent_and_set_exact():
    leaves = {
        '["optimizer","a"]': {"shape": [2], "dtype": "bfloat16", "sha256": "01"},
        '["optimizer","b"]': {"shape": [], "dtype": "uint32", "sha256": "02"},
        '["rng"]': {"shape": [2], "dtype": "uint32", "sha256": "03"},
    }
    permuted = dict(reversed(list(leaves.items())))
    expected = _aggregate_leaf_hash(leaves)
    assert _aggregate_leaf_hash(permuted) == expected

    missing = dict(leaves)
    missing.pop('["optimizer","b"]')
    assert _aggregate_leaf_hash(missing) != expected

    extra = dict(leaves)
    extra['["optimizer","c"]'] = {"shape": [1], "dtype": "float32", "sha256": "04"}
    assert _aggregate_leaf_hash(extra) != expected

    changed = {key: dict(value) for key, value in leaves.items()}
    changed['["rng"]']["sha256"] = "ff"
    assert _aggregate_leaf_hash(changed) != expected


def test_optimizer_counter_path_extractor_is_exact_and_fails_closed():
    pairs = [
        (("step", "value"), np.asarray(2000, dtype=np.uint32)),
        (("opt_state", "gradient_step", "value"), np.asarray(250, dtype=np.int32)),
        (("opt_state", "mini_step", "value"), np.asarray(0, dtype=np.int32)),
        (("unrelated",), np.asarray(99, dtype=np.int32)),
    ]
    assert _extract_optimizer_counters_from_pairs(pairs) == {
        "global_gradient_step": 250,
        "optimizer_micro_step": 2000,
        "gradient_accumulation_remainder": 0,
    }

    import pytest

    with pytest.raises(RuntimeError, match="missing optimizer counter"):
        _extract_optimizer_counters_from_pairs(pairs[:-2])
    with pytest.raises(RuntimeError, match="duplicate optimizer counter"):
        _extract_optimizer_counters_from_pairs(pairs + [pairs[0]])
    bad_type = list(pairs)
    bad_type[0] = (("step", "value"), np.asarray(2000.0, dtype=np.float32))
    with pytest.raises(RuntimeError, match="not an integer scalar"):
        _extract_optimizer_counters_from_pairs(bad_type)
    bad_shape = list(pairs)
    bad_shape[0] = (("step", "value"), np.asarray([2000], dtype=np.uint32))
    with pytest.raises(RuntimeError, match="not an integer scalar"):
        _extract_optimizer_counters_from_pairs(bad_shape)
