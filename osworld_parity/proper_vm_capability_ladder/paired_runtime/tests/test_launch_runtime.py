from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ..launch import audit
from ..runtime import _compact_action, _model_context, _native_action


PACKAGE = Path(__file__).resolve().parents[1]
OSWORLD_PARITY = Path(__file__).resolve().parents[3]
CONFIG = PACKAGE / "config" / "short_task_passk.template.json"
RECIPES = OSWORLD_PARITY / "labctl" / "recipes"


def test_strict_model_action_parsers() -> None:
    native = _native_action(
        "The target is down and to the right.\n10 20 0 ; +LMB -LMB", 1
    )
    assert native["semantic_step"] == 1
    compact = _compact_action(
        "The target is nearby.\n-3 4 0 ; +LMB -LMB", 2
    )
    assert compact["actions"] == ["-3 4 0 ; +LMB -LMB"]
    with pytest.raises(ValueError):
        _compact_action("One.\nTwo.\n0 0 0 ; +LMB -LMB", 2)
    with pytest.raises(ValueError):
        _native_action(
            '{"schema":"native_absolute_sequence_v1","semantic_step":1}', 1
        )


def test_model_context_binds_next_step_and_evaluator_history() -> None:
    history = ({"source": "gold_semantic_prefix", "semantic_step_index": 1},)
    context = _model_context("Do the task", 2, history)
    assert "Next semantic step index: 2" in context
    assert json.dumps(history, sort_keys=True, separators=(",", ":")) in context


def test_launch_bundle_is_resource_valid_and_explicitly_blocked() -> None:
    recipes = [RECIPES / f"paired_short_passk_shard_{index}_gpu_kvm.toml" for index in range(5)]
    report = audit(CONFIG, recipes, RECIPES / "paired_short_passk_aggregate_cpu.toml")
    assert report["status"] == "blocked_on_explicit_pins"
    assert report["pass_at_k"] == [1, 4, 8]
    assert report["shard_count"] == 5
    assert all(row["gpus"] == 2 for row in report["resources"][:-1])
    assert report["resources"][-1]["gpus"] == 0


def test_template_binds_runtime_and_prompts() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = Path(__file__).resolve().parents[1] / "runtime.py"
    assert raw["runtime"]["source_sha256"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
    prompts = Path(__file__).resolve().parents[1] / "prompts"
    by_name = {arm["name"]: arm for arm in raw["arms"]}
    for name, arm in by_name.items():
        prompt = prompts / f"{name}.txt"
        assert arm["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
