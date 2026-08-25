"""Tests for the checkpoint-lineage walk behind the W&B run naming.

The naming contract (``group=<producer_recipe>_<training run id>``,
``name=eval_<level>_<step>_<eval run id>``) rests entirely on
:func:`cua_micro_wandb.resolve_lineage` chasing the right two hops across
shared storage. These tests build a miniature labctl artifact tree so a change
in the sidecar/context shape fails here instead of silently mislabelling a
month of eval runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evals.micro_evals import cua_micro_wandb


def _artifact(
    path: Path,
    *,
    step: int,
    producer_recipe: str,
    producer_run_id: str | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "id": f"artifact_{producer_recipe}_{step}",
        "kind": "checkpoint",
        "metadata": {"step": step, "producer_recipe": producer_recipe},
    }
    if producer_run_id is not None:
        meta["producer_run_id"] = producer_run_id
    (path / ".meta.json").write_text(json.dumps(meta))
    return path


def _run(runs_user_dir: Path, run_id: str, *, checkpoint_input: Path | None) -> Path:
    lab = runs_user_dir / run_id / ".lab"
    lab.mkdir(parents=True, exist_ok=True)
    inputs = (
        [{"role": "checkpoint", "artifact_id": "x", "resolved_path": str(checkpoint_input)}]
        if checkpoint_input is not None
        else []
    )
    (lab / "context.json").write_text(json.dumps({"run_id": run_id, "inputs": inputs}))
    return lab / "context.json"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """The real shape: orbax checkpoint -> HF export run -> HF checkpoint.

    ``LABCTL_CONTEXT`` points at *our* run's context.json, which is how
    ``resolve_lineage`` locates the sibling runs of the same user.
    """
    runs = tmp_path / "labctl_runs" / "runs" / "alfred.nguyen"
    orbax = _artifact(
        tmp_path / "checkpoints" / "train_alias_run_aaa" / "003000",
        step=3000,
        producer_recipe="qwen3vl8b_lora_ds_v6",
        producer_run_id="run_train",
    )
    hf = _artifact(
        tmp_path / "checkpoints" / "bc_export_hf_v22_artifact_aaa" / "003000",
        step=3000,
        producer_recipe="bc_export_hf_per_checkpoint_v22",
        producer_run_id="run_export",
    )
    _run(runs, "run_export", checkpoint_input=orbax)
    _run(runs, "run_train", checkpoint_input=None)
    our_context = _run(runs, "run_eval", checkpoint_input=hf)
    monkeypatch.setenv("LABCTL_CONTEXT", str(our_context))
    return {"runs": runs, "orbax": orbax, "hf": hf}


def test_walks_export_hop_to_the_training_recipe(tree: dict[str, Path]) -> None:
    lineage = cua_micro_wandb.resolve_lineage(str(tree["hf"]))
    assert lineage.producer_recipe == "qwen3vl8b_lora_ds_v6"
    assert lineage.step == 3000
    assert lineage.chain == ("bc_export_hf_per_checkpoint_v22", "qwen3vl8b_lora_ds_v6")
    assert lineage.degraded is None
    # The GROUP's run id is the TRAINING run's (run_train), never the export
    # run's (run_export) and never this eval job's -- that is what makes it
    # one group per training run instead of one per eval.
    assert lineage.producer_run_id == "run_train"
    assert lineage.group == "qwen3vl8b_lora_ds_v6_run_train"
    # The NAME's run id is this eval job's instead, so two evals of one
    # checkpoint are two distinctly named runs inside that group.
    assert lineage.run_name("run_eval_a", "easy") == "eval_easy_3000_run_eval_a"
    assert lineage.run_name("run_eval_b", "easy") == "eval_easy_3000_run_eval_b"


def test_suite_level_separates_the_names_but_not_the_group(tree: dict[str, Path]) -> None:
    """easy and mid are two views of one training run: same group, distinct names."""
    lineage = cua_micro_wandb.resolve_lineage(str(tree["hf"]))
    easy = lineage.run_name("run_eval", "easy")
    mid = lineage.run_name("run_eval", "mid")
    assert (easy, mid) == ("eval_easy_3000_run_eval", "eval_mid_3000_run_eval")
    assert "easy" not in lineage.group and "mid" not in lineage.group


def test_name_falls_back_when_a_part_is_missing() -> None:
    lineage = cua_micro_wandb.Lineage("r", 3000, producer_run_id="run_train")
    assert lineage.run_name(None, "easy") == "eval_easy_3000"
    assert lineage.run_name("run_x", None) == "eval_3000_run_x"
    assert cua_micro_wandb.Lineage("r", None).run_name("run_x", "mid") == "eval_mid_run_x"
    assert cua_micro_wandb.Lineage(None, None).run_name(None, None) is None


@pytest.mark.parametrize(
    ("suite", "explicit", "expected"),
    [
        # The level lives only in the filename: the "suite" field inside both
        # JSONs is the same string, so inference has to key on the stem.
        ("/x/cua_micro_tasks_easy.json", None, "easy"),
        ("/x/cua_micro_tasks_mid.json", None, "mid"),
        # An explicit --suite_level always wins over the filename.
        ("/x/cua_micro_tasks_easy.json", "mid", "mid"),
        (None, "easy", "easy"),
        # No trailing _<segment> to infer from, and nothing passed.
        ("/x/suite.json", None, None),
        (None, None, None),
    ],
)
def test_resolve_suite_level(
    suite: str | None, explicit: str | None, expected: str | None
) -> None:
    args = argparse.Namespace(suite=suite, suite_level=explicit)
    assert cua_micro_wandb.resolve_suite_level(args) == expected


def test_two_runs_of_one_recipe_get_distinct_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the run id: same recipe trained twice must not merge."""
    runs = tmp_path / "labctl_runs" / "runs" / "alfred.nguyen"
    groups = set()
    for run_id in ("run_first", "run_second"):
        orbax = _artifact(
            tmp_path / "checkpoints" / f"alias_{run_id}" / "003000",
            step=3000,
            producer_recipe="same_recipe",
            producer_run_id=run_id,
        )
        hf = _artifact(
            tmp_path / "checkpoints" / f"export_{run_id}" / "003000",
            step=3000,
            producer_recipe="bc_export_hf_per_checkpoint_v22",
            producer_run_id=f"run_export_{run_id}",
        )
        _run(runs, f"run_export_{run_id}", checkpoint_input=orbax)
        _run(runs, run_id, checkpoint_input=None)
        monkeypatch.setenv(
            "LABCTL_CONTEXT", str(_run(runs, f"run_eval_{run_id}", checkpoint_input=hf))
        )
        groups.add(cua_micro_wandb.resolve_lineage(str(hf)).group)
    assert groups == {"same_recipe_run_first", "same_recipe_run_second"}


def test_evaluating_the_training_checkpoint_directly_needs_no_hop(tree: dict[str, Path]) -> None:
    """The walk is depth-agnostic: hand it the trainer's own output and stop there."""
    lineage = cua_micro_wandb.resolve_lineage(str(tree["orbax"]))
    assert lineage.run_name("run_eval", "easy") == "eval_easy_3000_run_eval"
    assert lineage.group == "qwen3vl8b_lora_ds_v6_run_train"
    assert lineage.chain == ("qwen3vl8b_lora_ds_v6",)


def test_without_labctl_context_falls_back_to_the_export_recipe(
    tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No run tree to walk (e.g. a manual invocation) still yields step + a label."""
    monkeypatch.delenv("LABCTL_CONTEXT")
    lineage = cua_micro_wandb.resolve_lineage(str(tree["hf"]))
    assert lineage.producer_recipe == "bc_export_hf_per_checkpoint_v22"
    assert lineage.step == 3000
    # Degraded, but still one group per export run rather than a global bucket.
    assert lineage.group == "bc_export_hf_per_checkpoint_v22_run_export"
    assert lineage.run_name("run_eval", "easy") == "eval_easy_3000_run_eval"
    assert lineage.degraded is not None


def test_bare_checkpoint_dir_recovers_the_step_from_the_dirname(tmp_path: Path) -> None:
    bare = tmp_path / "004500"
    bare.mkdir()
    lineage = cua_micro_wandb.resolve_lineage(str(bare))
    assert lineage.producer_recipe is None
    assert lineage.producer_run_id is None
    assert lineage.step == 4500
    assert lineage.group is None
    # No lineage at all, but step + eval run id still beat a W&B animal name.
    assert lineage.run_name("run_eval", "mid") == "eval_mid_4500_run_eval"
    assert lineage.degraded is not None


def test_no_model_path_is_not_an_error() -> None:
    lineage = cua_micro_wandb.resolve_lineage(None)
    assert (lineage.producer_recipe, lineage.step) == (None, None)
    assert lineage.group is None
    assert lineage.run_name("run_eval", None) == "eval_run_eval"


def test_cycle_in_the_lineage_terminates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt tree that points back at itself must stop, not spin."""
    runs = tmp_path / "labctl_runs" / "runs" / "alfred.nguyen"
    loop = _artifact(
        tmp_path / "checkpoints" / "loop" / "001000",
        step=1000,
        producer_recipe="loopy",
        producer_run_id="run_loop",
    )
    _run(runs, "run_loop", checkpoint_input=loop)
    monkeypatch.setenv("LABCTL_CONTEXT", str(_run(runs, "run_eval", checkpoint_input=loop)))
    lineage = cua_micro_wandb.resolve_lineage(str(loop))
    assert lineage.producer_recipe == "loopy"
    assert lineage.group == "loopy_run_loop"
    assert len(lineage.chain) == cua_micro_wandb._MAX_LINEAGE_HOPS
    assert lineage.degraded is not None


def test_disabled_run_is_a_total_no_op() -> None:
    """The eval calls log_aggregate/finish unconditionally; both must be safe."""
    run = cua_micro_wandb.WandbRun(None, cua_micro_wandb.Lineage(None, None))
    assert run.enabled is False
    assert run.url is None
    run.log_aggregate({"scores": {"overall/pass_at_1": 1.0}, "per_task": {"a": {"n": 1}}})
    run.finish(exit_code=1)
