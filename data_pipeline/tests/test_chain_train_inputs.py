"""The training chain must refuse an input path that is not on disk.

Nothing downstream catches one. A recipe whose input directory does not exist
validates clean and is submitted, and the failure surfaces as a job that dies on
an allocated node -- or, for a chained stage, as the whole tail going
DependencyNeverSatisfied. ``chain_annotate`` has statted its two external inputs
since it was written; ``chain_train`` had no such check at all, so ``GOALS_DIR``
(produced by a *different* chain, hence not creatable by this one) went straight
through.

``pmanager`` is not installed in this venv -- the ``pmanager launch`` process
loads these configs, not the stage venv -- so the schema it provides is faked
down to what the configs touch: nested attribute assignment.
"""

from __future__ import annotations

import sys
import types

import pytest


class _Bag:
    """Attribute tree that materialises children on first access."""

    def __init__(self) -> None:
        object.__setattr__(self, "_items", {})

    def __getattr__(self, name: str) -> "_Bag":
        items = object.__getattribute__(self, "_items")
        if name not in items:
            items[name] = _Bag()
        return items[name]

    def __setattr__(self, name: str, value: object) -> None:
        object.__getattribute__(self, "_items")[name] = value

    def to_dict(self) -> dict:
        return dict(object.__getattribute__(self, "_items"))


@pytest.fixture
def chain_train(monkeypatch):
    schema = types.ModuleType("pmanager.configs.schema")
    schema.pipeline_task = _Bag
    configs_mod = types.ModuleType("pmanager.configs")
    root = types.ModuleType("pmanager")
    for name, mod in (
        ("pmanager", root),
        ("pmanager.configs", configs_mod),
        ("pmanager.configs.schema", schema),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "configs.chain_train", raising=False)
    import configs.chain_train as mod  # noqa: PLC0415

    return mod


def test_a_missing_goals_dir_is_refused(chain_train, monkeypatch, tmp_path):
    monkeypatch.setattr(chain_train, "GOALS_DIR", str(tmp_path / "never-built"))
    with pytest.raises(RuntimeError, match="input path does not exist"):
        chain_train.stage_04_conversations()


def test_the_goal_free_path_still_builds(chain_train, monkeypatch):
    monkeypatch.setattr(chain_train, "GOALS_DIR", None)
    cfg = chain_train.stage_04_conversations()
    assert "goals_dir" not in cfg.entrypoint.args.to_dict()


def test_a_datasets_root_that_disagrees_with_the_annotate_chain_is_refused(
    chain_train, monkeypatch
):
    monkeypatch.setattr(chain_train, "DATASETS_ROOT", "/somewhere/else")
    with pytest.raises(RuntimeError, match="DATASETS_ROOT disagrees"):
        chain_train.stage_03_filter()
