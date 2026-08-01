from __future__ import annotations

import tomllib
from pathlib import Path

from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify import (
    _dispatch_gold,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.smoke_schema import (
    load_smoke,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import RecordingTransport


RECIPE_ROOT = Path(__file__).parents[3] / "labctl" / "recipes"


def _load(name: str) -> dict[str, object]:
    with (RECIPE_ROOT / name).open("rb") as handle:
        return tomllib.load(handle)


def test_labctl_args_render_as_supported_cli_flags() -> None:
    full = _load("natural_dev_cleanroom_cpu_kvm.toml")
    smoke = _load("natural_dev_cleanroom_plumbing_smoke_cpu_kvm.toml")
    calc = _load("natural_dev_cleanroom_calc_smoke_cpu_kvm.toml")
    supported = {
        "output", "work-dir", "qcow", "qemu", "provider", "shard-index", "task-id"
    }
    for recipe in (full, smoke, calc):
        args = recipe["args"]
        assert isinstance(args, dict)
        assert set(args) <= supported
        assert "work-dir" in args
        assert all("_" not in key for key in args)


def test_full_corpus_sweep_binds_the_hyphenated_shard_flag() -> None:
    full = _load("natural_dev_cleanroom_cpu_kvm.toml")
    sweep = full["sweep"]
    assert sweep == {"arg": "shard-index", "start": 0, "end": 3, "throttle": 4}
    args = full["args"]
    assert isinstance(args, dict) and args["shard-index"] == "0"


def test_calc_does_not_rebind_while_formula_edit_is_unconfirmed() -> None:
    task = next(task for task in load_smoke().tasks if task.app == "calc")
    transport = RecordingTransport()
    actions, runtime_bindings = _dispatch_gold(
        transport,
        task,
        {"cell": (125, 185)},
    )
    assert actions
    assert runtime_bindings == []
    step_two_classes = [
        row["action_class"] for row in actions if row["semantic_step"] == 2
    ]
    assert step_two_classes == [
        "key_chord",
        "key_chord",
        "key_chord",
        "coalesced_type",
    ]
    assert transport.audit.held_buttons == set()
    assert transport.audit.held_keys == set()
