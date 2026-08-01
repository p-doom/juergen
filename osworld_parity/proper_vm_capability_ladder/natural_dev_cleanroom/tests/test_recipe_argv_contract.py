from __future__ import annotations

import tomllib
from pathlib import Path


RECIPE_ROOT = Path(__file__).parents[3] / "labctl" / "recipes"


def _load(name: str) -> dict[str, object]:
    with (RECIPE_ROOT / name).open("rb") as handle:
        return tomllib.load(handle)


def test_labctl_args_render_as_supported_cli_flags() -> None:
    full = _load("natural_dev_cleanroom_cpu_kvm.toml")
    smoke = _load("natural_dev_cleanroom_plumbing_smoke_cpu_kvm.toml")
    supported = {"output", "work-dir", "qcow", "qemu", "provider", "shard-index"}
    for recipe in (full, smoke):
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
