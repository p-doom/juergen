from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("proper_task_pilot_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)
RECIPE_DIR = ROOT / "osworld_parity" / "labctl" / "recipes"
PILOT_TASKS = Path(__file__).with_name("pilot_tasks.txt")
SMOKE_TASKS = Path(__file__).with_name("smoke_tasks.txt")
RECIPES = (
    "proper_task_pilot_absolute_r32.toml",
    "proper_task_pilot_relative_r256.toml",
    "proper_task_smoke_absolute_r32.toml",
    "proper_task_smoke_relative_r256.toml",
)


def _recipe(name: str) -> dict[str, Any]:
    with (RECIPE_DIR / name).open("rb") as handle:
        return tomllib.load(handle)


def _interpolate(value: Any, recipe: dict[str, Any], output: Path) -> str:
    text = str(value)
    for name, config in recipe["inputs"].items():
        text = text.replace(f"{{inputs.{name}.path}}", config["path"])
    return text.replace("{outputs.result.path}", str(output))


def _runtime_config(recipe: dict[str, Any], tmp_path: Path) -> Any:
    argv = [
        f"--{key}={_interpolate(value, recipe, tmp_path / 'out')}"
        for key, value in recipe["args"].items()
        if key not in {"tasks_file", "canonical_tasks_file", "train_split", "heldout_split"}
    ]
    argv.extend(
        [
            f"--tasks_file={PILOT_TASKS if recipe['args']['mode'] in {'pilot', 'probe_seed'} else SMOKE_TASKS}",
            f"--canonical_tasks_file={PILOT_TASKS}",
            f"--train_split={ROOT / 'osworld_parity/split/osworld_train.json'}",
            f"--heldout_split={ROOT / 'osworld_parity/split/osworld_eval_heldout.json'}",
            "--base_url=http://127.0.0.1:18000/v1",
            "--model=policy",
            "--api_key=x",
        ]
    )
    return runtime.parse_config(argv)


@pytest.mark.parametrize("recipe_name", RECIPES)
def test_labctl_recipe_maps_exactly_to_runtime_config(
    recipe_name: str, tmp_path: Path
) -> None:
    recipe = _recipe(recipe_name)
    config = _runtime_config(recipe, tmp_path)

    assert recipe["repo"] == "juergen_rft"
    assert recipe["resources"]["gpus"] == 1
    assert recipe["resources"]["cpus"] == 32
    assert recipe["resources"]["mem"] == "128GB"
    assert "--no-requeue" in recipe["resources"]["sbatch_extra"]
    assert "--exclude=hai001,hai005" in recipe["resources"]["sbatch_extra"]
    assert "--deadline=2026-07-31T04:55:00" in recipe["resources"]["sbatch_extra"]
    assert config.screen_width == 1920
    assert config.screen_height == 1080
    assert config.snapshot_name == "osworld_ready"
    assert config.max_steps == 15
    assert config.n_history_frames == 4
    assert config.pause == 1.0
    assert config.temperature == 0.0
    assert config.server_max_model_len == 16384
    assert config.max_completion_tokens == 1024
    assert config.port_lock_dir == Path("/tmp/osworld_port_locks")
    assert config.port_base == 30000
    assert config.provider_sha256 == (
        "8d7e4a2602a81895a712bb275b327bd3270d43675acf6e61484be5775caefafc"
    )
    assert config.runtime_files_sha256 == (
        "abc961f70ad2278ab5edc2822759ba1f00e4e6106dcf0ee944e5e14283aa0bb3"
    )
    assert runtime._runtime_provenance(config.runtime_repo)["tree_sha256"] == (
        config.runtime_files_sha256
    )
    assert config.checkpoint_manifest_sha256 == recipe["args"][
        "checkpoint_manifest_sha256"
    ]
    assert config.expected_lora_rank == (32 if config.action_format == "absolute" else 256)
    assert config.reverse_tasks is (config.action_format == "move_rel")

    pool_config = runtime._desktop_pool_config(config)
    assert pool_config.min_ready_sessions == 1
    assert pool_config.max_sessions == 1
    assert pool_config.max_rollouts_per_session == 1
    assert pool_config.port_lock_dir == Path("/tmp/osworld_port_locks")


def test_pilot_and_smoke_task_inputs_are_exact_and_train_only() -> None:
    pilot_ids = PILOT_TASKS.read_text(encoding="utf-8").splitlines()
    smoke_ids = SMOKE_TASKS.read_text(encoding="utf-8").splitlines()
    train = runtime._json_object(ROOT / "osworld_parity/split/osworld_train.json")
    heldout = runtime._json_object(
        ROOT / "osworld_parity/split/osworld_eval_heldout.json"
    )

    assert runtime._sha256(PILOT_TASKS) == runtime.CANONICAL_PILOT_SHA256
    assert runtime._sha256(SMOKE_TASKS) == (
        "46e8a4340b022d15ac52984cd896d9bffdc3d922598c08c1c1f52c38fe0c6891"
    )
    assert len(pilot_ids) == 12
    assert len(smoke_ids) == 1
    assert set(smoke_ids).isdisjoint(pilot_ids)
    assert set(pilot_ids + smoke_ids) <= runtime._split_ids(train)
    assert set(pilot_ids + smoke_ids).isdisjoint(runtime._split_ids(heldout))


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"validity": "infra_invalid", "task_reward": None},
        {"validity": "valid", "task_reward": None},
        {"validity": "valid", "task_reward": float("nan")},
        {"validity": "valid", "task_reward": float("inf")},
    ],
)
def test_missing_nan_and_infra_results_fail_closed(result: Any) -> None:
    assert not runtime.result_is_infra_valid(result, trace_has_error=False)


def test_finite_reward_is_valid_only_without_trace_error() -> None:
    result = {"validity": "valid", "task_reward": 0.0}
    assert runtime.result_is_infra_valid(result, trace_has_error=False)
    assert not runtime.result_is_infra_valid(result, trace_has_error=True)


def test_phase2_probe_recipe_is_exact_and_gate_is_open(tmp_path: Path) -> None:
    recipe = _recipe("phase2_move_rel_probe_seed101.toml")
    config = _runtime_config(recipe, tmp_path)
    assert config.mode == "probe_seed"
    assert config.action_format == "move_rel"
    assert config.reverse_tasks is False
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.sampling_seed == 101
    assert recipe["resources"] == {
        "gpus": 1,
        "cpus": 8,
        "mem": "32GB",
        "time": "01:00:00",
        "qos": "low",
        "account": "hfmi_synergyunit",
        "partition": "standard",
        "sbatch_extra": [
            "--ntasks=1",
            "--ntasks-per-node=1",
            "--no-requeue",
            "--exclude=hai001,hai005",
            "--signal=B:TERM@120",
            "--deadline=2026-07-31T04:40:00",
        ],
    }
    assert ".source_path" in " ".join(recipe["command"])
    gate = runtime._validate_probe_gate(
        config, PILOT_TASKS.read_text(encoding="utf-8").splitlines()
    )
    assert gate["materiality_gate_open"] is True
    assert gate["absolute_successes"] == 6
    assert gate["relative_successes"] == 1


def test_seeded_client_forwards_preregistered_sampling() -> None:
    class Completions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, *args: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    completions = Completions()
    wrapped = runtime._SeededCompletions(completions, top_p=0.95, seed=101)
    result = asyncio.run(wrapped.create(model="policy", temperature=0.7))
    assert result == "ok"
    assert completions.kwargs == {
        "model": "policy",
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 101,
    }


def test_runtime_preflight_rejects_short_server_context(tmp_path: Path) -> None:
    config = _runtime_config(_recipe(RECIPES[0]), tmp_path)

    with pytest.raises(runtime.PilotError, match="max_model_len.*16384"):
        runtime.validate_preflight(replace(config, server_max_model_len=4096))


def test_long_artifact_path_cannot_leak_into_vllm_tmpdir(tmp_path: Path) -> None:
    long_output = tmp_path / ("artifact-segment-" * 12)
    config = _runtime_config(_recipe(RECIPES[0]), long_output)
    safe_tmpdir = Path("/tmp/ptp_135465_abcd12")

    assert len(str(config.output)) + 1 + 36 >= 107
    runtime.validate_zmq_tmpdir(safe_tmpdir)
    with pytest.raises(runtime.PilotError, match="too long"):
        runtime.validate_zmq_tmpdir(config.output / "tmp")


def test_runner_uses_shared_osworld_locks_without_deleting_them() -> None:
    stage = Path(__file__).with_name("run_stage.sh").read_text(encoding="utf-8")

    assert 'OSWORLD_PORT_BASE="${A[port_base]}"' in stage
    assert 'mkdir -m 700 -p "${A[port_lock_dir]}"' in stage
    assert 'rm -rf -- "${A[port_lock_dir]}"' not in stage
    assert 'rm -rf -- "$JOB_TMP"' in stage
    assert "trap 'exit 143' TERM INT" in stage


@pytest.mark.parametrize("recipe_name", RECIPES)
def test_labctl_semantically_validates_recipe(recipe_name: str) -> None:
    subprocess.run(
        ["labctl", "validate", str(RECIPE_DIR / recipe_name)],
        check=True,
        capture_output=True,
        text=True,
    )
