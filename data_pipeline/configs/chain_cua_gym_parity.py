"""Labctl chain for the CUA-Gym action-format parity stream."""

from __future__ import annotations

import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task

from pipeline.lib.omegalax import attest_omegalax, attest_processor_snapshot

TAG = "cuagym_action_format_parity_v1"


def _required_dir(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    path = Path(value).resolve()
    if not path.is_dir():
        raise RuntimeError(f"{name} is not a directory: {path}")
    return str(path)


def _required_file(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(f"{name} is not a file: {path}")
    return str(path)


def _repo() -> str:
    path = Path(_required_dir("JUERGEN_REPO"))
    if not (path / "pipeline").is_dir():
        raise RuntimeError(f"JUERGEN_REPO is not a Juergen checkout: {path}")
    return str(path)


def _entrypoint(repo: str, relative: str) -> str:
    if not (Path(repo) / relative).is_file():
        raise RuntimeError(f"missing Juergen entrypoint: {relative}")
    return relative


def _child(config) -> dict:
    value = config.to_dict()
    value["trigger"] = "on_complete"
    return value


def _task(name: str, repo: str, entrypoint: str):
    config = pipeline_task()
    config.name = name
    config.resources.n_gpus = 0
    config.resources.time = "6:00:00"
    config.resources.mem = "128GB"
    config.resources.cpus = 32
    config.entrypoint.repo_paths = {"berlin": repo}
    config.entrypoint.path = _entrypoint(repo, entrypoint)
    config.output.dataset_version = name
    config.dataset = ""
    return config


def get_config():
    repo = _repo()
    datasets = Path(_required_dir("LABCTL_DATASETS_ROOT"))
    screenshots = _required_dir("CUA_GYM_SCREENSHOTS_DIR")
    trajectories = _required_file("CUA_GYM_TRAJECTORIES")
    omegalax = _required_dir("OMEGALAX_REPO")
    attest_omegalax(Path(omegalax))
    processor_snapshot = attest_processor_snapshot(Path(_required_dir("SFT_PROCESSOR_SNAPSHOT")))
    for relative in (
        "scripts/measure_message_lengths_from_chat.py",
        "scripts/build_sft_records_from_chat.py",
    ):
        if not (Path(omegalax) / relative).is_file():
            raise RuntimeError(f"OMEGALAX_REPO is missing {relative}")

    stage_01_name = f"{TAG}_stage_01_images"
    stage_03_name = f"{TAG}_stage_03_curated"
    stage_04_name = f"{TAG}_stage_04_conversations"
    stage_05_name = f"{TAG}_stage_05_lengths"
    stage_06_name = f"{TAG}_stage_06_records"

    stage_01 = _task(stage_01_name, repo, "pipeline/cua_gym/stage_01_image_store.py")
    stage_01.entrypoint.args.screenshots_dir = screenshots
    stage_01.entrypoint.args.workers = 32
    stage_01.inputs.source = {"kind": "path", "path_per_cluster": {"berlin": screenshots}}

    stage_03 = _task(stage_03_name, repo, "pipeline/cua_gym/stage_03_curate_trajectories.py")
    stage_03.entrypoint.args.source_path = trajectories
    stage_03.inputs.source = {
        "kind": "path",
        "path_per_cluster": {"berlin": trajectories},
    }

    stage_04 = _task(stage_04_name, repo, "pipeline/cua_gym/stage_04_build_conversations.py")
    stage_04.entrypoint.args.curated_trajectories = str(datasets / stage_03_name)
    stage_04.entrypoint.args.image_store = str(datasets / stage_01_name)
    stage_04.inputs.source = {"kind": "dataset", "version": stage_03_name}

    stage_05 = _task(stage_05_name, repo, "pipeline/stage_05_measure_lengths.py")
    stage_05.entrypoint.args.source_path = str(datasets / stage_04_name)
    stage_05.entrypoint.args.omegalax_repo = omegalax
    stage_05.entrypoint.args.processor_snapshot = processor_snapshot["path"]
    stage_05.entrypoint.args.num_workers = 32
    stage_05.inputs.source = {"kind": "dataset", "version": stage_04_name}

    stage_06 = _task(stage_06_name, repo, "pipeline/stage_06_training_records.py")
    stage_06.entrypoint.args.source_path = str(datasets / stage_04_name)
    stage_06.entrypoint.args.message_lengths_path = str(datasets / stage_05_name)
    stage_06.entrypoint.args.omegalax_repo = omegalax
    stage_06.entrypoint.args.processor_snapshot = processor_snapshot["path"]
    stage_06.entrypoint.args.max_length = 32768
    stage_06.entrypoint.args.records_per_shard = 1024
    stage_06.entrypoint.args.num_workers = 32
    stage_06.entrypoint.args.val_fraction = 0.02
    stage_06.inputs.source = {"kind": "dataset", "version": stage_05_name}

    stage_05.children = [_child(stage_06)]
    stage_04.children = [_child(stage_05)]
    stage_03.children = [_child(stage_04)]
    stage_01.children = [_child(stage_03)]
    return stage_01
