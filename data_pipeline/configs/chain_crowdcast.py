"""Labctl chain for canonical Crowd-Cast SFT data."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task

from pipeline.lib.manifest import file_sha256_short
from pipeline.lib.realign import CLOSED_STATUSES

TAG = "crowdcast_canonical_v1"
ANNOTATION_MODEL = "Kimi-K2.6"
TRAINING_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ANNOTATION_FPS = 0.5
TRAINING_FPS = 1.0


def _required_dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    path = Path(value).resolve()
    if not path.is_dir():
        raise RuntimeError(f"{name} must be an existing directory: {path}")
    return path


def _required_file(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(f"{name} must be an existing file: {path}")
    return path


def _manifest(path: Path, expected: dict[str, object], *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label} manifest: {path}") from exc
    if not isinstance(value, dict) or {key: value.get(key) for key in expected} != expected:
        raise RuntimeError(f"{label} contract mismatch: {path}")
    return value


def _repo() -> str:
    path = _required_dir("JUERGEN_REPO")
    if not (path / "pipeline").is_dir():
        raise RuntimeError(f"JUERGEN_REPO is not a Juergen checkout: {path}")
    return str(path)


def _task(name: str, repo: str, entrypoint: str):
    path = Path(repo) / entrypoint
    if not path.is_file():
        raise RuntimeError(f"missing Juergen entrypoint: {entrypoint}")
    config = pipeline_task()
    config.name = name
    config.resources.n_gpus = 0
    config.resources.time = "6:00:00"
    config.resources.mem = "128GB"
    config.resources.cpus = 32
    config.entrypoint.repo_paths = {"berlin": repo}
    config.entrypoint.path = entrypoint
    config.output.dataset_version = name
    config.dataset = ""
    return config


def _child(config) -> dict:
    value = config.to_dict()
    value["trigger"] = "on_complete"
    return value


def get_config():
    repo = _repo()
    datasets = _required_dir("LABCTL_DATASETS_ROOT")
    master = _required_dir("CROWDCAST_MASTER_DIR")
    clips = _required_file("CROWDCAST_CLIPS_MANIFEST")
    omegalax = _required_dir("OMEGALAX_REPO")
    _manifest(
        master / "manifest.json",
        {
            "artifact_type": "juergen_annotation_frames_master",
            "schema_version": 1,
            "target_height": 720,
            "jpeg_quality": 92,
        },
        label="Crowd-Cast Stage01",
    )
    realigned = _manifest(
        clips.parent / "manifest.json",
        {
            "artifact_type": "juergen_annotation_clip_manifest_realigned",
            "schema_version": 1,
            "clips_file": "clips_manifest.jsonl",
        },
        label="Crowd-Cast Stage02",
    )
    if clips != (clips.parent / str(realigned["clips_file"])).resolve():
        raise RuntimeError("CROWDCAST_CLIPS_MANIFEST is not the Stage02 canonical clips file")
    if file_sha256_short(clips, n=64) != realigned.get("clips_sha256"):
        raise RuntimeError("CROWDCAST_CLIPS_MANIFEST digest does not match Stage02")
    clip_rows = [json.loads(line) for line in clips.read_text().splitlines() if line.strip()]
    if not clip_rows:
        raise RuntimeError("CROWDCAST_CLIPS_MANIFEST is empty")
    if any(
        row.get("alignment_closed") is not True
        or row.get("alignment_status") not in CLOSED_STATUSES
        for row in clip_rows
    ):
        raise RuntimeError("CROWDCAST_CLIPS_MANIFEST contains an unclosed alignment")
    for relative in (
        "scripts/measure_message_lengths_from_chat.py",
        "scripts/build_sft_records_from_chat.py",
    ):
        if not (omegalax / relative).is_file():
            raise RuntimeError(f"OMEGALAX_REPO is missing {relative}")

    filter_name = f"{TAG}_stage_03_filter"
    goals_name = f"{TAG}_stage_03b_describe_extract"
    conversations_name = f"{TAG}_stage_04_conversations"
    lengths_name = f"{TAG}_stage_05_lengths"
    records_name = f"{TAG}_stage_06_records"

    stage_03 = _task(filter_name, repo, "pipeline/stage_03_filter.py")
    stage_03.entrypoint.args.frames_master_dir = str(master)
    stage_03.entrypoint.args.clips_manifest = str(clips)
    stage_03.entrypoint.args.num_workers = 32
    stage_03.inputs.source = {"kind": "path", "path_per_cluster": {"berlin": str(master)}}

    stage_03b = _task(goals_name, repo, "pipeline/annotation/stage_annotate.py")
    stage_03b.resources.time = "24:00:00"
    stage_03b.entrypoint.args.filter_dir = str(datasets / filter_name)
    stage_03b.entrypoint.args.fps = ANNOTATION_FPS
    stage_03b.entrypoint.args.model = ANNOTATION_MODEL
    stage_03b.entrypoint.args.target_tpm = 1_800_000
    stage_03b.entrypoint.args.max_workers = 64
    stage_03b.inputs.source = {"kind": "dataset", "version": filter_name}

    stage_04 = _task(conversations_name, repo, "pipeline/stage_04_build_conversations.py")
    stage_04.entrypoint.args.filter_dir = str(datasets / filter_name)
    stage_04.entrypoint.args.goals_dir = str(datasets / goals_name)
    stage_04.entrypoint.args.fps = TRAINING_FPS
    stage_04.entrypoint.args.num_workers = 32
    stage_04.inputs.source = {"kind": "dataset", "version": goals_name}

    stage_05 = _task(lengths_name, repo, "pipeline/stage_05_measure_lengths.py")
    stage_05.entrypoint.args.source_path = str(datasets / conversations_name)
    stage_05.entrypoint.args.omegalax_repo = str(omegalax)
    stage_05.entrypoint.args.model_id = TRAINING_MODEL
    stage_05.entrypoint.args.processor = TRAINING_MODEL
    stage_05.entrypoint.args.num_workers = 32
    stage_05.inputs.source = {"kind": "dataset", "version": conversations_name}

    stage_06 = _task(records_name, repo, "pipeline/stage_06_training_records.py")
    stage_06.entrypoint.args.source_path = str(datasets / conversations_name)
    stage_06.entrypoint.args.message_lengths_path = str(datasets / lengths_name)
    stage_06.entrypoint.args.omegalax_repo = str(omegalax)
    stage_06.entrypoint.args.model_id = TRAINING_MODEL
    stage_06.entrypoint.args.processor = TRAINING_MODEL
    stage_06.entrypoint.args.max_length = 32768
    stage_06.entrypoint.args.records_per_shard = 1024
    stage_06.entrypoint.args.num_workers = 32
    stage_06.entrypoint.args.val_fraction = 0.02
    stage_06.inputs.source = {"kind": "dataset", "version": lengths_name}

    stage_05.children = [_child(stage_06)]
    stage_04.children = [_child(stage_05)]
    stage_03b.children = [_child(stage_04)]
    stage_03.children = [_child(stage_03b)]
    return stage_03
