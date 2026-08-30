"""Materialize OSWorld task assets from one local, repository-shaped bundle."""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

OSWORLD_ASSET_PREFIX = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
)
_OFFLINE_FILE_TYPE = "offline_file"


def stage_offline_task(
    task_config: dict[str, Any], asset_bundle: Path
) -> dict[str, Any]:
    """Resolve every network-backed task asset before a desktop is acquired."""
    task_id = str(task_config.get("id") or "task")
    staged = copy.deepcopy(task_config)
    config = staged.get("config", [])
    if not isinstance(config, list):
        raise TypeError(f"OSWorld task {task_id!r} has a non-list config")
    staged["config"] = [
        _stage_setup_step(step, asset_bundle, task_id, index)
        for index, step in enumerate(config)
    ]

    evaluator = staged.get("evaluator")
    if evaluator is not None:
        if not isinstance(evaluator, dict):
            raise TypeError(f"OSWorld task {task_id!r} has a non-object evaluator")
        postconfig = evaluator.get("postconfig")
        if postconfig is not None:
            if not isinstance(postconfig, list):
                raise TypeError(
                    f"OSWorld task {task_id!r} has a non-list evaluator postconfig"
                )
            evaluator["postconfig"] = [
                _stage_setup_step(step, asset_bundle, task_id, index)
                for index, step in enumerate(postconfig)
            ]
        staged["evaluator"] = _stage_cloud_files(evaluator, asset_bundle, task_id)
    assert_offline_task(staged)
    return staged


def assert_offline_task(task_config: dict[str, Any]) -> None:
    """Reject any setup or evaluator path that could reach the network."""
    task_id = str(task_config.get("id") or "task")
    for owner, steps in (
        ("config", task_config.get("config", [])),
        (
            "evaluator postconfig",
            task_config.get("evaluator", {}).get("postconfig", [])
            if isinstance(task_config.get("evaluator"), dict)
            else [],
        ),
    ):
        if not isinstance(steps, list):
            raise TypeError(f"OSWorld task {task_id!r} has a non-list {owner}")
        if any(
            isinstance(step, dict) and step.get("type") == "download" for step in steps
        ):
            raise ValueError(
                f"OSWorld task {task_id!r} has an unstaged download in {owner}"
            )
    if _contains_type(task_config.get("evaluator"), "cloud_file"):
        raise ValueError(
            f"OSWorld task {task_id!r} has an unstaged cloud_file evaluator"
        )


def get_offline_file(_env: Any, config: dict[str, Any]) -> str | list[str]:
    """OSWorld getter for files already validated in the local asset bundle."""
    raw_paths = config.get("path")
    paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
    if not paths or not all(
        isinstance(path, str) and Path(path).is_file() for path in paths
    ):
        raise FileNotFoundError(f"invalid staged OSWorld asset paths: {raw_paths!r}")
    gives = config.get("gives", [0])
    if not isinstance(gives, list) or not all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(paths)
        for index in gives
    ):
        raise ValueError(f"invalid staged OSWorld asset selection: {gives!r}")
    selected_indices = set(gives)
    selected = [path for index, path in enumerate(paths) if index in selected_indices]
    return selected[0] if len(selected) == 1 else selected


def _stage_setup_step(
    step: object,
    asset_bundle: Path,
    task_id: str,
    index: int,
) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise TypeError(f"OSWorld task {task_id!r} setup step {index} is not an object")
    if step.get("type") != "download":
        return step
    if set(step) != {"type", "parameters"} or not isinstance(step["parameters"], dict):
        raise ValueError(f"OSWorld task {task_id!r} download step {index} is malformed")
    parameters = step["parameters"]
    if set(parameters) != {"files"} or not isinstance(parameters["files"], list):
        raise ValueError(f"OSWorld task {task_id!r} download step {index} is malformed")
    if not parameters["files"]:
        raise ValueError(f"OSWorld task {task_id!r} download step {index} has no files")
    uploads = [
        _stage_download(entry, asset_bundle, task_id, index)
        for entry in parameters["files"]
    ]
    return {"type": "upload_file", "parameters": {"files": uploads}}


def _stage_download(
    entry: object,
    asset_bundle: Path,
    task_id: str,
    step_index: int,
) -> dict[str, str]:
    if not isinstance(entry, dict) or set(entry) != {"url", "path"}:
        raise ValueError(
            f"OSWorld task {task_id!r} download step {step_index} has a malformed file"
        )
    url, guest_path = entry["url"], entry["path"]
    if not isinstance(url, str) or not isinstance(guest_path, str) or not guest_path:
        raise ValueError(
            f"OSWorld task {task_id!r} download step {step_index} has invalid strings"
        )
    source = _bundle_file(url, asset_bundle, task_id)
    return {"local_path": str(source), "path": guest_path}


def _stage_cloud_files(value: Any, asset_bundle: Path, task_id: str) -> Any:
    if isinstance(value, list):
        return [_stage_cloud_files(item, asset_bundle, task_id) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") != "cloud_file":
        return {
            key: _stage_cloud_files(item, asset_bundle, task_id)
            for key, item in value.items()
        }

    allowed = {"type", "path", "dest", "multi", "gives"}
    if set(value) - allowed:
        raise ValueError(
            f"OSWorld task {task_id!r} has unsupported cloud_file fields: "
            f"{sorted(set(value) - allowed)}"
        )
    multi = value.get("multi", False)
    if not isinstance(multi, bool):
        raise TypeError(f"OSWorld task {task_id!r} has invalid cloud_file multi")
    raw_paths, raw_dests = value.get("path"), value.get("dest")
    paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
    dests = raw_dests if isinstance(raw_dests, list) else [raw_dests]
    if multi != isinstance(raw_paths, list) or multi != isinstance(raw_dests, list):
        raise ValueError(f"OSWorld task {task_id!r} has mismatched cloud_file shapes")
    if (
        not paths
        or len(paths) != len(dests)
        or not all(
            isinstance(path, str) and isinstance(dest, str) and dest
            for path, dest in zip(paths, dests, strict=True)
        )
    ):
        raise ValueError(f"OSWorld task {task_id!r} has invalid cloud_file entries")
    local_paths = [str(_bundle_file(path, asset_bundle, task_id)) for path in paths]
    gives = value.get("gives", [0])
    if not isinstance(gives, list) or not all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(local_paths)
        for index in gives
    ):
        raise ValueError(f"OSWorld task {task_id!r} has invalid cloud_file gives")
    return {
        "type": _OFFLINE_FILE_TYPE,
        "path": local_paths if multi else local_paths[0],
        "gives": sorted(set(gives)),
    }


def _bundle_file(url: str, asset_bundle: Path, task_id: str) -> Path:
    split = urlsplit(url)
    stripped = split._replace(query="", fragment="").geturl()
    if not stripped.startswith(OSWORLD_ASSET_PREFIX):
        raise ValueError(
            f"OSWorld task {task_id!r} uses an unsupported asset URL: {url!r}"
        )
    relative = PurePosixPath(unquote(stripped[len(OSWORLD_ASSET_PREFIX) :]))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"OSWorld task {task_id!r} has an unsafe asset URL: {url!r}")
    try:
        source = (asset_bundle / Path(*relative.parts)).resolve(strict=True)
        source.relative_to(asset_bundle)
    except (FileNotFoundError, ValueError) as exc:
        raise FileNotFoundError(
            f"OSWorld task {task_id!r} asset is absent from the bundle: {relative}"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(
            f"OSWorld task {task_id!r} asset is not a file: {relative}"
        )
    return source


def _contains_type(value: Any, expected: str) -> bool:
    if isinstance(value, dict):
        return value.get("type") == expected or any(
            _contains_type(item, expected) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_type(item, expected) for item in value)
    return False
