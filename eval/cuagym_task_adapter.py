"""Adapt one CUA-Gym task bundle into an OSWorld-compatible task_config.

Setup steps run verbatim through the pinned SetupController; relative
download URLs are satisfied by pre-seeding the controller's cache dir from
the bundle. The python evaluator is replaced with a no-op stub (always 0,
no VM access) so env.evaluate() never crashes; the real reward comes from
cuagym_reward after the episode. Round 0 covers desktop families only.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASKS_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/yll/gym_rollout_assets/dataset/tasks_v1"
)
DEFAULT_TASKS_PARQUET = Path(
    "/fast/project/HFMI_SynergyUnit/yll/gym_rollout_assets/dataset/CUA-Gym/data/tasks.parquet"
)

DESKTOP_FAMILIES = frozenset({"desktop_office", "desktop", "multi_apps", "other"})

NOOP_EVALUATOR = {
    "func": "exact_match",
    "result": {"type": "rule", "rules": "cuagym_reward_pending"},
    "expected": {"type": "rule", "rules": {"expected": "cuagym_reward_is_external"}},
}

METADATA_COLUMNS = [
    "id",
    "instruction",
    "app_type",
    "app_family",
    "platform",
    "difficulty",
    "setup_kind",
]


class UnsupportedTaskError(ValueError):
    pass


@dataclass(frozen=True)
class CacheSeed:
    url: str
    vm_path: str
    cache_relpath: str
    source_path: Path


@dataclass(frozen=True)
class AdaptedTask:
    task_id: str
    app_family: str
    app_type: str
    instruction: str
    difficulty: str
    bundle_dir: Path
    reward_script: Path
    task_config: dict
    cache_seeds: tuple[CacheSeed, ...]


def cache_relpath(task_id: str, url: str, vm_path: str) -> str:
    return "{}/{}_{}".format(
        task_id, uuid.uuid5(uuid.NAMESPACE_URL, url), Path(vm_path).name
    )


def _resolve_bundle_file(bundle_dir: Path, url: str) -> Path:
    rel = url[2:] if url.startswith("./") else url
    src = (bundle_dir / rel).resolve()
    if bundle_dir.resolve() not in src.parents:
        raise UnsupportedTaskError(f"setup url escapes bundle dir: {url}")
    if not src.is_file():
        raise FileNotFoundError(f"bundle file missing for setup url {url!r}: {src}")
    return src


def _collect_cache_seeds(task_id: str, bundle_dir: Path, config: list[dict]) -> tuple[CacheSeed, ...]:
    seeds: list[CacheSeed] = []
    for step in config:
        if step.get("type") != "download":
            continue
        for f in step.get("parameters", {}).get("files", []):
            url = f["url"]
            vm_path = f["path"]
            if url.startswith(("http://", "https://")):
                continue
            seeds.append(
                CacheSeed(
                    url=url,
                    vm_path=vm_path,
                    cache_relpath=cache_relpath(task_id, url, vm_path),
                    source_path=_resolve_bundle_file(bundle_dir, url),
                )
            )
    return tuple(seeds)


def load_metadata(task_id: str, parquet_path: Path | str = DEFAULT_TASKS_PARQUET) -> dict:
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=METADATA_COLUMNS)
    rows = df[df["id"] == task_id]
    if rows.empty:
        raise KeyError(f"task_id not found in tasks parquet: {task_id}")
    return {k: ("" if pd.isna(v) else v) for k, v in rows.iloc[0].to_dict().items()}


def load_task(
    task_id: str,
    app_family: str,
    tasks_root: Path | str = DEFAULT_TASKS_ROOT,
    difficulty: str = "",
) -> AdaptedTask:
    if app_family not in DESKTOP_FAMILIES:
        raise UnsupportedTaskError(
            f"app_family {app_family!r} out of round-0 scope "
            f"(supported: {sorted(DESKTOP_FAMILIES)}): {task_id}"
        )
    bundle_dir = Path(tasks_root) / task_id
    task_json_path = bundle_dir / "task.json"
    if not task_json_path.is_file():
        raise FileNotFoundError(f"task.json missing: {task_json_path}")
    task = json.loads(task_json_path.read_text())
    if task.get("id") != task_id:
        raise ValueError(f"task.json id {task.get('id')!r} != bundle dir {task_id!r}")
    reward_script = bundle_dir / "reward.py"
    if not reward_script.is_file():
        raise FileNotFoundError(f"reward.py missing: {reward_script}")

    config = task.get("config", [])
    evaluator = dict(NOOP_EVALUATOR)
    postconfig = task.get("evaluator", {}).get("postconfig")
    if postconfig:
        evaluator["postconfig"] = postconfig

    task_config = {
        "id": task["id"],
        "instruction": task["instruction"],
        "config": config,
        "evaluator": evaluator,
    }
    return AdaptedTask(
        task_id=task["id"],
        app_family=app_family,
        app_type=task.get("app_type", ""),
        instruction=task["instruction"],
        difficulty=difficulty or task.get("difficulty", ""),
        bundle_dir=bundle_dir,
        reward_script=reward_script,
        task_config=task_config,
        cache_seeds=_collect_cache_seeds(task["id"], bundle_dir, config),
    )


def seed_cache(adapted: AdaptedTask, cache_dir: Path | str) -> list[Path]:
    import shutil

    seeded: list[Path] = []
    for seed in adapted.cache_seeds:
        dst = Path(cache_dir) / seed.cache_relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed.source_path, dst)
        seeded.append(dst)
    return seeded
