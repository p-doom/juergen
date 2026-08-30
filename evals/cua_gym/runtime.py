"""CUA-Gym desktop tasks through the shared DesktopHarness lifecycle.

The dataset layer supplies immutable task bundles. This module is the one
runtime adapter: it selects desktop tasks, resets the harness'
leased VM, stages the bundle without a network route, and runs the bundle's
trusted reward script after the shared model loop. It does not own a pool,
model loop, or desktop proxy.
"""

from __future__ import annotations

import math
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import verifiers.v1 as vf
from desktop.vm import DesktopResetMode, GuestCommandResult
from pydantic import Field

from evals.cua_gym.manifest import PINNED_REVISION
from evals.cua_gym.models import (
    DatasetSnapshotConfig,
    SetupStep,
    TaskBundle,
    TaskId,
    TaskPlatform,
)
from evals.cua_gym.snapshot import CuaGymDatasetSnapshot
from evals.cua_gym.task_blocklist import TaskBlocklist, load_blocklist
from evals.tasks import DesktopTask, DesktopTaskData, register_preparer, valid_result
from evals.vm import DesktopFacade

__all__ = [
    "CUA_GYM_DESKTOP_KIND",
    "CuaGymDesktopPreparer",
    "CuaGymDesktopTask",
    "CuaGymDesktopTaskData",
    "CuaGymDesktopTaskset",
    "CuaGymDesktopTasksetConfig",
]

CUA_GYM_DESKTOP_KIND = "cua_gym_desktop"
_REWARD_SCRIPT_PATH = "/tmp/cua_gym_reward.py"
_COMMAND_TIMEOUT_S = 120.0
_GUEST_MODULES = (
    "PyPDF2",
    "docx",
    "fitz",
    "odf",
    "openpyxl",
    "pandas",
    "pdfplumber",
    "pptx",
)


class CuaGymDesktopTaskData(DesktopTaskData):
    """A row pointing at one immutable CUA-Gym desktop task bundle.

    DesktopHarness already owns the image/prompt history and `DesktopState`, so
    CUA-Gym needs only its bundle identity. A separate state or prompt format
    would create a second lifecycle for the same guest.
    """

    task_id: str
    dataset_root: str
    dataset_revision: str


class CuaGymDesktopTask(DesktopTask):
    """CUA-Gym's trusted in-guest reward, published by DesktopHarness."""

    @vf.reward
    async def cua_gym_reward(self, trace: vf.Trace) -> float:
        result = valid_result(trace, "CUA-Gym")
        raw = result.get("task_reward")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("CUA-Gym task reward is missing or non-numeric")
        score = float(raw)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"CUA-Gym task reward is invalid: {score!r}")
        return score


class CuaGymDesktopTasksetConfig(vf.TasksetConfig):
    """One manifest-pinned desktop snapshot and its measured blocklist."""

    dataset_root: str = ""
    blocklist_path: str = ""
    task_ids: list[str] = Field(default_factory=list)
    max_tasks: int = Field(default=0, ge=0)
    shuffle_seed: int = Field(default=-1, ge=-1)
    max_steps: int = Field(default=32, ge=1)


class CuaGymDesktopTaskset(vf.Taskset[CuaGymDesktopTask, CuaGymDesktopTasksetConfig]):
    """Enumerate non-blocklisted CUA-Gym desktop tasks.

    The blocklist is required. A task absent from it merely lacks a measured
    defect; it is not proof of correctness, but a training task with a known
    reset payout is never an acceptable default.
    """

    def load(self) -> Iterable[CuaGymDesktopTask]:
        root = _required_absolute_path(self.config.dataset_root, "dataset_root")
        blocklist_path = _required_absolute_path(
            self.config.blocklist_path, "blocklist_path"
        )
        snapshot = _snapshot(str(root), PINNED_REVISION)
        catalog = snapshot.load_catalog()
        blocklist = load_blocklist(blocklist_path)
        selected_ids = self._select_task_ids(
            tuple(task.id for task in catalog), blocklist
        )
        for idx, task_id in enumerate(selected_ids):
            metadata = catalog.get(task_id)
            yield CuaGymDesktopTask(
                CuaGymDesktopTaskData(
                    idx=idx,
                    name=str(task_id),
                    prompt=metadata.instruction,
                    instruction=metadata.instruction,
                    kind=CUA_GYM_DESKTOP_KIND,
                    max_steps=self.config.max_steps,
                    task_id=str(task_id),
                    dataset_root=str(root),
                    dataset_revision=snapshot.manifest.revision,
                ),
                self.config.task,
            )

    def _select_task_ids(
        self, available: tuple[TaskId, ...], blocklist: TaskBlocklist
    ) -> list[TaskId]:
        available_set = set(available)
        if self.config.task_ids:
            selected = [TaskId(task_id) for task_id in self.config.task_ids]
            if len(set(selected)) != len(selected):
                raise ValueError("CUA-Gym task_ids must not contain duplicates")
            missing = [task_id for task_id in selected if task_id not in available_set]
            if missing:
                raise ValueError(
                    "unknown or unsupported CUA-Gym task IDs: "
                    + ", ".join(str(task_id) for task_id in missing)
                )
            blocked = {
                task_id: blocklist.reasons(task_id)
                for task_id in selected
                if blocklist.reasons(task_id)
            }
            if blocked:
                detail = ", ".join(
                    f"{task_id} ({'/'.join(reasons)})"
                    for task_id, reasons in blocked.items()
                )
                raise ValueError(f"requested CUA-Gym tasks are blocklisted: {detail}")
        else:
            selected = [
                task_id for task_id in available if not blocklist.reasons(task_id)
            ]
        if not selected:
            raise ValueError("CUA-Gym selection contains no non-blocklisted tasks")
        if self.config.shuffle_seed >= 0:
            random.Random(self.config.shuffle_seed).shuffle(selected)
        if self.config.max_tasks:
            selected = selected[: self.config.max_tasks]
        return selected


class CuaGymDesktopPreparer:
    """The CUA-Gym half of DesktopHarness's one episode lifecycle."""

    kind = CUA_GYM_DESKTOP_KIND

    def prepare(self, session: DesktopFacade, task: DesktopTaskData) -> dict[str, Any]:
        snapshot, bundle = _bundle_for_task(task)

        session.reset(mode=DesktopResetMode.SNAPSHOT)
        _require_guest_modules(session)
        with tempfile.TemporaryDirectory(prefix="cua-gym-setup-") as staging:
            steps = _materialized_setup_steps(bundle.setup_steps, bundle, Path(staging))
            if steps:
                session.setup_steps(steps)
        return {
            "prepared": CUA_GYM_DESKTOP_KIND,
            "task_id": str(bundle.task_id),
            "dataset_revision": snapshot.manifest.revision,
            "setup_steps": len(bundle.setup_steps),
        }

    def probe(self, session: DesktopFacade, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {
            "cursor": list(session.cursor_position()),
            "screen": list(session.screen_size()),
            "postcondition_status": "ok",
        }

    def evaluate(
        self, session: DesktopFacade, task: DesktopTaskData, *, declared: str | None
    ) -> float:
        del declared
        snapshot, bundle = _bundle_for_task(task)
        result = _run_reward(session, bundle)
        if result.returncode != 0:
            raise RuntimeError(
                f"CUA-Gym reward process exited {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return snapshot.parse_reward_stdout(bundle.task_id, result.stdout)


@lru_cache(maxsize=32)
def _snapshot(dataset_root: str, revision: str) -> CuaGymDatasetSnapshot:
    return CuaGymDatasetSnapshot(
        DatasetSnapshotConfig(Path(dataset_root), revision=revision),
        platform=TaskPlatform.DESKTOP,
    )


def _bundle_for_task(
    task: DesktopTaskData,
) -> tuple[CuaGymDatasetSnapshot, TaskBundle]:
    if not isinstance(task, CuaGymDesktopTaskData):
        raise TypeError("CUA-Gym preparer requires CuaGymDesktopTaskData")
    root = _required_absolute_path(task.dataset_root, "dataset_root")
    if task.dataset_revision != PINNED_REVISION:
        raise ValueError(
            f"CUA-Gym task revision {task.dataset_revision!r} is not "
            f"the pinned revision {PINNED_REVISION!r}"
        )
    snapshot = _snapshot(str(root), task.dataset_revision)
    metadata = snapshot.load_catalog().get(task.task_id)
    if task.instruction != metadata.instruction:
        raise ValueError("CUA-Gym task instruction differs from the pinned dataset")
    return snapshot, snapshot.load_task_bundle(metadata.id)


def _require_guest_modules(session: DesktopFacade) -> None:
    result = session.run_guest_command(
        ["python3", "-c", "import " + ", ".join(_GUEST_MODULES)],
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        detail = detail or ", ".join(_GUEST_MODULES)
        raise RuntimeError(f"CUA-Gym guest image is missing required modules: {detail}")


def _materialized_setup_steps(
    steps: Sequence[SetupStep], bundle: TaskBundle, staging: Path
) -> list[dict[str, Any]]:
    staged: dict[str, Path] = {}
    for setup_file in bundle.setup_files:
        _validate_bundle_file_name(setup_file.name)
        local_path = staging / setup_file.name
        local_path.write_bytes(setup_file.content)
        staged[setup_file.name] = local_path

    rewritten: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        raw = step.to_mutable()
        parameters = raw.get("parameters")
        if not isinstance(parameters, dict):
            raise TypeError(f"CUA-Gym setup step {index} has invalid parameters")
        if raw.get("type") != "download":
            rewritten.append(raw)
            continue
        files = parameters.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"CUA-Gym download step {index} has no files")
        uploads = [_offline_upload(entry, staged) for entry in files]
        rewritten.append({"type": "upload_file", "parameters": {"files": uploads}})
    return rewritten


def _offline_upload(entry: object, staged: Mapping[str, Path]) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise TypeError(f"CUA-Gym download entry must be an object: {entry!r}")
    url = entry.get("url")
    path = entry.get("path")
    if not isinstance(url, str) or not url.startswith("./"):
        raise ValueError(
            f"CUA-Gym setup downloads are restricted to bundle files: {url!r}"
        )
    if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
        raise ValueError(f"CUA-Gym setup upload target must be absolute: {path!r}")
    name = url.removeprefix("./")
    _validate_bundle_file_name(name)
    try:
        local_path = staged[name]
    except KeyError as exc:
        raise ValueError(
            f"CUA-Gym setup references an unbundled file: {url!r}"
        ) from exc
    return {"local_path": str(local_path), "path": path}


def _run_reward(session: DesktopFacade, bundle: TaskBundle) -> GuestCommandResult:
    session.write_guest_file(_REWARD_SCRIPT_PATH, bundle.reward_source.encode("utf-8"))
    return session.run_guest_command(
        ["python3", _REWARD_SCRIPT_PATH], timeout_s=_COMMAND_TIMEOUT_S
    )


def _required_absolute_path(raw: str, name: str) -> Path:
    if not raw:
        raise ValueError(f"CUA-Gym {name} is required")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"CUA-Gym {name} must be an absolute path: {raw!r}")
    return path


def _validate_bundle_file_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid CUA-Gym bundle file name: {name!r}")


register_preparer(CuaGymDesktopPreparer())
