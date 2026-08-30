"""Typed models for the CUA-Gym dataset and task-bundle layer."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NewType

TaskId = NewType("TaskId", str)
EndpointName = NewType("EndpointName", str)
type JsonScalar = str | int | float | bool | None
type FrozenJsonValue = JsonScalar | FrozenJsonObject | tuple[FrozenJsonValue, ...]
type MutableJsonValue = (
    JsonScalar | list[MutableJsonValue] | dict[str, MutableJsonValue]
)
type MutableJsonObject = dict[str, MutableJsonValue]


@dataclass(frozen=True)
class FrozenJsonObject(Mapping[str, "FrozenJsonValue"]):
    """Recursively immutable JSON object with an explicit mutable export."""

    _values: MappingProxyType[str, FrozenJsonValue]

    def __getitem__(self, key: str) -> FrozenJsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_mutable(self) -> MutableJsonObject:
        """Return a recursively independent dict/list JSON representation."""

        return {key: thaw_json(value) for key, value in self._values.items()}


def freeze_json(value: object) -> FrozenJsonValue:
    """Validate and recursively freeze a value produced by a JSON parser."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return FrozenJsonObject(
            MappingProxyType({key: freeze_json(item) for key, item in value.items()})
        )
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def freeze_json_object(value: object) -> FrozenJsonObject:
    """Validate that ``value`` is a JSON object and recursively freeze it."""

    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenJsonObject):
        raise TypeError("JSON root must be an object")
    return frozen


def thaw_json(value: FrozenJsonValue) -> MutableJsonValue:
    """Return a recursively mutable JSON representation."""

    if isinstance(value, FrozenJsonObject):
        return value.to_mutable()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def clone_frozen_json_object(value: FrozenJsonObject) -> FrozenJsonObject:
    """Return a deeply independent immutable copy of a frozen JSON object."""

    return freeze_json_object(value.to_mutable())


@dataclass(frozen=True)
class DatasetSnapshotConfig:
    """Location and expected immutable revision of a CUA-Gym snapshot."""

    dataset_root: Path
    revision: str = "3c021d06e2b01bbb6ad28cfddec41d856e4a78c5"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        if not self.revision or "/" in self.revision:
            raise ValueError("revision must be a non-empty snapshot identifier")

    @property
    def snapshot_root(self) -> Path:
        """Resolve either an exact snapshot directory or a HF cache repo root."""

        root = self.dataset_root.expanduser().absolute()
        cache_snapshot = root / "snapshots" / self.revision
        if cache_snapshot.is_dir():
            return cache_snapshot
        return root


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    sha256: str


@dataclass(frozen=True)
class MetadataField:
    name: str
    arrow_type: str
    nullable: bool


@dataclass(frozen=True)
class EndpointSpec:
    """Known placeholders and release-hosted values for one mock service."""

    name: EndpointName
    url_tokens: tuple[str, ...]
    host_tokens: tuple[str, ...]


class TaskPlatform(StrEnum):
    """Dataset platform label selecting one snapshot task family."""

    WEB = "web"
    DESKTOP = "desktop"


class SetupStepType(StrEnum):
    """Task-bundle setup step kinds the runtime knows how to execute."""

    DOWNLOAD = "download"
    EXECUTE = "execute"
    LAUNCH = "launch"
    OPEN = "open"
    SLEEP = "sleep"


@dataclass(frozen=True)
class SetupStep:
    """One validated entry of a task bundle's ``config`` list."""

    type: SetupStepType
    parameters: FrozenJsonObject

    def to_mutable(self) -> MutableJsonObject:
        """Return the OSWorld setup-controller representation of this step."""

        return {"type": str(self.type), "parameters": self.parameters.to_mutable()}

    def download_targets(self) -> tuple[str, ...]:
        """Return the guest paths a ``download`` step writes, if it is one."""

        if self.type is not SetupStepType.DOWNLOAD:
            return ()
        files = self.parameters.get("files")
        if not isinstance(files, tuple):
            return ()
        return tuple(
            str(entry["path"]) for entry in files if isinstance(entry, FrozenJsonObject)
        )


class RewardOutputFormat(StrEnum):
    """The one stdout reward format accepted for a compatible task."""

    REWARD_PREFIX = "reward_prefix"
    BARE_NUMBER = "bare_number"
    TOTAL_SCORE = "total_score"


@dataclass(frozen=True)
class TaskCompatibility:
    """Pinned task quirks derived from executable bundle contents."""

    task_id: TaskId
    required_endpoints: tuple[EndpointName, ...]
    reward_output_format: RewardOutputFormat = RewardOutputFormat.REWARD_PREFIX
    setup_target_path: str = "/home/user/initial_setup.py"
    hard_coded_endpoint_urls: tuple[tuple[EndpointName, str], ...] = ()
    hard_coded_sid: str | None = None
    allow_zero_reward_diagnostic: bool = False

    @property
    def allow_bare_reward(self) -> bool:
        """Whether this task's declared reward format is a bare number."""

        return self.reward_output_format is RewardOutputFormat.BARE_NUMBER


@dataclass(frozen=True)
class TaskMetadata:
    """A validated eligible row from the viewer-friendly Parquet table."""

    id: TaskId
    instruction: str
    app_type: str
    app_family: str
    platform: str
    difficulty: str | None
    setup_kind: str
    num_setup_steps: int
    num_setup_files: int
    has_ground_truth: bool
    setup_files: tuple[str, ...]
    archive_path: str
    archive_member: str
    task_json_member: str
    reward_member: str
    setup_file_members: tuple[str, ...]
    compatibility: TaskCompatibility


@dataclass(frozen=True)
class BundleFile:
    """One in-memory file from a task bundle."""

    name: str
    content: bytes

    def text(self) -> str:
        return self.content.decode("utf-8")


@dataclass(frozen=True)
class TaskBundle:
    """Raw, unexecuted task sources loaded from the compressed artifact."""

    metadata: TaskMetadata
    task_config: FrozenJsonObject
    reward_source: str
    setup_files: tuple[BundleFile, ...]
    setup_steps: tuple[SetupStep, ...] = ()

    @property
    def task_id(self) -> TaskId:
        return self.metadata.id

    def mutable_task_config(self) -> MutableJsonObject:
        """Return an independent dict/list config for a future runtime adapter."""

        return self.task_config.to_mutable()


@dataclass(frozen=True)
class MaterializedTaskBundle:
    """Per-episode sources with deployment endpoints replaced in memory."""

    metadata: TaskMetadata
    task_config: FrozenJsonObject
    reward_source: str
    setup_files: tuple[BundleFile, ...]
    gateway_urls: MappingProxyType[EndpointName, str]

    def mutable_task_config(self) -> MutableJsonObject:
        """Return an independent dict/list config for a future runtime adapter."""

        return self.task_config.to_mutable()


@dataclass(frozen=True)
class TaskCatalog:
    """Immutable, deterministic catalog of compatible browser-only tasks."""

    tasks: tuple[TaskMetadata, ...]
    _by_id: MappingProxyType[TaskId, TaskMetadata]

    @classmethod
    def from_tasks(cls, tasks: tuple[TaskMetadata, ...]) -> TaskCatalog:
        by_id = {task.id: task for task in tasks}
        if len(by_id) != len(tasks):
            raise ValueError("task catalog contains duplicate IDs")
        return cls(tasks=tasks, _by_id=MappingProxyType(by_id))

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[TaskMetadata]:
        return iter(self.tasks)

    def get(self, task_id: str | TaskId) -> TaskMetadata:
        try:
            return self._by_id[TaskId(str(task_id))]
        except KeyError as error:
            raise KeyError(f"Unknown compatible CUA-Gym task: {task_id}") from error
