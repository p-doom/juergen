from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from ..manifest import PINNED_REVISION, CompatibilityManifest
from ..models import EndpointName, TaskId

WEB_RUNTIME_MANIFEST_VERSION = 1

_APP_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_HOST_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
_SCRATCH_PATH_RE = re.compile(r"/tmp/[A-Za-z0-9._-]+")
_WRITABLE_DIRECTORY_RE = re.compile(r"\.mock-[a-z0-9-]+")
_MANIFEST_KEYS = {
    "dataset_revision",
    "endpoints",
    "hub_revision",
    "manifest_version",
    "supported_task_count",
    "task_scratch_paths",
    "unsupported_endpoints",
    "unsupported_tasks",
    "writable_directories",
}


@dataclass(frozen=True)
class CuaGymWebRuntimeManifest:
    manifest_version: int
    dataset_revision: str
    hub_revision: str
    endpoint_apps: Mapping[EndpointName, str]
    writable_directories: tuple[str, ...]
    unsupported_endpoints: Mapping[EndpointName, str]
    unsupported_tasks: Mapping[TaskId, str]
    task_scratch_paths: tuple[str, ...]
    supported_task_count: int

    def __post_init__(self) -> None:
        if self.manifest_version != WEB_RUNTIME_MANIFEST_VERSION:
            raise ValueError(
                "Unsupported CUA-Gym web runtime manifest version: "
                f"{self.manifest_version!r}"
            )
        if _COMMIT_RE.fullmatch(self.dataset_revision) is None:
            raise ValueError("dataset_revision must be a full lowercase Git revision")
        if _COMMIT_RE.fullmatch(self.hub_revision) is None:
            raise ValueError("hub_revision must be a full lowercase Git revision")
        if not self.endpoint_apps:
            raise ValueError("CUA-Gym web runtime must contain at least one endpoint")

        endpoint_apps: dict[EndpointName, str] = {}
        hostnames: set[str] = set()
        for raw_endpoint, app in self.endpoint_apps.items():
            if not isinstance(raw_endpoint, str) or not raw_endpoint:
                raise ValueError("CUA-Gym web manifest contains an invalid endpoint")
            endpoint = EndpointName(raw_endpoint)
            hostname = self.hostname(endpoint)
            if hostname in hostnames:
                raise ValueError(f"Duplicate CUA-Gym gateway hostname: {hostname}")
            hostnames.add(hostname)
            if not isinstance(app, str) or _APP_NAME_RE.fullmatch(app) is None:
                raise ValueError(f"Invalid CUA-Gym-Hub app name: {app!r}")
            endpoint_apps[endpoint] = app

        if (
            not self.writable_directories
            or tuple(sorted(self.writable_directories)) != self.writable_directories
            or len(set(self.writable_directories)) != len(self.writable_directories)
            or any(
                _WRITABLE_DIRECTORY_RE.fullmatch(directory) is None
                for directory in self.writable_directories
            )
        ):
            raise ValueError(
                "writable_directories must be sorted, unique .mock-* names"
            )

        unsupported_endpoints: dict[EndpointName, str] = {}
        for raw_endpoint, reason in self.unsupported_endpoints.items():
            if not isinstance(raw_endpoint, str) or not raw_endpoint:
                raise ValueError("Unsupported CUA-Gym endpoint is invalid")
            unsupported_endpoints[EndpointName(raw_endpoint)] = reason
        unexpected = set(unsupported_endpoints) - set(endpoint_apps)
        if unexpected:
            raise ValueError(
                "Unsupported CUA-Gym endpoints are missing from endpoint_apps: "
                f"{sorted(unexpected)}"
            )
        if any(
            not isinstance(reason, str) or not reason.strip()
            for reason in unsupported_endpoints.values()
        ):
            raise ValueError("Unsupported CUA-Gym endpoint reasons must not be empty")

        unsupported_tasks: dict[TaskId, str] = {}
        for raw_task_id, reason in self.unsupported_tasks.items():
            if not isinstance(raw_task_id, str) or not raw_task_id:
                raise ValueError("Unsupported CUA-Gym task ID is invalid")
            unsupported_tasks[TaskId(raw_task_id)] = reason
        if any(
            not isinstance(reason, str) or not reason.strip()
            for reason in unsupported_tasks.values()
        ):
            raise ValueError("Unsupported CUA-Gym task reasons must not be empty")

        if (
            not self.task_scratch_paths
            or tuple(sorted(self.task_scratch_paths)) != self.task_scratch_paths
            or len(set(self.task_scratch_paths)) != len(self.task_scratch_paths)
            or any(
                _SCRATCH_PATH_RE.fullmatch(path) is None
                or path.removeprefix("/tmp/") in {".", ".."}
                for path in self.task_scratch_paths
            )
        ):
            raise ValueError(
                "task_scratch_paths must be sorted, unique, flat /tmp paths"
            )
        if type(self.supported_task_count) is not int or self.supported_task_count < 1:
            raise ValueError("supported_task_count must be a positive integer")

        object.__setattr__(self, "endpoint_apps", MappingProxyType(endpoint_apps))
        object.__setattr__(
            self,
            "unsupported_endpoints",
            MappingProxyType(unsupported_endpoints),
        )
        object.__setattr__(
            self,
            "unsupported_tasks",
            MappingProxyType(unsupported_tasks),
        )

    @property
    def apps(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.endpoint_apps.values())))

    def hostname(self, endpoint: str | EndpointName) -> str:
        name = str(endpoint).replace("_", "-")
        if _HOST_LABEL_RE.fullmatch(name) is None:
            raise ValueError(f"Endpoint cannot form an internal hostname: {endpoint}")
        return f"{name}.cua.internal"

    def unsupported_for(
        self, endpoints: tuple[EndpointName, ...]
    ) -> tuple[EndpointName, ...]:
        return tuple(
            endpoint for endpoint in endpoints if endpoint in self.unsupported_endpoints
        )

    def incompatibilities_for(
        self,
        task_id: str | TaskId,
        endpoints: tuple[EndpointName, ...],
    ) -> tuple[str, ...]:
        normalized_task_id = TaskId(str(task_id))
        incompatibilities: list[str] = []
        task_reason = self.unsupported_tasks.get(normalized_task_id)
        if task_reason is not None:
            incompatibilities.append(f"task {normalized_task_id}: {task_reason}")
        incompatibilities.extend(
            f"endpoint {endpoint}: {self.unsupported_endpoints[endpoint]}"
            for endpoint in endpoints
            if endpoint in self.unsupported_endpoints
        )
        return tuple(incompatibilities)

    def validate_dataset(self, manifest: CompatibilityManifest) -> None:
        if manifest.revision != self.dataset_revision:
            raise ValueError(
                "CUA-Gym web runtime and dataset revisions differ: "
                f"{self.dataset_revision} != {manifest.revision}"
            )
        dataset_endpoints = set(manifest.endpoint_specs)
        runtime_endpoints = set(self.endpoint_apps)
        if dataset_endpoints != runtime_endpoints:
            raise ValueError(
                "CUA-Gym web endpoint mapping differs from the dataset manifest "
                f"(missing={sorted(dataset_endpoints - runtime_endpoints)}, "
                f"unexpected={sorted(runtime_endpoints - dataset_endpoints)})"
            )
        unknown_task_ids = set(self.unsupported_tasks) - set(manifest.tasks)
        if unknown_task_ids:
            raise ValueError(
                "Unsupported CUA-Gym task IDs are missing from the dataset manifest: "
                f"{sorted(unknown_task_ids)}"
            )
        excluded_task_ids = {
            task_id
            for task_id, compatibility in manifest.tasks.items()
            if self.incompatibilities_for(task_id, compatibility.required_endpoints)
        }
        computed_count = manifest.eligible_task_count - len(excluded_task_ids)
        if self.supported_task_count != computed_count:
            raise ValueError(
                "CUA-Gym supported task count differs from endpoint and task "
                f"exclusions ({self.supported_task_count} != {computed_count})"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CuaGymWebRuntimeManifest:
        actual_keys = set(payload)
        if actual_keys != _MANIFEST_KEYS:
            raise ValueError(
                "CUA-Gym web runtime manifest keys differ from the schema "
                f"(missing={sorted(_MANIFEST_KEYS - actual_keys)}, "
                f"unexpected={sorted(actual_keys - _MANIFEST_KEYS)})"
            )
        endpoints = _string_mapping(payload, "endpoints")
        return cls(
            manifest_version=_integer(payload, "manifest_version"),
            dataset_revision=_string(payload, "dataset_revision"),
            hub_revision=_string(payload, "hub_revision"),
            endpoint_apps={
                EndpointName(endpoint): app for endpoint, app in endpoints.items()
            },
            writable_directories=_strings(payload, "writable_directories"),
            unsupported_endpoints={
                EndpointName(endpoint): reason
                for endpoint, reason in _string_mapping(
                    payload, "unsupported_endpoints"
                ).items()
            },
            unsupported_tasks={
                TaskId(task_id): reason
                for task_id, reason in _string_mapping(
                    payload, "unsupported_tasks"
                ).items()
            },
            task_scratch_paths=_strings(payload, "task_scratch_paths"),
            supported_task_count=_integer(payload, "supported_task_count"),
        )


def load_default_web_runtime_manifest() -> CuaGymWebRuntimeManifest:
    resource = files(__package__).joinpath("compatibility", f"{PINNED_REVISION}.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("CUA-Gym web runtime manifest must be an object")
    return CuaGymWebRuntimeManifest.from_dict(payload)


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"CUA-Gym web runtime manifest has invalid {key}")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"CUA-Gym web runtime manifest has invalid {key}")
    return value


def _strings(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"CUA-Gym web runtime manifest has invalid {key}")
    return tuple(value)


def _string_mapping(payload: Mapping[str, Any], key: str) -> dict[str, str]:
    value = payload[key]
    if not isinstance(value, dict) or not all(
        isinstance(item, str) and item and isinstance(reason, str) and reason.strip()
        for item, reason in value.items()
    ):
        raise ValueError(f"CUA-Gym web runtime manifest has invalid {key}")
    return value
