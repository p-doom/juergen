"""Revision-locked CUA-Gym compatibility manifest loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from .errors import SnapshotValidationError
from .models import (
    EndpointName,
    EndpointSpec,
    FileFingerprint,
    MetadataField,
    RewardOutputFormat,
    SetupStepType,
    TaskCompatibility,
    TaskId,
)

PINNED_REVISION = "3c021d06e2b01bbb6ad28cfddec41d856e4a78c5"
MANIFEST_VERSION = 4
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class DesktopCompatibility:
    """Checked-in facts for the desktop task family of one dataset revision."""

    task_count: int
    eligible_task_count: int
    excluded_tasks: MappingProxyType[TaskId, str]
    setup_step_types: tuple[SetupStepType, ...]
    reward_output_overrides: MappingProxyType[TaskId, RewardOutputFormat]
    evaluator_postconfig_sha256: str
    evaluator_postconfig_task_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DesktopCompatibility:
        _expect_exact_keys(
            payload,
            {
                "task_count",
                "eligible_task_count",
                "excluded_tasks",
                "setup_step_types",
                "reward_output_overrides",
                "evaluator_postconfig_sha256",
                "evaluator_postconfig_task_count",
            },
            "desktop",
        )
        task_count = _positive_int(payload["task_count"], "desktop.task_count")
        eligible_task_count = _positive_int(
            payload["eligible_task_count"], "desktop.eligible_task_count"
        )
        excluded = _string_map(payload["excluded_tasks"], "desktop.excluded_tasks")
        if eligible_task_count != task_count - len(excluded):
            raise SnapshotValidationError(
                "desktop.eligible_task_count must be task_count minus exclusions"
            )
        raw_step_types = _strings(
            payload["setup_step_types"], "desktop.setup_step_types"
        )
        if raw_step_types != tuple(sorted(set(raw_step_types))):
            raise SnapshotValidationError(
                "desktop.setup_step_types must be sorted and unique"
            )
        try:
            step_types = tuple(SetupStepType(value) for value in raw_step_types)
        except ValueError as error:
            raise SnapshotValidationError(
                f"Unknown desktop setup step type: {error}"
            ) from error
        raw_overrides = _string_map(
            payload["reward_output_overrides"], "desktop.reward_output_overrides"
        )
        overrides: dict[TaskId, RewardOutputFormat] = {}
        for raw_task_id, raw_format in raw_overrides.items():
            task_id = TaskId(raw_task_id)
            if raw_task_id in excluded:
                raise SnapshotValidationError(
                    f"Excluded desktop task carries a reward override: {task_id}"
                )
            try:
                overrides[task_id] = RewardOutputFormat(raw_format)
            except ValueError as error:
                raise SnapshotValidationError(
                    f"Unknown reward output format for {task_id}: {raw_format!r}"
                ) from error
        digest = _string(
            payload["evaluator_postconfig_sha256"],
            "desktop.evaluator_postconfig_sha256",
        )
        if _SHA256_RE.fullmatch(digest) is None:
            raise SnapshotValidationError(
                "Invalid SHA-256 for desktop.evaluator_postconfig_sha256"
            )
        postconfig_task_count = _non_negative_int(
            payload["evaluator_postconfig_task_count"],
            "desktop.evaluator_postconfig_task_count",
        )
        if postconfig_task_count > task_count:
            raise SnapshotValidationError(
                "desktop.evaluator_postconfig_task_count exceeds task_count"
            )
        return cls(
            task_count=task_count,
            eligible_task_count=eligible_task_count,
            excluded_tasks=MappingProxyType(
                {TaskId(task_id): reason for task_id, reason in excluded.items()}
            ),
            setup_step_types=step_types,
            reward_output_overrides=MappingProxyType(overrides),
            evaluator_postconfig_sha256=digest,
            evaluator_postconfig_task_count=postconfig_task_count,
        )

    def task(self, task_id: str | TaskId) -> TaskCompatibility:
        """Synthesize one desktop task's compatibility from the pinned rules."""

        normalized = TaskId(str(task_id))
        if normalized in self.excluded_tasks:
            raise KeyError(
                f"Task is excluded by this manifest: {task_id} "
                f"({self.excluded_tasks[normalized]})"
            )
        return TaskCompatibility(
            task_id=normalized,
            required_endpoints=(),
            reward_output_format=self.reward_output_overrides.get(
                normalized, RewardOutputFormat.REWARD_PREFIX
            ),
        )


@dataclass(frozen=True)
class CompatibilityManifest:
    """Checked-in compatibility facts for one immutable dataset revision."""

    revision: str
    dataset: str
    files: MappingProxyType[str, FileFingerprint]
    metadata_schema: tuple[MetadataField, ...]
    metadata_row_count: int
    eligible_task_count: int
    excluded_tasks: MappingProxyType[TaskId, str]
    endpoint_specs: MappingProxyType[EndpointName, EndpointSpec]
    tasks: MappingProxyType[TaskId, TaskCompatibility]
    desktop: DesktopCompatibility

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompatibilityManifest:
        _expect_exact_keys(
            payload,
            {
                "manifest_version",
                "dataset",
                "revision",
                "files",
                "metadata_schema",
                "metadata_row_count",
                "eligible_task_count",
                "excluded_tasks",
                "endpoint_specs",
                "eligible_task_ids",
                "endpoint_task_ids",
                "bare_reward_task_ids",
                "zero_reward_diagnostic_task_ids",
                "reward_output_overrides",
                "setup_target_overrides",
                "hard_coded_endpoint_overrides",
                "hard_coded_sid_overrides",
                "desktop",
            },
            "manifest",
        )
        if payload["manifest_version"] != MANIFEST_VERSION:
            raise SnapshotValidationError(
                f"Unsupported CUA-Gym manifest version: {payload['manifest_version']!r}"
            )

        revision = _string(payload["revision"], "revision")
        dataset = _string(payload["dataset"], "dataset")
        raw_files = _dict(payload["files"], "files")
        fingerprints: dict[str, FileFingerprint] = {}
        for name, raw_fingerprint in raw_files.items():
            fingerprint = _dict(raw_fingerprint, f"files.{name}")
            _expect_exact_keys(fingerprint, {"path", "sha256"}, f"files.{name}")
            digest = _string(fingerprint["sha256"], f"files.{name}.sha256")
            if _SHA256_RE.fullmatch(digest) is None:
                raise SnapshotValidationError(f"Invalid SHA-256 for files.{name}")
            fingerprints[name] = FileFingerprint(
                path=_string(fingerprint["path"], f"files.{name}.path"),
                sha256=digest,
            )

        raw_schema = _list(payload["metadata_schema"], "metadata_schema")
        schema: list[MetadataField] = []
        for index, raw_field in enumerate(raw_schema):
            field = _dict(raw_field, f"metadata_schema[{index}]")
            _expect_exact_keys(
                field,
                {"name", "arrow_type", "nullable"},
                f"metadata_schema[{index}]",
            )
            nullable = field["nullable"]
            if not isinstance(nullable, bool):
                raise SnapshotValidationError(
                    f"metadata_schema[{index}].nullable must be a boolean"
                )
            schema.append(
                MetadataField(
                    name=_string(field["name"], f"metadata_schema[{index}].name"),
                    arrow_type=_string(
                        field["arrow_type"],
                        f"metadata_schema[{index}].arrow_type",
                    ),
                    nullable=nullable,
                )
            )

        raw_endpoint_specs = _dict(payload["endpoint_specs"], "endpoint_specs")
        endpoint_specs: dict[EndpointName, EndpointSpec] = {}
        token_owners: dict[str, EndpointName] = {}
        for raw_name, raw_spec in raw_endpoint_specs.items():
            name = EndpointName(raw_name)
            spec = _dict(raw_spec, f"endpoint_specs.{raw_name}")
            _expect_exact_keys(
                spec,
                {"url_tokens", "host_tokens"},
                f"endpoint_specs.{raw_name}",
            )
            url_tokens = _strings(spec["url_tokens"], f"{raw_name}.url_tokens")
            host_tokens = _strings(spec["host_tokens"], f"{raw_name}.host_tokens")
            if len(set((*url_tokens, *host_tokens))) != len(
                (*url_tokens, *host_tokens)
            ):
                raise SnapshotValidationError(
                    f"Endpoint {raw_name} contains duplicate replacement tokens"
                )
            for token in (*url_tokens, *host_tokens):
                previous_owner = token_owners.setdefault(token, name)
                if previous_owner != name:
                    raise SnapshotValidationError(
                        f"Endpoint token {token!r} belongs to multiple endpoints"
                    )
            endpoint_specs[name] = EndpointSpec(
                name=name,
                url_tokens=url_tokens,
                host_tokens=host_tokens,
            )

        eligible_ids = tuple(
            TaskId(value)
            for value in _strings(payload["eligible_task_ids"], "eligible_task_ids")
        )
        if eligible_ids != tuple(sorted(eligible_ids)) or len(set(eligible_ids)) != len(
            eligible_ids
        ):
            raise SnapshotValidationError(
                "eligible_task_ids must be sorted and contain no duplicates"
            )
        eligible_set = set(eligible_ids)

        endpoints_by_task: dict[TaskId, list[EndpointName]] = {
            task_id: [] for task_id in eligible_ids
        }
        raw_endpoint_tasks = _dict(payload["endpoint_task_ids"], "endpoint_task_ids")
        if set(raw_endpoint_tasks) != {str(name) for name in endpoint_specs}:
            raise SnapshotValidationError(
                "endpoint_task_ids keys must exactly match endpoint_specs"
            )
        for raw_endpoint, raw_task_ids in raw_endpoint_tasks.items():
            endpoint = EndpointName(raw_endpoint)
            task_ids = _strings(raw_task_ids, f"endpoint_task_ids.{raw_endpoint}")
            if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(
                task_ids
            ):
                raise SnapshotValidationError(
                    f"endpoint_task_ids.{raw_endpoint} must be sorted and unique"
                )
            for raw_task_id in task_ids:
                task_id = TaskId(raw_task_id)
                if task_id not in eligible_set:
                    raise SnapshotValidationError(
                        f"Endpoint {raw_endpoint} references ineligible task {task_id}"
                    )
                endpoints_by_task[task_id].append(endpoint)

        bare_reward_ids = _task_id_set(
            payload["bare_reward_task_ids"], eligible_set, "bare_reward_task_ids"
        )
        zero_reward_diagnostic_ids = _task_id_set(
            payload["zero_reward_diagnostic_task_ids"],
            eligible_set,
            "zero_reward_diagnostic_task_ids",
        )
        raw_reward_overrides = _string_map(
            payload["reward_output_overrides"], "reward_output_overrides"
        )
        reward_overrides: dict[TaskId, RewardOutputFormat] = {}
        for raw_task_id, raw_format in raw_reward_overrides.items():
            task_id = TaskId(raw_task_id)
            if task_id not in eligible_set:
                raise SnapshotValidationError(
                    f"Reward output override references ineligible task {task_id}"
                )
            try:
                reward_overrides[task_id] = RewardOutputFormat(raw_format)
            except ValueError as error:
                raise SnapshotValidationError(
                    f"Unknown reward output format for {task_id}: {raw_format!r}"
                ) from error
        overlap = bare_reward_ids & set(reward_overrides)
        if overlap:
            raise SnapshotValidationError(
                f"Tasks cannot have bare and explicit reward output policies: {sorted(overlap)}"
            )
        non_prefixed_reward_ids = bare_reward_ids | {
            task_id
            for task_id, output_format in reward_overrides.items()
            if output_format is not RewardOutputFormat.REWARD_PREFIX
        }
        incompatible_diagnostic_ids = (
            zero_reward_diagnostic_ids & non_prefixed_reward_ids
        )
        if incompatible_diagnostic_ids:
            raise SnapshotValidationError(
                "zero_reward_diagnostic_task_ids may only contain reward_prefix tasks: "
                f"{sorted(incompatible_diagnostic_ids)}"
            )
        setup_overrides = _string_map(
            payload["setup_target_overrides"], "setup_target_overrides"
        )
        sid_overrides = _string_map(
            payload["hard_coded_sid_overrides"], "hard_coded_sid_overrides"
        )
        raw_endpoint_overrides = _dict(
            payload["hard_coded_endpoint_overrides"],
            "hard_coded_endpoint_overrides",
        )
        endpoint_overrides: dict[TaskId, tuple[tuple[EndpointName, str], ...]] = {}
        for raw_task_id, raw_overrides in raw_endpoint_overrides.items():
            task_id = TaskId(raw_task_id)
            if task_id not in eligible_set:
                raise SnapshotValidationError(
                    f"Endpoint overrides reference ineligible task {task_id}"
                )
            override_map = _dict(
                raw_overrides, f"hard_coded_endpoint_overrides.{raw_task_id}"
            )
            values: list[tuple[EndpointName, str]] = []
            for raw_endpoint, raw_url in sorted(override_map.items()):
                endpoint = EndpointName(raw_endpoint)
                if endpoint not in endpoint_specs:
                    raise SnapshotValidationError(
                        f"Unknown endpoint override {raw_endpoint} for {task_id}"
                    )
                values.append((endpoint, _string(raw_url, f"override.{task_id}")))
                if endpoint not in endpoints_by_task.get(task_id, []):
                    endpoints_by_task[task_id].append(endpoint)
            endpoint_overrides[task_id] = tuple(values)

        override_ids = (
            {TaskId(value) for value in setup_overrides}
            | {TaskId(value) for value in sid_overrides}
            | set(endpoint_overrides)
            | set(reward_overrides)
        )
        unknown_overrides = override_ids - eligible_set
        if unknown_overrides:
            raise SnapshotValidationError(
                f"Compatibility overrides reference ineligible tasks: {sorted(unknown_overrides)}"
            )

        task_compatibility = {
            task_id: TaskCompatibility(
                task_id=task_id,
                required_endpoints=tuple(sorted(endpoints_by_task[task_id])),
                reward_output_format=reward_overrides.get(
                    task_id,
                    RewardOutputFormat.BARE_NUMBER
                    if task_id in bare_reward_ids
                    else RewardOutputFormat.REWARD_PREFIX,
                ),
                setup_target_path=setup_overrides.get(
                    str(task_id), "/home/user/initial_setup.py"
                ),
                hard_coded_endpoint_urls=endpoint_overrides.get(task_id, ()),
                hard_coded_sid=sid_overrides.get(str(task_id)),
                allow_zero_reward_diagnostic=(task_id in zero_reward_diagnostic_ids),
            )
            for task_id in eligible_ids
        }

        excluded = _string_map(payload["excluded_tasks"], "excluded_tasks")
        metadata_row_count = _positive_int(
            payload["metadata_row_count"], "metadata_row_count"
        )
        eligible_task_count = _positive_int(
            payload["eligible_task_count"], "eligible_task_count"
        )
        if eligible_task_count != len(eligible_ids):
            raise SnapshotValidationError(
                "eligible_task_count does not match eligible_task_ids"
            )
        return cls(
            revision=revision,
            dataset=dataset,
            files=MappingProxyType(fingerprints),
            metadata_schema=tuple(schema),
            metadata_row_count=metadata_row_count,
            eligible_task_count=eligible_task_count,
            excluded_tasks=MappingProxyType(
                {TaskId(task_id): reason for task_id, reason in excluded.items()}
            ),
            endpoint_specs=MappingProxyType(endpoint_specs),
            tasks=MappingProxyType(task_compatibility),
            desktop=DesktopCompatibility.from_dict(
                _dict(payload["desktop"], "desktop")
            ),
        )

    def task(self, task_id: str | TaskId) -> TaskCompatibility:
        try:
            return self.tasks[TaskId(str(task_id))]
        except KeyError as error:
            raise KeyError(
                f"Task is not compatible with this manifest: {task_id}"
            ) from error


def load_default_manifest() -> CompatibilityManifest:
    resource = files("evals.cua_gym").joinpath(
        "compatibility", f"{PINNED_REVISION}.json"
    )
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotValidationError(
            f"Could not load checked-in CUA-Gym compatibility manifest: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SnapshotValidationError("Compatibility manifest root must be an object")
    return CompatibilityManifest.from_dict(payload)


def _dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SnapshotValidationError(f"{field} must be an object with string keys")
    return value


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{field} must be an array")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotValidationError(f"{field} must be a non-empty string")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    values = _list(value, field)
    if not all(isinstance(item, str) and item for item in values):
        raise SnapshotValidationError(f"{field} must contain non-empty strings")
    return tuple(values)


def _non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SnapshotValidationError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SnapshotValidationError(f"{field} must be a positive integer")
    return value


def _string_map(value: object, field: str) -> dict[str, str]:
    values = _dict(value, field)
    if not all(isinstance(item, str) and item for item in values.values()):
        raise SnapshotValidationError(f"{field} values must be non-empty strings")
    return values


def _task_id_set(value: object, eligible_ids: set[TaskId], field: str) -> set[TaskId]:
    raw_ids = _strings(value, field)
    task_ids = {TaskId(raw_id) for raw_id in raw_ids}
    if len(task_ids) != len(raw_ids) or not task_ids <= eligible_ids:
        raise SnapshotValidationError(f"{field} must contain unique eligible task IDs")
    return task_ids


def _expect_exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise SnapshotValidationError(
            f"{field} keys do not match schema (missing={missing}, extra={extra})"
        )
