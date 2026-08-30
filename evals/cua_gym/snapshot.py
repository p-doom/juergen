"""Fail-closed loading of CUA-Gym Parquet metadata and raw task bundles."""

from __future__ import annotations

import hashlib
import json
import posixpath
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from threading import Lock
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import BundleValidationError, SnapshotValidationError
from .manifest import (
    CompatibilityManifest,
    DesktopCompatibility,
    load_default_manifest,
)
from .materialization import derive_required_endpoints
from .materialization import materialize_task_bundle as _materialize_task_bundle
from .models import (
    BundleFile,
    DatasetSnapshotConfig,
    EndpointName,
    FrozenJsonObject,
    MaterializedTaskBundle,
    SetupStep,
    SetupStepType,
    TaskBundle,
    TaskCatalog,
    TaskId,
    TaskMetadata,
    TaskPlatform,
    freeze_json,
    freeze_json_object,
)
from .reward import parse_reward_output as _parse_reward_output


class CuaGymDatasetSnapshot:
    """Read-only access to one manifest-pinned CUA-Gym dataset snapshot."""

    def __init__(
        self,
        config: DatasetSnapshotConfig,
        manifest: CompatibilityManifest | None = None,
        platform: TaskPlatform = TaskPlatform.WEB,
    ) -> None:
        self.config = config
        self.platform = TaskPlatform(platform)
        self.manifest = manifest or load_default_manifest()
        self._catalog: TaskCatalog | None = None
        self._bundle_catalog: Mapping[TaskId, TaskBundle] | None = None
        self._bundle_catalog_lock = Lock()

    @property
    def root(self) -> Path:
        return self.config.snapshot_root

    def load_catalog(self) -> TaskCatalog:
        """Validate the snapshot and return exactly the compatible task rows."""

        if self._catalog is not None:
            return self._catalog
        self._validate_snapshot_files()
        metadata_path = self._file_path("metadata")
        try:
            table = pq.read_table(metadata_path)
        except Exception as error:
            raise SnapshotValidationError(
                f"Could not read CUA-Gym Parquet metadata: {error}"
            ) from error
        actual_schema = tuple(
            (field.name, str(field.type), field.nullable) for field in table.schema
        )
        expected_schema = tuple(
            (field.name, field.arrow_type, field.nullable)
            for field in self.manifest.metadata_schema
        )
        if actual_schema != expected_schema:
            raise SnapshotValidationError(
                "CUA-Gym metadata schema does not match the compatibility manifest"
            )
        if table.num_rows != self.manifest.metadata_row_count:
            raise SnapshotValidationError(
                f"Expected {self.manifest.metadata_row_count} metadata rows, "
                f"found {table.num_rows}"
            )

        rows = table.to_pylist()
        rows_by_id: dict[TaskId, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            raw_id = row.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                raise SnapshotValidationError(f"Metadata row {index} has an invalid ID")
            task_id = TaskId(raw_id)
            if task_id in rows_by_id:
                raise SnapshotValidationError(f"Duplicate metadata task ID: {task_id}")
            rows_by_id[task_id] = row

        selected_ids = (
            self._web_task_ids(rows_by_id)
            if self.platform is TaskPlatform.WEB
            else self._desktop_task_ids(rows_by_id)
        )
        tasks = tuple(
            self._parse_task_metadata(rows_by_id[task_id]) for task_id in selected_ids
        )
        self._catalog = TaskCatalog.from_tasks(tasks)
        return self._catalog

    def load_task_bundle(self, task_id: str | TaskId) -> TaskBundle:
        """Load one bundle into memory without extracting or executing it."""

        return self.load_task_bundles((task_id,))[0]

    def load_bundle_catalog(self) -> Mapping[TaskId, TaskBundle]:
        """Load every compatible bundle once and return an immutable ID index.

        The first call streams the compressed archive once. Later calls on this
        snapshot instance return the same mapping and perform no file-system work.
        """

        if self._bundle_catalog is not None:
            return self._bundle_catalog
        with self._bundle_catalog_lock:
            if self._bundle_catalog is not None:
                return self._bundle_catalog
            task_ids = tuple(task.id for task in self.load_catalog())
            bundles = self.load_task_bundles(task_ids)
            by_id = {
                task_id: bundle
                for task_id, bundle in zip(task_ids, bundles, strict=True)
            }
            if len(by_id) != len(task_ids):
                raise BundleValidationError(
                    "Bundle catalog contains duplicate task IDs"
                )
            self._bundle_catalog = MappingProxyType(by_id)
            return self._bundle_catalog

    def load_task_bundles(
        self,
        task_ids: Iterable[str | TaskId],
    ) -> tuple[TaskBundle, ...]:
        """Load several bundles in one streaming pass over the zstd tar archive."""

        requested = tuple(TaskId(str(task_id)) for task_id in task_ids)
        if not requested:
            return ()
        if len(set(requested)) != len(requested):
            raise ValueError("task_ids must not contain duplicates")
        active_catalog = self.load_catalog()
        metadata = {task_id: active_catalog.get(task_id) for task_id in requested}
        expected_members: dict[str, tuple[TaskId, str]] = {}
        for task_id, task in metadata.items():
            members = {
                task.task_json_member: "task.json",
                task.reward_member: "reward.py",
                **{
                    member: PurePosixPath(member).name
                    for member in task.setup_file_members
                },
            }
            for member_path, logical_name in members.items():
                if member_path in expected_members:
                    raise BundleValidationError(
                        f"Duplicate expected archive member: {member_path}"
                    )
                expected_members[member_path] = (task_id, logical_name)

        contents: dict[TaskId, dict[str, bytes]] = {
            task_id: {} for task_id in requested
        }
        # A cached catalog must never serve as proof that the artifact is still pinned.
        self._validate_snapshot_files()
        archive_path = self._file_path("artifact")
        try:
            with pa.input_stream(str(archive_path), compression="zstd") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        expected = expected_members.get(member.name)
                        if expected is None:
                            continue
                        if not member.isfile():
                            raise BundleValidationError(
                                f"Expected regular archive file: {member.name}"
                            )
                        task_id, logical_name = expected
                        if logical_name in contents[task_id]:
                            raise BundleValidationError(
                                f"Duplicate archive member: {member.name}"
                            )
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise BundleValidationError(
                                f"Could not read archive member: {member.name}"
                            )
                        contents[task_id][logical_name] = extracted.read()
                        if sum(len(files) for files in contents.values()) == len(
                            expected_members
                        ):
                            break
        except BundleValidationError:
            raise
        except Exception as error:
            raise BundleValidationError(
                f"Could not stream CUA-Gym artifact: {error}"
            ) from error

        bundles: dict[TaskId, TaskBundle] = {}
        for task_id in requested:
            expected_count = 2 + len(metadata[task_id].setup_file_members)
            if len(contents[task_id]) != expected_count:
                missing = sorted(
                    {
                        "task.json",
                        "reward.py",
                        *(
                            PurePosixPath(path).name
                            for path in metadata[task_id].setup_file_members
                        ),
                    }
                    - set(contents[task_id])
                )
                raise BundleValidationError(
                    f"Task {task_id} is missing archive members: {missing}"
                )
            bundles[task_id] = self._build_bundle(metadata[task_id], contents[task_id])
        return tuple(bundles[task_id] for task_id in requested)

    def materialize_task_bundle(
        self,
        bundle: TaskBundle,
        gateway_urls: Mapping[str | EndpointName, str],
    ) -> MaterializedTaskBundle:
        """Materialize one raw bundle using this snapshot's compatibility data."""

        return _materialize_task_bundle(bundle, gateway_urls, self.manifest)

    def parse_reward_stdout(self, task_id: str | TaskId, stdout: str) -> float:
        """Parse reward stdout under this snapshot's per-task compatibility policy."""

        compatibility = self.load_catalog().get(task_id).compatibility
        return _parse_reward_output(task_id, stdout, compatibility)

    def _web_task_ids(
        self, rows_by_id: Mapping[TaskId, dict[str, Any]]
    ) -> tuple[TaskId, ...]:
        excluded_ids = set(self.manifest.excluded_tasks)
        web_ids = {
            task_id
            for task_id, row in rows_by_id.items()
            if row.get("platform") == TaskPlatform.WEB
        }
        eligible_ids = web_ids - excluded_ids
        manifest_ids = set(self.manifest.tasks)
        if eligible_ids != manifest_ids:
            missing = sorted(manifest_ids - eligible_ids)
            unexpected = sorted(eligible_ids - manifest_ids)
            raise SnapshotValidationError(
                "Browser-only metadata selection differs from the compatibility manifest "
                f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
            )
        if not excluded_ids <= web_ids:
            raise SnapshotValidationError(
                "Manifest exclusions must refer to metadata rows labeled web"
            )
        if len(manifest_ids) != self.manifest.eligible_task_count:
            raise SnapshotValidationError(
                "Validated task count differs from eligible_task_count"
            )
        return tuple(sorted(manifest_ids))

    def _desktop_task_ids(
        self, rows_by_id: Mapping[TaskId, dict[str, Any]]
    ) -> tuple[TaskId, ...]:
        desktop = self.manifest.desktop
        desktop_ids = {
            task_id
            for task_id, row in rows_by_id.items()
            if row.get("platform") == TaskPlatform.DESKTOP
        }
        if len(desktop_ids) != desktop.task_count:
            raise SnapshotValidationError(
                f"Expected {desktop.task_count} desktop metadata rows, "
                f"found {len(desktop_ids)}"
            )
        excluded_ids = set(desktop.excluded_tasks)
        if not excluded_ids <= desktop_ids:
            raise SnapshotValidationError(
                "Manifest exclusions must refer to metadata rows labeled desktop"
            )
        if not set(desktop.reward_output_overrides) <= desktop_ids - excluded_ids:
            raise SnapshotValidationError(
                "Desktop reward overrides must refer to eligible desktop tasks"
            )
        eligible_ids = desktop_ids - excluded_ids
        if len(eligible_ids) != desktop.eligible_task_count:
            raise SnapshotValidationError(
                "Validated desktop task count differs from eligible_task_count"
            )
        return tuple(sorted(eligible_ids))

    def _validate_snapshot_files(self) -> None:
        if self.config.revision != self.manifest.revision:
            raise SnapshotValidationError(
                f"Configured revision {self.config.revision!r} does not match "
                f"manifest revision {self.manifest.revision!r}"
            )
        if not self.root.is_dir():
            raise SnapshotValidationError(
                f"CUA-Gym snapshot directory does not exist: {self.root}"
            )
        required_file_keys = {"metadata", "artifact", "stats", "url_variables"}
        if set(self.manifest.files) != required_file_keys:
            raise SnapshotValidationError(
                "Manifest files must contain metadata, artifact, stats, and url_variables"
            )
        for name, fingerprint in self.manifest.files.items():
            path = self._file_path(name)
            if not path.is_file():
                raise SnapshotValidationError(f"Missing CUA-Gym snapshot file: {path}")
            digest = _sha256(path)
            if digest != fingerprint.sha256:
                raise SnapshotValidationError(
                    f"SHA-256 mismatch for {fingerprint.path}: "
                    f"expected {fingerprint.sha256}, found {digest}"
                )

    def _file_path(self, name: str) -> Path:
        try:
            relative = PurePosixPath(self.manifest.files[name].path)
        except KeyError as error:
            raise SnapshotValidationError(f"Manifest has no {name!r} file") from error
        if relative.is_absolute() or ".." in relative.parts:
            raise SnapshotValidationError(
                f"Manifest file path must stay inside the snapshot: {relative}"
            )
        return self.root.joinpath(*relative.parts)

    def _parse_task_metadata(self, row: dict[str, Any]) -> TaskMetadata:
        task_id = TaskId(_required_string(row, "id"))
        compatibility = (
            self.manifest.task(task_id)
            if self.platform is TaskPlatform.WEB
            else self.manifest.desktop.task(task_id)
        )
        setup_files = _string_tuple(row, "setup_files")
        setup_members = _string_tuple(row, "setup_file_members")
        archive_path = _required_string(row, "archive_path")
        archive_member = _required_string(row, "archive_member")
        task_json_member = _required_string(row, "task_json_member")
        reward_member = _required_string(row, "reward_member")
        expected_archive_path = self.manifest.files["artifact"].path
        expected_setup_members = tuple(f"{task_id}/{name}" for name in setup_files)
        if (
            archive_path != expected_archive_path
            or archive_member != task_id
            or task_json_member != f"{task_id}/task.json"
            or reward_member != f"{task_id}/reward.py"
            or setup_members != expected_setup_members
        ):
            raise SnapshotValidationError(
                f"Archive manifest fields are inconsistent for task {task_id}"
            )
        num_setup_files = _required_int(row, "num_setup_files")
        if num_setup_files != len(setup_files):
            raise SnapshotValidationError(
                f"num_setup_files is inconsistent for task {task_id}"
            )
        if self.platform is TaskPlatform.WEB and (
            row.get("platform") != "web" or row.get("app_family") != "mock_web"
        ):
            raise SnapshotValidationError(
                f"Compatible task {task_id} is not browser-only mock web metadata"
            )
        if self.platform is TaskPlatform.DESKTOP and row.get("platform") != "desktop":
            raise SnapshotValidationError(
                f"Compatible task {task_id} is not desktop metadata"
            )
        difficulty = row.get("difficulty")
        if difficulty is not None and not isinstance(difficulty, str):
            raise SnapshotValidationError(f"Invalid difficulty for task {task_id}")
        has_ground_truth = row.get("has_ground_truth")
        if not isinstance(has_ground_truth, bool):
            raise SnapshotValidationError(
                f"Invalid has_ground_truth for task {task_id}"
            )
        return TaskMetadata(
            id=task_id,
            instruction=_required_string(row, "instruction"),
            app_type=_required_string(row, "app_type"),
            app_family=_required_string(row, "app_family"),
            platform=_required_string(row, "platform"),
            difficulty=difficulty,
            setup_kind=_required_string(row, "setup_kind"),
            num_setup_steps=_required_int(row, "num_setup_steps"),
            num_setup_files=num_setup_files,
            has_ground_truth=has_ground_truth,
            setup_files=setup_files,
            archive_path=archive_path,
            archive_member=archive_member,
            task_json_member=task_json_member,
            reward_member=reward_member,
            setup_file_members=setup_members,
            compatibility=compatibility,
        )

    def _build_bundle(
        self, metadata: TaskMetadata, contents: dict[str, bytes]
    ) -> TaskBundle:
        try:
            task_config = json.loads(contents["task.json"].decode("utf-8"))
            reward_source = contents["reward.py"].decode("utf-8")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BundleValidationError(
                f"Task {metadata.id} contains invalid UTF-8 or task JSON"
            ) from error
        if not isinstance(task_config, dict):
            raise BundleValidationError(f"Task config is not an object: {metadata.id}")
        for field, expected in (
            ("id", str(metadata.id)),
            ("instruction", metadata.instruction),
            ("app_type", metadata.app_type),
        ):
            if task_config.get(field) != expected:
                raise BundleValidationError(
                    f"Task config field {field!r} mismatches metadata for {metadata.id}"
                )
        setup_files = tuple(
            BundleFile(name=name, content=contents[name])
            for name in metadata.setup_files
        )
        if self.platform is TaskPlatform.WEB:
            _validate_execution_config(task_config, metadata)
            self._validate_web_endpoints(metadata, setup_files, reward_source)
        else:
            _validate_desktop_execution_config(
                task_config, metadata, self.manifest.desktop
            )
        return TaskBundle(
            metadata=metadata,
            task_config=freeze_json_object(task_config),
            reward_source=reward_source,
            setup_files=setup_files,
            setup_steps=_parse_setup_steps(task_config),
        )

    def _validate_web_endpoints(
        self,
        metadata: TaskMetadata,
        setup_files: tuple[BundleFile, ...],
        reward_source: str,
    ) -> None:
        try:
            setup_sources = tuple(setup_file.text() for setup_file in setup_files)
        except UnicodeDecodeError as error:
            raise BundleValidationError(
                f"Task {metadata.id} setup source is not UTF-8"
            ) from error
        derived_endpoints = derive_required_endpoints(
            metadata.compatibility,
            setup_sources,
            reward_source,
            self.manifest,
        )
        if derived_endpoints != metadata.compatibility.required_endpoints:
            raise BundleValidationError(
                f"Source-derived endpoints for {metadata.id} do not match manifest: "
                f"derived={derived_endpoints}, "
                f"manifest={metadata.compatibility.required_endpoints}"
            )


def _validate_execution_config(
    task_config: dict[str, Any], metadata: TaskMetadata
) -> None:
    evaluator = task_config.get("evaluator")
    if evaluator != {"type": "python", "url": "./reward.py"}:
        raise BundleValidationError(
            f"Unsupported evaluator config for task {metadata.id}: {evaluator!r}"
        )
    config = task_config.get("config")
    if not isinstance(config, list) or len(config) != metadata.num_setup_steps:
        raise BundleValidationError(f"Invalid setup config for task {metadata.id}")
    if len(config) != 2:
        raise BundleValidationError(
            f"Task {metadata.id} does not use the supported download/execute setup"
        )
    download, execute = config
    expected_files = [
        {"url": f"./{name}", "path": metadata.compatibility.setup_target_path}
        for name in metadata.setup_files
    ]
    if download != {"type": "download", "parameters": {"files": expected_files}}:
        raise BundleValidationError(
            f"Download setup config mismatches compatibility manifest for {metadata.id}"
        )
    expected_command = f"python3 {metadata.compatibility.setup_target_path}"
    if execute != {
        "type": "execute",
        "parameters": {"command": expected_command},
    }:
        raise BundleValidationError(
            f"Execute setup config mismatches compatibility manifest for {metadata.id}"
        )


_DESKTOP_EVALUATOR_KEYS = frozenset({"type", "url"})
_MAX_SETUP_SLEEP_S = 60.0


def _validate_desktop_execution_config(
    task_config: dict[str, Any],
    metadata: TaskMetadata,
    desktop: DesktopCompatibility,
) -> None:
    evaluator = task_config.get("evaluator")
    if not isinstance(evaluator, dict) or not _DESKTOP_EVALUATOR_KEYS <= set(evaluator):
        raise BundleValidationError(
            f"Unsupported evaluator config for task {metadata.id}: {evaluator!r}"
        )
    if (evaluator["type"], evaluator["url"]) != ("python", "./reward.py"):
        raise BundleValidationError(
            f"Unsupported evaluator config for task {metadata.id}: {evaluator!r}"
        )
    unknown_fields = set(evaluator) - _DESKTOP_EVALUATOR_KEYS - {"postconfig"}
    if unknown_fields:
        raise BundleValidationError(
            f"Unknown evaluator fields for task {metadata.id}: {sorted(unknown_fields)}"
        )
    if "postconfig" in evaluator and (
        _canonical_digest(evaluator["postconfig"])
        != desktop.evaluator_postconfig_sha256
    ):
        raise BundleValidationError(
            f"Task {metadata.id} carries an unpinned evaluator postconfig"
        )
    config = task_config.get("config")
    if not isinstance(config, list) or len(config) != metadata.num_setup_steps:
        raise BundleValidationError(f"Invalid setup config for task {metadata.id}")
    if not config or _step_type(config[0], metadata) is not SetupStepType.DOWNLOAD:
        raise BundleValidationError(
            f"Task {metadata.id} does not start with a download setup step"
        )
    allowed = frozenset(desktop.setup_step_types)
    for step in config:
        step_type = _step_type(step, metadata)
        if step_type not in allowed:
            raise BundleValidationError(
                f"Task {metadata.id} uses unsupported setup step {step_type}"
            )
        _validate_desktop_setup_parameters(step["parameters"], step_type, metadata)


def _step_type(step: object, metadata: TaskMetadata) -> SetupStepType:
    if (
        not isinstance(step, dict)
        or set(step) != {"type", "parameters"}
        or not isinstance(step["parameters"], dict)
    ):
        raise BundleValidationError(f"Malformed setup step for task {metadata.id}")
    try:
        return SetupStepType(step["type"])
    except ValueError as error:
        raise BundleValidationError(
            f"Unknown setup step type for task {metadata.id}: {step['type']!r}"
        ) from error


def _validate_desktop_setup_parameters(
    parameters: dict[str, Any],
    step_type: SetupStepType,
    metadata: TaskMetadata,
) -> None:
    expected_keys = {
        SetupStepType.DOWNLOAD: {"files"},
        SetupStepType.EXECUTE: {"command"},
        SetupStepType.LAUNCH: {"command"},
        SetupStepType.OPEN: {"path"},
        SetupStepType.SLEEP: {"seconds"},
    }[step_type]
    if set(parameters) != expected_keys:
        raise BundleValidationError(
            f"Task {metadata.id} {step_type} step has parameters {sorted(parameters)}"
        )
    if step_type is SetupStepType.DOWNLOAD:
        _validate_desktop_download(parameters["files"], metadata)
    elif step_type is SetupStepType.OPEN:
        _validate_guest_path(parameters["path"], metadata)
    elif step_type is SetupStepType.SLEEP:
        seconds = parameters["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not 0 < float(seconds) <= _MAX_SETUP_SLEEP_S
        ):
            raise BundleValidationError(
                f"Task {metadata.id} sleep step waits {seconds!r} seconds"
            )
    else:
        _validate_setup_command(parameters["command"], metadata)


def _validate_desktop_download(files: object, metadata: TaskMetadata) -> None:
    if not isinstance(files, list) or not files:
        raise BundleValidationError(f"Task {metadata.id} download step lists no files")
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"url", "path"}:
            raise BundleValidationError(
                f"Task {metadata.id} download entry is malformed: {entry!r}"
            )
        url = entry["url"]
        if (
            not isinstance(url, str)
            or not url.startswith("./")
            or url[2:] not in metadata.setup_files
        ):
            raise BundleValidationError(
                f"Task {metadata.id} downloads {url!r}, which is not a bundled "
                "setup file"
            )
        _validate_guest_path(entry["path"], metadata)


def _validate_setup_command(command: object, metadata: TaskMetadata) -> None:
    if isinstance(command, str):
        if command.strip():
            return
    elif (
        isinstance(command, list)
        and command
        and all(isinstance(value, str) and value for value in command)
    ):
        return
    raise BundleValidationError(
        f"Task {metadata.id} setup command is not a shell-free argv or string"
    )


def _validate_guest_path(path: object, metadata: TaskMetadata) -> None:
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or posixpath.normpath(path) != path
    ):
        raise BundleValidationError(
            f"Task {metadata.id} references an unsafe guest path: {path!r}"
        )


def _parse_setup_steps(task_config: dict[str, Any]) -> tuple[SetupStep, ...]:
    steps: list[SetupStep] = []
    for step in task_config.get("config", ()):
        parameters = freeze_json(step["parameters"])
        if not isinstance(parameters, FrozenJsonObject):
            raise BundleValidationError("Setup step parameters must be an object")
        steps.append(SetupStep(type=SetupStepType(step["type"]), parameters=parameters))
    return tuple(steps)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise SnapshotValidationError(f"Metadata field {field!r} must be a string")
    return value


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SnapshotValidationError(
            f"Metadata field {field!r} must be a non-negative integer"
        )
    return value


def _string_tuple(row: dict[str, Any], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise SnapshotValidationError(
            f"Metadata field {field!r} must be an array of strings"
        )
    return tuple(value)
