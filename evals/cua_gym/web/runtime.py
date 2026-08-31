from __future__ import annotations

import random
import secrets
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from desktop.vm import PortRangeLease, acquire_port_range
from pydantic import Field

from evals.cua_gym.manifest import PINNED_REVISION
from evals.cua_gym.models import (
    DatasetSnapshotConfig,
    MaterializedTaskBundle,
    TaskBundle,
    TaskId,
    TaskPlatform,
)
from evals.cua_gym.runtime import (
    CuaGymDesktopTask,
    _materialized_setup_steps,
    _required_absolute_path,
    _run_reward,
    _run_setup_steps,
)
from evals.cua_gym.snapshot import CuaGymDatasetSnapshot
from evals.cua_gym.web.desktop import CuaGymDesktopBrowser, CuaGymDesktopConfig
from evals.cua_gym.web.gateway import (
    CuaGymEpisodeGateway,
    CuaGymGatewayConfig,
    GatewayPhase,
)
from evals.cua_gym.web.hub import (
    CuaGymHubConfig,
    CuaGymHubDescriptor,
    CuaGymHubSupervisor,
)
from evals.cua_gym.web.manifest import (
    CuaGymWebRuntimeManifest,
    load_default_web_runtime_manifest,
)
from evals.tasks import DesktopTaskData, register_preparer

__all__ = [
    "CUA_GYM_WEB_KIND",
    "CuaGymWebPreparer",
    "CuaGymWebTask",
    "CuaGymWebTaskData",
    "CuaGymWebTaskset",
    "CuaGymWebTasksetConfig",
]

CUA_GYM_WEB_KIND = "cua_gym_web"
_HUB_PORT_RANGE = (30_000, 30_999)
_GATEWAY_PORT_RANGE = (31_000, 31_999)
_GATEWAY_BIND_HOST = "0.0.0.0"
_GUEST_GATEWAY_HOST = "10.0.2.2"
_HUB_STARTUP_TIMEOUT_S = 300.0
_HUB_REQUEST_TIMEOUT_S = 20.0
_BROWSER_READY_TIMEOUT_S = 120.0


class CuaGymWebTaskData(DesktopTaskData):
    task_id: str
    dataset_root: str
    dataset_revision: str


class CuaGymWebTask(CuaGymDesktopTask):
    pass


class CuaGymWebTasksetConfig(vf.TasksetConfig):
    dataset_root: str = ""
    task_ids: list[str] = Field(default_factory=list)
    max_tasks: int = Field(default=0, ge=0)
    shuffle_seed: int = Field(default=-1, ge=-1)
    max_steps: int = Field(default=32, ge=1)


class CuaGymWebTaskset(vf.Taskset[CuaGymWebTask, CuaGymWebTasksetConfig]):
    def load(self) -> Iterable[CuaGymWebTask]:
        root = _required_absolute_path(self.config.dataset_root, "dataset_root")
        snapshot = _web_snapshot(str(root), PINNED_REVISION)
        web_manifest = load_default_web_runtime_manifest()
        web_manifest.validate_dataset(snapshot.manifest)
        catalog = snapshot.load_catalog()
        supported = tuple(
            task.id
            for task in catalog
            if not web_manifest.incompatibilities_for(
                task.id, task.compatibility.required_endpoints
            )
        )
        selected = self._select_task_ids(tuple(task.id for task in catalog), supported)
        for idx, task_id in enumerate(selected):
            metadata = catalog.get(task_id)
            yield CuaGymWebTask(
                CuaGymWebTaskData(
                    idx=idx,
                    name=str(task_id),
                    prompt=metadata.instruction,
                    instruction=metadata.instruction,
                    kind=CUA_GYM_WEB_KIND,
                    max_steps=self.config.max_steps,
                    task_id=str(task_id),
                    dataset_root=str(root),
                    dataset_revision=snapshot.manifest.revision,
                ),
                self.config.task,
            )

    def _select_task_ids(
        self,
        available: tuple[TaskId, ...],
        supported: tuple[TaskId, ...],
    ) -> list[TaskId]:
        available_set = set(available)
        supported_set = set(supported)
        if self.config.task_ids:
            selected = [TaskId(task_id) for task_id in self.config.task_ids]
            if len(set(selected)) != len(selected):
                raise ValueError("CUA-Gym web task_ids must not contain duplicates")
            missing = [task_id for task_id in selected if task_id not in available_set]
            if missing:
                raise ValueError(
                    "unknown CUA-Gym web task IDs: "
                    + ", ".join(str(task_id) for task_id in missing)
                )
            unsupported = [
                task_id for task_id in selected if task_id not in supported_set
            ]
            if unsupported:
                raise ValueError(
                    "unsupported CUA-Gym web task IDs: "
                    + ", ".join(str(task_id) for task_id in unsupported)
                )
        else:
            selected = list(supported)
        if not selected:
            raise ValueError("CUA-Gym web selection contains no supported tasks")
        if self.config.shuffle_seed >= 0:
            random.Random(self.config.shuffle_seed).shuffle(selected)
        if self.config.max_tasks:
            selected = selected[: self.config.max_tasks]
        return selected


class CuaGymWebPreparer:
    kind = CUA_GYM_WEB_KIND

    def episode(
        self,
        *,
        session: Any,
        task: CuaGymWebTaskData,
        episode_id: str,
        artifacts: Path,
        hub_image: Path,
        apptainer_binary: Path,
        port_lock_dir: Path,
        guest_password: str,
    ) -> _CuaGymWebEpisode:
        return _CuaGymWebEpisode(
            session=session,
            task=task,
            episode_id=episode_id,
            artifacts=artifacts,
            hub_image=hub_image,
            apptainer_binary=apptainer_binary,
            port_lock_dir=port_lock_dir,
            guest_password=guest_password,
        )


class _CuaGymWebEpisode:
    def __init__(
        self,
        *,
        session: Any,
        task: CuaGymWebTaskData,
        episode_id: str,
        artifacts: Path,
        hub_image: Path,
        apptainer_binary: Path,
        port_lock_dir: Path,
        guest_password: str,
    ) -> None:
        self.session = session
        self.task = task
        self.episode_id = episode_id
        self.artifacts = artifacts.absolute()
        self.hub_image = hub_image
        self.apptainer_binary = apptainer_binary
        self.port_lock_dir = port_lock_dir
        self.guest_password = guest_password
        self.manifest = load_default_web_runtime_manifest()
        self.snapshot, self.bundle = _bundle_for_task(task, self.manifest)
        self.materialized: MaterializedTaskBundle | None = None
        self.hub: CuaGymHubSupervisor | None = None
        self.descriptor: CuaGymHubDescriptor | None = None
        self.gateway_lease: PortRangeLease | None = None
        self.gateway: CuaGymEpisodeGateway | None = None
        self.browser: CuaGymDesktopBrowser | None = None
        self.browser_identity: str | None = None

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if session is not self.session or task is not self.task:
            raise RuntimeError("CUA-Gym web episode was called with another lease")
        web_root = self.artifacts / "web"
        self.hub = CuaGymHubSupervisor(
            config=CuaGymHubConfig(
                image_path=self.hub_image,
                state_root=web_root / "hub-state",
                log_root=web_root / "hub-logs",
                descriptor_path=web_root / "hub-descriptor.json",
                port_range_start=_HUB_PORT_RANGE[0],
                port_range_end=_HUB_PORT_RANGE[1],
                port_lock_dir=self.port_lock_dir,
                startup_timeout_s=_HUB_STARTUP_TIMEOUT_S,
                request_timeout_s=_HUB_REQUEST_TIMEOUT_S,
                apptainer_binary=self.apptainer_binary,
            ),
            manifest=self.manifest,
        )
        self.descriptor = self.hub.start()
        self.hub.assert_healthy()
        self.gateway_lease = acquire_port_range(
            count=1,
            purpose="cua-gym-web-episode",
            range_start=_GATEWAY_PORT_RANGE[0],
            range_end=_GATEWAY_PORT_RANGE[1],
            lock_dir=self.port_lock_dir,
            bind_host=_GATEWAY_BIND_HOST,
        )
        endpoints = self.bundle.metadata.compatibility.required_endpoints
        self.gateway = CuaGymEpisodeGateway(
            config=CuaGymGatewayConfig(
                bind_host=_GATEWAY_BIND_HOST,
                port=self.gateway_lease.ports[0],
                endpoints=endpoints,
                episode_id=self.episode_id,
                namespace_key=secrets.token_bytes(32),
                hub=self.descriptor,
            ),
            manifest=self.manifest,
        )
        self.gateway.start()
        self.gateway.assert_healthy()
        self.browser = CuaGymDesktopBrowser(
            guest=session,
            config=CuaGymDesktopConfig(
                browser_debugging_url=_chromium_debugging_url(session),
                guest_gateway_host=_GUEST_GATEWAY_HOST,
                guest_hostnames=tuple(sorted(self.gateway.gateway_hostnames.values())),
                guest_password=self.guest_password,
                browser_ready_timeout_s=_BROWSER_READY_TIMEOUT_S,
            ),
        )
        self.browser.configure_guest_hosts()
        self.browser_identity = self.browser.ensure_browser()
        self.materialized = self.snapshot.materialize_task_bundle(
            self.bundle, self.gateway.gateway_urls
        )
        with tempfile.TemporaryDirectory(prefix="cua-gym-web-setup-") as staging:
            steps = _materialized_setup_steps(
                self.bundle.setup_steps, self.materialized, Path(staging)
            )
            if steps:
                _run_setup_steps(session, steps)
        self.browser.verify_after_setup(self.browser_identity)
        self.gateway.wait_for_browser_session(_BROWSER_READY_TIMEOUT_S)
        self.gateway.transition(GatewayPhase.ROLLOUT)
        return {
            "prepared": CUA_GYM_WEB_KIND,
            "task_id": str(self.bundle.task_id),
            "dataset_revision": self.snapshot.manifest.revision,
            "hub_revision": self.manifest.hub_revision,
            "endpoints": [str(endpoint) for endpoint in endpoints],
            "setup_steps": len(self.bundle.setup_steps),
        }

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if session is not self.session or task is not self.task:
            raise RuntimeError("CUA-Gym web episode was called with another lease")
        return {
            "cursor": list(session.cursor_position()),
            "screen": list(session.screen_size()),
            "postcondition_status": "ok",
        }

    def evaluate(
        self,
        session: Any,
        task: DesktopTaskData,
        *,
        declared: str | None,
    ) -> float:
        del declared
        if session is not self.session or task is not self.task:
            raise RuntimeError("CUA-Gym web episode was called with another lease")
        if self.gateway is None or self.materialized is None:
            raise RuntimeError("CUA-Gym web episode was not prepared")
        self.gateway.transition(GatewayPhase.EVALUATE)
        result = _run_reward(session, self.materialized)
        if result.returncode != 0:
            raise RuntimeError(
                f"CUA-Gym reward process exited {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return self.snapshot.parse_reward_stdout(self.bundle.task_id, result.stdout)

    def close(self) -> None:
        errors: list[Exception] = []

        def attempt(function: Any, *args: Any, **kwargs: Any) -> None:
            try:
                function(*args, **kwargs)
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        gateway = self.gateway
        descriptor = self.descriptor
        private_sessions = gateway.private_sessions if gateway is not None else ()
        if (
            self.browser is not None
            and self.browser_identity is not None
            and gateway is not None
        ):
            attempt(
                self.browser.cleanup_browser,
                origins=tuple(sorted(gateway.gateway_urls.values())),
                expected_identity=self.browser_identity,
            )
        if gateway is not None:
            attempt(gateway.close)
            self.gateway = None
        if descriptor is not None and private_sessions:
            attempt(
                descriptor.cleanup_private_sessions,
                private_sessions,
                manifest=self.manifest,
            )
        if self.hub is not None:
            attempt(self.hub.close)
            self.hub = None
            self.descriptor = None
        if self.gateway_lease is not None:
            attempt(self.gateway_lease.release)
            self.gateway_lease = None
        if errors:
            raise RuntimeError(
                f"CUA-Gym web cleanup failed with {len(errors)} error(s)"
            ) from errors[0]


@lru_cache(maxsize=32)
def _web_snapshot(dataset_root: str, revision: str) -> CuaGymDatasetSnapshot:
    return CuaGymDatasetSnapshot(
        DatasetSnapshotConfig(Path(dataset_root), revision=revision),
        platform=TaskPlatform.WEB,
    )


def _bundle_for_task(
    task: CuaGymWebTaskData,
    web_manifest: CuaGymWebRuntimeManifest,
) -> tuple[CuaGymDatasetSnapshot, TaskBundle]:
    root = _required_absolute_path(task.dataset_root, "dataset_root")
    if task.dataset_revision != PINNED_REVISION:
        raise ValueError(
            f"CUA-Gym task revision {task.dataset_revision!r} is not "
            f"the pinned revision {PINNED_REVISION!r}"
        )
    snapshot = _web_snapshot(str(root), task.dataset_revision)
    web_manifest.validate_dataset(snapshot.manifest)
    metadata = snapshot.load_catalog().get(task.task_id)
    if task.instruction != metadata.instruction:
        raise ValueError("CUA-Gym task instruction differs from the pinned dataset")
    incompatibilities = web_manifest.incompatibilities_for(
        task.task_id, metadata.compatibility.required_endpoints
    )
    if incompatibilities:
        raise ValueError(
            "Unsupported CUA-Gym web task: " + "; ".join(incompatibilities)
        )
    return snapshot, snapshot.load_task_bundle(metadata.id)


def _chromium_debugging_url(session: Any) -> str:
    port = getattr(session.ports, "chromium", None)
    if type(port) is not int or port <= 0:
        raise RuntimeError("desktop session has no forwarded Chromium port")
    return f"http://127.0.0.1:{port}"


register_preparer(CuaGymWebPreparer())
