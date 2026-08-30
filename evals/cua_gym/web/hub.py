from __future__ import annotations

import ipaddress
import json
import math
import os
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from desktop.vm import PortRangeLease, acquire_port_range

from ..models import EndpointName
from .image import (
    HUB_BASE_IMAGE,
    HUB_SOURCE_PATCH_SHA256,
    CuaGymHubImageManifest,
    hub_image_manifest_path,
)
from .manifest import CuaGymWebRuntimeManifest

CUA_GYM_HUB_DESCRIPTOR_VERSION = 1

_BIND_HOST = "127.0.0.1"
_DESCRIPTOR_KEYS = {
    "admin_token",
    "backend_urls",
    "dataset_revision",
    "descriptor_version",
    "hub_revision",
    "request_timeout_s",
    "state_root",
}
_MAX_DESCRIPTOR_BYTES = 1_048_576
_ADMIN_TOKEN_LENGTH = 64
_PRIVATE_SID_LENGTH = 64


@dataclass(frozen=True)
class CuaGymHubConfig:
    image_path: Path
    state_root: Path
    log_root: Path
    descriptor_path: Path
    port_range_start: int
    port_range_end: int
    port_lock_dir: Path
    startup_timeout_s: float
    request_timeout_s: float
    apptainer_binary: Path

    def __post_init__(self) -> None:
        for name in (
            "image_path",
            "state_root",
            "log_root",
            "descriptor_path",
            "port_lock_dir",
            "apptainer_binary",
        ):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
            object.__setattr__(self, name, path)
        if (
            type(self.port_range_start) is not int
            or type(self.port_range_end) is not int
            or not 1 <= self.port_range_start <= self.port_range_end <= 65_535
        ):
            raise ValueError("Hub port range must be within 1-65535")
        for name in ("startup_timeout_s", "request_timeout_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class CuaGymHubDescriptor:
    dataset_revision: str
    hub_revision: str
    backend_urls: Mapping[EndpointName, str]
    state_root: Path
    request_timeout_s: float
    admin_token: str = field(repr=False)
    descriptor_version: int = CUA_GYM_HUB_DESCRIPTOR_VERSION

    def __post_init__(self) -> None:
        if self.descriptor_version != CUA_GYM_HUB_DESCRIPTOR_VERSION:
            raise ValueError(
                f"Unsupported CUA-Gym Hub descriptor version: {self.descriptor_version!r}"
            )
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision:
            raise ValueError("Hub descriptor dataset revision must not be empty")
        if not isinstance(self.hub_revision, str) or not self.hub_revision:
            raise ValueError("Hub descriptor revision must not be empty")
        if not self.backend_urls:
            raise ValueError("Hub descriptor must contain backend URLs")
        backend_urls: dict[EndpointName, str] = {}
        for raw_endpoint, backend_url in self.backend_urls.items():
            if not isinstance(raw_endpoint, str) or not raw_endpoint:
                raise ValueError("Hub descriptor contains an invalid endpoint")
            endpoint = EndpointName(raw_endpoint)
            if not isinstance(backend_url, str):
                raise TypeError("Hub descriptor contains an invalid backend URL")
            _validate_loopback_backend_url(backend_url)
            backend_urls[endpoint] = backend_url
        state_root = Path(self.state_root)
        if not state_root.is_absolute():
            raise ValueError("Hub descriptor state_root must be absolute")
        if (
            isinstance(self.request_timeout_s, bool)
            or not isinstance(self.request_timeout_s, (int, float))
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("Hub descriptor request timeout must be positive")
        if (
            not isinstance(self.admin_token, str)
            or len(self.admin_token) != _ADMIN_TOKEN_LENGTH
            or any(
                character not in "0123456789abcdef" for character in self.admin_token
            )
        ):
            raise ValueError(
                "Hub descriptor admin token must be 64 lowercase hex digits"
            )
        object.__setattr__(self, "backend_urls", MappingProxyType(backend_urls))
        object.__setattr__(self, "state_root", state_root)
        object.__setattr__(self, "request_timeout_s", float(self.request_timeout_s))

    def validate_manifest(self, manifest: CuaGymWebRuntimeManifest) -> None:
        if self.dataset_revision != manifest.dataset_revision:
            raise ValueError("Hub descriptor dataset revision does not match manifest")
        if self.hub_revision != manifest.hub_revision:
            raise ValueError("Hub descriptor revision does not match manifest")
        if set(self.backend_urls) != set(manifest.endpoint_apps):
            raise ValueError("Hub descriptor must cover exactly the manifest endpoints")
        urls_by_app: dict[str, str] = {}
        apps_by_url: dict[str, str] = {}
        for endpoint, app in manifest.endpoint_apps.items():
            backend_url = self.backend_urls[endpoint]
            previous_url = urls_by_app.setdefault(app, backend_url)
            if previous_url != backend_url:
                raise ValueError(
                    "Hub descriptor maps one Hub app to multiple backend URLs"
                )
            previous_app = apps_by_url.setdefault(backend_url, app)
            if previous_app != app:
                raise ValueError(
                    "Hub descriptor maps multiple Hub apps to one backend URL"
                )

    def cleanup_private_sessions(
        self,
        private_sessions: tuple[tuple[EndpointName, str], ...],
        *,
        manifest: CuaGymWebRuntimeManifest,
    ) -> None:
        """Reset and remove the exact private state created by one episode."""

        self.validate_manifest(manifest)
        if type(private_sessions) is not tuple:
            raise TypeError("private_sessions must be a tuple of endpoint/SID pairs")

        sessions: list[tuple[EndpointName, str]] = []
        for pair in private_sessions:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("private_sessions must contain endpoint/SID pairs")
            raw_endpoint, private_sid = pair
            if (
                type(raw_endpoint) is not str
                or raw_endpoint not in manifest.endpoint_apps
            ):
                raise ValueError(f"Unknown CUA-Gym endpoint: {raw_endpoint!r}")
            if (
                type(private_sid) is not str
                or len(private_sid) != _PRIVATE_SID_LENGTH
                or any(character not in "0123456789abcdef" for character in private_sid)
            ):
                raise ValueError(
                    "private Hub session IDs must be 64 lowercase hex digits"
                )
            sessions.append((EndpointName(raw_endpoint), private_sid))

        if self.state_root.is_symlink() or not self.state_root.is_dir():
            raise RuntimeError("Hub state root must be a real directory")
        resolved_state_root = self.state_root.resolve(strict=True)
        roots_by_app: dict[str, tuple[Path, ...]] = {}
        candidates: list[tuple[Path, ...]] = []
        for endpoint, private_sid in sessions:
            app = manifest.endpoint_apps[endpoint]
            roots = roots_by_app.get(app)
            if roots is None:
                app_root = self.state_root / app
                if app_root.is_symlink() or not app_root.is_dir():
                    raise RuntimeError(f"Hub app state root is not a directory: {app}")
                if app_root.resolve(strict=True).parent != resolved_state_root:
                    raise RuntimeError(f"Hub app state root escaped state_root: {app}")
                writable_roots: list[Path] = []
                for directory in manifest.writable_directories:
                    writable_root = app_root / directory
                    if writable_root.is_symlink() or not writable_root.is_dir():
                        raise RuntimeError(
                            f"Hub writable state root is not a directory: {app}/{directory}"
                        )
                    if writable_root.resolve(strict=True).parent != app_root.resolve(
                        strict=True
                    ):
                        raise RuntimeError(
                            f"Hub writable state root escaped its app root: "
                            f"{app}/{directory}"
                        )
                    writable_roots.append(writable_root)
                roots = tuple(writable_roots)
                roots_by_app[app] = roots

            exact_candidates: list[Path] = []
            for writable_root in roots:
                for candidate in (
                    writable_root / private_sid,
                    writable_root / f"{private_sid}.json",
                    writable_root / f"{private_sid}_initial.json",
                    writable_root / f"{private_sid}.initial.json",
                ):
                    if candidate.parent.resolve(strict=True) != writable_root.resolve(
                        strict=True
                    ):
                        raise RuntimeError(
                            f"Hub cleanup path escaped its writable root: {candidate}"
                        )
                    if (
                        candidate.exists()
                        and not candidate.is_symlink()
                        and not candidate.is_file()
                        and not candidate.is_dir()
                    ):
                        raise RuntimeError(
                            f"Hub cleanup target has an unsupported type: {candidate}"
                        )
                    exact_candidates.append(candidate)
            candidates.append(tuple(exact_candidates))

        for endpoint, private_sid in sessions:
            self._reset_private_session(endpoint, private_sid)
        for exact_candidates in candidates:
            for candidate in exact_candidates:
                if candidate.is_symlink() or candidate.is_file():
                    candidate.unlink(missing_ok=True)
                elif candidate.is_dir():
                    shutil.rmtree(candidate)

    def _reset_private_session(
        self,
        endpoint: EndpointName,
        private_sid: str,
    ) -> None:
        request = Request(
            f"{self.backend_urls[endpoint]}/post?sid={private_sid}",
            data=json.dumps({"action": "reset"}).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-CUA-Admin-Token": self.admin_token,
            },
        )
        try:
            with urlopen(request, timeout=self.request_timeout_s) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Hub reset failed for {endpoint}: HTTP {response.status}"
                    )
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Hub reset failed for {endpoint}: {error}") from error

    def publish(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("Hub descriptor path must be absolute")
        payload: dict[str, object] = {
            "admin_token": self.admin_token,
            "backend_urls": {
                str(endpoint): url for endpoint, url in self.backend_urls.items()
            },
            "dataset_revision": self.dataset_revision,
            "descriptor_version": self.descriptor_version,
            "hub_revision": self.hub_revision,
            "request_timeout_s": self.request_timeout_s,
            "state_root": str(self.state_root),
        }
        _publish_private_json(path, payload)

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        manifest: CuaGymWebRuntimeManifest,
    ) -> CuaGymHubDescriptor:
        if not path.is_absolute():
            raise ValueError("Hub descriptor path must be absolute")
        payload = _read_private_json(path)
        if set(payload) != _DESCRIPTOR_KEYS:
            raise ValueError("Hub descriptor has an unexpected schema")
        backend_payload = payload["backend_urls"]
        if not isinstance(backend_payload, dict) or not all(
            isinstance(endpoint, str) for endpoint in backend_payload
        ):
            raise ValueError("Hub descriptor backend_urls must be an object")
        descriptor = cls(
            descriptor_version=_json_integer(payload, "descriptor_version"),
            dataset_revision=_json_string(payload, "dataset_revision"),
            hub_revision=_json_string(payload, "hub_revision"),
            backend_urls={
                EndpointName(endpoint): _json_value_string(value, "backend URL")
                for endpoint, value in backend_payload.items()
            },
            state_root=Path(_json_string(payload, "state_root")),
            request_timeout_s=_json_number(payload, "request_timeout_s"),
            admin_token=_json_string(payload, "admin_token"),
        )
        descriptor.validate_manifest(manifest)
        return descriptor


class CuaGymHubSupervisor:
    def __init__(
        self,
        *,
        config: CuaGymHubConfig,
        manifest: CuaGymWebRuntimeManifest,
    ) -> None:
        if config.port_range_end - config.port_range_start + 1 < len(manifest.apps):
            raise ValueError("Hub port range is smaller than the manifest app set")
        self.config = config
        self.manifest = manifest
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: list[BinaryIO] = []
        self._descriptor: CuaGymHubDescriptor | None = None
        self._port_lease: PortRangeLease | None = None
        self._published_descriptor = False

    def start(self) -> CuaGymHubDescriptor:
        if self._descriptor is not None or self._processes:
            raise RuntimeError("CUA-Gym-Hub supervisor has already been started")
        self._validate_inputs()
        self._validate_image()
        self.config.state_root.mkdir(parents=True)
        self.config.log_root.mkdir(parents=True)
        self.config.descriptor_path.parent.mkdir(parents=True, exist_ok=True)

        apps = self.manifest.apps
        try:
            self._port_lease = acquire_port_range(
                count=len(apps),
                purpose="cua-gym-hub",
                range_start=self.config.port_range_start,
                range_end=self.config.port_range_end,
                lock_dir=self.config.port_lock_dir,
                bind_host=_BIND_HOST,
            )
            app_urls = {
                app: f"http://{_BIND_HOST}:{port}"
                for app, port in zip(apps, self._port_lease.ports, strict=True)
            }
            self._prepare_state_directories(apps)
            admin_token = secrets.token_hex(32)
            for app in apps:
                self._start_app(app, app_urls[app], admin_token)
            self._wait_until_healthy(app_urls, admin_token)
            descriptor = CuaGymHubDescriptor(
                dataset_revision=self.manifest.dataset_revision,
                hub_revision=self.manifest.hub_revision,
                backend_urls={
                    endpoint: app_urls[app]
                    for endpoint, app in self.manifest.endpoint_apps.items()
                },
                state_root=self.config.state_root,
                request_timeout_s=self.config.request_timeout_s,
                admin_token=admin_token,
            )
            descriptor.validate_manifest(self.manifest)
            self._assert_processes_alive()
            descriptor.publish(self.config.descriptor_path)
            self._published_descriptor = True
            self._descriptor = descriptor
            self._assert_processes_alive()
            return descriptor
        except Exception as startup_error:
            try:
                self.close()
            except RuntimeError as cleanup_error:
                startup_error.add_note(
                    f"Hub startup cleanup also failed: {cleanup_error}"
                )
            raise

    def assert_healthy(self) -> None:
        if self._descriptor is None:
            raise RuntimeError("CUA-Gym-Hub deployment is not running")
        self._assert_processes_alive()
        for app, url in self._app_urls(self._descriptor).items():
            if not self._probe(url, self._descriptor.admin_token):
                raise RuntimeError(f"CUA-Gym-Hub app {app} failed its health check")

    def close(self) -> None:
        self._descriptor = None
        errors: list[Exception] = []
        if self._published_descriptor:
            try:
                self.config.descriptor_path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(error)
            self._published_descriptor = False
        for process in self._processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    errors.append(error)
        deadline = time.monotonic() + 15.0
        for process in self._processes.values():
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    errors.append(error)
            except OSError as error:
                errors.append(error)
        self._processes.clear()
        for log in self._logs:
            try:
                log.close()
            except OSError as error:
                errors.append(error)
        self._logs.clear()
        if self._port_lease is not None:
            try:
                self._port_lease.release()
            except OSError as error:
                errors.append(error)
            self._port_lease = None
        if errors:
            raise RuntimeError(
                f"CUA-Gym-Hub cleanup failed with {len(errors)} error(s)"
            ) from errors[0]

    def __enter__(self) -> CuaGymHubDescriptor:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _validate_inputs(self) -> None:
        if not self.config.image_path.is_file():
            raise FileNotFoundError(
                f"CUA-Gym-Hub image is missing: {self.config.image_path}"
            )
        if not self.config.apptainer_binary.is_file() or not os.access(
            self.config.apptainer_binary, os.X_OK
        ):
            raise FileNotFoundError(
                "CUA-Gym-Hub apptainer executable is missing or not executable: "
                f"{self.config.apptainer_binary}"
            )
        if self.config.state_root.exists():
            raise FileExistsError(
                f"CUA-Gym-Hub state root already exists: {self.config.state_root}"
            )
        if self.config.log_root.exists():
            raise FileExistsError(
                f"CUA-Gym-Hub log root already exists: {self.config.log_root}"
            )
        if (
            self.config.descriptor_path.exists()
            or self.config.descriptor_path.is_symlink()
        ):
            raise FileExistsError(
                f"CUA-Gym-Hub descriptor already exists: {self.config.descriptor_path}"
            )

    def _validate_image(self) -> None:
        image_manifest = CuaGymHubImageManifest.read(
            hub_image_manifest_path(self.config.image_path)
        )
        image_manifest.validate(
            self.config.image_path,
            web_manifest=self.manifest,
        )
        result = subprocess.run(
            [
                str(self.config.apptainer_binary),
                "inspect",
                "--json",
                str(self.config.image_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not inspect CUA-Gym-Hub image: " + result.stderr.strip()
            )
        try:
            payload = json.loads(result.stdout)
            labels = payload["data"]["attributes"]["labels"]
            revision = labels["org.opencontainers.image.revision"]
            base_digest = labels["org.opencontainers.image.base.digest"]
            source_patch = labels["io.juergen.cua-gym.source-patch.sha256"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("CUA-Gym-Hub image has invalid metadata") from error
        if revision != self.manifest.hub_revision:
            raise RuntimeError(
                "CUA-Gym-Hub image revision differs from runtime manifest: "
                f"{revision!r} != {self.manifest.hub_revision!r}"
            )
        if base_digest != HUB_BASE_IMAGE.removeprefix("node@"):
            raise RuntimeError("CUA-Gym-Hub image base digest differs from the pin")
        if source_patch != HUB_SOURCE_PATCH_SHA256:
            raise RuntimeError("CUA-Gym-Hub source patch differs from the pin")

    def _prepare_state_directories(self, apps: tuple[str, ...]) -> None:
        for app in apps:
            for directory in self.manifest.writable_directories:
                path = self.config.state_root / app / directory
                path.mkdir(parents=True)
                path.chmod(0o700)

    def _start_app(self, app: str, app_url: str, admin_token: str) -> None:
        port = urlsplit(app_url).port
        if port is None:
            raise RuntimeError(f"Hub app URL has no port: {app_url}")
        log = (self.config.log_root / f"{app}.log").open("xb", buffering=0)
        self._logs.append(log)
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("APPTAINERENV_")
        }
        environment.update(
            {
                "APPTAINERENV_CUA_GYM_ADMIN_TOKEN": admin_token,
                "APPTAINERENV_CUA_GYM_HARDENED": "1",
                "APPTAINERENV_NODE_ENV": "production",
            }
        )
        process = subprocess.Popen(
            [
                str(self.config.apptainer_binary),
                "exec",
                "--cleanenv",
                "--writable-tmpfs",
                "--bind",
                f"{self.config.state_root}:/var/lib/cua-gym",
                str(self.config.image_path),
                "npm",
                "--prefix",
                f"/opt/cua-gym-hub/websites/{app}",
                "run",
                "preview",
                "--",
                "--host",
                _BIND_HOST,
                "--port",
                str(port),
                "--strictPort",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self._processes[app] = process

    def _wait_until_healthy(
        self,
        app_urls: Mapping[str, str],
        admin_token: str,
    ) -> None:
        pending = set(app_urls)
        deadline = time.monotonic() + self.config.startup_timeout_s
        while pending and time.monotonic() < deadline:
            self._assert_processes_alive()
            for app in tuple(pending):
                if self._probe(app_urls[app], admin_token):
                    pending.remove(app)
            if pending:
                time.sleep(0.1)
        if pending:
            raise TimeoutError(
                "Timed out waiting for CUA-Gym-Hub apps: " + ", ".join(sorted(pending))
            )

    def _assert_processes_alive(self) -> None:
        expected = set(self.manifest.apps)
        if set(self._processes) != expected:
            raise RuntimeError("CUA-Gym-Hub child process set is incomplete")
        exited = {
            app: returncode
            for app, process in self._processes.items()
            if (returncode := process.poll()) is not None
        }
        if exited:
            statuses = ", ".join(
                f"{app}={returncode}" for app, returncode in sorted(exited.items())
            )
            raise RuntimeError(f"CUA-Gym-Hub child exited ({statuses})")

    @staticmethod
    def _probe(app_url: str, admin_token: str) -> bool:
        request = Request(
            f"{app_url}/state?sid=_health",
            headers={"X-CUA-Admin-Token": admin_token},
        )
        try:
            with urlopen(request, timeout=0.5) as response:
                return response.status == 200
        except (HTTPError, URLError, TimeoutError):
            return False

    def _app_urls(self, descriptor: CuaGymHubDescriptor) -> dict[str, str]:
        return {
            app: descriptor.backend_urls[endpoint]
            for endpoint, app in self.manifest.endpoint_apps.items()
        }


def _validate_loopback_backend_url(backend_url: str) -> None:
    try:
        backend = urlsplit(backend_url)
        address = ipaddress.ip_address(backend.hostname or "")
        port = backend.port
    except ValueError:
        raise ValueError("Hub descriptor contains an invalid backend URL") from None
    if (
        backend.scheme != "http"
        or address != ipaddress.ip_address(_BIND_HOST)
        or port is None
        or backend.username is not None
        or backend.password is not None
        or backend.path not in {"", "/"}
        or backend.query
        or backend.fragment
        or backend.netloc != f"{_BIND_HOST}:{port}"
        or backend.geturl() != backend_url
    ):
        raise ValueError(
            "Hub descriptor backend URLs must be canonical loopback origins"
        )


def _publish_private_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = -1
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_private_json(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if path.is_symlink():
            raise ValueError("Refusing to read a symlinked Hub descriptor") from None
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Hub descriptor must be a regular file")
        if metadata.st_uid != os.getuid():
            raise PermissionError("Hub descriptor must be owned by the current UID")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PermissionError(
                "Hub descriptor must not grant group or other permissions"
            )
        if metadata.st_size > _MAX_DESCRIPTOR_BYTES:
            raise ValueError("Hub descriptor is too large")
        with os.fdopen(descriptor, "rb") as input_file:
            descriptor = -1
            raw_payload = input_file.read(_MAX_DESCRIPTOR_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw_payload) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("Hub descriptor is too large")
    try:
        payload = json.loads(
            raw_payload.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Hub descriptor is not valid UTF-8 JSON") from None
    if not isinstance(payload, dict):
        raise TypeError("Hub descriptor root must be an object")
    return payload


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Hub descriptor contains a duplicate object key")
        payload[key] = value
    return payload


def _json_string(payload: Mapping[str, Any], key: str) -> str:
    return _json_value_string(payload[key], key)


def _json_value_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hub descriptor {name} must be a non-empty string")
    return value


def _json_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"Hub descriptor {key} must be an integer")
    return value


def _json_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Hub descriptor {key} must be a number")
    return float(value)
