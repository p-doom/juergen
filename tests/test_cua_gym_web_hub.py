from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import stat
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Self
from urllib.request import Request

import pytest
from desktop.vm import acquire_port_range

from evals.cua_gym.models import EndpointName
from evals.cua_gym.web import hub as hub_module
from evals.cua_gym.web.hub import (
    CUA_GYM_HUB_DESCRIPTOR_VERSION,
    CuaGymHubConfig,
    CuaGymHubDescriptor,
    CuaGymHubSupervisor,
)
from evals.cua_gym.web.image import (
    HUB_BASE_IMAGE,
    HUB_SOURCE_PATCH_SHA256,
    CuaGymHubImageManifest,
    hub_image_manifest_path,
)
from evals.cua_gym.web.manifest import CuaGymWebRuntimeManifest

_HUB_REVISION = "e" * 40


def _manifest(*, hub_revision: str = _HUB_REVISION) -> CuaGymWebRuntimeManifest:
    return CuaGymWebRuntimeManifest(
        manifest_version=1,
        dataset_revision="d" * 40,
        hub_revision=hub_revision,
        endpoint_apps={EndpointName("gmail"): "gmail_mock"},
        writable_directories=(
            ".mock-files",
            ".mock-secure-states",
            ".mock-state",
            ".mock-states",
            ".mock-uploads",
        ),
        unsupported_endpoints={},
        unsupported_tasks={},
        task_scratch_paths=("/tmp/task_sid",),
        supported_task_count=1,
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _fake_apptainer(tmp_path: Path, *, revision: str = _HUB_REVISION) -> Path:
    executable = tmp_path / "apptainer"
    executable.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            if sys.argv[1] == "inspect":
                print(json.dumps({{"data": {{"attributes": {{"labels": {{
                    "org.opencontainers.image.revision": {revision!r},
                    "org.opencontainers.image.base.digest": {HUB_BASE_IMAGE.removeprefix("node@")!r},
                    "io.juergen.cua-gym.source-patch.sha256": {HUB_SOURCE_PATCH_SHA256!r}
                }}}}}}}}))
                raise SystemExit(0)

            assert "--contain" in sys.argv
            assert "--writable-tmpfs" in sys.argv
            assert sys.argv[sys.argv.index("--cwd") + 1] == "/opt/cua-gym-hub"
            assert "REVIEW_SECRET" not in os.environ
            assert "APPTAINER_BIND" not in os.environ
            assert "APPTAINERENV_CUA_GYM_LEGACY_COMPAT" not in os.environ
            port = int(sys.argv[sys.argv.index("--port") + 1])
            token = os.environ["APPTAINERENV_CUA_GYM_ADMIN_TOKEN"]

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    healthy = (
                        self.path == "/state?sid=_health"
                        and self.headers.get("X-CUA-Admin-Token") == token
                    )
                    self.send_response(200 if healthy else 403)
                    self.end_headers()

                def log_message(self, *_args):
                    pass

            ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _config(tmp_path: Path, port: int, apptainer: Path) -> CuaGymHubConfig:
    image = tmp_path / "hub.sif"
    image.write_bytes(b"test CUA-Gym Hub image")
    CuaGymHubImageManifest.for_image(
        image,
        web_manifest=_manifest(),
    ).write(hub_image_manifest_path(image))
    return CuaGymHubConfig(
        image_path=image,
        state_root=tmp_path / "state",
        log_root=tmp_path / "logs",
        descriptor_path=tmp_path / "hub.json",
        port_range_start=port,
        port_range_end=port,
        port_lock_dir=tmp_path / "locks",
        startup_timeout_s=5.0,
        request_timeout_s=1.0,
        apptainer_binary=apptainer,
    )


def test_hub_runs_real_children_publishes_descriptor_and_releases_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_SECRET", "must-not-reach-hub")
    monkeypatch.setenv("APPTAINER_BIND", "/:/host")
    port = _free_port()
    config = _config(tmp_path, port, _fake_apptainer(tmp_path))
    supervisor = CuaGymHubSupervisor(config=config, manifest=_manifest())

    descriptor = supervisor.start()
    processes = tuple(supervisor._processes.values())
    assert supervisor.assert_healthy() is None
    assert config.descriptor_path.stat().st_uid == os.getuid()
    assert stat.S_IMODE(config.descriptor_path.stat().st_mode) == 0o600
    assert config.state_root.stat().st_uid == os.getuid()
    assert stat.S_IMODE(config.state_root.stat().st_mode) == 0o700
    assert config.log_root.stat().st_uid == os.getuid()
    assert stat.S_IMODE(config.log_root.stat().st_mode) == 0o700
    assert config.port_lock_dir.stat().st_uid == os.getuid()
    assert stat.S_IMODE(config.port_lock_dir.stat().st_mode) == 0o700
    assert (config.log_root / "gmail_mock.log").stat().st_uid == os.getuid()
    assert stat.S_IMODE((config.log_root / "gmail_mock.log").stat().st_mode) == 0o600
    assert (
        CuaGymHubDescriptor.read(config.descriptor_path, manifest=_manifest())
        == descriptor
    )
    assert descriptor.backend_urls == {
        EndpointName("gmail"): f"http://127.0.0.1:{port}"
    }
    state_directory = config.state_root / "gmail_mock" / ".mock-states"
    assert stat.S_IMODE(state_directory.stat().st_mode) == 0o700
    with pytest.raises(RuntimeError, match="unavailable"):
        acquire_port_range(
            count=1,
            purpose="collision probe",
            range_start=port,
            range_end=port,
            lock_dir=config.port_lock_dir,
            exact_start=port,
            bind_host="127.0.0.1",
        )

    supervisor.close()

    assert not config.descriptor_path.exists()
    assert all(process.poll() is not None for process in processes)
    with (config.port_lock_dir / f"port-{port}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def test_hub_config_keeps_private_outputs_outside_the_guest_mount(
    tmp_path: Path,
) -> None:
    port = _free_port()
    config = _config(tmp_path, port, _fake_apptainer(tmp_path))

    with pytest.raises(ValueError, match="descriptor_path must not overlap"):
        replace(config, descriptor_path=config.state_root / "hub.json")
    with pytest.raises(ValueError, match="log_root must not overlap"):
        replace(config, log_root=config.state_root / "logs")
    with pytest.raises(ValueError, match="port_lock_dir must not overlap"):
        replace(config, port_lock_dir=config.state_root / "locks")
    with pytest.raises(ValueError, match="log_root must not overlap"):
        replace(config, state_root=config.log_root / "state")
    with pytest.raises(ValueError, match="port_lock_dir must not overlap"):
        replace(config, state_root=config.port_lock_dir / "state")
    with pytest.raises(ValueError, match="descriptor_path must not overlap"):
        replace(config, state_root=config.descriptor_path / "state")
    with pytest.raises(ValueError, match="must not contain"):
        replace(config, state_root=tmp_path / "state,extra")


def test_hub_health_fails_when_a_child_dies(tmp_path: Path) -> None:
    port = _free_port()
    supervisor = CuaGymHubSupervisor(
        config=_config(tmp_path, port, _fake_apptainer(tmp_path)),
        manifest=_manifest(),
    )
    supervisor.start()
    process = next(iter(supervisor._processes.values()))
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=5.0)

    with pytest.raises(RuntimeError, match="child exited"):
        supervisor.assert_healthy()
    supervisor.close()


def test_hub_rejects_image_revision_before_starting_children(tmp_path: Path) -> None:
    port = _free_port()
    config = _config(tmp_path, port, _fake_apptainer(tmp_path, revision="f" * 40))
    supervisor = CuaGymHubSupervisor(config=config, manifest=_manifest())

    with pytest.raises(RuntimeError, match="image revision differs"):
        supervisor.start()

    assert supervisor._processes == {}
    assert not config.state_root.exists()
    assert not config.log_root.exists()
    assert not config.descriptor_path.exists()


def test_hub_rejects_image_hash_tampering_before_inspection(tmp_path: Path) -> None:
    port = _free_port()
    apptainer = _fake_apptainer(tmp_path)
    config = _config(tmp_path, port, apptainer)
    config.image_path.write_bytes(b"tampered")
    supervisor = CuaGymHubSupervisor(config=config, manifest=_manifest())

    with pytest.raises(RuntimeError, match="provenance or SHA-256"):
        supervisor.start()

    assert supervisor._processes == {}
    assert not config.state_root.exists()
    assert not config.log_root.exists()


def test_hub_descriptor_rejects_malformed_and_unsafe_documents(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    descriptor = CuaGymHubDescriptor(
        descriptor_version=CUA_GYM_HUB_DESCRIPTOR_VERSION,
        dataset_revision=manifest.dataset_revision,
        hub_revision=manifest.hub_revision,
        backend_urls={EndpointName("gmail"): "http://127.0.0.1:45000"},
        state_root=tmp_path / "state",
        request_timeout_s=1.0,
        admin_token="a" * 64,
    )
    path = tmp_path / "hub.json"
    descriptor.publish(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.unlink()
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="unexpected schema"):
        CuaGymHubDescriptor.read(path, manifest=manifest)

    payload.pop("unexpected")
    payload["backend_urls"] = {"gmail": "http://10.0.0.1:45000"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="loopback origins"):
        CuaGymHubDescriptor.read(path, manifest=manifest)

    path.write_text('{"admin_token":"a","admin_token":"b"}', encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate object key"):
        CuaGymHubDescriptor.read(path, manifest=manifest)


def test_hub_descriptor_rejects_public_permissions_and_symlinks(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    descriptor = CuaGymHubDescriptor(
        dataset_revision=manifest.dataset_revision,
        hub_revision=manifest.hub_revision,
        backend_urls={EndpointName("gmail"): "http://127.0.0.1:45000"},
        state_root=tmp_path / "state",
        request_timeout_s=1.0,
        admin_token="a" * 64,
    )
    path = tmp_path / "hub.json"
    descriptor.publish(path)
    path.chmod(0o640)
    with pytest.raises(PermissionError, match="group or other"):
        CuaGymHubDescriptor.read(path, manifest=manifest)

    path.chmod(0o600)
    link = tmp_path / "hub-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlinked"):
        CuaGymHubDescriptor.read(link, manifest=manifest)


class _ResetResponse:
    status = 200

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        pass


def _cleanup_descriptor(tmp_path: Path) -> CuaGymHubDescriptor:
    manifest = _manifest()
    return CuaGymHubDescriptor(
        dataset_revision=manifest.dataset_revision,
        hub_revision=manifest.hub_revision,
        backend_urls={EndpointName("gmail"): "http://127.0.0.1:45000"},
        state_root=tmp_path / "state",
        request_timeout_s=1.0,
        admin_token="a" * 64,
    )


def _prepare_cleanup_state(
    descriptor: CuaGymHubDescriptor,
    manifest: CuaGymWebRuntimeManifest,
    private_sid: str,
) -> tuple[Path, ...]:
    app_root = descriptor.state_root / "gmail_mock"
    targets: list[Path] = []
    for directory in manifest.writable_directories:
        writable_root = app_root / directory
        writable_root.mkdir(parents=True)
        session_directory = writable_root / private_sid
        session_directory.mkdir()
        (session_directory / "payload").write_text("private", encoding="utf-8")
        targets.append(session_directory)
        for suffix in (".json", "_initial.json", ".initial.json"):
            target = writable_root / f"{private_sid}{suffix}"
            target.write_text("private", encoding="utf-8")
            targets.append(target)
        (writable_root / "unrelated.json").write_text("keep", encoding="utf-8")
    return tuple(targets)


def test_hub_cleanup_resets_and_removes_exact_state_across_manifest_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    descriptor = _cleanup_descriptor(tmp_path)
    private_sid = "1" * 64
    targets = _prepare_cleanup_state(descriptor, manifest, private_sid)
    descriptor_path = tmp_path / "worker-hub.json"
    descriptor.publish(descriptor_path)
    descriptor = CuaGymHubDescriptor.read(descriptor_path, manifest=manifest)
    requests: list[tuple[Request, float]] = []

    def reset(request: Request, *, timeout: float) -> _ResetResponse:
        requests.append((request, timeout))
        return _ResetResponse()

    monkeypatch.setattr(hub_module, "urlopen", reset)
    descriptor.cleanup_private_sessions(
        ((EndpointName("gmail"), private_sid),),
        manifest=manifest,
    )

    assert all(not target.exists() for target in targets)
    assert all(
        (descriptor.state_root / "gmail_mock" / directory / "unrelated.json").is_file()
        for directory in manifest.writable_directories
    )
    request, timeout = requests.pop()
    assert request.full_url == f"http://127.0.0.1:45000/post?sid={private_sid}"
    assert request.get_header("X-cua-admin-token") == "a" * 64
    assert json.loads(request.data or b"") == {"action": "reset"}
    assert timeout == 1.0


def test_hub_cleanup_validates_every_session_before_mutating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    descriptor = _cleanup_descriptor(tmp_path)
    private_sid = "1" * 64
    targets = _prepare_cleanup_state(descriptor, manifest, private_sid)
    reset_called = False

    def reset(_request: Request, *, timeout: float) -> _ResetResponse:
        nonlocal reset_called
        reset_called = True
        return _ResetResponse()

    monkeypatch.setattr(hub_module, "urlopen", reset)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        descriptor.cleanup_private_sessions(
            (
                (EndpointName("gmail"), private_sid),
                (EndpointName("gmail"), "../escape"),
            ),
            manifest=manifest,
        )

    assert not reset_called
    assert all(target.exists() for target in targets)


def test_hub_cleanup_rejects_unknown_endpoints_before_backend_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _cleanup_descriptor(tmp_path)
    monkeypatch.setattr(
        hub_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("backend must not be accessed"),
    )

    with pytest.raises(ValueError, match="Unknown CUA-Gym endpoint"):
        descriptor.cleanup_private_sessions(
            ((EndpointName("unknown"), "1" * 64),),
            manifest=_manifest(),
        )


def test_hub_cleanup_rejects_symlinked_writable_roots_without_touching_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    descriptor = _cleanup_descriptor(tmp_path)
    app_root = descriptor.state_root / "gmail_mock"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / f"{'1' * 64}.json"
    outside_target.write_text("keep", encoding="utf-8")
    app_root.mkdir(parents=True)
    for directory in manifest.writable_directories:
        writable_root = app_root / directory
        if directory == ".mock-files":
            writable_root.symlink_to(outside, target_is_directory=True)
        else:
            writable_root.mkdir()
    monkeypatch.setattr(
        hub_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("backend must not be accessed"),
    )

    with pytest.raises(RuntimeError, match="writable state root is not a directory"):
        descriptor.cleanup_private_sessions(
            ((EndpointName("gmail"), "1" * 64),),
            manifest=manifest,
        )

    assert outside_target.read_text(encoding="utf-8") == "keep"
