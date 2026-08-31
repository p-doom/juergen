"""Consumer contracts for the one CUA-Gym desktop runtime adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import verifiers.v1 as vf
from desktop.vm import GuestCommandResult
from juergen_doubles import FakeSession, make_ctx
from test_cua_gym_dataset import (
    DESKTOP_ID,
    DESKTOP_SETUP_FILE,
    DESKTOP_TARGET_PATH,
    _synthetic_snapshot,
)

from evals.cua_gym import PINNED_REVISION, TaskPlatform, runtime
from evals.cua_gym.models import EndpointName
from evals.cua_gym.web import runtime as web_runtime
from evals.cua_gym.web.gateway import GatewayPhase
from evals.cua_gym.web.manifest import CuaGymWebRuntimeManifest
from evals.harness import (
    ArtifactConfig,
    CuaGymWebConfig,
    DesktopHarness,
    DesktopHarnessConfig,
    DesktopPoolConfig,
    SettleConfig,
)
from evals.tasks import RESULT_KEY, DesktopState


class _CuaSession(FakeSession):
    def __init__(
        self,
        *,
        packages_ready: bool = True,
        reward_returncode: int = 0,
        reward_stdout: str = "REWARD: 0.25\n",
        reward_stderr: str = "",
        frames: list[bytes] | None = None,
    ) -> None:
        super().__init__(frames=frames)
        self.packages_ready = packages_ready
        self.reward_returncode = reward_returncode
        self.reward_stdout = reward_stdout
        self.reward_stderr = reward_stderr
        self.setup_calls: list[list[dict[str, Any]]] = []
        self.staged_contents: list[bytes] = []
        self.written_files: dict[str, bytes] = {}
        self.guest_commands: list[list[str]] = []
        self.ports = SimpleNamespace(chromium=9222, vlc=8080)

    def record_setup_steps(self, steps: list[dict[str, Any]]) -> int:
        self.setup_calls.append(steps)
        upload = steps[0]["parameters"]["files"][0]
        self.staged_contents.append(Path(upload["local_path"]).read_bytes())
        return len(steps)

    def write_guest_file(self, path: str, content: bytes) -> None:
        self.written_files[path] = content

    def run_guest(
        self,
        argv: list[str],
        *,
        timeout_s: float | None = None,
    ) -> GuestCommandResult:
        del timeout_s
        command = list(argv)
        self.guest_commands.append(command)
        if command[:2] == ["python3", "-c"]:
            return GuestCommandResult(
                0 if self.packages_ready else 1, "", "ModuleNotFoundError: docx"
            )
        if command == ["python3", "/tmp/cua_gym_reward.py"]:
            return GuestCommandResult(
                self.reward_returncode, self.reward_stdout, self.reward_stderr
            )
        raise AssertionError(f"unexpected guest command: {command}")


@pytest.fixture(autouse=True)
def _replace_osworld_setup_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    def record(session: _CuaSession, steps: list[dict[str, Any]]) -> int:
        return session.record_setup_steps(steps)

    monkeypatch.setattr(runtime, "_run_setup_steps", record)
    monkeypatch.setattr(web_runtime, "_run_setup_steps", record)


def _blocklist(path: Path, *, blocked: dict[str, list[str]] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "revision": PINNED_REVISION,
                "blocklist_version": 1,
                "blocked_tasks": {
                    task_id: {"reasons": reasons}
                    for task_id, reasons in (blocked or {}).items()
                },
                "measured": {"reset_probe_task_ids": []},
            }
        )
    )
    return path


def _task(root: Path) -> runtime.CuaGymDesktopTaskData:
    return runtime.CuaGymDesktopTaskData(
        idx=0,
        name=DESKTOP_ID,
        prompt="Add a two-column table.",
        instruction="Add a two-column table.",
        kind=runtime.CUA_GYM_DESKTOP_KIND,
        max_steps=1,
        task_id=DESKTOP_ID,
        dataset_root=str(root),
        dataset_revision=PINNED_REVISION,
    )


def test_taskset_preserves_requested_order_and_rejects_blocklisted_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    blocklist = _blocklist(tmp_path / "blocklist.json")
    taskset = runtime.CuaGymDesktopTaskset(
        runtime.CuaGymDesktopTasksetConfig(
            dataset_root=str(root),
            blocklist_path=str(blocklist),
            task_ids=[DESKTOP_ID],
        )
    )
    (task,) = tuple(taskset.load())
    assert task.data.task_id == DESKTOP_ID
    assert task.data.kind == runtime.CUA_GYM_DESKTOP_KIND
    assert task.data.instruction == "Add a two-column table."

    blocked = _blocklist(
        tmp_path / "blocked.json", blocked={DESKTOP_ID: ["pays_before_agent_acts"]}
    )
    refused = runtime.CuaGymDesktopTaskset(
        runtime.CuaGymDesktopTasksetConfig(
            dataset_root=str(root), blocklist_path=str(blocked), task_ids=[DESKTOP_ID]
        )
    )
    with pytest.raises(ValueError, match="blocklisted"):
        tuple(refused.load())


def test_preparer_stages_bundle_files_offline_on_the_checked_out_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    session = _CuaSession()

    evidence = runtime.CuaGymDesktopPreparer().prepare(session, _task(root))

    assert evidence["setup_steps"] == 3
    assert session.staged_contents == [b"PK\x03\x04 binary docx"]
    (steps,) = session.setup_calls
    assert steps[0]["type"] == "upload_file"
    (upload,) = steps[0]["parameters"]["files"]
    assert upload["path"] == DESKTOP_TARGET_PATH
    assert Path(upload["local_path"]).name == DESKTOP_SETUP_FILE
    assert [step["type"] for step in steps[1:]] == ["open", "sleep"]
    assert "./initial_setup.docx" not in str(steps)


def test_missing_guest_packages_refuse_the_episode_without_runtime_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    session = _CuaSession(packages_ready=False)

    with pytest.raises(RuntimeError, match="guest image is missing required modules"):
        runtime.CuaGymDesktopPreparer().prepare(session, _task(root))

    assert session.setup_calls == []
    assert not any(command[1:3] == ["-m", "pip"] for command in session.guest_commands)


def test_trusted_reward_uses_the_exact_prepared_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    task = _task(root)
    session = _CuaSession()
    preparer = runtime.CuaGymDesktopPreparer()

    preparer.prepare(session, task)
    assert preparer.evaluate(session, task, declared=None) == 0.25
    assert session.written_files["/tmp/cua_gym_reward.py"] == b"print('REWARD: 0.25')\n"


def test_an_external_setup_download_is_refused_before_setup_controller_runs() -> None:
    with pytest.raises(ValueError, match="restricted to bundle files"):
        runtime._offline_upload(
            {"url": "https://example.invalid/file", "path": "/home/user/file"}, {}
        )


def _harness_config(tmp_path: Path) -> DesktopHarnessConfig:
    return DesktopHarnessConfig(
        id="cua_gym_runtime_test",
        settle=SettleConfig(min_delay_s=0.0, per_kind={}),
        artifacts=ArtifactConfig(
            output_dir=str(tmp_path),
            save_frames=True,
            save_prompts=True,
            write_gif=False,
        ),
        pool=DesktopPoolConfig(
            key=f"cua-gym-{tmp_path.name}",
            max_node_slots=1,
            slot_dir=str(tmp_path / "slots"),
            pool_target="juergen_harness_pool:Pool",
            hide_gpu_during_boot=False,
        ),
        evaluate_on_finish=True,
        require_unsolved_start=True,
    )


def _launch_harness(
    tmp_path: Path, task: runtime.CuaGymDesktopTaskData, session: _CuaSession
) -> vf.Trace:
    import juergen_harness_pool

    import agent.desktop as dsk

    juergen_harness_pool.Pool.session = session
    trace = vf.Trace(
        task=vf.TraceTask(type="CuaGymDesktopTask", data=task),
        state=DesktopState(),
    )
    try:
        asyncio.run(
            DesktopHarness(_harness_config(tmp_path)).launch(
                make_ctx(replies=["NO_OP"]),
                trace,
                SimpleNamespace(is_local=True),
                "",
                "",
                {},
            )
        )
    finally:
        dsk.close_all_pools()
        juergen_harness_pool.Pool.session = None
    return trace


def test_desktop_harness_uses_the_cua_preparer_for_final_trusted_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    session = _CuaSession()
    trace = _launch_harness(tmp_path, _task(root), session)
    result = trace.info[RESULT_KEY]
    assert result["validity"] == "valid"
    assert result["task_reward"] == 0.25
    assert result["setup"]["prepared"] == runtime.CUA_GYM_DESKTOP_KIND


def test_desktop_harness_publishes_missing_image_packages_as_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    session = _CuaSession(packages_ready=False)

    result = _launch_harness(tmp_path, _task(root), session).info[RESULT_KEY]

    assert result["validity"] == "infra_invalid"
    assert result["infra_error"]["stage"] == "episode"
    assert "guest image is missing required modules" in result["infra_error"]["message"]
    assert session.setup_calls == []


def test_desktop_harness_never_turns_a_grader_failure_into_zero_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.DESKTOP)
    monkeypatch.setattr(runtime, "_snapshot", lambda *_: snapshot)
    session = _CuaSession(reward_returncode=7, reward_stderr="grader crashed")

    result = _launch_harness(tmp_path, _task(root), session).info[RESULT_KEY]

    assert result["validity"] == "infra_invalid"
    assert result["infra_error"]["stage"] == "evaluate"
    assert result["task_reward"] is None


def test_verifiers_loaders_resolve_the_flat_cua_gym_front_door() -> None:
    from verifiers.v1.loaders import default_harness_id, harness_class, taskset_class

    assert taskset_class("cua_gym") is runtime.CuaGymDesktopTaskset
    assert taskset_class("cua_gym_web") is web_runtime.CuaGymWebTaskset
    assert harness_class("cua_gym") is DesktopHarness
    assert harness_class("cua_gym_web") is DesktopHarness
    assert default_harness_id("cua_gym") == "cua_gym"
    assert default_harness_id("cua_gym_web") == "cua_gym_web"


def test_runtime_accepts_only_the_streams_trained_action_grammar() -> None:
    from agent.agent import load_codec

    assert load_codec("ordered_events_v3").name == "ordered_events_v3"
    with pytest.raises(LookupError, match="required: 'ordered_events_v3'"):
        load_codec("deltatype_v2")


def test_web_runtime_uses_the_harness_renderer_and_closes_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path, platform=TaskPlatform.WEB)
    manifest = CuaGymWebRuntimeManifest(
        manifest_version=1,
        dataset_revision=PINNED_REVISION,
        hub_revision="1" * 40,
        endpoint_apps={EndpointName("gmail"): "gmail_mock"},
        writable_directories=(".mock-state",),
        unsupported_endpoints={},
        unsupported_tasks={},
        task_scratch_paths=("/tmp/task_reward",),
        supported_task_count=1,
    )
    events: list[object] = []
    private_sessions = ((EndpointName("gmail"), "b" * 64),)
    capability_host = f"{'a' * 52}.gmail.cua.internal"

    class Descriptor:
        def cleanup_private_sessions(self, sessions, *, manifest) -> None:
            events.append(("private_cleanup", sessions, manifest.hub_revision))

    descriptor = Descriptor()

    class Hub:
        def __init__(self, *, config, manifest) -> None:
            events.append(("hub_init", config.image_path.name, manifest.hub_revision))

        def start(self):
            events.append("hub_start")
            return descriptor

        def assert_healthy(self) -> None:
            events.append("hub_healthy")

        def close(self) -> None:
            events.append("hub_close")

    class Lease:
        ports = (31_337,)

        def release(self) -> None:
            events.append("port_release")

    class Gateway:
        def __init__(self, *, config, manifest) -> None:
            self.config = config
            self.gateway_hostnames = {EndpointName("gmail"): capability_host}
            self.gateway_urls = {
                EndpointName("gmail"): f"http://{capability_host}:31337"
            }
            self.private_sessions = private_sessions
            events.append(("gateway_init", config.endpoints, manifest.hub_revision))

        def start(self) -> None:
            events.append("gateway_start")

        def assert_healthy(self) -> None:
            events.append("gateway_healthy")

        def wait_for_browser_session(self, timeout_s: float) -> None:
            events.append(("browser_session", timeout_s))

        def transition(self, phase: GatewayPhase) -> None:
            events.append(("gateway_phase", phase))

        def close(self) -> None:
            events.append("gateway_close")

    class Browser:
        def __init__(self, *, guest, config) -> None:
            assert guest is session
            events.append(("browser_init", config.guest_hostnames))

        def configure_guest_hosts(self) -> None:
            events.append("guest_hosts")

        def ensure_browser(self) -> str:
            events.append("browser_ready")
            return "ws://localhost:9222/devtools/browser/cua-gym"

        def verify_after_setup(self, expected_identity: str) -> None:
            events.append(("browser_verified", expected_identity))

        def cleanup_browser(self, *, origins, expected_identity) -> None:
            events.append(("browser_cleanup", origins, expected_identity))

    monkeypatch.setattr(web_runtime, "_web_snapshot", lambda *_: snapshot)
    monkeypatch.setattr(
        web_runtime, "load_default_web_runtime_manifest", lambda: manifest
    )
    monkeypatch.setattr(web_runtime, "CuaGymHubSupervisor", Hub)
    monkeypatch.setattr(web_runtime, "CuaGymEpisodeGateway", Gateway)
    monkeypatch.setattr(web_runtime, "CuaGymDesktopBrowser", Browser)
    monkeypatch.setattr(web_runtime, "acquire_port_range", lambda **_: Lease())

    session = _CuaSession(reward_stdout="REWARD: 0.5\n")
    task = web_runtime.CuaGymWebTaskData(
        idx=0,
        name="web-consumer",
        prompt="Send a test email.",
        instruction="Send a test email.",
        kind=web_runtime.CUA_GYM_WEB_KIND,
        max_steps=1,
        task_id="00000000-0000-0000-0000-000000000001",
        dataset_root=str(root),
        dataset_revision=PINNED_REVISION,
    )
    image = tmp_path / "hub.sif"
    image.write_bytes(b"test image")
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    apptainer.chmod(0o700)
    config = _harness_config(tmp_path).model_copy(
        update={
            "web": CuaGymWebConfig(
                hub_image=str(image),
                apptainer_binary=str(apptainer),
                port_lock_dir=str(tmp_path / "port-locks"),
                guest_password="password",
            )
        }
    )

    import juergen_harness_pool

    import agent.desktop as dsk

    juergen_harness_pool.Pool.session = session
    trace = vf.Trace(
        task=vf.TraceTask(type="CuaGymWebTask", data=task), state=DesktopState()
    )
    try:
        asyncio.run(
            DesktopHarness(config).launch(
                make_ctx(replies=["NO_OP"]),
                trace,
                SimpleNamespace(is_local=True),
                "",
                "",
                {},
            )
        )
    finally:
        dsk.close_all_pools()
        juergen_harness_pool.Pool.session = None

    result = trace.info[RESULT_KEY]
    assert result["validity"] == "valid"
    assert result["task_reward"] == 0.5
    assert result["setup"]["prepared"] == web_runtime.CUA_GYM_WEB_KIND
    assert result["render"]["max_completed_turns"] == 4
    assert result["images"]["image_domain"] == (
        "osworld_cursor_jpeg_q92_420_1920x1080_v1"
    )
    assert capability_host.encode() in session.staged_contents[0]
    assert capability_host.encode() in session.written_files["/tmp/cua_gym_reward.py"]
    assert ("gateway_phase", GatewayPhase.ROLLOUT) in events
    assert ("gateway_phase", GatewayPhase.EVALUATE) in events
    assert any(
        isinstance(event, tuple) and event[0] == "browser_cleanup" for event in events
    )
    assert ("private_cleanup", private_sessions, manifest.hub_revision) in events
    assert events[-4:] == [
        "gateway_close",
        ("private_cleanup", private_sessions, manifest.hub_revision),
        "hub_close",
        "port_release",
    ]
