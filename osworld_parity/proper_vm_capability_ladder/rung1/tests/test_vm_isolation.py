from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.vm import (
    FIXTURE_READINESS_MAX_ATTEMPTS,
    FixtureReadinessError,
    KvmFixtureSession,
    VmHarnessError,
    task_unique_vm_id,
)
from osworld_parity.proper_vm_capability_ladder.rung1.server import (
    FixtureServerError,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    InputAudit,
    Operation,
)


class _ReadinessTransport:
    def __init__(self, *, launch_delay_s: float = 0.0) -> None:
        self.audit = InputAudit()
        self.commands: list[list[str]] = []
        self.timeout_s = 30.0
        self.launch_delay_s = launch_delay_s
        self.observed_timeouts: list[float] = []

    def execute_argv(self, argv: list[str]) -> dict:
        self.commands.append(argv)
        self.observed_timeouts.append(self.timeout_s)
        if self.launch_delay_s > 0:
            time.sleep(min(self.launch_delay_s, self.timeout_s))
            if self.launch_delay_s > self.timeout_s:
                raise TimeoutError("simulated launch RPC timeout")
        return {"status": "success", "returncode": 0, "error": None}


class _ReadinessStore:
    def __init__(self, outcomes: list[object], *, sleep_for_budget: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.sleep_for_budget = sleep_for_budget
        self.generation = 0
        self.wait_timeouts: list[float] = []
        self.last_error: str | None = None

    def reset(self, _fixture) -> int:
        self.generation += 1
        self.last_error = None
        return self.generation

    def wait_ready(self, _fixture_id: str, *, timeout_s: float) -> dict:
        self.wait_timeouts.append(timeout_s)
        if self.sleep_for_budget:
            time.sleep(timeout_s + 0.002)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            self.last_error = str(outcome)
            raise outcome
        return dict(outcome)

    def snapshot(self, _fixture_id: str) -> dict:
        return {
            "generation": self.generation,
            "ready": self.last_error is None,
            "geometry_stabilization_error": self.last_error,
            "geometry_observations": [],
            "events": [],
            "last_client_sequence": 0,
            "browser_audit_events": [],
            "browser_audit_dropped": 0,
            "diagnostic_journal_dropped": 0,
            "diagnostic_journal": [
                {
                    "journal_sequence": self.generation,
                    "stage": "test_readiness",
                    "details": {"error": self.last_error},
                }
            ],
        }


class _ReadinessServer:
    def __init__(self, store: _ReadinessStore) -> None:
        self.store = store

    @staticmethod
    def guest_url(fixture) -> str:
        return f"http://10.0.2.2:12345/fixture/{fixture.id}"


def _readiness_session(
    tmp_path: Path, *, launch_delay_s: float = 0.0
) -> tuple[KvmFixtureSession, _ReadinessTransport]:
    session = KvmFixtureSession(vm_log_dir=tmp_path / "vm_logs")
    transport = _ReadinessTransport(launch_delay_s=launch_delay_s)
    session.transport = transport
    session.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    session.metadata_path.write_text(
        json.dumps(
            {
                "fixture_readiness": {"schema_version": 1, "launches": []},
                "closed": False,
            }
        ),
        encoding="utf-8",
    )
    return session, transport


def test_vm_ids_are_unique_and_auditable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("LABCTL_RUN_ID", "run_abc")
    first = task_unique_vm_id()
    second = task_unique_vm_id()
    assert first != second
    assert first.startswith("4242-3-run_abc-")


def test_one_live_vm_per_task_and_scratch_under_slurm_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    first = KvmFixtureSession(vm_id="first", scratch_root=tmp_path)
    second = KvmFixtureSession(vm_id="second", scratch_root=tmp_path)
    first._prepare_isolation()
    assert first.vm_scratch_dir is not None
    assert first.vm_scratch_dir.is_relative_to(tmp_path)
    with pytest.raises(VmHarnessError, match="another VM"):
        second._prepare_isolation()
    second.close()
    first.close()
    assert not (tmp_path / "proper-vm-first").exists()


def test_job_unique_tmp_fallback_when_slurm_tmpdir_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "vm-fallback-test")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    session = KvmFixtureSession(vm_id="fallback")
    session._prepare_isolation()
    scratch = session.vm_scratch_dir
    assert scratch is not None
    assert session.scratch_source == "job_unique_tmp_fallback"
    assert scratch.is_relative_to(Path("/tmp/proper-vm-job-0-vm-fallback-test-7")) or (
        "proper-vm-job-" in str(scratch)
    )
    session.close()
    assert not scratch.exists()


def test_vm_metadata_rejects_port_collisions_and_long_qmp(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "result" / "vm_logs"
    log_dir.mkdir(parents=True)
    log = log_dir / "qemu.log"
    log.write_text("", encoding="utf-8")
    session = KvmFixtureSession(vm_log_dir=log_dir, vm_id="metadata")
    session.vm_scratch_dir = tmp_path / "scratch"
    base = {
        "ports": {"server": 10001, "chromium": 10002, "vnc": 10003, "vlc": 10004},
        "qmp_path": "/tmp/oswqmp_short.sock",
        "log": log,
    }
    metadata = session._metadata(state=base, overlay_fds=[str(session.vm_scratch_dir / "overlay (deleted)")])
    assert len(set(metadata["ports"].values())) == 4
    assert metadata["fixture_readiness"] == {"schema_version": 1, "launches": []}
    collided = dict(base)
    collided["ports"] = {**base["ports"], "vlc": 10001}
    with pytest.raises(VmHarnessError, match="colliding"):
        session._metadata(state=collided, overlay_fds=[])
    long_qmp = dict(base)
    long_qmp["qmp_path"] = "/tmp/" + "x" * 90
    with pytest.raises(VmHarnessError, match="QMP"):
        session._metadata(state=long_qmp, overlay_fds=[])


def test_fixture_readiness_recovers_transient_fetch_before_dispatch(
    tmp_path: Path,
) -> None:
    fixture = SimpleNamespace(id="r1a-click-dev-1101", fixture_sha256="fixture-sha")
    store = _ReadinessStore(
        [
            FixtureServerError(
                "r1a-click-dev-1101: TypeError: Failed to fetch"
            ),
            {"ready": True, "generation": 2},
        ]
    )
    session, transport = _readiness_session(tmp_path)

    ready = session.launch_fixture(
        _ReadinessServer(store), fixture, timeout_s=5.0
    )

    evidence = ready["session_readiness"]
    assert ready["generation"] == 2
    assert evidence["status"] == "ready"
    assert evidence["attempt_count"] == 2
    assert evidence["successful_attempt_index"] == 2
    assert evidence["attempts"][0]["error_type"] == "FixtureServerError"
    assert evidence["attempts"][0]["recoverable"] is True
    assert evidence["attempts"][0]["browser_restart"] is False
    assert evidence["attempts"][1]["status"] == "ready"
    assert evidence["attempts"][1]["browser_restart"] is True
    assert evidence["attempts"][1]["deadline_remaining_ns_at_start"] < evidence[
        "attempts"
    ][0]["deadline_remaining_ns_at_start"]
    assert evidence["input_audit_unchanged"] is True
    assert evidence["model_access"] is False
    assert evidence["trial_retry_count"] == 0
    assert evidence["trial_replacement_count"] == 0
    assert len(transport.commands) == 2
    assert "pkill" not in transport.commands[0][2]
    assert "pkill" in transport.commands[1][2]
    assert all("pyautogui" not in command[2] for command in transport.commands)
    assert transport.audit.operations == []
    metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
    assert metadata["fixture_readiness"]["launches"] == [evidence]


def test_fixture_readiness_does_not_retry_deterministic_geometry_error(
    tmp_path: Path,
) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    store = _ReadinessStore(
        [FixtureServerError("fixture: ready geometry missing")]
    )
    session, transport = _readiness_session(tmp_path)

    with pytest.raises(FixtureReadinessError, match="fixed 1.000s") as caught:
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=1.0
        )

    evidence = caught.value.evidence
    assert evidence["status"] == "failed"
    assert evidence["attempt_count"] == 1
    assert evidence["attempts"][0]["recoverable"] is False
    assert evidence["attempts"][0]["host_state"][
        "geometry_stabilization_error"
    ] == "fixture: ready geometry missing"
    assert evidence["input_audit_unchanged"] is True
    assert len(transport.commands) == 1


def test_fixture_readiness_attempts_are_bounded_by_one_setup_deadline(
    tmp_path: Path,
) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    store = _ReadinessStore(
        [TimeoutError("not ready")] * FIXTURE_READINESS_MAX_ATTEMPTS,
        sleep_for_budget=True,
    )
    session, transport = _readiness_session(tmp_path)

    with pytest.raises(FixtureReadinessError) as caught:
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=0.01
        )

    evidence = caught.value.evidence
    assert evidence["attempt_count"] == 1
    assert len(store.wait_timeouts) == 1
    assert 0 < store.wait_timeouts[0] <= 0.01
    assert evidence["setup_completed_host_monotonic_ns"] >= evidence[
        "setup_deadline_host_monotonic_ns"
    ]
    assert evidence["attempts"][0]["recoverable"] is True
    assert len(transport.commands) == 1


def test_fixture_readiness_launch_rpc_is_clamped_to_remaining_deadline(
    tmp_path: Path,
) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    store = _ReadinessStore([{"ready": True, "generation": 1}])
    session, transport = _readiness_session(tmp_path, launch_delay_s=0.06)

    started = time.monotonic()
    with pytest.raises(FixtureReadinessError) as caught:
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=0.01
        )
    elapsed = time.monotonic() - started

    evidence = caught.value.evidence
    assert elapsed < 0.05
    assert evidence["attempt_count"] == 1
    assert evidence["attempts"][0]["error_type"] == "TimeoutError"
    assert evidence["attempts"][0]["recoverable"] is True
    assert store.wait_timeouts == []
    assert 0 < transport.observed_timeouts[0] <= 0.01
    assert transport.timeout_s == 30.0


def test_fixture_readiness_timeout_allowlist_walks_wrapped_causes() -> None:
    try:
        try:
            raise TimeoutError("socket timed out")
        except TimeoutError as exc:
            raise RuntimeError("transport wrapper") from exc
    except RuntimeError as wrapped:
        assert KvmFixtureSession._recoverable_fixture_readiness_error(
            wrapped, fixture_id="fixture"
        )
    assert not KvmFixtureSession._recoverable_fixture_readiness_error(
        FixtureServerError("fixture: NetworkError without a Chromium fetch failure"),
        fixture_id="fixture",
    )
    assert not KvmFixtureSession._recoverable_fixture_readiness_error(
        FixtureServerError(
            "fixture: deterministic validation: TypeError: Failed to fetch"
        ),
        fixture_id="fixture",
    )


def test_fixture_readiness_has_a_fixed_attempt_ceiling(tmp_path: Path) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    store = _ReadinessStore(
        [TimeoutError("not ready")] * FIXTURE_READINESS_MAX_ATTEMPTS
    )
    session, transport = _readiness_session(tmp_path)

    with pytest.raises(FixtureReadinessError) as caught:
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=10.0
        )

    evidence = caught.value.evidence
    assert evidence["attempt_count"] == FIXTURE_READINESS_MAX_ATTEMPTS
    assert len(evidence["attempts"]) == FIXTURE_READINESS_MAX_ATTEMPTS
    assert all(attempt["recoverable"] is True for attempt in evidence["attempts"])
    assert len(transport.commands) == FIXTURE_READINESS_MAX_ATTEMPTS
    assert all("pkill" in command[2] for command in transport.commands[1:])


def test_fixture_readiness_rejects_post_action_use(tmp_path: Path) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    store = _ReadinessStore([{"ready": True, "generation": 1}])
    session, transport = _readiness_session(tmp_path)
    transport.audit.operations.append(Operation("move_to", (1, 2)))

    with pytest.raises(VmHarnessError, match="before any action dispatch"):
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=5.0
        )

    assert transport.commands == []
    assert store.generation == 0


def test_fixture_readiness_persisted_errors_have_a_fixed_size_bound(
    tmp_path: Path,
) -> None:
    fixture = SimpleNamespace(id="fixture", fixture_sha256="fixture-sha")
    oversized_error = "deterministic:" + "x" * 200_000
    store = _ReadinessStore([FixtureServerError(oversized_error)])
    session, _transport = _readiness_session(tmp_path)

    with pytest.raises(FixtureReadinessError) as caught:
        session.launch_fixture(
            _ReadinessServer(store), fixture, timeout_s=1.0
        )

    evidence = caught.value.evidence
    attempt = evidence["attempts"][0]
    assert "original_bytes=200014" in attempt["error"]
    assert "sha256=" in attempt["error"]
    assert len(attempt["error"].encode("utf-8")) <= 512
    geometry_error = attempt["host_state"]["geometry_stabilization_error"]
    assert "original_bytes=200014" in geometry_error
    diagnostic_entry = attempt["host_state"]["diagnostic_tail"][0]
    assert diagnostic_entry["truncated"] is True
    assert diagnostic_entry["original_json_bytes"] > 200_000
    assert len(diagnostic_entry["json_prefix"].encode("utf-8")) <= 512
    compact = json.dumps(
        evidence, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert evidence["serialized_size_bytes"] == len(compact)
    assert len(compact) <= evidence["serialized_size_limit_bytes"] == 64 * 1024
    persisted = session.metadata_path.read_bytes()
    assert len(persisted) < 70 * 1024
