"""The owned local server keeps one listener and one reapable process group."""

from __future__ import annotations

import contextlib
import errno
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_OWNED_SERVER = r"""
import http.server
import signal
import subprocess
import sys

descendant = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
)
print(f"DESCENDANT {descendant.pid}", flush=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""

_EXITING_SERVER = r"""
import signal
import subprocess
import sys

descendant = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
)
print(f"DESCENDANT {descendant.pid}", flush=True)
raise SystemExit(7)
"""


@contextlib.contextmanager
def _lifecycle_timeouts(monkeypatch):
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(dispatcher, "_SGLANG_TERM_TIMEOUT_S", 0.1)
    monkeypatch.setattr(dispatcher, "_SGLANG_KILL_TIMEOUT_S", 3.0)
    monkeypatch.setattr(dispatcher, "_SGLANG_GROUP_POLL_S", 0.01)
    monkeypatch.setattr(dispatcher, "_SGLANG_READY_POLL_S", 0.01)
    monkeypatch.setattr(dispatcher, "_sglang_environment", dict)
    yield


def _launch(
    tmp_path: Path,
    monkeypatch,
    script: str,
    *,
    port: int | None = None,
    ready_timeout_s: float = 3.0,
):
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_sglang_command",
        lambda **kwargs: [sys.executable, "-c", script, str(kwargs["port"])],
    )
    if port is None:
        port = _available_test_port()
    return dispatcher._sglang(
        python=sys.executable,
        model_path=tmp_path / "model",
        log_path=tmp_path / "sglang.log",
        port=port,
        mem_fraction_static=0.5,
        ready_timeout_s=ready_timeout_s,
        served_model="test-model",
    )


def _available_test_port() -> int:
    for _ in range(64):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]
        if port <= 55535:
            return port
    raise RuntimeError("no test port in SGLang's admitted range")


def _descendant_pid(log_path: Path) -> int:
    for line in log_path.read_text().splitlines():
        if line.startswith("DESCENDANT "):
            return int(line.split()[1])
    raise AssertionError(log_path.read_text())


def _run_signal_controller(
    root: str,
    port: int,
    custody_path: str,
    teardown_path: str,
) -> None:
    import evals.signoflife.__main__ as dispatcher

    dispatcher._SGLANG_TERM_TIMEOUT_S = 1.0
    dispatcher._SGLANG_KILL_TIMEOUT_S = 3.0
    dispatcher._SGLANG_GROUP_POLL_S = 0.01
    dispatcher._SGLANG_READY_POLL_S = 0.01
    dispatcher._sglang_environment = dict
    dispatcher._sglang_command = lambda **kwargs: [
        sys.executable,
        "-c",
        _OWNED_SERVER,
        str(kwargs["port"]),
    ]
    signal_group = dispatcher._signal_process_group

    def record_teardown(pgid: int, signal_number: int) -> None:
        if signal_number == signal.SIGTERM:
            Path(teardown_path).write_text(str(pgid), encoding="utf-8")
        signal_group(pgid, signal_number)

    dispatcher._signal_process_group = record_teardown
    root_path = Path(root)
    with dispatcher._sglang(
        python=sys.executable,
        model_path=root_path / "model",
        log_path=root_path / "sglang.log",
        port=port,
        mem_fraction_static=0.5,
        ready_timeout_s=3.0,
        served_model="test-model",
    ) as server:
        Path(custody_path).write_text(
            json.dumps(server.launch["process"]), encoding="utf-8"
        )
        time.sleep(300)


def _wait_for_path(path: Path, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_owned_listener_is_leased_and_stubborn_descendant_is_reaped(
    tmp_path, monkeypatch
) -> None:
    with _lifecycle_timeouts(monkeypatch):
        with pytest.raises(RuntimeError, match="synthetic body failure"):
            with _launch(tmp_path, monkeypatch, _OWNED_SERVER) as server:
                process = server.launch["process"]
                assert process["pid"] == process["sid"] == process["pgid"]
                assert process["listener"]["host"] == "127.0.0.1"
                assert process["listener"]["port"] > 0
                pgid = process["pgid"]
                raise RuntimeError("synthetic body failure")

    descendant = _descendant_pid(tmp_path / "sglang.log")
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    assert not Path(f"/proc/{descendant}").exists()


def test_repeated_parent_termination_cannot_interrupt_session_drain(tmp_path) -> None:
    import multiprocessing

    custody_path = tmp_path / "custody.json"
    teardown_path = tmp_path / "teardown"
    log_path = tmp_path / "sglang.log"
    process = multiprocessing.get_context("fork").Process(
        target=_run_signal_controller,
        args=(str(tmp_path), _available_test_port(), str(custody_path), str(teardown_path)),
    )
    process.start()
    custody: dict[str, object] = {}
    try:
        _wait_for_path(custody_path)
        custody = json.loads(custody_path.read_text(encoding="utf-8"))
        assert custody["pid"] == custody["sid"] == custody["pgid"]
        os.kill(process.pid, signal.SIGTERM)
        _wait_for_path(teardown_path)
        os.kill(process.pid, signal.SIGTERM)
        process.join(timeout=10)
        assert process.exitcode not in (None, 0)
        descendant = _descendant_pid(log_path)
        with pytest.raises(ProcessLookupError):
            os.killpg(int(custody["pgid"]), 0)
        assert not Path(f"/proc/{descendant}").exists()
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if custody:
            try:
                os.killpg(int(custody["pgid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_listener_from_an_unrelated_process_is_rejected(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    impostor = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=impostor.serve_forever, daemon=True)
    thread.start()
    launched: list[int] = []
    popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process.pid)
        return process

    monkeypatch.setattr(dispatcher.subprocess, "Popen", capture_process)
    try:
        with _lifecycle_timeouts(monkeypatch):
            with pytest.raises(RuntimeError, match="not owned by the launched"):
                with _launch(
                    tmp_path,
                    monkeypatch,
                    (
                        "import os,time; "
                        "print(f'LEADER {os.getpid()}', flush=True); time.sleep(300)"
                    ),
                    port=impostor.server_address[1],
                ):
                    pytest.fail("foreign readiness listener was accepted")
    finally:
        impostor.shutdown()
        impostor.server_close()
        thread.join(timeout=3)

    assert len(launched) == 1
    assert not Path(f"/proc/{launched[0]}").exists()


def test_early_leader_exit_still_reaps_its_stubborn_descendant(
    tmp_path, monkeypatch
) -> None:
    with _lifecycle_timeouts(monkeypatch):
        with pytest.raises(RuntimeError, match="exited before ready"):
            with _launch(
                tmp_path,
                monkeypatch,
                _EXITING_SERVER,
                ready_timeout_s=1.0,
            ):
                pytest.fail("exited server was accepted")

    descendant = _descendant_pid(tmp_path / "sglang.log")
    assert not Path(f"/proc/{descendant}").exists()


def test_readiness_timeout_reaps_the_private_process_group(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    launched: list[int] = []
    popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process.pid)
        return process

    monkeypatch.setattr(dispatcher.subprocess, "Popen", capture_process)
    with _lifecycle_timeouts(monkeypatch):
        with pytest.raises(TimeoutError, match="not ready"):
            with _launch(
                tmp_path,
                monkeypatch,
                "import time; time.sleep(300)",
                ready_timeout_s=0.05,
            ):
                pytest.fail("unready server was accepted")

    assert len(launched) == 1
    assert not Path(f"/proc/{launched[0]}").exists()


@pytest.mark.parametrize("failure", [errno.ENOSYS, errno.EPERM])
def test_pidfd_getfd_failure_precedes_process_and_log_acquisition(
    tmp_path, monkeypatch, failure
) -> None:
    import evals.signoflife.__main__ as dispatcher

    def unavailable(pidfd, target_fd):
        raise OSError(failure, os.strerror(failure))

    monkeypatch.setattr(dispatcher, "_pidfd_getfd", unavailable)
    monkeypatch.setattr(
        dispatcher.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen ran before pidfd_getfd preflight"),
    )
    log_path = tmp_path / "logs" / "sglang.log"

    with pytest.raises(RuntimeError, match="pidfd_getfd preflight failed"):
        with dispatcher._sglang(
            python=sys.executable,
            model_path=tmp_path / "model",
            log_path=log_path,
            port=29500,
            mem_fraction_static=0.5,
            ready_timeout_s=1.0,
            served_model="test-model",
        ):
            pytest.fail("unavailable pidfd_getfd was accepted")

    assert not log_path.parent.exists()


def test_automatic_port_selection_is_refused_before_custody_acquisition(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_preflight_pidfd_getfd",
        lambda: pytest.fail("pidfd preflight ran for an unowned port"),
    )
    log_path = tmp_path / "logs" / "sglang.log"

    with pytest.raises(RuntimeError, match="explicit port"):
        with dispatcher._sglang(
            python=sys.executable,
            model_path=tmp_path / "model",
            log_path=log_path,
            port=0,
            mem_fraction_static=0.5,
            ready_timeout_s=1.0,
            served_model="test-model",
        ):
            pytest.fail("automatic port selection was accepted")

    assert not log_path.parent.exists()


def test_descriptor_enumeration_is_bounded(monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(dispatcher, "_SGLANG_MAX_FDS", 1)
    with pytest.raises(RuntimeError, match="exceeds 1 open descriptors"):
        dispatcher._bounded_process_fds(os.getpid())


def test_stale_descriptor_number_cannot_authorize_a_listener(monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pidfd = dispatcher._pidfd_open(os.getpid())
    monkeypatch.setattr(dispatcher, "_bounded_process_fds", lambda pid: [read_fd])
    try:
        with pytest.raises(RuntimeError, match="not owned by the launched"):
            dispatcher._leader_listener_lease(
                pidfd=pidfd,
                leader_pid=os.getpid(),
                port=65535,
            )
    finally:
        os.close(pidfd)
        os.close(read_fd)
        os.close(write_fd)
