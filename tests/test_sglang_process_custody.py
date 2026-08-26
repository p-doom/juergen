from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_OWNED_SERVER = """
import http.server
import signal
import socket
import subprocess
import sys

listener = socket.socket(fileno=int(sys.argv[1]))
host, port = listener.getsockname()
descendant = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
])
print(f"DESCENDANT {descendant.pid}", flush=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass

server = http.server.HTTPServer((host, port), Handler, bind_and_activate=False)
server.socket = listener
server.server_address = (host, port)
server.serve_forever()
"""

_EXITING_SERVER = """
import signal
import subprocess
import sys

descendant = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
])
print(f"DESCENDANT {descendant.pid}", flush=True)
raise SystemExit(7)
"""

_UNREADY_SERVER = "import time; time.sleep(300)"


@contextlib.contextmanager
def _lifecycle_timeouts(monkeypatch):
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(dispatcher, "_SGLANG_TERM_TIMEOUT_S", 0.1)
    monkeypatch.setattr(dispatcher, "_SGLANG_KILL_TIMEOUT_S", 3.0)
    monkeypatch.setattr(dispatcher, "_SGLANG_GROUP_POLL_S", 0.01)
    monkeypatch.setattr(dispatcher, "_SGLANG_READY_POLL_S", 0.01)
    monkeypatch.setattr(dispatcher, "_sglang_environment", dict)
    yield


def _launch(tmp_path: Path, monkeypatch, script: str, *, ready_timeout_s: float = 3):
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_sglang_command",
        lambda **kwargs: [
            sys.executable,
            "-c",
            script,
            str(kwargs["listener_fd"]),
        ],
    )
    return dispatcher._sglang(
        python=sys.executable,
        model_path=tmp_path / "model",
        log_path=tmp_path / "sglang.log",
        mem_fraction_static=0.5,
        ready_timeout_s=ready_timeout_s,
    )


def _descendant_pid(log_path: Path) -> int:
    for line in log_path.read_text().splitlines():
        if line.startswith("DESCENDANT "):
            return int(line.split()[1])
    raise AssertionError(log_path.read_text())


def test_owned_listener_and_process_group_are_released(tmp_path, monkeypatch) -> None:
    with _lifecycle_timeouts(monkeypatch):
        with pytest.raises(RuntimeError, match="synthetic body failure"):
            with _launch(tmp_path, monkeypatch, _OWNED_SERVER) as server:
                assert server.launch["pid"] == server.launch["sid"]
                assert server.launch["pid"] == server.launch["pgid"]
                assert server.launch["listener"]["host"] == "127.0.0.1"
                pgid = server.launch["pgid"]
                raise RuntimeError("synthetic body failure")

    descendant = _descendant_pid(tmp_path / "sglang.log")
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    assert not Path(f"/proc/{descendant}").exists()


def test_reserved_listener_cannot_be_stolen(tmp_path, monkeypatch) -> None:
    with _lifecycle_timeouts(monkeypatch):
        with _launch(tmp_path, monkeypatch, _OWNED_SERVER) as server:
            listener = server.launch["listener"]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as thief:
                with pytest.raises(OSError, match="Address already in use"):
                    thief.bind((listener["host"], listener["port"]))


def test_early_exit_reaps_the_process_group(tmp_path, monkeypatch) -> None:
    with _lifecycle_timeouts(monkeypatch):
        with pytest.raises(RuntimeError, match="exited before ready"):
            with _launch(tmp_path, monkeypatch, _EXITING_SERVER):
                pytest.fail("exited server was accepted")

    descendant = _descendant_pid(tmp_path / "sglang.log")
    assert not Path(f"/proc/{descendant}").exists()


def test_readiness_timeout_reaps_the_leader(tmp_path, monkeypatch) -> None:
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
                _UNREADY_SERVER,
                ready_timeout_s=0.05,
            ):
                pytest.fail("unready server was accepted")

    assert len(launched) == 1
    assert not Path(f"/proc/{launched[0]}").exists()


def test_listener_reservation_failure_precedes_process_acquisition(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_reserve_listener",
        lambda: (_ for _ in ()).throw(RuntimeError("reservation failed")),
    )
    monkeypatch.setattr(
        dispatcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen ran after reservation failed"),
    )

    with pytest.raises(RuntimeError, match="reservation failed"):
        with dispatcher._sglang(
            python=sys.executable,
            model_path=tmp_path / "model",
            log_path=tmp_path / "sglang.log",
            mem_fraction_static=0.5,
            ready_timeout_s=1,
        ):
            pytest.fail("server became ready")
