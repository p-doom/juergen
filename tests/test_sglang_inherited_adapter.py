from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from types import ModuleType, SimpleNamespace

import pytest

from evals.signoflife import sglang_server


def _reserved_listener(host: str = "127.0.0.1") -> socket.socket:
    for _ in range(64):
        listener = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_CLOEXEC
        )
        listener.bind((host, 0))
        if listener.getsockname()[1] <= sglang_server._MAX_PORT:
            listener.listen()
            return listener
        listener.close()
    raise RuntimeError("kernel did not allocate an admitted test port")


def test_uvicorn_proxy_uses_only_the_inherited_listener() -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda app, **kwargs: calls.append((app, kwargs))
    record = {"host": "127.0.0.1", "port": 31415}
    proxy = sglang_server._UvicornProxy(uvicorn, listener_fd=7, listener_record=record)
    app = object()

    proxy.run(app, host="127.0.0.1", port=31415, loop="uvloop")

    assert calls == [(app, {"fd": 7, "loop": "uvloop"})]
    with pytest.raises(RuntimeError, match="more than one"):
        proxy.run(app, host="127.0.0.1", port=31415)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"host": "0.0.0.0"}, "listener host"),
        ({"port": 31416}, "listener port"),
        ({"fd": 8}, "listener path"),
        ({"uds": "/tmp/server.sock"}, "listener path"),
        ({"workers": 2}, "listener path"),
    ],
)
def test_uvicorn_proxy_rejects_other_listener_paths(changes, error) -> None:
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: pytest.fail(
        "invalid listener reached Uvicorn"
    )
    proxy = sglang_server._UvicornProxy(
        uvicorn,
        listener_fd=7,
        listener_record={"host": "127.0.0.1", "port": 31415},
    )

    kwargs = {"host": "127.0.0.1", "port": 31415, **changes}
    with pytest.raises(RuntimeError, match=error):
        proxy.run(object(), **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder_only", True),
        ("grpc_mode", True),
        ("use_ray", True),
        ("tokenizer_worker_num", 2),
        ("enable_ssl_refresh", True),
        ("ssl_keyfile", "key.pem"),
        ("ssl_certfile", "cert.pem"),
        ("ssl_ca_certs", "ca.pem"),
        ("ssl_keyfile_password", "secret"),
        ("nnodes", 2),
        ("node_rank", 1),
    ],
)
def test_noncanonical_sglang_server_branches_are_rejected(field, value) -> None:
    fields = {
        "encoder_only": False,
        "grpc_mode": False,
        "use_ray": False,
        "tokenizer_worker_num": 1,
        "enable_ssl_refresh": False,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_ca_certs": None,
        "ssl_keyfile_password": None,
        "nnodes": 1,
        "node_rank": 0,
    }
    fields[field] = value

    assert sglang_server._unsupported_server_branches(SimpleNamespace(**fields)) == [
        field
    ]


def test_listener_validation_rejects_non_loopback_and_non_listening_sockets() -> None:
    wildcard = _reserved_listener("0.0.0.0")
    unbound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        for candidate in (wildcard, unbound):
            duplicate = os.dup(candidate.fileno())
            os.set_inheritable(duplicate, True)
            with pytest.raises(RuntimeError, match="reserved loopback"):
                sglang_server._listener(duplicate)
            os.close(duplicate)
    finally:
        wildcard.close()
        unbound.close()


def test_uvicorn_serves_http_from_the_inherited_listener() -> None:
    listener = _reserved_listener()
    port = listener.getsockname()[1]
    script = """
import sys
import uvicorn

async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"inherited"})

uvicorn.run(app, fd=int(sys.argv[1]), log_level="error")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(listener.fileno())],
        pass_fds=(listener.fileno(),),
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Uvicorn exited with {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=0.2
                ) as response:
                    assert response.read() == b"inherited"
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        listener.close()
