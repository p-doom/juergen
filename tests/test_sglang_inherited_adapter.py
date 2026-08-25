from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
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


def test_uvicorn_proxy_replaces_only_the_attested_listener_arguments() -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    uvicorn = ModuleType("uvicorn")
    uvicorn.marker = object()
    uvicorn.run = lambda app, **kwargs: calls.append((app, kwargs))
    record = {"host": "127.0.0.1", "port": 31415}
    proxy = sglang_server._UvicornProxy(
        uvicorn,
        listener_fd=7,
        listener_record=record,
    )

    app = object()
    proxy.run(
        app,
        host=record["host"],
        port=record["port"],
        root_path="/root",
        loop="uvloop",
        timeout_keep_alive=5,
    )

    assert calls == [
        (
            app,
            {
                "fd": 7,
                "root_path": "/root",
                "loop": "uvloop",
                "timeout_keep_alive": 5,
            },
        )
    ]
    assert proxy.marker is uvicorn.marker
    with pytest.raises(RuntimeError, match="more than one"):
        proxy.run(app, host=record["host"], port=record["port"])


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
def test_uvicorn_proxy_rejects_every_other_listener_path(changes, error) -> None:
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: pytest.fail("invalid listener reached Uvicorn")
    record = {"host": "127.0.0.1", "port": 31415}
    proxy = sglang_server._UvicornProxy(
        uvicorn,
        listener_fd=7,
        listener_record=record,
    )
    kwargs = {"host": record["host"], "port": record["port"], **changes}

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
def test_every_noncanonical_sglang_server_branch_is_rejected(field, value) -> None:
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


def test_gpu_identity_binds_the_visible_minor_to_driver_and_hardware(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "nvidia"
    information = root / "gpus" / "0000:01:00.0" / "information"
    information.parent.mkdir(parents=True)
    (root / "version").write_text("NVRM version: test-driver\n")
    information.write_text(
        "Model: H100-SXM5-80GB\n"
        "GPU UUID: GPU-01234567-89ab-cdef-0123-456789abcdef\n"
        "Bus Location: 0000:01:00.0\n"
        "Device Minor: 3\n"
    )
    monkeypatch.setattr(sglang_server, "_NVIDIA_PROC_ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    identity = sglang_server._gpu_identity()

    assert identity["cuda_visible_devices"] == "3"
    assert identity["Model"] == "H100-SXM5-80GB"
    assert identity["GPU UUID"] == "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert identity["Bus Location"] == "0000:01:00.0"


@pytest.mark.parametrize("visible", ["", "03", "3,4", "GPU-0123", " 3"])
def test_gpu_identity_rejects_every_other_visible_device_shape(
    tmp_path, monkeypatch, visible
) -> None:
    root = tmp_path / "nvidia"
    (root / "gpus").mkdir(parents=True)
    (root / "version").write_text("driver\n")
    monkeypatch.setattr(sglang_server, "_NVIDIA_PROC_ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        sglang_server._gpu_identity()


def test_runtime_tree_identity_changes_with_any_package_byte(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    payload = package / "module.py"
    payload.write_bytes(b"first")
    before = sglang_server._tree_identity(package)

    payload.write_bytes(b"second")
    after = sglang_server._tree_identity(package)

    assert before["file_count"] == after["file_count"] == 1
    assert before["inventory_sha256"] != after["inventory_sha256"]


def test_uvicorn_047_serves_http_from_the_inherited_af_inet_fd() -> None:
    listener = _reserved_listener()
    os.set_inheritable(listener.fileno(), False)
    port = listener.getsockname()[1]
    script = r"""
import sys
import uvicorn

async def app(scope, receive, send):
    if scope["type"] != "http":
        raise RuntimeError("unexpected ASGI scope")
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
        deadline = time.monotonic() + 10.0
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Uvicorn exited with {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=0.2
                ) as response:
                    assert response.status == 200
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


def test_sigterm_cleanup_reaps_a_descendant_that_escaped_with_setsid(tmp_path) -> None:
    listener = _reserved_listener()
    receipt_read, receipt_write = os.pipe2(os.O_CLOEXEC)
    model = tmp_path / "model"
    model.mkdir()
    child_pid_path = tmp_path / "child.pid"
    cleanup_path = tmp_path / "cleanup"
    adapter_path = Path(sglang_server.__file__).resolve()
    model_identity = {
        "path": str(model),
        "artifact_id": "artifact-test",
        "producer_run_id": "run-test",
        "registration": "registration-test",
        "registration_sha256": "1" * 64,
        "artifact_manifest": "manifest-test",
        "manifest_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "config_identity": {"model_type": "test"},
        "file_count": 1,
        "total_bytes": 1,
        "served_model": "test-model",
    }
    script = r"""
import importlib.util
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

adapter_path, child_pid_path, cleanup_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("adapter_under_test", adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter._installed_runtime_identity = lambda: {"runtime": "exact"}
adapter._gpu_identity = lambda: {"gpu": "exact"}
adapter.importlib.metadata.version = lambda name: {
    "sglang": "0.5.10.post1",
    "uvicorn": "0.47.0",
}[name]

uvicorn = types.ModuleType("uvicorn")
uvicorn.run = lambda *args, **kwargs: None
http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
http_server.uvicorn = uvicorn
entrypoints = types.ModuleType("sglang.srt.entrypoints")
entrypoints.http_server = http_server
server_args_module = types.ModuleType("sglang.srt.server_args")
server_args_module.prepare_server_args = lambda argv: types.SimpleNamespace(
    encoder_only=False,
    grpc_mode=False,
    use_ray=False,
    tokenizer_worker_num=1,
    enable_ssl_refresh=False,
    ssl_keyfile=None,
    ssl_certfile=None,
    ssl_ca_certs=None,
    ssl_keyfile_password=None,
    nnodes=1,
    node_rank=0,
)
child = None
launch_server = types.ModuleType("sglang.launch_server")
def run_server(_args):
    global child
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,signal,time; os.setsid(); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
        ]
    )
    Path(child_pid_path).write_text(str(child.pid))
    time.sleep(300)
launch_server.run_server = run_server
utils = types.ModuleType("sglang.srt.utils")
def kill_process_tree(_pid, include_parent=False):
    if child is not None:
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    Path(cleanup_path).write_text("complete")
utils.kill_process_tree = kill_process_tree
sglang = types.ModuleType("sglang")
srt = types.ModuleType("sglang.srt")
sys.modules.update({
    "uvicorn": uvicorn,
    "sglang": sglang,
    "sglang.launch_server": launch_server,
    "sglang.srt": srt,
    "sglang.srt.entrypoints": entrypoints,
    "sglang.srt.entrypoints.http_server": http_server,
    "sglang.srt.server_args": server_args_module,
    "sglang.srt.utils": utils,
})
adapter._run(sys.argv[4:])
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(adapter_path),
            str(child_pid_path),
            str(cleanup_path),
            "--listener-fd",
            str(listener.fileno()),
            "--receipt-fd",
            str(receipt_write),
            "--model-path",
            str(model),
            "--model-identity-json",
            json.dumps(model_identity, sort_keys=True, separators=(",", ":")),
            "--served-model-name",
            "test-model",
            "--mem-fraction-static",
            "0.5",
            "--runtime-identity-sha256",
            "663e8400f99c940458b79b042478baf46e8798f5089109ec1d5cc7eace8e69e2",
            "--gpu-identity-sha256",
            "8ec3a5dc747bb2a620afff7b8bf3eaad87c3693203dba2e762885dfe9ec58405",
        ],
        pass_fds=(listener.fileno(), receipt_write),
        start_new_session=True,
    )
    os.close(receipt_write)
    child_pid = None
    try:
        assert os.read(receipt_read, 65536).endswith(b"\n")
        deadline = time.monotonic() + 10.0
        while not child_pid_path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("escaped child did not start")
            time.sleep(0.02)
        child_pid = int(child_pid_path.read_text())
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        assert process.returncode != 0
        assert cleanup_path.read_text() == "complete"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.close(receipt_read)
        listener.close()
