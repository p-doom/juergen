from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import socket
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SGLANG_VERSION = "0.5.10.post1"
_UVICORN_VERSION = "0.47.0"
_MAX_PORT = 55535


class _UvicornProxy:
    def __init__(
        self,
        module: ModuleType,
        *,
        listener_fd: int,
        listener_record: dict[str, Any],
    ) -> None:
        self._module = module
        self._listener_fd = listener_fd
        self._listener_record = listener_record
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def run(self, app: Any, **kwargs: Any) -> None:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("SGLang attempted more than one HTTP server launch")
        if kwargs.pop("host", None) != self._listener_record["host"]:
            raise RuntimeError("SGLang changed the listener host")
        if kwargs.pop("port", None) != self._listener_record["port"]:
            raise RuntimeError("SGLang changed the listener port")
        if any(key in kwargs for key in ("fd", "uds", "sockets", "workers")):
            raise RuntimeError("SGLang selected an unsupported Uvicorn listener path")
        self._module.run(app, fd=self._listener_fd, **kwargs)


def _unsupported_server_branches(server_args: Any) -> list[str]:
    unsupported = {
        "encoder_only": server_args.encoder_only,
        "grpc_mode": server_args.grpc_mode,
        "use_ray": server_args.use_ray,
        "tokenizer_worker_num": server_args.tokenizer_worker_num != 1,
        "enable_ssl_refresh": server_args.enable_ssl_refresh,
        "ssl_keyfile": server_args.ssl_keyfile is not None,
        "ssl_certfile": server_args.ssl_certfile is not None,
        "ssl_ca_certs": server_args.ssl_ca_certs is not None,
        "ssl_keyfile_password": server_args.ssl_keyfile_password is not None,
        "nnodes": server_args.nnodes != 1,
        "node_rank": server_args.node_rank != 0,
    }
    return sorted(key for key, enabled in unsupported.items() if enabled)


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listener-fd", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    return parser.parse_args(argv)


def _listener(fd: int) -> tuple[socket.socket, dict[str, Any]]:
    if fd < 3 or not os.get_inheritable(fd):
        raise RuntimeError("listener fd is not inherited")
    listener = socket.socket(fileno=fd)
    try:
        metadata = os.fstat(fd)
        address = listener.getsockname()
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or listener.family != socket.AF_INET
            or listener.type & socket.SOCK_STREAM != socket.SOCK_STREAM
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) != 0
            or not isinstance(address, tuple)
            or len(address) != 2
            or address[0] != "127.0.0.1"
            or isinstance(address[1], bool)
            or not isinstance(address[1], int)
            or not 1 <= address[1] <= _MAX_PORT
        ):
            raise RuntimeError("listener fd is not the reserved loopback socket")
        return listener, {"host": address[0], "port": address[1]}
    except BaseException:
        listener.detach()
        raise


def _server_arguments(args: argparse.Namespace, listener: dict[str, Any]) -> list[str]:
    return [
        "--model-path",
        str(args.model_path),
        "--host",
        listener["host"],
        "--port",
        str(listener["port"]),
        "--enable-deterministic-inference",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        "2048",
    ]


def _run(argv: list[str]) -> None:
    args = _arguments(argv)
    if not args.model_path.is_absolute() or not args.model_path.is_dir():
        raise RuntimeError("model path must be an absolute directory")
    if (
        not math.isfinite(args.mem_fraction_static)
        or not 0 < args.mem_fraction_static <= 1
    ):
        raise RuntimeError("mem-fraction-static must be finite and in (0, 1]")
    versions = {
        "sglang": importlib.metadata.version("sglang"),
        "uvicorn": importlib.metadata.version("uvicorn"),
    }
    if versions != {"sglang": _SGLANG_VERSION, "uvicorn": _UVICORN_VERSION}:
        raise RuntimeError(f"unsupported serving runtime: {versions}")

    listener, listener_record = _listener(args.listener_fd)
    try:
        os.set_inheritable(args.listener_fd, False)
        import uvicorn
        from sglang.launch_server import run_server
        from sglang.srt.entrypoints import http_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree

        server_args = prepare_server_args(_server_arguments(args, listener_record))
        rejected = _unsupported_server_branches(server_args)
        if rejected:
            raise RuntimeError(
                "unsupported SGLang server branch: " + ", ".join(rejected)
            )
        original = http_server.uvicorn
        proxy = _UvicornProxy(
            uvicorn,
            listener_fd=args.listener_fd,
            listener_record=listener_record,
        )
        http_server.uvicorn = proxy
        try:
            try:
                run_server(server_args)
            finally:
                kill_process_tree(os.getpid(), include_parent=False)
        finally:
            http_server.uvicorn = original
        if proxy.calls != 1:
            raise RuntimeError("SGLang did not launch the inherited HTTP listener")
    finally:
        listener.close()


def main() -> int:
    _run(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
