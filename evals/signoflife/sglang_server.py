from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import socket
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SGLANG_VERSION = "0.5.10.post1"
_UVICORN_VERSION = "0.47.0"
_RUNTIME_DISTRIBUTION_COUNT = 344
_RUNTIME_DISTRIBUTION_SET_SHA256 = (
    "d68a25f22ce9baaaf1dbcb2d0c13c47af4187726a8fcad2cc543c95103077cd9"
)
_MAX_PORT = 55535
_NVIDIA_PROC_ROOT = Path("/proc/driver/nvidia")
_NVIDIA_PROC_MAX_BYTES = 64 * 1024
_RUNTIME_DISTRIBUTIONS = {
    "sglang": ("0.5.10.post1", ("sglang",)),
    "torch": ("2.9.1+cu128", ("torch", "functorch", "torchgen")),
    "transformers": ("5.3.0", ("transformers",)),
    "uvicorn": ("0.47.0", ("uvicorn",)),
}
_PROXY_ENVIRONMENT = frozenset(
    {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
)


class _Termination(BaseException):
    pass


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
            raise RuntimeError("SGLang changed the attested listener host")
        if kwargs.pop("port", None) != self._listener_record["port"]:
            raise RuntimeError("SGLang changed the attested listener port")
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


def _server_arguments(
    args: argparse.Namespace, listener_record: dict[str, Any]
) -> list[str]:
    return [
        "--model-path",
        str(args.model_path),
        "--host",
        listener_record["host"],
        "--port",
        str(listener_record["port"]),
        "--served-model-name",
        args.served_model_name,
        "--enable-deterministic-inference",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        "2048",
    ]


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listener-fd", type=int, required=True)
    parser.add_argument("--receipt-fd", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-identity-json", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--gpu-identity-sha256", required=True)
    return parser.parse_args(argv)


def _model_identity(value: str, *, model_path: Path, served_model: str) -> dict[str, Any]:
    try:
        identity = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("model identity is not valid JSON") from error
    expected_fields = {
        "path",
        "artifact_id",
        "producer_run_id",
        "registration",
        "registration_sha256",
        "artifact_manifest",
        "manifest_sha256",
        "artifact_sha256",
        "config_sha256",
        "config_identity",
        "file_count",
        "total_bytes",
        "served_model",
    }
    if not isinstance(identity, dict) or set(identity) != expected_fields:
        raise RuntimeError("model identity has invalid fields")
    if identity["path"] != str(model_path) or identity["served_model"] != served_model:
        raise RuntimeError("model identity does not name the requested model")
    for field in (
        "registration_sha256",
        "manifest_sha256",
        "artifact_sha256",
        "config_sha256",
    ):
        if not isinstance(identity[field], str) or re.fullmatch(
            r"[0-9a-f]{64}", identity[field]
        ) is None:
            raise RuntimeError(f"model identity has invalid {field}")
    for field in ("artifact_id", "producer_run_id", "registration", "artifact_manifest"):
        if not isinstance(identity[field], str) or not identity[field]:
            raise RuntimeError(f"model identity has invalid {field}")
    for field in ("file_count", "total_bytes"):
        if (
            isinstance(identity[field], bool)
            or not isinstance(identity[field], int)
            or identity[field] <= 0
        ):
            raise RuntimeError(f"model identity has invalid {field}")
    if not isinstance(identity["config_identity"], dict):
        raise RuntimeError("model identity has invalid config_identity")
    return identity


def _listener(fd: int) -> tuple[socket.socket, dict[str, Any]]:
    if fd < 3 or not os.get_inheritable(fd):
        raise RuntimeError("listener fd is not an inherited application descriptor")
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
        return listener, {
            "fd": fd,
            "host": address[0],
            "port": address[1],
            "socket_device": metadata.st_dev,
            "socket_inode": metadata.st_ino,
        }
    except BaseException:
        listener.detach()
        raise


def _write_receipt(fd: int, value: dict[str, Any]) -> None:
    if fd < 3 or not os.get_inheritable(fd):
        raise RuntimeError("receipt fd is not an inherited application descriptor")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    view = memoryview(payload)
    try:
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("receipt pipe made no progress")
            view = view[written:]
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def fail(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(
        root, followlinks=False, onerror=fail
    ):
        directory_path = Path(directory)
        for name in directories:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(f"runtime package contains a non-directory: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"runtime package contains a non-regular file: {candidate}")
            rows.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    rows.sort(key=lambda row: row["path"])
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "inventory_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _installed_runtime_identity() -> dict[str, Any]:
    installed = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    installed_sha256 = hashlib.sha256("\n".join(installed).encode()).hexdigest()
    if (
        len(installed) != _RUNTIME_DISTRIBUTION_COUNT
        or installed_sha256 != _RUNTIME_DISTRIBUTION_SET_SHA256
    ):
        raise RuntimeError(
            "unsupported serving distribution set: "
            f"{len(installed)} distributions, sha256 {installed_sha256}"
        )
    distributions: dict[str, Any] = {}
    for name, (expected_version, package_roots) in _RUNTIME_DISTRIBUTIONS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected_version:
            raise RuntimeError(
                f"unsupported {name} version: {distribution.version!r}"
            )
        record = distribution.read_text("RECORD")
        if record is None:
            raise RuntimeError(f"{name} installation has no RECORD")
        roots = {}
        for package_root in package_roots:
            root = Path(distribution.locate_file(package_root))
            if root.is_symlink() or not root.is_dir():
                raise RuntimeError(f"invalid {name} package root: {root}")
            roots[package_root] = _tree_identity(root)
        distributions[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
            "packages": roots,
        }
    executable = Path(sys.executable).resolve()
    return {
        "python": str(executable),
        "python_sha256": _sha256_file(executable),
        "python_version": list(sys.version_info[:3]),
        "distribution_count": len(installed),
        "distribution_set_sha256": installed_sha256,
        "distributions": distributions,
    }


def _bounded_proc_file(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"GPU identity source is not a regular file: {path}")
        with path.open("rb", buffering=0) as handle:
            payload = handle.read(_NVIDIA_PROC_MAX_BYTES + 1)
    except OSError as error:
        raise RuntimeError(f"cannot read GPU identity source: {path}") from error
    if len(payload) > _NVIDIA_PROC_MAX_BYTES:
        raise RuntimeError(f"GPU identity source exceeds its bound: {path}")
    return payload


def _gpu_identity() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isascii() or visible != visible.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one GPU")
    if "," in visible or not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one GPU")
    version = _bounded_proc_file(_NVIDIA_PROC_ROOT / "version")
    candidates = []
    for path in sorted((_NVIDIA_PROC_ROOT / "gpus").glob("*/information")):
        payload = _bounded_proc_file(path)
        fields = {}
        for line in payload.decode("utf-8", errors="strict").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        candidates.append((path, payload, fields))
    if not visible.isdecimal() or visible != str(int(visible)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES has an unsupported identity shape")
    selected = [row for row in candidates if row[2].get("Device Minor") == visible]
    if len(selected) != 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES {visible!r} does not select one NVIDIA proc device"
        )
    path, payload, fields = selected[0]
    required = {key: fields.get(key) for key in ("Model", "GPU UUID", "Bus Location")}
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise RuntimeError(f"incomplete NVIDIA GPU identity: {required}")
    return {
        "cuda_visible_devices": visible,
        "driver_version_sha256": hashlib.sha256(version).hexdigest(),
        "gpu_information_path": str(path),
        "gpu_information_sha256": hashlib.sha256(payload).hexdigest(),
        **required,
    }


def _run(argv: list[str]) -> None:
    args = _arguments(argv)
    proxy_variables = sorted(
        key for key, value in os.environ.items() if value and key.upper() in _PROXY_ENVIRONMENT
    )
    if proxy_variables:
        raise RuntimeError(
            "proxy environment is forbidden in the SGLang runtime: "
            + ", ".join(proxy_variables)
        )
    if not args.model_path.is_absolute() or not args.model_path.is_dir():
        raise RuntimeError("model path must be an absolute directory")
    if not args.served_model_name:
        raise RuntimeError("served model name must be non-empty")
    model_identity = _model_identity(
        args.model_identity_json,
        model_path=args.model_path,
        served_model=args.served_model_name,
    )
    if (
        not math.isfinite(args.mem_fraction_static)
        or not 0.0 < args.mem_fraction_static <= 1.0
    ):
        raise RuntimeError("mem-fraction-static must be finite and in (0, 1]")

    listener, listener_record = _listener(args.listener_fd)
    try:
        os.set_inheritable(args.listener_fd, False)
        versions = {
            "sglang": importlib.metadata.version("sglang"),
            "uvicorn": importlib.metadata.version("uvicorn"),
        }
        if versions != {
            "sglang": _SGLANG_VERSION,
            "uvicorn": _UVICORN_VERSION,
        }:
            raise RuntimeError(f"unsupported serving runtime: {versions}")
        runtime_identity = _installed_runtime_identity()
        gpu_identity = _gpu_identity()
        for label, expected, observed in (
            (
                "runtime",
                args.runtime_identity_sha256,
                hashlib.sha256(
                    json.dumps(
                        runtime_identity, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            ),
            (
                "GPU",
                args.gpu_identity_sha256,
                hashlib.sha256(
                    json.dumps(gpu_identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            ),
        ):
            if expected != observed:
                raise RuntimeError(
                    f"{label} identity changed before SGLang startup: "
                    f"expected {expected!r}, observed {observed!r}"
                )

        import uvicorn
        from sglang.launch_server import run_server
        from sglang.srt.entrypoints import http_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree

        server_args = prepare_server_args(_server_arguments(args, listener_record))
        rejected = _unsupported_server_branches(server_args)
        if rejected:
            raise RuntimeError("unsupported SGLang server branch: " + ", ".join(rejected))

        original_http_uvicorn = http_server.uvicorn
        proxy = _UvicornProxy(
            uvicorn,
            listener_fd=args.listener_fd,
            listener_record=listener_record,
        )
        http_server.uvicorn = proxy
        receipt = {
            "schema_version": 1,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pid": os.getpid(),
            "sid": os.getsid(0),
            "pgid": os.getpgrp(),
            "listener": listener_record,
            "model": model_identity,
            "runtime": runtime_identity,
            "gpu": gpu_identity,
        }
        _write_receipt(args.receipt_fd, receipt)
        previous_handlers: dict[int, Any] = {}
        teardown_started = False

        def terminate(signum: int, _frame: Any) -> None:
            if teardown_started:
                return
            raise _Termination(f"received signal {signum}")

        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, terminate)
            try:
                run_server(server_args)
            finally:
                teardown_started = True
                kill_process_tree(os.getpid(), include_parent=False)
        finally:
            http_server.uvicorn = original_http_uvicorn
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if proxy.calls != 1:
            raise RuntimeError("SGLang did not launch the inherited HTTP listener")
    finally:
        listener.close()


def main() -> int:
    _run(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
