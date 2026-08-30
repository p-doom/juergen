"""`python -m evals.signoflife` — the gate's dispatcher.

`verifiers`' own `vf eval` CLI cannot stand in, because three things this gate
needs are not expressible as CLI flags —

  * the arm. An arm is a whole `DesktopHarnessConfig` (codec, history policy,
    image budget, settle profile, scripted/negative, artifact policy) that lives in
    `cells.py` so an arm cannot be redefined at a command line. `--arm` names one;
    it does not rebuild one.
  * the VM. `DesktopPoolConfig.session_kwargs` has to be filled with an image,
    a qemu binary and a pool target that the harness can actually drive
    (`evals/vm.py`).
  * the aggregate. labctl's `eval_result` output wants one `result.json` at a
    fixed marker path, and what a multi-trial run has to yield is pass_rate per
    cell, not one pass count.

Everything else is verifiers': task loading, the episode, interception, the
client, `traces.jsonl`.

Each `(cell ordinal, trial)` is a separate `run_eval` pass in a supervised spawn
worker, not a `num_rollouts` fan-out. The worker creates, uses and closes its own
single-session desktop pool, so its QEMU descendants share the worker's process
group and the dispatcher can enforce a derived outer deadline without abandoning
a thread or killing a parent-owned VM. Unique attempt roots keep every frame,
prompt, trace and result; the parent dynamically refills up to `--vm-slots` and
sorts completed rows back into canonical cell/trial order before publishing.

`--tier` picks the cell set, and one run is one tier: `scored` is the calibrated
cells, `candidate` is the ones whose own oracle has not been measured on hardware
yet. Averaging the two would publish exactly the uncalibrated number the controls
exist to prevent, so the flag is a choice and never a union.

`controls_ok` is emitted for control arms only, and is `null` for a model arm:
nothing derived from a model arm can calibrate that arm, so cite the scripted
oracle/negative runs instead.

Exit status: 0 fine, 2 a control arm did not read its calibrated value, 3
infrastructure failure (a result that must not be read as a model number).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    # The flat plugin id `signoflife` is a module at the repo root; verifiers
    # imports it by name, so the root must be importable however we were invoked.
    sys.path.insert(0, str(_REPO_ROOT))

import verifiers.v1 as vf  # noqa: E402
from verifiers.v1.cli.eval.runner import run_eval  # noqa: E402
from verifiers.v1.configs.eval import EvalConfig  # noqa: E402

from evals.signoflife.cells import ARMS, verify_phaseb_provenance  # noqa: E402
from evals.signoflife.guest import SETUP_GUEST_REQUESTS  # noqa: E402
from evals.signoflife.suite import TIERS, DevelopmentTask, load_suite  # noqa: E402
from evals.tasks import RESULT_KEY  # noqa: E402
from signoflife import PLUGIN_ID  # noqa: E402

_LOGGER = logging.getLogger("signoflife")

_SERVED_MODEL_PREFIX = "sign-of-life-sha256-"
_ATTESTATION_TIMEOUT_S = 10.0
_ATTESTATION_REQUESTS = 2
_ATTESTATION_MAX_BYTES = 65536
_SEED_PROBE_SEEDS = (19088743, 230973796, 427587855, 1985229328)
_SEED_PROBE_REQUESTS = len(_SEED_PROBE_SEEDS) + 1

API_KEY_VAR = "SIGN_OF_LIFE_API_KEY"
"""Removed external-client credential. Its presence is a dispatch error."""

_LOCAL_NO_AUTH_API_KEY_VAR = "JUERGEN_LOCAL_SGLANG_NO_AUTH_MUST_BE_UNSET"
"""Verifiers maps an absent API-key variable to its fixed in-memory ``EMPTY`` value."""

POOL_TARGET = "evals.vm:kvm_desktop_pool"
"""The production constructor. Tests replace the value before building workers."""

_POOL_ACQUIRE_TIMEOUT_S = 1800.0
_POOL_CHECKOUT_TIMEOUT_S = 1800.0
_POOL_STARTUP_TIMEOUT_S = 1200.0
_GUEST_REQUEST_TIMEOUT_S = 60.0
_QEMU_SHUTDOWN_TIMEOUT_S = 15.0
_SCONTROL_PATH = Path("/usr/bin/scontrol")
_SCONTROL_TIMEOUT_S = 10.0
_SCONTROL_KILL_TIMEOUT_S = 1.0
_SCONTROL_MAX_OUTPUT_BYTES = 64 * 1024
_SUPERVISOR_REAP_TIMEOUT_S = 20.0
_SUPERVISOR_KILL_TIMEOUT_S = 10.0
_ATTEMPT_LAUNCH_MARGIN_S = 20.0
_ATTEMPT_SESSION_READY_TIMEOUT_S = _ATTEMPT_LAUNCH_MARGIN_S
_SCHEDULER_POLL_S = 0.1
_WORKER_START_METHOD = "spawn"
_RESULT_COMMITTED = "RESULT_COMMITTED.json"
_RESULT_COMMIT_SOURCE = ".RESULT_COMMITTED.source"
_OUTPUT_MAX_MARKER_BYTES = 64 * 1024
_OUTPUT_MAX_RESULT_BYTES = 64 * 1024 * 1024
_OUTPUT_MAX_FILES = 65536
_OUTPUT_MAX_DEPTH = 32
_OUTPUT_MAX_COMPONENT_BYTES = 255
_OUTPUT_MAX_PATH_BYTES = 4096
_OUTPUT_MAX_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
_OUTPUT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SAMPLING_SEED_DOMAIN = b"juergen-signoflife-sampling-seed-v1\0"
_DETERMINISTIC_ATTENTION_BACKENDS = frozenset({"fa3", "flashinfer", "triton"})
_PIDFD_SEND_SIGNAL_SYSCALL_X86_64 = 424
_SGLANG_VERSION = "0.5.10.post1"
_SGLANG_PIDFD_OPEN_SYSCALL_X86_64 = 434
_SGLANG_PIDFD_GETFD_SYSCALL_X86_64 = 438
_SGLANG_MAX_FDS = 4096
_SGLANG_MAX_FD_COMPONENT_BYTES = 10
_PROC_MAX_ENTRIES = 32768
_PROC_MAX_STAT_BYTES = 4096
_ATTEMPT_MAX_SESSION_IDENTITIES = 4096
_SGLANG_TERM_TIMEOUT_S = 30.0
_SGLANG_KILL_TIMEOUT_S = 10.0
_SGLANG_GROUP_POLL_S = 0.05
_SGLANG_READY_POLL_S = 2.0
_SGLANG_INHERITED_ENV = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "PATH",
        "TMPDIR",
    }
)
_SGLANG_SEMANTIC_ENV_PREFIXES = (
    "CUDA_",
    "FLASHINFER_",
    "HF_",
    "NCCL_",
    "NVIDIA_",
    "PYTORCH_",
    "SGLANG_",
    "TORCH_",
    "TRANSFORMERS_",
    "TRITON_",
    "VLLM_",
)
_SGLANG_SEMANTIC_ENV_NAMES = frozenset({"PYTHONHOME", "PYTHONPATH"})


@dataclass(frozen=True)
class _LocalServer:
    base_url: str
    launch: dict[str, Any]


class _SglangTermination(BaseException):
    pass


@dataclass
class _SglangSignalGuard:
    previous: dict[int, Any]
    signal_number: int | None = None
    armed: bool = False
    teardown_started: bool = False


@dataclass
class _RunOutput:
    final: Path
    path: Path
    parent_fd: int
    staging_fd: int
    published: bool = False
    durable: bool = False

    def publish(self, *, forbidden_values: tuple[str, ...]) -> None:
        inventory = _seal_output_tree(
            self.staging_fd, forbidden_values=forbidden_values
        )
        _require_private_output_root(self.staging_fd)
        marker = _commit_marker(inventory)
        source_fd = os.open(
            _RESULT_COMMIT_SOURCE,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=self.staging_fd,
        )
        linked = False
        try:
            _write_all(source_fd, _canonical_json(marker) + b"\n")
            os.fsync(source_fd)
            os.fchmod(source_fd, 0o400)
            os.fsync(source_fd)
            source_metadata = os.fstat(source_fd)
            source_path_metadata = os.stat(
                _RESULT_COMMIT_SOURCE,
                dir_fd=self.staging_fd,
                follow_symlinks=False,
            )
            _require_same_object(
                source_metadata, source_path_metadata, label="output commit-marker source"
            )
            os.link(
                _RESULT_COMMIT_SOURCE,
                _RESULT_COMMITTED,
                src_dir_fd=self.staging_fd,
                dst_dir_fd=self.staging_fd,
                follow_symlinks=False,
            )
            linked = True
            self.published = True
            linked_source_metadata = os.fstat(source_fd)
            marker_metadata = os.stat(
                _RESULT_COMMITTED, dir_fd=self.staging_fd, follow_symlinks=False
            )
            _require_same_object(
                linked_source_metadata, marker_metadata, label="output commit marker"
            )
            os.fsync(self.staging_fd)
        except BaseException as error:
            if not linked:
                try:
                    os.unlink(_RESULT_COMMIT_SOURCE, dir_fd=self.staging_fd)
                except FileNotFoundError:
                    pass
            if linked:
                raise RuntimeError(
                    "output commit marker is visible but its link/source state was "
                    f"not durably sealed; quarantine {self.final}"
                ) from error
            raise
        finally:
            os.close(source_fd)
        marker_metadata = os.stat(
            _RESULT_COMMITTED, dir_fd=self.staging_fd, follow_symlinks=False
        )
        source_metadata = os.stat(
            _RESULT_COMMIT_SOURCE, dir_fd=self.staging_fd, follow_symlinks=False
        )
        _require_same_object(source_metadata, marker_metadata, label="output commit link")
        _require_private_output_root(self.staging_fd)
        self.durable = True
        self._close_fds()

    def cleanup(self) -> None:
        self._close_fds()

    def _close_fds(self) -> None:
        if self.staging_fd >= 0:
            os.close(self.staging_fd)
            self.staging_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def _validate_output_target(final: Path) -> Path:
    if not final.is_absolute():
        raise RuntimeError(f"--output must be an absolute run-unique path: {final}")
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"--output must not already exist: {final}")
    parent = final.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise RuntimeError(
            f"output parent must be an existing canonical directory without symlinks: {parent}"
        )
    return final


def _create_uncommitted_output(final: Path) -> _RunOutput:
    final = _validate_output_target(final)
    parent_before = final.parent.stat()
    parent_fd = os.open(
        final.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    staging_fd = -1
    try:
        parent_open = os.fstat(parent_fd)
        _require_same_object(parent_before, parent_open, label="output parent")
        os.mkdir(final.name, mode=0o700, dir_fd=parent_fd)
        try:
            staging_fd = os.open(
                final.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except BaseException:
                os.rmdir(final.name, dir_fd=parent_fd)
                raise
        os.fchmod(staging_fd, 0o700)
        staging_metadata = os.fstat(staging_fd)
        if (
            staging_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(staging_metadata.st_mode) != 0o700
        ):
            os.close(staging_fd)
            staging_fd = -1
            os.rmdir(final.name, dir_fd=parent_fd)
            raise RuntimeError("uncommitted output is not private to the evaluator identity")
        os.fsync(staging_fd)
        os.fsync(parent_fd)
        return _RunOutput(
            final=final,
            path=final,
            parent_fd=parent_fd,
            staging_fd=staging_fd,
        )
    except BaseException:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)
        raise


def _object_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_same_object(
    expected: os.stat_result, observed: os.stat_result, *, label: str
) -> None:
    if _object_identity(expected) != _object_identity(observed):
        raise RuntimeError(f"output changed while sealing: {label}")


def _seal_output_tree(
    root_fd: int, *, forbidden_values: tuple[str, ...]
) -> list[dict[str, Any]]:
    needles = tuple(value.encode() for value in forbidden_values if value)
    root_before = _require_private_output_root(root_fd)
    names = os.listdir(root_fd)
    reserved = sorted({_RESULT_COMMITTED, _RESULT_COMMIT_SOURCE} & set(names))
    if reserved:
        raise RuntimeError(
            f"output commit entries already exist before publication: {reserved}"
        )
    entries: list[dict[str, Any]] = []
    state = {"file_count": 0, "total_bytes": 0}
    _seal_output_directory(
        root_fd,
        relative=".",
        depth=0,
        needles=needles,
        entries=entries,
        capture=None,
        sync=True,
        excluded_root_names=frozenset(),
        state=state,
    )
    root_after = os.fstat(root_fd)
    _require_same_object(root_before, root_after, label="uncommitted output")
    if "result.json" not in os.listdir(root_fd):
        raise RuntimeError("uncommitted output has no result.json payload")
    return entries


def _require_private_output_root(root_fd: int) -> os.stat_result:
    metadata = os.fstat(root_fd)
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError("uncommitted output lost its private 0700 ownership")
    return metadata


def _seal_output_directory(
    directory_fd: int,
    *,
    relative: str,
    depth: int,
    needles: tuple[bytes, ...],
    entries: list[dict[str, Any]],
    capture: dict[str, bytes] | None,
    sync: bool,
    excluded_root_names: frozenset[str],
    state: dict[str, int],
) -> None:
    if depth > _OUTPUT_MAX_DEPTH:
        raise RuntimeError(f"output directory depth exceeds {_OUTPUT_MAX_DEPTH}: {relative}")
    directory_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise RuntimeError(f"output entry is not a directory: {relative}")
    names_before = sorted(os.listdir(directory_fd))
    for name in names_before:
        if relative == "." and name in excluded_root_names:
            continue
        if (
            _OUTPUT_COMPONENT.fullmatch(name) is None
            or len(name.encode()) > _OUTPUT_MAX_COMPONENT_BYTES
        ):
            raise RuntimeError(f"output contains a noncanonical component: {name!r}")
        child_relative = name if relative == "." else f"{relative}/{name}"
        if len(child_relative.encode()) > _OUTPUT_MAX_PATH_BYTES:
            raise RuntimeError(
                f"output path exceeds {_OUTPUT_MAX_PATH_BYTES} bytes: {child_relative}"
            )
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(before.st_mode):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(descriptor)
                _require_same_object(before, opened, label=child_relative)
                if opened.st_uid != os.geteuid() or opened.st_nlink != 1:
                    raise RuntimeError(
                        f"output regular file has unsafe owner/link count: {child_relative}"
                    )
                if (
                    child_relative == "result.json"
                    and opened.st_size > _OUTPUT_MAX_RESULT_BYTES
                ):
                    raise RuntimeError(
                        f"result.json exceeds {_OUTPUT_MAX_RESULT_BYTES} bytes"
                    )
                state["file_count"] += 1
                state["total_bytes"] += opened.st_size
                if state["file_count"] > _OUTPUT_MAX_FILES:
                    raise RuntimeError(
                        f"output file count exceeds {_OUTPUT_MAX_FILES}"
                    )
                if state["total_bytes"] > _OUTPUT_MAX_TOTAL_BYTES:
                    raise RuntimeError(
                        f"output bytes exceed {_OUTPUT_MAX_TOTAL_BYTES}"
                    )
                digest, contents, byte_count = _scan_and_fsync_file(
                    descriptor,
                    relative=child_relative,
                    needles=needles,
                    capture=capture is not None and child_relative == "result.json",
                    capture_limit=(
                        _OUTPUT_MAX_RESULT_BYTES
                        if child_relative == "result.json"
                        else 0
                    ),
                    sync=sync,
                )
                if byte_count != opened.st_size:
                    raise RuntimeError(
                        f"output byte count changed while sealing: {child_relative}"
                    )
                entries.append(
                    {
                        "path": child_relative,
                        "size": opened.st_size,
                        "sha256": digest,
                    }
                )
                if contents is not None:
                    assert capture is not None
                    capture[child_relative] = contents
                after = os.fstat(descriptor)
                _require_same_object(opened, after, label=child_relative)
                path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                _require_same_object(after, path_after, label=child_relative)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                _require_same_object(before, opened, label=child_relative)
                if opened.st_uid != os.geteuid():
                    raise RuntimeError(
                        f"output directory has an unsafe owner: {child_relative}"
                    )
                _seal_output_directory(
                    child_fd,
                    relative=child_relative,
                    depth=depth + 1,
                    needles=needles,
                    entries=entries,
                    capture=capture,
                    sync=sync,
                    excluded_root_names=frozenset(),
                    state=state,
                )
                after = os.fstat(child_fd)
                _require_same_object(opened, after, label=child_relative)
                path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                _require_same_object(after, path_after, label=child_relative)
            finally:
                os.close(child_fd)
        else:
            raise RuntimeError(f"output contains a non-regular entry: {child_relative}")
    if sorted(os.listdir(directory_fd)) != names_before:
        raise RuntimeError(f"output directory entries changed while sealing: {relative}")
    if sync:
        os.fsync(directory_fd)
    directory_after = os.fstat(directory_fd)
    _require_same_object(directory_before, directory_after, label=relative)


def _scan_and_fsync_file(
    descriptor: int,
    *,
    relative: str,
    needles: tuple[bytes, ...],
    capture: bool,
    capture_limit: int,
    sync: bool,
) -> tuple[str, bytes | None, int]:
    overlap = max((len(needle) for needle in needles), default=1) - 1
    previous = b""
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if capture else None
    byte_count = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        byte_count += len(chunk)
        if capture and byte_count > capture_limit:
            raise RuntimeError(f"output capture exceeds {capture_limit} bytes: {relative}")
        digest.update(chunk)
        if captured is not None:
            captured.append(chunk)
        contents = previous + chunk
        if any(needle in contents for needle in needles):
            raise RuntimeError(f"credential value found in staged output: {relative}")
        previous = contents[-overlap:] if overlap else b""
    if sync:
        os.fsync(descriptor)
    return (
        digest.hexdigest(),
        b"".join(captured) if captured is not None else None,
        byte_count,
    )


def _commit_marker(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: entry["path"])
    result = next((entry for entry in ordered if entry["path"] == "result.json"), None)
    if result is None:
        raise RuntimeError("uncommitted output has no result.json payload")
    return {
        "schema_version": 1,
        "status": "committed",
        "authority": "transport_completion_only",
        "promotion_evidence": False,
        "inventory": {
            "file_count": len(ordered),
            "total_bytes": sum(entry["size"] for entry in ordered),
            "sha256": hashlib.sha256(_canonical_json(ordered)).hexdigest(),
        },
        "result_json": {
            "size": result["size"],
            "sha256": result["sha256"],
        },
    }


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while creating output commit marker")
        offset += written


def read_committed_result(output: Path) -> dict[str, Any]:
    """Read a complete transport generation through its exhaustive marker.

    This does not authorize promotion. Promotion additionally requires the
    Labctl DB-bound receipt named by ``result["promotion_evidence"]``.
    """
    if not output.is_absolute():
        raise RuntimeError(f"committed output path must be absolute: {output}")
    parent = output.parent
    if parent.resolve(strict=True) != parent:
        raise RuntimeError(f"committed output parent is not canonical: {parent}")
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        before = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(output_fd)
            _require_same_object(before, opened, label="committed output directory")
            if opened.st_uid != os.geteuid() or stat.S_IMODE(opened.st_mode) != 0o700:
                raise RuntimeError("committed output directory lost private 0700 ownership")
            marker_bytes = _read_output_marker(output_fd)
            try:
                marker = json.loads(marker_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("invalid output commit marker JSON") from error
            if marker_bytes != _canonical_json(marker) + b"\n":
                raise RuntimeError("output commit marker is not canonical JSON")
            entries: list[dict[str, Any]] = []
            captured: dict[str, bytes] = {}
            state = {"file_count": 0, "total_bytes": 0}
            _seal_output_directory(
                output_fd,
                relative=".",
                depth=0,
                needles=(),
                entries=entries,
                capture=captured,
                sync=False,
                excluded_root_names=frozenset(
                    {_RESULT_COMMITTED, _RESULT_COMMIT_SOURCE}
                ),
                state=state,
            )
            expected_marker = _commit_marker(entries)
            if marker != expected_marker:
                raise RuntimeError("output commit marker does not match generation bytes")
            result_bytes = captured.get("result.json")
            if result_bytes is None:
                raise RuntimeError("committed output has no captured result.json")
            try:
                result = json.loads(result_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("committed result.json is invalid") from error
            if not isinstance(result, dict):
                raise RuntimeError("committed result.json is not a JSON object")
            path_after = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            _require_same_object(opened, path_after, label="committed output directory")
            return result
        finally:
            os.close(output_fd)
    finally:
        os.close(parent_fd)


def _read_output_marker(output_fd: int) -> bytes:
    before = os.stat(_RESULT_COMMITTED, dir_fd=output_fd, follow_symlinks=False)
    source = os.stat(
        _RESULT_COMMIT_SOURCE, dir_fd=output_fd, follow_symlinks=False
    )
    _require_same_object(source, before, label="output commit source/marker")
    marker_fd = os.open(
        _RESULT_COMMITTED,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=output_fd,
    )
    try:
        opened = os.fstat(marker_fd)
        _require_same_object(before, opened, label=_RESULT_COMMITTED)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o400
        ):
            raise RuntimeError("output commit marker has unsafe metadata")
        if opened.st_size > _OUTPUT_MAX_MARKER_BYTES:
            raise RuntimeError(
                f"output commit marker exceeds {_OUTPUT_MAX_MARKER_BYTES} bytes"
            )
        _, contents, byte_count = _scan_and_fsync_file(
            marker_fd,
            relative=_RESULT_COMMITTED,
            needles=(),
            capture=True,
            capture_limit=_OUTPUT_MAX_MARKER_BYTES,
            sync=False,
        )
        if byte_count != opened.st_size:
            raise RuntimeError("output commit marker byte count changed while reading")
        after = os.fstat(marker_fd)
        _require_same_object(opened, after, label=_RESULT_COMMITTED)
        path_after = os.stat(
            _RESULT_COMMITTED, dir_fd=output_fd, follow_symlinks=False
        )
        _require_same_object(after, path_after, label=_RESULT_COMMITTED)
        assert contents is not None
        return contents
    finally:
        os.close(marker_fd)


def _linux_syscall(number: int, *arguments: int) -> int:
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("SGLang custody requires Linux x86_64 pidfds")
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    result = syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in arguments))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _pidfd_open(pid: int) -> int:
    pidfd = _linux_syscall(_SGLANG_PIDFD_OPEN_SYSCALL_X86_64, pid, 0)
    fcntl.fcntl(pidfd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    return pidfd


def _pidfd_getfd(pidfd: int, target_fd: int) -> int:
    duplicated = _linux_syscall(
        _SGLANG_PIDFD_GETFD_SYSCALL_X86_64, pidfd, target_fd, 0
    )
    fcntl.fcntl(duplicated, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    return duplicated


def _pidfd_send_signal(pidfd: int, signal_number: int) -> None:
    try:
        _linux_syscall(_PIDFD_SEND_SIGNAL_SYSCALL_X86_64, pidfd, signal_number, 0, 0)
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise


def _preflight_pidfd_getfd() -> None:
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pidfd = -1
    duplicated = -1
    try:
        pidfd = _pidfd_open(os.getpid())
        duplicated = _pidfd_getfd(pidfd, read_fd)
        before = os.fstat(read_fd)
        after = os.fstat(duplicated)
        if (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode)) != (
            after.st_dev,
            after.st_ino,
            stat.S_IFMT(after.st_mode),
        ):
            raise RuntimeError("pidfd_getfd preflight duplicated the wrong object")
    except OSError as error:
        raise RuntimeError(
            f"SGLang custody pidfd_getfd preflight failed: {error}"
        ) from error
    finally:
        if duplicated >= 0:
            os.close(duplicated)
        if pidfd >= 0:
            os.close(pidfd)
        os.close(read_fd)
        os.close(write_fd)


def _bounded_process_fds(pid: int) -> list[int]:
    descriptor_numbers: list[int] = []
    try:
        entries = os.scandir(f"/proc/{pid}/fd")
    except OSError as error:
        raise RuntimeError(f"cannot enumerate SGLang leader {pid} descriptors") from error
    with entries:
        for count, entry in enumerate(entries, start=1):
            if count > _SGLANG_MAX_FDS:
                raise RuntimeError(
                    f"SGLang leader exceeds {_SGLANG_MAX_FDS} open descriptors"
                )
            name = entry.name
            if (
                not name.isascii()
                or not name.isdecimal()
                or len(name.encode()) > _SGLANG_MAX_FD_COMPONENT_BYTES
                or name != str(int(name))
            ):
                raise RuntimeError(f"noncanonical SGLang descriptor name: {name!r}")
            descriptor_numbers.append(int(name))
    return sorted(descriptor_numbers)


def _leader_listener_lease(
    *, pidfd: int, leader_pid: int, port: int
) -> tuple[socket.socket, dict[str, Any]]:
    retained: socket.socket | None = None
    retained_identity: tuple[int, int] | None = None
    retained_fd_number: int | None = None
    try:
        for target_fd in _bounded_process_fds(leader_pid):
            try:
                duplicated = _pidfd_getfd(pidfd, target_fd)
            except OSError as error:
                if error.errno in {errno.EBADF, errno.ESRCH}:
                    continue
                raise RuntimeError(
                    f"cannot attest SGLang leader descriptor {target_fd}: {error}"
                ) from error
            try:
                candidate = socket.socket(fileno=duplicated)
            except OSError:
                os.close(duplicated)
                continue
            try:
                if (
                    candidate.family != socket.AF_INET
                    or candidate.type & socket.SOCK_STREAM != socket.SOCK_STREAM
                    or candidate.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1
                    or candidate.getsockname() != ("127.0.0.1", port)
                    or candidate.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) != 0
                ):
                    continue
                metadata = os.fstat(candidate.fileno())
                if not stat.S_ISSOCK(metadata.st_mode):
                    raise RuntimeError("attested SGLang listener is not a socket")
                identity = (metadata.st_dev, metadata.st_ino)
                if retained is None:
                    retained = candidate
                    retained_identity = identity
                    retained_fd_number = target_fd
                    candidate = None
                elif identity != retained_identity:
                    raise RuntimeError(
                        "SGLang leader owns multiple matching loopback listeners"
                    )
            finally:
                if candidate is not None:
                    candidate.close()
        if retained is None or retained_identity is None or retained_fd_number is None:
            raise RuntimeError(
                "SGLang readiness endpoint is not owned by the launched process-group leader"
            )
        return retained, {
            "leader_fd": retained_fd_number,
            "socket_device": retained_identity[0],
            "socket_inode": retained_identity[1],
            "host": "127.0.0.1",
            "port": port,
        }
    except BaseException:
        if retained is not None:
            retained.close()
        raise


def _pidfd_exited(pidfd: int) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def _process_identity(pid: int) -> tuple[str, int, int, int]:
    stat_fd = os.open(
        f"/proc/{pid}/stat",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        raw = os.read(stat_fd, _PROC_MAX_STAT_BYTES + 1)
    finally:
        os.close(stat_fd)
    if len(raw) > _PROC_MAX_STAT_BYTES:
        raise RuntimeError("oversized /proc process stat during process teardown")
    closing = raw.rfind(b") ")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20:
        raise RuntimeError("malformed /proc process stat during process teardown")
    try:
        state = fields[0].decode("ascii", errors="strict")
        process_group = int(fields[2])
        session = int(fields[3])
        start_time = int(fields[19])
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("invalid /proc process identity during teardown") from error
    return state, process_group, session, start_time


def _process_group_members(pgid: int) -> dict[int, str]:
    members: dict[int, str] = {}
    with os.scandir("/proc") as entries:
        for count, entry in enumerate(entries, start=1):
            if count > _PROC_MAX_ENTRIES:
                raise RuntimeError(
                    f"/proc exceeds {_PROC_MAX_ENTRIES} entries during process teardown"
                )
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            try:
                state, process_group, _session, _start_time = _process_identity(
                    int(entry.name)
                )
            except (FileNotFoundError, ProcessLookupError):
                continue
            if process_group == pgid:
                members[int(entry.name)] = state
    return members


def _wait_for_reserved_leader_only(
    *, pgid: int, leader_pid: int, timeout_s: float
) -> dict[int, str]:
    deadline = time.monotonic() + timeout_s
    while True:
        members = _process_group_members(pgid)
        if not {
            pid: state
            for pid, state in members.items()
            if pid != leader_pid or state != "Z"
        }:
            return members
        if time.monotonic() >= deadline:
            return members
        time.sleep(_SGLANG_GROUP_POLL_S)


def _wait_for_empty_process_group(pgid: int, timeout_s: float) -> dict[int, str]:
    deadline = time.monotonic() + timeout_s
    while True:
        members = _process_group_members(pgid)
        if not members:
            return members
        if time.monotonic() >= deadline:
            return members
        time.sleep(_SGLANG_GROUP_POLL_S)


def _signal_process_group(pgid: int, signal_number: int) -> None:
    try:
        os.killpg(pgid, signal_number)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise RuntimeError(f"cannot signal SGLang process group {pgid}") from error


def _terminate_process_group(process: subprocess.Popen[Any], *, pgid: int) -> None:
    if pgid != process.pid or pgid == os.getpgrp():
        raise RuntimeError("refusing to signal an unowned SGLang process group")
    cleanup_error: RuntimeError | None = None
    _signal_process_group(pgid, signal.SIGTERM)
    try:
        members = _wait_for_reserved_leader_only(
            pgid=pgid,
            leader_pid=process.pid,
            timeout_s=_SGLANG_TERM_TIMEOUT_S,
        )
    except RuntimeError as error:
        members = {pgid: "unknown"}
        cleanup_error = error
    if members and members != {process.pid: "Z"}:
        _signal_process_group(pgid, signal.SIGKILL)
        try:
            members = _wait_for_reserved_leader_only(
                pgid=pgid,
                leader_pid=process.pid,
                timeout_s=_SGLANG_KILL_TIMEOUT_S,
            )
        except RuntimeError as error:
            members = {pgid: "unknown"}
            cleanup_error = cleanup_error or error
    if members and members != {process.pid: "Z"}:
        cleanup_error = cleanup_error or RuntimeError(
            f"SGLang process group {pgid} survived SIGKILL: {members}"
        )
    try:
        process.wait(timeout=_SGLANG_KILL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _signal_process_group(pgid, signal.SIGKILL)
        process.kill()
        process.wait(timeout=_SGLANG_KILL_TIMEOUT_S)
        cleanup_error = cleanup_error or RuntimeError(
            f"SGLang leader {process.pid} did not become reapable"
        )
    try:
        remaining = _wait_for_empty_process_group(pgid, _SGLANG_KILL_TIMEOUT_S)
    except RuntimeError as error:
        remaining = {pgid: "unknown"}
        cleanup_error = cleanup_error or error
    if remaining:
        cleanup_error = cleanup_error or RuntimeError(
            f"SGLang process group {pgid} remains after leader reap: {remaining}"
        )
    if cleanup_error is not None:
        raise cleanup_error


def _install_sglang_signal_guard() -> _SglangSignalGuard:
    guard = _SglangSignalGuard(previous={})

    def terminate(signum: int, _frame: Any) -> None:
        guard.signal_number = signum
        if not guard.armed or guard.teardown_started:
            return
        guard.teardown_started = True
        raise _SglangTermination(f"received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            guard.previous[signum] = signal.signal(signum, terminate)
    except BaseException:
        _restore_sglang_signal_handlers(guard)
        raise
    return guard


def _arm_sglang_signal_guard(guard: _SglangSignalGuard) -> None:
    guard.armed = True
    if guard.signal_number is not None and not guard.teardown_started:
        guard.teardown_started = True
        raise _SglangTermination(f"received signal {guard.signal_number}")


def _restore_sglang_signal_handlers(guard: _SglangSignalGuard) -> None:
    for signum, handler in guard.previous.items():
        signal.signal(signum, handler)


def _reject_disabled_cudnn_check() -> None:
    if "SGLANG_DISABLE_CUDNN_CHECK" in os.environ:
        raise RuntimeError("refusing disabled SGLang cuDNN compatibility validation")


@contextlib.contextmanager
def _sglang(
    *,
    python: str,
    model_path: Path,
    log_path: Path,
    port: int,
    mem_fraction_static: float,
    ready_timeout_s: float,
    served_model: str,
) -> Iterator[_LocalServer]:
    """Serve `model_path` and yield its OpenAI base URL.

    `python` is an explicit interpreter, not `sys.executable`: the harness needs
    `verifiers`, sglang needs a 14 GB CUDA stack, and the two do not have to be the
    same venv.
    """
    _reject_disabled_cudnn_check()
    if not 1 <= port <= 55535:
        raise RuntimeError("SGLang requires an explicit port in [1, 55535]")
    _preflight_pidfd_getfd()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _sglang_command(
        python=python,
        model_path=model_path,
        port=port,
        mem_fraction_static=mem_fraction_static,
        served_model=served_model,
    )
    environment = _sglang_environment()
    launch = {
        "argv": command,
        "argv_sha256": hashlib.sha256(_canonical_json(command)).hexdigest(),
        "environment": [
            {
                "key": key,
                "value_sha256": hashlib.sha256(value.encode()).hexdigest(),
            }
            for key, value in sorted(environment.items())
        ],
    }
    _LOGGER.info("sglang launch command sha256: %s", launch["argv_sha256"])
    guard = _install_sglang_signal_guard()
    handle: Any | None = None
    process: subprocess.Popen[Any] | None = None
    pidfd = -1
    listener: socket.socket | None = None
    try:
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _arm_sglang_signal_guard(guard)
        pgid = process.pid
        try:
            observed_sid = os.getsid(process.pid)
            observed_pgid = os.getpgid(process.pid)
            if (observed_sid, observed_pgid) != (process.pid, process.pid):
                raise RuntimeError(
                    "SGLang child is not the leader of its private session"
                )
            pidfd = _pidfd_open(process.pid)
        except OSError as error:
            raise RuntimeError("cannot establish SGLang process custody") from error
        deadline = time.monotonic() + ready_timeout_s
        probe = f"http://127.0.0.1:{port}/health_generate"
        while time.monotonic() < deadline:
            if _pidfd_exited(pidfd):
                tail = "\n".join(log_path.read_text().splitlines()[-40:])
                raise RuntimeError(f"sglang exited before ready:\n{tail}")
            try:
                with urllib.request.urlopen(probe, timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:  # noqa: BLE001 - not up yet is the normal case
                pass
            time.sleep(_SGLANG_READY_POLL_S)
        else:
            raise TimeoutError(f"sglang not ready after {ready_timeout_s}s")
        listener, listener_record = _leader_listener_lease(
            pidfd=pidfd,
            leader_pid=process.pid,
            port=port,
        )
        launch["process"] = {
            "pid": process.pid,
            "sid": observed_sid,
            "pgid": pgid,
            "listener": listener_record,
        }
        url = f"http://127.0.0.1:{port}/v1"
        _LOGGER.info("sglang ready at %s", url)
        yield _LocalServer(base_url=url, launch=launch)
    finally:
        guard.teardown_started = True
        try:
            if process is not None:
                _terminate_process_group(process, pgid=process.pid)
        finally:
            if listener is not None:
                listener.close()
            if pidfd >= 0:
                os.close(pidfd)
            if handle is not None:
                handle.close()
            _restore_sglang_signal_handlers(guard)


def _sglang_command(
    *,
    python: str,
    model_path: Path,
    port: int,
    mem_fraction_static: float,
    served_model: str,
) -> list[str]:
    return [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model,
        "--enable-deterministic-inference",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        "2048",
    ]
def _sglang_environment() -> dict[str, str]:
    rejected = sorted(
        key
        for key, value in os.environ.items()
        if value
        and key not in _SGLANG_INHERITED_ENV
        and (
            key in _SGLANG_SEMANTIC_ENV_NAMES
            or key.startswith(_SGLANG_SEMANTIC_ENV_PREFIXES)
        )
    )
    if rejected:
        raise RuntimeError(
            "refusing unbound semantic variables in local SGLang environment: "
            + ", ".join(rejected)
        )
    environment = {
        key: os.environ[key] for key in sorted(_SGLANG_INHERITED_ENV) if key in os.environ
    }
    return environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _ModelArtifact:
    model_path: Path
    artifact_id: str
    producer_run_id: str
    registration_path: Path
    registration_sha256: str
    manifest_path: Path
    manifest_sha256: str
    artifact_sha256: str
    config_sha256: str
    config_identity: dict[str, Any]
    file_count: int
    total_bytes: int
    served_model: str

    def record(self, *, attestation: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(self.model_path),
            "artifact_id": self.artifact_id,
            "producer_run_id": self.producer_run_id,
            "registration": str(self.registration_path),
            "registration_sha256": self.registration_sha256,
            "artifact_manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "artifact_sha256": self.artifact_sha256,
            "config_sha256": self.config_sha256,
            "config_identity": self.config_identity,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "served_model": self.served_model,
            "attestation": attestation,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {label} {path}: expected a JSON object")
    return value


def _manifest_file_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise RuntimeError("invalid artifact manifest file inventory row")
    relative = value["path"]
    parsed = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        parsed is None
        or not relative
        or parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise RuntimeError(f"invalid artifact manifest path {relative!r}")
    size = value["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"invalid artifact manifest size for {relative!r}")
    digest = value["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"invalid artifact manifest sha256 for {relative!r}")
    return {"path": relative, "size": size, "sha256": digest}


def _model_inventory(model_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(
        model_path, followlinks=False, onerror=_raise_walk_error
    ):
        directory_path = Path(directory)
        for name in directories:
            candidate = directory_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise RuntimeError(f"model artifact contains a non-directory: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"model artifact contains a non-regular file: {candidate}")
            rows.append(
                {
                    "path": candidate.relative_to(model_path).as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    return sorted(rows, key=lambda row: row["path"])


def _verify_model_artifact(model_path: Path) -> _ModelArtifact:
    """Verify the registered exhaustive byte inventory before allocating a VM."""
    if not model_path.is_absolute():
        raise RuntimeError(f"model artifact path must be absolute: {model_path}")
    if model_path.is_symlink() or not model_path.is_dir():
        raise RuntimeError(f"model artifact is not a regular directory: {model_path}")
    root = model_path.parent
    metadata_path = root / ".meta.json"
    manifest_path = root / "artifact_manifest.json"
    for path, label in (
        (metadata_path, "artifact registration"),
        (manifest_path, "artifact manifest"),
    ):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise RuntimeError(f"missing {label}: {path}") from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")

    metadata = _json_object(metadata_path, label="artifact registration")
    manifest_sha256 = _sha256_file(manifest_path)
    if metadata.get("artifact_manifest_sha256") != manifest_sha256:
        raise RuntimeError(
            "artifact manifest registration mismatch: "
            f"expected {metadata.get('artifact_manifest_sha256')!r}, "
            f"observed {manifest_sha256!r}"
        )
    artifact_id = metadata.get("id")
    producer_run_id = metadata.get("producer_run_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise RuntimeError("artifact registration requires a non-empty id")
    if not isinstance(producer_run_id, str) or not producer_run_id:
        raise RuntimeError("artifact registration requires a non-empty producer_run_id")

    manifest = _json_object(manifest_path, label="artifact manifest")
    if set(manifest) != {
        "schema_version",
        "status",
        "artifact_sha256",
        "config_sha256",
        "files",
    }:
        raise RuntimeError("invalid artifact manifest fields")
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise RuntimeError("artifact manifest is not schema 1 complete")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise RuntimeError("artifact manifest requires a non-empty file inventory")
    expected = [_manifest_file_row(row) for row in manifest["files"]]
    expected_paths = [row["path"] for row in expected]
    if expected_paths != sorted(set(expected_paths)):
        raise RuntimeError("artifact manifest inventory must be sorted and unique")
    expected_artifact_sha256 = hashlib.sha256(_canonical_json(expected)).hexdigest()
    if manifest["artifact_sha256"] != expected_artifact_sha256:
        raise RuntimeError("artifact manifest has an invalid artifact_sha256")

    observed = _model_inventory(model_path)
    if observed != expected:
        raise RuntimeError(
            f"artifact inventory mismatch: expected {expected!r}, observed {observed!r}"
        )
    config_rows = [row for row in observed if row["path"] == "config.json"]
    if len(config_rows) != 1 or manifest["config_sha256"] != config_rows[0]["sha256"]:
        raise RuntimeError("artifact manifest has an invalid config_sha256")
    config = _json_object(model_path / "config.json", label="model config")
    config_identity = {
        key: config.get(key) for key in ("model_type", "architectures")
    }
    return _ModelArtifact(
        model_path=model_path,
        artifact_id=artifact_id,
        producer_run_id=producer_run_id,
        registration_path=metadata_path,
        registration_sha256=_sha256_file(metadata_path),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        artifact_sha256=expected_artifact_sha256,
        config_sha256=config_rows[0]["sha256"],
        config_identity=config_identity,
        file_count=len(observed),
        total_bytes=sum(row["size"] for row in observed),
        served_model=f"{_SERVED_MODEL_PREFIX}{expected_artifact_sha256}",
    )


def _server_json(
    request: str | urllib.request.Request, *, timeout_s: float
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise ValueError(f"HTTP status {response.status}")
            raw = response.read(_ATTESTATION_MAX_BYTES + 1)
        if len(raw) > _ATTESTATION_MAX_BYTES:
            raise ValueError("server response exceeds 65536 bytes")
        value = json.loads(raw)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"local SGLang attestation request failed: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("local SGLang attestation response is not a JSON object")
    return value


def _attest_local_server(
    base_url: str, *, artifact: _ModelArtifact
) -> dict[str, Any]:
    root = base_url.removesuffix("/v1")
    models = _server_json(f"{base_url}/models", timeout_s=_ATTESTATION_TIMEOUT_S)
    data = models.get("data")
    model_ids = (
        [row.get("id") for row in data if isinstance(row, dict)]
        if isinstance(data, list)
        else []
    )
    if model_ids != [artifact.served_model]:
        raise RuntimeError(
            f"local SGLang model identity mismatch: expected {[artifact.served_model]!r}, "
            f"observed {model_ids!r}"
        )
    info = _server_json(f"{root}/server_info", timeout_s=_ATTESTATION_TIMEOUT_S)
    attested = {
        "version": info.get("version"),
        "enable_deterministic_inference": info.get("enable_deterministic_inference"),
        "sampling_backend": info.get("sampling_backend"),
        "attention_backend": info.get("attention_backend"),
    }
    if (
        attested["version"] != _SGLANG_VERSION
        or attested["enable_deterministic_inference"] is not True
        or attested["sampling_backend"] != "pytorch"
        or attested["attention_backend"] not in _DETERMINISTIC_ATTENTION_BACKENDS
    ):
        raise RuntimeError(f"local SGLang deterministic server mismatch: {attested}")
    return {
        "source": "local_verified_launch",
        "artifact_sha256": artifact.artifact_sha256,
        "config_sha256": artifact.config_sha256,
        "served_model": artifact.served_model,
        "server": attested,
    }


def _seed_probe_completion(
    base_url: str, *, served_model: str, seed: int, timeout_s: float
) -> tuple[str, str | None]:
    payload = {
        "model": served_model,
        "messages": [
            {
                "role": "user",
                "content": "Produce exactly 16 random lowercase ASCII letters.",
            }
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=_canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = _server_json(request, timeout_s=timeout_s)
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"invalid SGLang seed-probe response: {response}") from error
    if not isinstance(content, str):
        raise RuntimeError("invalid SGLang seed-probe content")
    return content, finish_reason


def _probe_seeded_sampling(
    base_url: str, *, served_model: str, timeout_s: float
) -> dict[str, Any]:
    first_seed = _SEED_PROBE_SEEDS[0]
    repeated = [
        _seed_probe_completion(
            base_url,
            served_model=served_model,
            seed=first_seed,
            timeout_s=timeout_s,
        )
        for _ in range(2)
    ]
    if repeated[0] != repeated[1]:
        raise RuntimeError("SGLang repeated same-seed probe was not byte-identical")
    controls = [
        _seed_probe_completion(
            base_url,
            served_model=served_model,
            seed=seed,
            timeout_s=timeout_s,
        )
        for seed in _SEED_PROBE_SEEDS[1:]
    ]
    if len(set(controls)) == 1:
        raise RuntimeError("SGLang different-seed controls were all identical")
    return {
        "schema_version": 1,
        "scope": "repeatability_and_variation_preflight",
        "individual_seed_consumption_proven": False,
        "bitwise_cross_hardware_determinism": False,
        "request_count": _SEED_PROBE_REQUESTS,
        "seeds": list(_SEED_PROBE_SEEDS),
        "same_seed_output_sha256": hashlib.sha256(
            _canonical_json(repeated[0])
        ).hexdigest(),
        "different_seed_output_sha256": [
            hashlib.sha256(_canonical_json(output)).hexdigest() for output in controls
        ],
    }


def _attempt_wall_bound_s(task: DevelopmentTask, arm: Any) -> float:
    """Hard outer bound from the exact blocking phases one worker can enter."""
    try:
        setup_requests = SETUP_GUEST_REQUESTS[task.kind]
    except KeyError as error:
        raise ValueError(f"no setup request bound for task kind {task.kind!r}") from error
    max_steps = arm.max_steps or task.max_steps
    settle_s = arm.settle.per_kind.get(task.kind, arm.settle.min_delay_s)
    observation_bound_s = settle_s + arm.settle.stability_timeout_s
    scripted_render_requests = 2 if arm.scripted.enabled else 0
    per_turn_guest_requests = 5 + scripted_render_requests
    per_turn_model_s = 0.0 if arm.scripted.enabled else arm.model_request_timeout_s
    return float(
        _POOL_ACQUIRE_TIMEOUT_S
        + _POOL_CHECKOUT_TIMEOUT_S
        + _ATTEMPT_LAUNCH_MARGIN_S
        + setup_requests * _GUEST_REQUEST_TIMEOUT_S
        # Geometry, the initial state probe and the initial screenshot.
        + 3 * _GUEST_REQUEST_TIMEOUT_S
        + observation_bound_s
        + max_steps
        * (
            per_turn_guest_requests * _GUEST_REQUEST_TIMEOUT_S
            + per_turn_model_s
            + observation_bound_s
        )
        # One batched held-input release and the runtime-backed reward probe.
        + 2 * _GUEST_REQUEST_TIMEOUT_S
        + _QEMU_SHUTDOWN_TIMEOUT_S
    )


def _suite_wall_bound_s(
    tasks: list[DevelopmentTask],
    *,
    arm: Any,
    trials: int,
    vm_slots: int,
    local_sglang: bool,
    sglang_ready_timeout_s: float,
) -> float:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if vm_slots < 1:
        raise ValueError("vm_slots must be >= 1")
    slot_bounds = [0.0] * vm_slots
    attempt_count = 0
    for task in tasks:
        for _trial in range(trials):
            slot = min(range(vm_slots), key=lambda index: (slot_bounds[index], index))
            slot_bounds[slot] += _attempt_wall_bound_s(task, arm)
            attempt_count += 1
    supervisor_drain_bound_s = attempt_count * (
        _SUPERVISOR_REAP_TIMEOUT_S + 3 * _SUPERVISOR_KILL_TIMEOUT_S
    )
    local_server_bound_s = 0.0
    if local_sglang:
        local_server_bound_s = (
            sglang_ready_timeout_s
            + _ATTESTATION_REQUESTS * _ATTESTATION_TIMEOUT_S
            + _SEED_PROBE_REQUESTS * arm.model_request_timeout_s
            + _SGLANG_TERM_TIMEOUT_S
            + 3 * _SGLANG_KILL_TIMEOUT_S
        )
    return (
        max(slot_bounds, default=0.0)
        + supervisor_drain_bound_s
        + local_server_bound_s
    )


def _parse_slurm_duration_s(value: str) -> float:
    if value == "UNLIMITED":
        return math.inf
    day_text, separator, clock_text = value.partition("-")
    days = int(day_text) if separator else 0
    if not separator:
        clock_text = day_text
    fields = [int(field) for field in clock_text.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours = 0
        minutes, seconds = fields
    elif len(fields) == 1:
        hours = 0
        minutes = fields[0]
        seconds = 0
    else:
        raise ValueError(f"invalid SLURM duration {value!r}")
    if days < 0 or hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid SLURM duration {value!r}")
    return float(((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def _bounded_scontrol(job_id: str) -> str:
    try:
        process = subprocess.Popen(
            [str(_SCONTROL_PATH), "show", "job", job_id, "-o"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot start scontrol for job {job_id}") from error
    streams: dict[int, str] = {}
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        if process.pid is None:
            raise RuntimeError("scontrol process has no pid")
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("scontrol output pipes were not created")
        poller = select.poll()
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            poller.register(
                descriptor,
                select.POLLIN | select.POLLHUP | select.POLLERR,
            )
            streams[descriptor] = name
        deadline = time.monotonic() + _SCONTROL_TIMEOUT_S
        while streams or process.poll() is None:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(f"scontrol query timed out for job {job_id}")
            for descriptor, _event in poller.poll(
                max(1, min(100, math.ceil(remaining_s * 1000)))
            ):
                name = streams[descriptor]
                while True:
                    try:
                        chunk = os.read(descriptor, 8192)
                    except BlockingIOError:
                        break
                    if not chunk:
                        poller.unregister(descriptor)
                        del streams[descriptor]
                        break
                    captured[name].extend(chunk)
                    if len(captured[name]) > _SCONTROL_MAX_OUTPUT_BYTES:
                        raise RuntimeError(
                            f"scontrol {name} exceeds its bound for job {job_id}"
                        )
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"scontrol query failed for job {job_id}")
        return captured["stdout"].decode("utf-8", errors="strict")
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise RuntimeError(f"bounded scontrol query failed for job {job_id}") from error
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            try:
                if process.pid is not None and process.pid != os.getpgrp():
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=_SCONTROL_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"scontrol process did not reap for job {job_id}"
                ) from error


def _slurm_remaining_wall_s(job_id: str) -> float:
    if not job_id.isascii() or not job_id.isdecimal() or len(job_id) > 20:
        raise RuntimeError(f"invalid SLURM_JOB_ID {job_id!r}")
    stdout = _bounded_scontrol(job_id)
    fields = {
        key: value
        for token in stdout.split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    try:
        limit = _parse_slurm_duration_s(fields["TimeLimit"])
        runtime = _parse_slurm_duration_s(fields["RunTime"])
    except KeyError as error:
        raise RuntimeError(f"scontrol omitted {error.args[0]} for job {job_id}") from error
    return max(0.0, limit - runtime)


def _preflight_slurm_wall_budget(required_s: float) -> float:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        _LOGGER.info("derived suite wall bound: %.0fs", required_s)
        return required_s
    remaining_s = _slurm_remaining_wall_s(job_id)
    if remaining_s < required_s:
        raise RuntimeError(
            f"SLURM job {job_id} has {remaining_s:.0f}s remaining and cannot cover "
            f"declared suite bound {required_s:.0f}s"
        )
    _LOGGER.info(
        "SLURM wall preflight: job=%s remaining=%.0fs required=%.0fs",
        job_id,
        remaining_s,
        required_s,
    )
    return required_s


def _harness_payload(arm: str, *, artifacts: Path, pool: dict[str, Any]) -> dict[str, Any]:
    """One arm's `DesktopHarnessConfig` as verifiers wants it.

    `id` is overwritten with the plugin id, and it has to be: `HarnessConfig.id` is
    the *plugin* id `harness_class()` imports (`loaders.py:87-88`), while `cells.py`
    uses it as the arm's human name (`sol_native_oracle`, ...). Resolving
    `sol_native_oracle` as a package fails, so the arm name moves into the run
    record and the field goes back to meaning what verifiers means by it.
    """
    payload = ARMS[arm].model_dump()
    payload["id"] = PLUGIN_ID
    payload["artifacts"] = {**payload["artifacts"], "output_dir": str(artifacts)}
    payload["pool"] = {**payload["pool"], **pool}
    return payload


def _eval_config(
    *,
    arm: str,
    tier: str,
    task_ids: list[str],
    artifacts: Path,
    traces_dir: Path,
    pool: dict[str, Any],
    base_url: str,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int,
    served_model: str,
    seed: int | None,
) -> EvalConfig:
    sampling = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        sampling["seed"] = seed
    return EvalConfig(
        taskset={"id": PLUGIN_ID, "tier": tier, "task_ids": task_ids},
        harness=_harness_payload(arm, artifacts=artifacts, pool=pool),
        model=served_model,
        client={"base_url": base_url, "api_key_var": _LOCAL_NO_AUTH_API_KEY_VAR},
        sampling=sampling,
        num_rollouts=1,
        # One episode per supervised worker. Parallelism belongs to the dispatcher,
        # where every process owns and releases its one desktop.
        max_concurrent=1,
        output_dir=traces_dir,
        rich=False,  # a live dashboard in a batch job is a log full of escape codes
        push=False,  # never upload a gate run to the Prime platform
    )


def _harness_error(trace: Any) -> dict[str, Any] | None:
    """The reason an episode published nothing at all.

    `DesktopHarness._run` turns an episode failure into `validity="infra_invalid"`
    with an `infra_error`, but a raise *before* it — a bad pool spec, an unknown
    grammar, an unregistered task kind — never reaches that code and leaves
    `trace.info` empty. The row then said `validity: null, infra_error: null`, so
    the run exited 3 with no reason recorded anywhere a reader would look, and the
    only copy of the message was a log line in a batch job's stdout.

    Reads `trace.errors` / `trace.stop_condition`, which verifiers fills in for this
    case. Diagnostic only: an episode with no result is already excluded from every
    rate (`_aggregate` counts `validity == "valid"`), so nothing here can move a
    pass count.
    """
    errors = getattr(trace, "errors", None) or []
    if not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        first = getattr(first, "model_dump", lambda: {"message": str(first)})()
    return {
        "stage": "harness",
        "type": str(first.get("type") or "HarnessError"),
        "message": str(first.get("message") or first),
        "stop_condition": getattr(trace, "stop_condition", None),
    }


def _episode_row(trace: Any, trial: int) -> dict[str, Any]:
    episode = dict(trace.info.get(RESULT_KEY) or {})
    prompt = dict(trace.info.get("prompt") or {})
    infra_error = episode.get("infra_error") or _harness_error(trace)
    validity = episode.get("validity")
    if validity is None and infra_error is not None:
        validity = "infra_invalid"
    return {
        "trial": trial,
        "cell": trace.task.data.name,
        "kind": trace.task.data.kind,
        "trace_id": trace.id,
        "success": None if validity == "infra_invalid" else episode.get("success"),
        "validity": validity,
        "outcome": episode.get("outcome") or (
            "infrastructure_error" if validity == "infra_invalid" else None
        ),
        "steps": episode.get("steps"),
        "parse_errors": episode.get("parse_errors"),
        "action_errors": episode.get("action_errors"),
        "executor_errors": episode.get("executor_errors"),
        "control_terminate": episode.get("control_terminate"),
        "terminate_step": episode.get("terminate_step"),
        "control_ok": episode.get("control_ok"),
        "infra_error": infra_error,
        "final_probe": episode.get("final_probe"),
        "sampling": episode.get("sampling"),
        "host": episode.get("host"),
        "prompt_sha256": prompt.get("prompt_sha256"),
        "comparable_to_sealed_baseline": prompt.get("comparable_to_sealed_baseline"),
        "steps_detail": episode.get("steps_detail"),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    cell_ids: list[str],
    expected_trials: int,
    scripted: bool,
    negative: bool,
) -> dict[str, Any]:
    """pass_rate per cell over trials.

    A single-trial score cannot be read on `desktop_open_chrome`: the suite's own
    `instrument_limits` note says a Chrome that starts but never maps a window flips
    PASS to FAIL, so the cell must be read as a rate over trials. A scalar
    `passed/4` hides that, and one such race was once reported as an arm difference.
    """
    per_cell: dict[str, Any] = {}
    for cell in cell_ids:
        draws = [row for row in rows if row["cell"] == cell]
        valid = [row for row in draws if row["validity"] == "valid"]
        passed = sum(1 for row in valid if row["success"] is True)
        complete = (
            sorted(row.get("trial") for row in draws) == list(range(1, expected_trials + 1))
            and len(valid) == expected_trials
        )
        per_cell[cell] = {
            "trials": len(draws),
            "valid_trials": len(valid),
            "passed": passed,
            "pass_rate": (passed / expected_trials) if complete else None,
            "trial_contract_complete": complete,
            "outcomes": [row["outcome"] for row in draws],
        }
    valid_rows = [row for row in rows if row["validity"] == "valid"]
    conformant = (
        None
        if not scripted
        else all(row.get("control_ok") == 1.0 for row in rows) and len(valid_rows) == len(rows)
    )
    return {
        "per_cell": per_cell,
        "episodes": len(rows),
        "valid_episodes": len(valid_rows),
        "valid_trial_contract_complete": all(
            cell["trial_contract_complete"] for cell in per_cell.values()
        ),
        "expected_per_cell_pass_rate": (0.0 if negative else 1.0) if scripted else None,
        "controls_ok": conformant,
        "controls_ok_note": (
            "null for a model arm on purpose. A model arm has no expected value, so "
            "any 'controls_ok' computed from its own rows only restates the pass "
            "count; calibration comes from the separate scripted oracle/negative runs."
        ),
    }


def _trial_contract_errors(
    rows: list[dict[str, Any]], *, cell_ids: list[str], expected_trials: int
) -> list[dict[str, Any]]:
    expected = list(range(1, expected_trials + 1))
    errors: list[dict[str, Any]] = []
    unknown = sorted({row.get("cell") for row in rows} - set(cell_ids), key=str)
    if unknown:
        errors.append(
            {
                "type": "ValidTrialCountError",
                "message": f"rows contain cells outside the sealed selection: {unknown}",
            }
        )
    for cell in cell_ids:
        draws = [row for row in rows if row.get("cell") == cell]
        observed = sorted(row.get("trial") for row in draws)
        valid = sorted(
            row.get("trial") for row in draws if row.get("validity") == "valid"
        )
        if observed != expected or valid != expected:
            errors.append(
                {
                    "cell": cell,
                    "type": "ValidTrialCountError",
                    "message": (
                        f"expected exact trial identities {expected}; "
                        f"observed={observed}, valid={valid}"
                    ),
                }
            )
    return errors


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class _AttemptSpec:
    index: int
    cell_ordinal: int
    trial: int
    task: DevelopmentTask
    wall_bound_s: float
    sampling_seed: int | None = None


@dataclass(frozen=True)
class _WorkerRuntime:
    arm: str
    tier: str
    output: Path
    base_url: str
    temperature: float | None
    top_p: float | None
    max_tokens: int
    served_model: str
    model: dict[str, Any] | None
    qcow: Path
    qemu: Path | None
    qemu_img: Path | None
    vm_smp: int | None
    vm_mem: str | None
    vm_slots: int
    pool_target: str


@dataclass
class _ActiveAttempt:
    spec: _AttemptSpec
    process: multiprocessing.Process
    started_at: float


def _attempt_identity(*, cell_ordinal: int, trial: int, trials: int) -> int:
    if cell_ordinal < 0 or not 1 <= trial <= trials:
        raise ValueError(
            f"invalid attempt identity: cell_ordinal={cell_ordinal}, "
            f"trial={trial}, trials={trials}"
        )
    return cell_ordinal * trials + trial - 1


def _sampling_seed(*, suite_manifest_sha256: str, cell_id: str, trial: int) -> int:
    if re.fullmatch(r"[0-9a-f]{64}", suite_manifest_sha256) is None:
        raise ValueError("suite manifest digest must be 64 lowercase hexadecimal characters")
    if not cell_id or trial < 1:
        raise ValueError(f"invalid sampling seed identity: cell={cell_id!r}, trial={trial}")
    material = (
        _SAMPLING_SEED_DOMAIN
        + suite_manifest_sha256.encode()
        + b"\0"
        + cell_id.encode()
        + b"\0"
        + str(trial).encode()
    )
    seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF
    return seed or 1


def _task_slug(task_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", task_id.lower()).strip("-")


def _attempt_root(runtime: _WorkerRuntime, spec: _AttemptSpec) -> Path:
    return (
        runtime.output
        / f"trial_{spec.trial:02d}"
        / f"attempt_{spec.index:03d}_{_task_slug(spec.task.id)}"
    )


def _attempt_result_path(runtime: _WorkerRuntime, spec: _AttemptSpec) -> Path:
    return _attempt_root(runtime, spec) / "attempt.json"


def _attempt_row_identity(
    runtime: _WorkerRuntime, spec: _AttemptSpec
) -> dict[str, Any]:
    return {
        "index": spec.index,
        "cell_ordinal": spec.cell_ordinal,
        "trial": spec.trial,
        "cell": spec.task.id,
        "kind": spec.task.kind,
        "attempt_wall_bound_s": spec.wall_bound_s,
        "sampling_seed": spec.sampling_seed,
        "artifact_subdir": str(_attempt_root(runtime, spec).relative_to(runtime.output)),
    }


def _infra_invalid_attempt_row(
    runtime: _WorkerRuntime,
    spec: _AttemptSpec,
    *,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        **_attempt_row_identity(runtime, spec),
        "trace_id": None,
        "success": None,
        "validity": "infra_invalid",
        "outcome": "attempt_wall_timeout"
        if error_type == "AttemptWallTimeout"
        else "attempt_process_error",
        "steps": None,
        "parse_errors": None,
        "action_errors": None,
        "executor_errors": None,
        "control_terminate": None,
        "terminate_step": None,
        "control_ok": None,
        "infra_error": {"stage": "attempt", "type": error_type, "message": message},
        "final_probe": None,
        "sampling": {
            "temperature": runtime.temperature,
            "top_p": runtime.top_p,
            "max_tokens": runtime.max_tokens,
            "seed": spec.sampling_seed,
        },
        "host": socket.gethostname(),
        "prompt_sha256": None,
        "comparable_to_sealed_baseline": None,
        "steps_detail": None,
        "model": runtime.model,
    }


def _worker_pool(runtime: _WorkerRuntime, spec: _AttemptSpec) -> dict[str, Any]:
    arm = ARMS[runtime.arm]
    attempt_root = _attempt_root(runtime, spec)
    session_kwargs: dict[str, Any] = {
        "image": str(runtime.qcow),
        "root_dir": str(attempt_root / "vm"),
        "accelerator": "kvm",
        "transport_timeout_s": _GUEST_REQUEST_TIMEOUT_S,
        "min_ready_sessions": 1,
        "max_sessions": 1,
        "max_rollouts_per_session": 1,
        "checkout_timeout_s": _POOL_CHECKOUT_TIMEOUT_S,
        "startup_timeout_s": _POOL_STARTUP_TIMEOUT_S,
    }
    for key, value in (
        ("qemu_binary", runtime.qemu),
        ("qemu_img_binary", runtime.qemu_img),
        ("smp", runtime.vm_smp),
        ("memory", runtime.vm_mem),
    ):
        if value is not None:
            session_kwargs[key] = str(value) if isinstance(value, Path) else value
    return {
        "key": f"signoflife-{runtime.arm}-attempt-{spec.index}",
        "max_node_slots": runtime.vm_slots,
        "slot_dir": str(runtime.output / "vm_slots"),
        "pool_idle_ttl_s": arm.pool.pool_idle_ttl_s,
        "acquire_timeout_s": _POOL_ACQUIRE_TIMEOUT_S,
        "reap_interval_s": arm.pool.reap_interval_s,
        "pool_target": runtime.pool_target,
        "session_kwargs": session_kwargs,
    }


def _attempt_process_main(
    runtime: _WorkerRuntime,
    spec: _AttemptSpec,
    session_ready: Any | None = None,
) -> None:
    os.setsid()
    if os.getpid() != os.getpgrp():
        raise RuntimeError("attempt worker did not establish its own process group")
    if session_ready is not None:
        session_ready.send(os.getpid())
        session_ready.close()
    attempt_root = _attempt_root(runtime, spec)
    attempt_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(attempt_root / "worker.log", encoding="utf-8"),
        ],
        force=True,
    )
    from agent.desktop import close_all_pools

    try:
        row: dict[str, Any]
        try:
            config = _eval_config(
                arm=runtime.arm,
                tier=runtime.tier,
                task_ids=[spec.task.id],
                artifacts=attempt_root / "artifacts",
                traces_dir=attempt_root / "traces",
                pool=_worker_pool(runtime, spec),
                base_url=runtime.base_url,
                temperature=runtime.temperature,
                top_p=runtime.top_p,
                max_tokens=runtime.max_tokens,
                served_model=runtime.served_model,
                seed=spec.sampling_seed,
            )
            environment = vf.Environment(config)
            traces = asyncio.run(run_eval(environment, config))
            if len(traces) != 1:
                raise RuntimeError(
                    f"attempt {spec.index} selected one cell but produced {len(traces)} traces"
                )
            row = {
                **_episode_row(traces[0], spec.trial),
                **_attempt_row_identity(runtime, spec),
                "model": runtime.model,
            }
        except Exception as error:
            _LOGGER.exception(
                "attempt worker failed: index=%d cell=%s trial=%d",
                spec.index,
                spec.task.id,
                spec.trial,
            )
            row = _infra_invalid_attempt_row(
                runtime,
                spec,
                error_type=type(error).__name__,
                message=str(error),
            )
        _atomic_json(_attempt_result_path(runtime, spec), row)
    finally:
        close_all_pools()


def _spawn_attempt_process(
    runtime: _WorkerRuntime, spec: _AttemptSpec
) -> multiprocessing.Process:
    context = multiprocessing.get_context(_WORKER_START_METHOD)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_attempt_process_main,
        args=(runtime, spec, sender),
        name=f"signoflife-{spec.index:03d}",
    )
    process.start()
    sender.close()
    poller = select.poll()
    poller.register(receiver.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
    poller.register(process.sentinel, select.POLLIN | select.POLLHUP | select.POLLERR)
    events = poller.poll(int(_ATTEMPT_SESSION_READY_TIMEOUT_S * 1000))
    try:
        if not events or not receiver.poll():
            raise RuntimeError("attempt worker did not establish its session in time")
        leader_pid = receiver.recv()
        if type(leader_pid) is not int or leader_pid != process.pid:
            raise RuntimeError(
                f"attempt worker reported invalid session leader {leader_pid!r}"
            )
        if (
            os.getsid(process.pid) != process.pid
            or os.getpgid(process.pid) != process.pid
        ):
            raise RuntimeError("attempt worker session identity changed after readiness")
    except BaseException:
        try:
            if (
                process.pid is not None
                and os.getsid(process.pid) == process.pid
                and os.getpgid(process.pid) == process.pid
            ):
                _terminate_attempt_process_session(process)
            else:
                process.terminate()
                if not _wait_for_attempt_exit(process, _SUPERVISOR_KILL_TIMEOUT_S):
                    process.kill()
                process.join()
        except ProcessLookupError:
            process.join()
        raise
    finally:
        receiver.close()
    return process


def _attempt_process_exited(process: multiprocessing.Process) -> bool:
    poller = select.poll()
    poller.register(process.sentinel, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def _wait_for_attempt_exit(
    process: multiprocessing.Process, timeout_s: float
) -> bool:
    poller = select.poll()
    poller.register(process.sentinel, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(max(0, int(timeout_s * 1000))))


def _retain_attempt_session_members(
    session_id: int,
    retained: dict[tuple[int, int], int],
) -> dict[tuple[int, int], str]:
    members: dict[tuple[int, int], str] = {}
    with os.scandir("/proc") as entries:
        for count, entry in enumerate(entries, start=1):
            if count > _PROC_MAX_ENTRIES:
                raise RuntimeError(
                    f"/proc exceeds {_PROC_MAX_ENTRIES} entries during attempt teardown"
                )
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            try:
                before = _process_identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if before[2] != session_id:
                continue
            identity = (pid, before[3])
            observed_state = before[0]
            if identity not in retained:
                if len(retained) >= _ATTEMPT_MAX_SESSION_IDENTITIES:
                    raise RuntimeError(
                        "attempt session exceeded its retained process-identity bound"
                    )
                pidfd = -1
                try:
                    pidfd = _pidfd_open(pid)
                    after = _process_identity(pid)
                except (FileNotFoundError, ProcessLookupError):
                    if pidfd >= 0:
                        os.close(pidfd)
                    continue
                except BaseException:
                    if pidfd >= 0:
                        os.close(pidfd)
                    raise
                if after[2:] != before[2:]:
                    os.close(pidfd)
                    continue
                retained[identity] = pidfd
                observed_state = after[0]
            members[identity] = observed_state
    return members


def _wait_for_reserved_session_leader(
    *,
    session_id: int,
    leader_pid: int,
    retained: dict[tuple[int, int], int],
    signal_number: int,
    timeout_s: float,
) -> dict[tuple[int, int], str]:
    deadline = time.monotonic() + timeout_s
    while True:
        members = _retain_attempt_session_members(session_id, retained)
        live = {
            identity: state
            for identity, state in members.items()
            if identity[0] != leader_pid or state != "Z"
        }
        for identity in live:
            _pidfd_send_signal(retained[identity], signal_number)
        if not live or time.monotonic() >= deadline:
            return members
        time.sleep(_SGLANG_GROUP_POLL_S)


def _wait_for_empty_session(
    session_id: int,
    retained: dict[tuple[int, int], int],
    timeout_s: float,
) -> dict[tuple[int, int], str]:
    deadline = time.monotonic() + timeout_s
    while True:
        members = _retain_attempt_session_members(session_id, retained)
        if not members or time.monotonic() >= deadline:
            return members
        time.sleep(_SGLANG_GROUP_POLL_S)


def _terminate_attempt_process_session(
    process: multiprocessing.Process, *, terminate: bool = True
) -> bool:
    if process.pid is None:
        raise RuntimeError("attempt process has no pid")
    session_id = os.getsid(process.pid)
    if (
        session_id != process.pid
        or os.getpgid(process.pid) != process.pid
        or session_id == os.getsid(0)
    ):
        raise RuntimeError(
            f"attempt process {process.pid} is not its owned session leader"
        )
    retained: dict[tuple[int, int], int] = {}
    try:
        members = _retain_attempt_session_members(session_id, retained)
        leaked_descendants = any(
            identity[0] != process.pid for identity in members
        )
        if terminate or leaked_descendants:
            members = _wait_for_reserved_session_leader(
                session_id=session_id,
                leader_pid=process.pid,
                retained=retained,
                signal_number=signal.SIGTERM,
                timeout_s=_SUPERVISOR_REAP_TIMEOUT_S,
            )
        live = {
            identity: state
            for identity, state in members.items()
            if identity[0] != process.pid or state != "Z"
        }
        if live:
            members = _wait_for_reserved_session_leader(
                session_id=session_id,
                leader_pid=process.pid,
                retained=retained,
                signal_number=signal.SIGKILL,
                timeout_s=_SUPERVISOR_KILL_TIMEOUT_S,
            )
        live = {
            identity: state
            for identity, state in members.items()
            if identity[0] != process.pid or state != "Z"
        }
        if live:
            raise RuntimeError(
                f"attempt session {session_id} survived SIGKILL: {live}"
            )
        if not _attempt_process_exited(process):
            _wait_for_reserved_session_leader(
                session_id=session_id,
                leader_pid=process.pid,
                retained=retained,
                signal_number=signal.SIGKILL,
                timeout_s=_SUPERVISOR_KILL_TIMEOUT_S,
            )
            if not _attempt_process_exited(process):
                raise RuntimeError(
                    f"attempt session leader {process.pid} did not become reapable"
                )
        process.join()
        remaining = _wait_for_empty_session(
            session_id, retained, _SUPERVISOR_KILL_TIMEOUT_S
        )
        if remaining:
            raise RuntimeError(
                f"attempt session {session_id} remains after leader reap: {remaining}"
            )
        return leaked_descendants
    finally:
        for pidfd in retained.values():
            os.close(pidfd)


def _read_attempt_row(runtime: _WorkerRuntime, spec: _AttemptSpec) -> dict[str, Any]:
    path = _attempt_result_path(runtime, spec)
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise TypeError("attempt result is not an object")
        expected = _attempt_row_identity(runtime, spec)
        mismatches = {
            key: {"expected": value, "observed": row.get(key)}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(f"attempt identity mismatch: {mismatches}")
        if row.get("validity") not in {"valid", "infra_invalid"}:
            raise ValueError(f"invalid validity {row.get('validity')!r}")
        if row["validity"] == "infra_invalid" and row.get("success") is not None:
            raise ValueError("an infrastructure-invalid attempt must have success=null")
        return row
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _infra_invalid_attempt_row(
            runtime,
            spec,
            error_type="AttemptArtifactError",
            message=str(error),
        )


def _run_attempts(
    runtime: _WorkerRuntime, specs: list[_AttemptSpec]
) -> list[dict[str, Any]]:
    pending = iter(specs)
    active: dict[int, _ActiveAttempt] = {}
    reaped: set[int] = set()
    rows: list[dict[str, Any]] = []
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < runtime.vm_slots and not exhausted:
                try:
                    spec = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                _attempt_root(runtime, spec).mkdir(parents=True, exist_ok=True)
                started_at = time.monotonic()
                try:
                    process = _spawn_attempt_process(runtime, spec)
                except (OSError, RuntimeError) as error:
                    row = _infra_invalid_attempt_row(
                        runtime,
                        spec,
                        error_type="AttemptSpawnError",
                        message=str(error),
                    )
                    _atomic_json(_attempt_result_path(runtime, spec), row)
                    rows.append(row)
                    _LOGGER.exception(
                        "attempt spawn failed: index=%d cell=%s trial=%d",
                        spec.index,
                        spec.task.id,
                        spec.trial,
                    )
                    continue
                active[spec.index] = _ActiveAttempt(
                    spec=spec,
                    process=process,
                    started_at=started_at,
                )
                _LOGGER.info(
                    "attempt start: index=%d cell=%s trial=%d pid=%s bound=%.0fs",
                    spec.index,
                    spec.task.id,
                    spec.trial,
                    process.pid,
                    spec.wall_bound_s,
                )
            finished: list[int] = []
            now = time.monotonic()
            for index, attempt in active.items():
                process = attempt.process
                spec = attempt.spec
                if _attempt_process_exited(process):
                    leaked_descendants = _terminate_attempt_process_session(
                        process, terminate=False
                    )
                    reaped.add(index)
                    if leaked_descendants:
                        row = _infra_invalid_attempt_row(
                            runtime,
                            spec,
                            error_type="AttemptDescendantLeak",
                            message="attempt process exited with live descendants",
                        )
                        _atomic_json(_attempt_result_path(runtime, spec), row)
                    elif process.exitcode == 0:
                        row = _read_attempt_row(runtime, spec)
                    else:
                        row = _infra_invalid_attempt_row(
                            runtime,
                            spec,
                            error_type="AttemptProcessExit",
                            message=f"attempt process exited with status {process.exitcode}",
                        )
                        _atomic_json(_attempt_result_path(runtime, spec), row)
                elif now - attempt.started_at >= spec.wall_bound_s:
                    _LOGGER.error(
                        "attempt wall deadline exceeded: index=%d cell=%s trial=%d "
                        "pid=%s bound=%.0fs",
                        spec.index,
                        spec.task.id,
                        spec.trial,
                        process.pid,
                        spec.wall_bound_s,
                    )
                    _terminate_attempt_process_session(process)
                    reaped.add(index)
                    row = _infra_invalid_attempt_row(
                        runtime,
                        spec,
                        error_type="AttemptWallTimeout",
                        message=f"attempt exceeded derived wall bound of {spec.wall_bound_s:.0f}s",
                    )
                    _atomic_json(_attempt_result_path(runtime, spec), row)
                else:
                    continue
                rows.append(row)
                finished.append(index)
                _LOGGER.info(
                    "attempt done: index=%d cell=%s trial=%d validity=%s success=%s",
                    spec.index,
                    spec.task.id,
                    spec.trial,
                    row["validity"],
                    row["success"],
                )
            for index in finished:
                del active[index]
            if active and not finished:
                time.sleep(_SCHEDULER_POLL_S)
    finally:
        cleanup_errors: list[BaseException] = []
        for index, attempt in active.items():
            if index not in reaped:
                try:
                    _terminate_attempt_process_session(attempt.process)
                except BaseException as error:
                    cleanup_errors.append(error)
        if cleanup_errors:
            active_error = sys.exc_info()[1]
            if active_error is None and len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            errors = (
                cleanup_errors
                if active_error is None
                else [active_error, *cleanup_errors]
            )
            raise BaseExceptionGroup("attempt scheduler cleanup failed", errors)
    return sorted(rows, key=lambda row: int(row["index"]))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.signoflife")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tier",
        default="scored",
        choices=list(TIERS),
        help="which tier to run; one run is one tier. `scored` is the calibrated "
        "set, `candidate` is the cells whose own oracle is not measured yet — a "
        "mean over both would be the uncalibrated number this gate prevents.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-index",
        type=int,
        action="append",
        default=None,
        help="zero-based cell within the tier; repeatable. Omitted = the whole tier.",
    )
    selection.add_argument("--cell", action="append", default=None, help="cell id; repeatable")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="independent draws per cell. >=3 for any model arm: the gate is "
        "single-trial only by historical accident, and one race-prone cell makes a "
        "single draw uninterpretable.",
    )
    parser.add_argument("--qcow", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, default=None)
    parser.add_argument("--qemu-img", type=Path, default=None)
    parser.add_argument("--vm-smp", type=int, default=None)
    parser.add_argument("--vm-mem", default=None)
    parser.add_argument("--vm-slots", type=int, default=1)
    parser.add_argument(
        "--vm-rollouts-per-session",
        type=int,
        choices=[1],
        default=1,
        help="fixed at 1: each supervised attempt owns and tears down its VM pool",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--sglang-python", default=None)
    parser.add_argument("--sglang-port", type=int, default=0)
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.65)
    parser.add_argument("--sglang-ready-timeout-s", type=float, default=1500.0)
    parser.add_argument(
        "--temperature", type=float, default=None, help="override the arm's own value"
    )
    parser.add_argument("--top-p", type=float, default=None, help="override the arm's own value")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="override the arm's own value"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _reject_disabled_cudnn_check()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    if API_KEY_VAR in os.environ:
        raise SystemExit(
            f"{API_KEY_VAR} is unsupported: canonical evaluation launches only the "
            "owned local no-auth SGLang server"
        )
    if _LOCAL_NO_AUTH_API_KEY_VAR in os.environ:
        raise SystemExit(
            f"reserved variable {_LOCAL_NO_AUTH_API_KEY_VAR} must be absent"
        )
    arm = ARMS[args.arm]
    scripted = arm.scripted.enabled
    negative = arm.scripted.negative
    suite = load_suite()

    tier_cells = suite.for_tier(args.tier)
    selected = list(tier_cells)
    if args.task_index is not None:
        invalid = [index for index in args.task_index if not 0 <= index < len(tier_cells)]
        if invalid:
            raise SystemExit(
                f"--task-index {invalid} outside the {args.tier!r} tier's "
                f"0..{len(tier_cells) - 1} range"
            )
        selected = [tier_cells[index] for index in args.task_index]
    if args.cell is not None:
        selected = [suite.by_id(cell) for cell in args.cell]
    off_tier = [task.id for task in selected if task.tier != args.tier]
    if off_tier:
        raise SystemExit(
            f"--cell {off_tier} is not in the {args.tier!r} tier; a run is one tier"
        )
    cell_ids = [task.id for task in selected]
    duplicates = sorted({cell for cell in cell_ids if cell_ids.count(cell) > 1})
    if duplicates:
        raise SystemExit(f"duplicate cells are not independent attempts: {duplicates}")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.vm_slots < 1:
        raise SystemExit("--vm-slots must be >= 1")
    if (
        not math.isfinite(args.sglang_ready_timeout_s)
        or args.sglang_ready_timeout_s <= 0.0
    ):
        raise SystemExit("--sglang-ready-timeout-s must be a finite value > 0")

    if not scripted and args.trials < 3:
        _LOGGER.warning(
            "model arm %s with trials=%d: a single draw cannot separate a model "
            "difference from the open_chrome window-mapping race",
            args.arm,
            args.trials,
        )
    if scripted and args.model_path:
        raise SystemExit(
            f"arm {args.arm} is scripted and never calls a model; --model-path "
            "would be recorded in the run and ignored"
        )
    if scripted and (args.temperature is not None or args.top_p is not None):
        raise SystemExit(
            f"arm {args.arm} is scripted and never calls a model; --temperature / "
            "--top-p would be recorded in the run and ignored"
        )
    temperature = arm.temperature if args.temperature is None else args.temperature
    top_p = arm.top_p if args.top_p is None else args.top_p
    max_tokens = arm.max_tokens if args.max_tokens is None else args.max_tokens
    if not scripted and temperature is None:
        raise SystemExit(
            f"arm {args.arm} names no temperature: set one on the arm or pass "
            "--temperature. An unnamed one is whatever the server defaults to, and "
            "greedy scores the decoder rather than the checkpoint"
        )
    if not scripted and top_p is None:
        raise SystemExit(
            f"arm {args.arm} names no top_p: set one on the arm or pass --top-p"
        )
    if not scripted and (not math.isfinite(temperature) or temperature < 0.0):
        raise SystemExit(f"invalid temperature {temperature!r}: expected a finite value >= 0")
    if not scripted and (not math.isfinite(top_p) or not 0.0 < top_p <= 1.0):
        raise SystemExit(f"invalid top_p {top_p!r}: expected a finite value in (0, 1]")
    if not scripted and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        raise SystemExit(f"invalid max_tokens {max_tokens!r}: expected a positive integer")
    if not scripted and args.model_path is None:
        raise SystemExit(
            f"arm {args.arm} is a model arm: pass --model-path for the registered "
            "bytes the server must attest"
        )
    if scripted and args.sglang_python is not None:
        raise SystemExit(
            f"arm {args.arm} is scripted and never launches SGLang; "
            "--sglang-python would be ignored"
        )
    if not scripted:
        if args.sglang_python is None:
            raise SystemExit("local causal evaluation requires explicit --sglang-python")
        sglang_python = Path(args.sglang_python)
        if (
            not sglang_python.is_absolute()
            or not sglang_python.is_file()
            or not os.access(sglang_python, os.X_OK)
        ):
            raise SystemExit(
                "--sglang-python must be an absolute executable regular file"
            )
    if not scripted and not 1 <= args.sglang_port <= 55535:
        raise SystemExit("model arms require --sglang-port in [1, 55535]")
    if scripted and args.sglang_port != 0:
        raise SystemExit(
            f"arm {args.arm} never launches SGLang; --sglang-port would be ignored"
        )
    if (
        not math.isfinite(args.sglang_mem_fraction)
        or not 0.0 < args.sglang_mem_fraction <= 1.0
    ):
        raise SystemExit("--sglang-mem-fraction must be finite and in (0, 1]")
    final_output = _validate_output_target(args.output)
    artifact: _ModelArtifact | None = None
    model_record: dict[str, Any] | None = None
    served_model = "scripted-no-model"
    if not scripted:
        artifact = _verify_model_artifact(args.model_path)
        served_model = artifact.served_model
        if args.arm == "phaseb_compact":
            # This arm names one historical checkpoint, not a family of compatible
            # architectures. Its registration is therefore always part of dispatch.
            verify_phaseb_provenance(args.model_path)

    suite_wall_bound_s = _suite_wall_bound_s(
        selected,
        arm=arm,
        trials=args.trials,
        vm_slots=args.vm_slots,
        local_sglang=not scripted,
        sglang_ready_timeout_s=args.sglang_ready_timeout_s,
    )
    _preflight_slurm_wall_budget(suite_wall_bound_s)

    rows: list[dict[str, Any]] = []

    specs = [
        _AttemptSpec(
            index=_attempt_identity(
                cell_ordinal=cell_ordinal,
                trial=trial,
                trials=args.trials,
            ),
            cell_ordinal=cell_ordinal,
            trial=trial,
            task=task,
            wall_bound_s=_attempt_wall_bound_s(task, arm),
            sampling_seed=(
                None
                if scripted
                else _sampling_seed(
                    suite_manifest_sha256=suite.manifest_sha256,
                    cell_id=task.id,
                    trial=trial,
                )
            ),
        )
        for cell_ordinal, task in enumerate(selected)
        for trial in range(1, args.trials + 1)
    ]

    publication = _create_uncommitted_output(final_output)
    output = publication.path

    def _run_selected(base_url: str) -> None:
        runtime = _WorkerRuntime(
            arm=args.arm,
            tier=args.tier,
            output=output,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            served_model=served_model,
            model=model_record,
            qcow=args.qcow,
            qemu=args.qemu,
            qemu_img=args.qemu_img,
            vm_smp=args.vm_smp,
            vm_mem=args.vm_mem,
            vm_slots=args.vm_slots,
            pool_target=POOL_TARGET,
        )
        rows.extend(_run_attempts(runtime, specs))

    try:
        if scripted:
            _run_selected("http://127.0.0.1:1/v1")
        else:
            assert artifact is not None
            with tempfile.TemporaryDirectory(prefix="juergen-sglang-") as temporary:
                log_path = Path(temporary) / "sglang.log"
                try:
                    with _sglang(
                        python=args.sglang_python,
                        model_path=args.model_path,
                        log_path=log_path,
                        port=args.sglang_port,
                        mem_fraction_static=args.sglang_mem_fraction,
                        ready_timeout_s=args.sglang_ready_timeout_s,
                        served_model=served_model,
                    ) as server:
                        reverified = _verify_model_artifact(args.model_path)
                        if reverified != artifact:
                            raise RuntimeError(
                                "model artifact changed while sglang was loading it"
                            )
                        attestation = _attest_local_server(
                            server.base_url, artifact=artifact
                        )
                        attestation["launch"] = server.launch
                        attestation["seeded_sampling_probe"] = _probe_seeded_sampling(
                            server.base_url,
                            served_model=artifact.served_model,
                            timeout_s=arm.model_request_timeout_s,
                        )
                        model_record = artifact.record(attestation=attestation)
                        _run_selected(server.base_url)
                finally:
                    if log_path.is_file():
                        shutil.copy2(log_path, output / "sglang.log")
    except BaseException:
        publication.cleanup()
        raise

    infrastructure_errors = [
        {
            "index": row["index"],
            "trial": row["trial"],
            "cell": row["cell"],
            "error": row["infra_error"],
        }
        for row in rows
        if row["validity"] != "valid"
    ]
    infrastructure_errors.extend(
        _trial_contract_errors(
            rows, cell_ids=cell_ids, expected_trials=args.trials
        )
    )

    aggregate = _aggregate(
        rows,
        cell_ids=cell_ids,
        expected_trials=args.trials,
        scripted=scripted,
        negative=negative,
    )
    result = {
        "schema_version": 3,
        "arm": args.arm,
        "arm_id": arm.id,
        "arm_kind": "scripted_negative" if negative else "scripted_oracle" if scripted else "model",
        "claim_scope": "control_calibration" if scripted else "local_causal_seeded",
        "codec": arm.codec,
        "history_policy": arm.history.name,
        "trials": args.trials,
        "suite_id": suite.suite_id,
        "suite_role": suite.role,
        "final_benchmark": suite.final_benchmark,
        "suite_manifest_sha256": suite.manifest_sha256,
        "suite_scored_sha256": suite.scored_sha256,
        "tier": args.tier,
        "selection": {"task_ids": cell_ids, "full_tier_task_count": len(tier_cells)},
        "suite_wall_bound_s": suite_wall_bound_s,
        "status": "complete" if not infrastructure_errors else "infrastructure_failure",
        "promotion_evidence": {
            "status": "unregistered",
            "eligible": False,
            "required_receipt": "labctl_db_bound_exhaustive_result_receipt_v1",
            "note": (
                "RESULT_COMMITTED.json proves atomic transport completion only. "
                "Promotion requires a separately authorized Labctl DB record that "
                "binds this exhaustive generation inventory."
            ),
        },
        "aggregate": aggregate,
        "model": model_record,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed_contract": (
                None
                if scripted
                else {
                    "schema_version": 1,
                    "domain": _SAMPLING_SEED_DOMAIN.decode().rstrip("\0"),
                    "identity": ["suite_manifest_sha256", "cell", "trial"],
                    "arm_independent": True,
                    "bitwise_cross_hardware_determinism": False,
                }
            ),
        },
        "vm": {
            "qcow": str(args.qcow),
            "qemu": str(args.qemu) if args.qemu else None,
            "smp": args.vm_smp,
            "memory": args.vm_mem,
            "slots": args.vm_slots,
            "sessions_per_worker": 1,
            "rollouts_per_session": args.vm_rollouts_per_session,
            "worker_start_method": _WORKER_START_METHOD,
            "attempt_workers": args.vm_slots,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "labctl_run_id": os.environ.get("LABCTL_RUN_ID"),
        },
        "indicators_note": (
            "A/B/C/D do not exist in this schema. The eov3 indicator set is 16 flat "
            "named keys and is not computed here; what this file carries per episode "
            "is parse_errors / action_errors / executor_errors. The over-submission "
            "indicator is structurally zero on this suite regardless -- see "
            "evals/signoflife/suite.py, which is the one place that classification "
            "lives and the only module allowed to name it."
        ),
        "baseline_note": (
            "Re-baseline, not a reproduction. codec.describe() is not byte-identical "
            "to the sealed prompts, so every episode carries "
            "comparable_to_sealed_baseline=false and a difference from a sealed "
            "number is not a regression."
        ),
        "infrastructure_errors": infrastructure_errors,
        "episodes": rows,
    }
    try:
        _atomic_json(output / "result.json", result)
        publication.publish(forbidden_values=())
        if read_committed_result(final_output) != result:
            raise RuntimeError("fresh committed-output readback changed result.json")
    except BaseException:
        publication.cleanup()
        raise
    print(
        json.dumps(
            {
                "arm": args.arm,
                "tier": args.tier,
                "episodes": aggregate["episodes"],
                "valid_episodes": aggregate["valid_episodes"],
                **{k: v["pass_rate"] for k, v in aggregate["per_cell"].items()},
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if infrastructure_errors:
        return 3
    if scripted and aggregate["controls_ok"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
