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
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
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
from evals.signoflife.suite import DevelopmentTask, TIERS, load_suite  # noqa: E402
from evals.tasks import RESULT_KEY  # noqa: E402
from signoflife import PLUGIN_ID  # noqa: E402

_LOGGER = logging.getLogger("signoflife")

_SERVED_MODEL_PREFIX = "sign-of-life-sha256-"
_ATTESTATION_PATH = "/model-attestation"
_ATTESTATION_TIMEOUT_S = 10.0
_ATTESTATION_MAX_BYTES = 65536
_SEED_PROBE_SEEDS = (19088743, 230973796, 427587855, 1985229328)
_SEED_PROBE_REQUESTS = len(_SEED_PROBE_SEEDS) + 1

API_KEY_VAR = "SIGN_OF_LIFE_API_KEY"
"""`resolve_api_key` reads the key from an env var named by the client config
(`clients/config.py:91-102`); it is never a CLI field, so we name our own."""

POOL_TARGET = "evals.vm:kvm_desktop_pool"
"""The production constructor. Tests replace the value before building workers."""

_POOL_ACQUIRE_TIMEOUT_S = 1800.0
_POOL_CHECKOUT_TIMEOUT_S = 1800.0
_POOL_STARTUP_TIMEOUT_S = 1200.0
_GUEST_REQUEST_TIMEOUT_S = 60.0
_QEMU_SHUTDOWN_TIMEOUT_S = 15.0
_SUPERVISOR_REAP_TIMEOUT_S = 20.0
_ATTEMPT_LAUNCH_MARGIN_S = 20.0
_SCHEDULER_POLL_S = 0.1
_WORKER_START_METHOD = "spawn"
_SAMPLING_SEED_DOMAIN = b"juergen-signoflife-sampling-seed-v1\0"
_DETERMINISTIC_ATTENTION_BACKENDS = frozenset({"fa3", "flashinfer", "triton"})
_SGLANG_VERSION = "0.5.10.post1"
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
    if port == 0:
        # sglang derives grpc_port = port + 10000 and rejects > 65535, and its own
        # warmup probes the *requested* port, so `--port 0` cannot be handed
        # through: pick a real free port in the safe sub-range.
        for _ in range(64):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                candidate = sock.getsockname()[1]
            if candidate + 10000 <= 65535:
                port = candidate
                break
        else:
            raise RuntimeError("no sglang-safe free port (port + 10000 <= 65535)")
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
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        handle.close()
        raise
    try:
        deadline = time.monotonic() + ready_timeout_s
        probe = f"http://127.0.0.1:{port}/health_generate"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = "\n".join(log_path.read_text().splitlines()[-40:])
                raise RuntimeError(
                    f"sglang exited before ready (rc={process.returncode}):\n{tail}"
                )
            try:
                with urllib.request.urlopen(probe, timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:  # noqa: BLE001 - not up yet is the normal case
                pass
            time.sleep(2.0)
        else:
            raise TimeoutError(f"sglang not ready after {ready_timeout_s}s")
        url = f"http://127.0.0.1:{port}/v1"
        _LOGGER.info("sglang ready at %s", url)
        yield _LocalServer(base_url=url, launch=launch)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        handle.close()


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
        and key != "SGLANG_DISABLE_CUDNN_CHECK"
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
    environment["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
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


def _attestation_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(f"invalid external model base URL {base_url!r}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, _ATTESTATION_PATH, "", ""))


def _attest_external_server(
    base_url: str, artifact: _ModelArtifact
) -> dict[str, Any]:
    """Require the external server to echo the exact registered artifact identity."""
    endpoint = _attestation_url(base_url)
    nonce = secrets.token_hex(32)
    api_key = os.environ.get(API_KEY_VAR)
    if not api_key:
        raise RuntimeError(f"external model server requires {API_KEY_VAR}")
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Attestation-Nonce": nonce,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_ATTESTATION_TIMEOUT_S) as response:
            raw = response.read(_ATTESTATION_MAX_BYTES + 1)
        if len(raw) > _ATTESTATION_MAX_BYTES:
            raise ValueError("attestation response exceeds 65536 bytes")
        observed = json.loads(raw)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"external model attestation failed at {endpoint}: {error}") from error
    expected = {
        "schema_version": 1,
        "nonce": nonce,
        "served_model": artifact.served_model,
        "artifact_id": artifact.artifact_id,
        "artifact_sha256": artifact.artifact_sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "config_sha256": artifact.config_sha256,
    }
    if not isinstance(observed, dict):
        raise RuntimeError("model attestation mismatch: expected a JSON object")
    if observed != expected:
        mismatches = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        extra = sorted(set(observed) - set(expected))
        if extra:
            mismatches["extra_fields"] = {"expected": [], "observed": extra}
        raise RuntimeError(f"model attestation mismatch: {mismatches}")
    return {
        "source": "external_endpoint",
        "url": endpoint,
        "artifact_sha256": artifact.artifact_sha256,
        "config_sha256": artifact.config_sha256,
        "served_model": artifact.served_model,
    }


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
    for task in tasks:
        for _trial in range(trials):
            slot = min(range(vm_slots), key=lambda index: (slot_bounds[index], index))
            slot_bounds[slot] += _attempt_wall_bound_s(task, arm)
    return max(slot_bounds, default=0.0) + (
        sglang_ready_timeout_s
        + _SEED_PROBE_REQUESTS * arm.model_request_timeout_s
        if local_sglang
        else 0.0
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


def _slurm_remaining_wall_s(job_id: str) -> float:
    completed = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = {
        key: value
        for token in completed.stdout.split()
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
        client={"base_url": base_url, "api_key_var": API_KEY_VAR},
        sampling=sampling,
        num_rollouts=1,
        # One episode per supervised worker. Parallelism belongs to the dispatcher,
        # where every process owns the one desktop its deadline may reap.
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
    rows: list[dict[str, Any]], *, cell_ids: list[str], scripted: bool, negative: bool
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
        per_cell[cell] = {
            "trials": len(draws),
            "valid_trials": len(valid),
            "passed": passed,
            "pass_rate": (passed / len(valid)) if valid else None,
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
        "expected_per_cell_pass_rate": (0.0 if negative else 1.0) if scripted else None,
        "controls_ok": conformant,
        "controls_ok_note": (
            "null for a model arm on purpose. A model arm has no expected value, so "
            "any 'controls_ok' computed from its own rows only restates the pass "
            "count; calibration comes from the separate scripted oracle/negative runs."
        ),
    }


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
    scoring_grace_s: float
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
        "lease_timeout_s": arm.pool.episode_ttl_s,
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
        "episode_ttl_s": arm.pool.episode_ttl_s,
        "scoring_grace_s": runtime.scoring_grace_s,
        "pool_idle_ttl_s": arm.pool.pool_idle_ttl_s,
        "acquire_timeout_s": _POOL_ACQUIRE_TIMEOUT_S,
        "reap_interval_s": arm.pool.reap_interval_s,
        "pool_target": runtime.pool_target,
        "session_kwargs": session_kwargs,
    }


def _attempt_process_main(runtime: _WorkerRuntime, spec: _AttemptSpec) -> None:
    os.setsid()
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
    try:
        _atomic_json(_attempt_result_path(runtime, spec), row)
    finally:
        from agent.desktop import close_all_pools

        close_all_pools()


def _spawn_attempt_process(
    runtime: _WorkerRuntime, spec: _AttemptSpec
) -> multiprocessing.Process:
    process = multiprocessing.get_context(_WORKER_START_METHOD).Process(
        target=_attempt_process_main,
        args=(runtime, spec),
        name=f"signoflife-{spec.index:03d}",
    )
    process.start()
    return process


def _terminate_attempt_process_group(process: multiprocessing.Process) -> None:
    if process.pid is None:
        raise RuntimeError("attempt process has no pid")
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process.join()
        return
    if process_group != process.pid:
        process.terminate()
        process.join(timeout=_SUPERVISOR_REAP_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(timeout=_SUPERVISOR_REAP_TIMEOUT_S)
        raise RuntimeError(
            f"attempt process {process.pid} did not establish its own process group"
        )
    os.killpg(process_group, signal.SIGTERM)
    process.join(timeout=_SUPERVISOR_REAP_TIMEOUT_S)
    if process.is_alive():
        os.killpg(process_group, signal.SIGKILL)
        process.join(timeout=_SUPERVISOR_REAP_TIMEOUT_S)
    if process.is_alive():
        raise RuntimeError(f"attempt process group {process_group} survived SIGKILL")
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return
    raise RuntimeError(f"attempt process group {process_group} still exists after reap")


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
    rows: list[dict[str, Any]] = []
    exhausted = False
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
            if not process.is_alive():
                process.join()
                row = (
                    _read_attempt_row(runtime, spec)
                    if process.exitcode == 0
                    else _infra_invalid_attempt_row(
                        runtime,
                        spec,
                        error_type="AttemptProcessExit",
                        message=f"attempt process exited with status {process.exitcode}",
                    )
                )
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
                _terminate_attempt_process_group(process)
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
    parser.add_argument(
        "--scoring-grace-s",
        type=float,
        default=120.0,
        help="how long the desktop stays leased after the episode so a "
        "runtime-declaring reward can probe live guest state. Pure wall clock per "
        "cell, and until now it was pinned in code with no way to name it.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="serve externally instead")
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
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
    if not math.isfinite(args.scoring_grace_s) or args.scoring_grace_s < 0.0:
        raise SystemExit("--scoring-grace-s must be a finite value >= 0")
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
    if scripted and (args.model_path or args.base_url):
        raise SystemExit(
            f"arm {args.arm} is scripted and never calls a model; --model-path / "
            "--base-url would be recorded in the run and ignored"
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
    artifact: _ModelArtifact | None = None
    model_record: dict[str, Any] | None = None
    served_model = "scripted-no-model"
    if not scripted:
        if args.base_url is None:
            os.environ[API_KEY_VAR] = "local-loopback-no-auth"
        elif not os.environ.get(API_KEY_VAR):
            raise SystemExit(
                f"external model server credentials must be supplied only through "
                f"the {API_KEY_VAR} environment variable"
            )
        artifact = _verify_model_artifact(args.model_path)
        served_model = artifact.served_model
        if args.arm == "phaseb_compact":
            # This arm names one historical checkpoint, not a family of compatible
            # architectures. Its registration is therefore always part of dispatch.
            verify_phaseb_provenance(args.model_path)
        if args.base_url is not None:
            attestation = _attest_external_server(args.base_url, artifact)
            model_record = artifact.record(attestation=attestation)

    local_sglang = not scripted and args.base_url is None
    suite_wall_bound_s = _suite_wall_bound_s(
        selected,
        arm=arm,
        trials=args.trials,
        vm_slots=args.vm_slots,
        local_sglang=local_sglang,
        sglang_ready_timeout_s=args.sglang_ready_timeout_s,
    )
    _preflight_slurm_wall_budget(suite_wall_bound_s)

    output: Path = args.output

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
            scoring_grace_s=args.scoring_grace_s,
            pool_target=POOL_TARGET,
        )
        rows.extend(_run_attempts(runtime, specs))

    if scripted or args.base_url:
        output.mkdir(parents=True, exist_ok=True)
        _run_selected(args.base_url or "http://127.0.0.1:1/v1")
    else:
        assert artifact is not None
        if args.sglang_python is None:
            raise SystemExit("local causal evaluation requires explicit --sglang-python")
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
                        raise RuntimeError("model artifact changed while sglang was loading it")
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
                    output.mkdir(parents=True, exist_ok=True)
                    _run_selected(server.base_url)
            finally:
                if output.is_dir() and log_path.is_file():
                    shutil.copy2(log_path, output / "sglang.log")

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

    aggregate = _aggregate(rows, cell_ids=cell_ids, scripted=scripted, negative=negative)
    result = {
        "schema_version": 3,
        "arm": args.arm,
        "arm_id": arm.id,
        "arm_kind": "scripted_negative" if negative else "scripted_oracle" if scripted else "model",
        "claim_scope": (
            "control_calibration"
            if scripted
            else "external_diagnostic_non_causal"
            if args.base_url
            else "local_causal_seeded"
        ),
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
    _atomic_json(output / "result.json", result)
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
