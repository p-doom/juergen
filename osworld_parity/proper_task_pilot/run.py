#!/usr/bin/env python3
"""Run one frozen OSWorld proper-task arm and publish a hardened artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import verifiers.v1 as vf

import rl.osworld.harness_task as harness_task_module
from rl.osworld.config import OSWorldDesktopRuntimeConfig
from rl.osworld.desktop.pool import DesktopPoolConfig
from rl.osworld.harness_task import OSWorldTaskHarness, OSWorldTaskHarnessConfig
from rl.osworld.task_loading import load_json
from rl.osworld.taskset import OSWorldTaskData
from rl.osworld.taskset_task import (
    OSWORLD_FULL_SUCCESS_THRESHOLD,
    OSWORLD_TASK_RESULT_KEY,
    OSWorldRealState,
    OSWorldRealTask,
)

SCHEMA_VERSION = 1
CANONICAL_PILOT_SHA256 = (
    "53c6750ec8bbc9d1705ea770bcc1c8216c028a88d008d45c040802e1fd96a100"
)
EXPECTED_ESTIMAND = (
    "Descriptive proper-task performance parity between the best current absolute "
    "r32 checkpoint and the selected relative r256 checkpoint; not a causal "
    "equal-rank action-format effect."
)
PROBE_ESTIMAND = (
    "Preregistered stochastic best-of-8 data-yield probe of the normalized "
    "move_rel intermediate policy on the frozen 12 train-only proper tasks."
)
PROBE_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
TRUSTED_MARKER = "run_manifest.json"
RUNTIME_PROVENANCE_FILES = (
    "rl/runtime/ports.py",
    "rl/osworld/config.py",
    "rl/osworld/harness_task.py",
    "rl/osworld/task_loading.py",
    "rl/osworld/taskset.py",
    "rl/osworld/taskset_task.py",
    "rl/osworld/absolute_system_prompt.txt",
    "rl/osworld/desktop/deployment.py",
    "rl/osworld/desktop/factory.py",
    "rl/osworld/desktop/pool.py",
    "rl/osworld/desktop/proxy.py",
    "rl/osworld/desktop/qemu_snapshot.py",
    "rl/osworld/desktop/readiness.py",
    "rl/computer_use/__init__.py",
    "rl/computer_use/actions.py",
    "rl/computer_use/parsing.py",
    "rl/computer_use/tools.py",
)


class PilotError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunConfig:
    mode: Literal[
        "smoke", "pilot", "pilot_shard", "probe_seed", "probe_continuation"
    ]
    arm: str
    action_format: Literal["absolute", "move_rel"]
    checkpoint: Path
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str
    expected_lora_rank: int
    expected_lora_alpha: int
    runtime_repo: Path
    runtime_files_sha256: str
    tasks_file: Path
    tasks_sha256: str
    canonical_tasks_file: Path
    reverse_tasks: bool
    shard_index: int | None
    shard_count: int | None
    task_base: Path
    train_split: Path
    train_split_sha256: str
    heldout_split: Path
    heldout_split_sha256: str
    output: Path
    base_url: str
    model: str
    api_key: str
    qcow_path: Path
    qemu_bin: Path
    provider_source: Path
    provider_sha256: str
    port_lock_dir: Path
    port_base: int
    osworld_root: Path
    host_python: Path
    apptainer_image: Path
    screen_width: int
    screen_height: int
    snapshot_name: str
    max_steps: int
    n_history_frames: int
    pause: float
    server_max_model_len: int
    max_completion_tokens: int
    temperature: float
    top_p: float
    sampling_seed: int | None
    gate_absolute_manifest: Path | None
    gate_absolute_manifest_sha256: str | None
    gate_absolute_payload_sha256: str | None
    gate_relative_manifest: Path | None
    gate_relative_manifest_sha256: str | None
    gate_relative_payload_sha256: str | None
    continuation_start_index: int | None
    continuation_parent: Path | None
    continuation_parent_failure_sha256: str | None
    continuation_parent_result0_sha256: str | None
    continuation_parent_trace0_sha256: str | None
    continuation_parent_result1_sha256: str | None
    continuation_parent_trace1_sha256: str | None


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return lowered == "true"


def parse_config(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "pilot",
            "pilot_shard",
            "probe_seed",
            "probe_continuation",
        ),
        required=True,
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--action_format", choices=("absolute", "move_rel"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint_manifest", type=Path, required=True)
    parser.add_argument("--checkpoint_manifest_sha256", required=True)
    parser.add_argument("--expected_lora_rank", type=int, required=True)
    parser.add_argument("--expected_lora_alpha", type=int, required=True)
    parser.add_argument("--runtime_repo", type=Path, required=True)
    parser.add_argument("--runtime_files_sha256", required=True)
    parser.add_argument("--tasks_file", type=Path, required=True)
    parser.add_argument("--tasks_sha256", required=True)
    parser.add_argument("--canonical_tasks_file", type=Path, required=True)
    parser.add_argument("--reverse_tasks", type=_bool, required=True)
    parser.add_argument("--shard_index", type=int)
    parser.add_argument("--shard_count", type=int)
    parser.add_argument("--task_base", type=Path, required=True)
    parser.add_argument("--train_split", type=Path, required=True)
    parser.add_argument("--train_split_sha256", required=True)
    parser.add_argument("--heldout_split", type=Path, required=True)
    parser.add_argument("--heldout_split_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--api_key", default="x")
    parser.add_argument("--qcow_path", type=Path, required=True)
    parser.add_argument("--qemu_bin", type=Path, required=True)
    parser.add_argument("--provider_source", type=Path, required=True)
    parser.add_argument("--provider_sha256", required=True)
    parser.add_argument("--port_lock_dir", type=Path, required=True)
    parser.add_argument("--port_base", type=int, required=True)
    parser.add_argument("--osworld_root", type=Path, required=True)
    parser.add_argument("--host_python", type=Path, required=True)
    parser.add_argument("--apptainer_image", type=Path, required=True)
    parser.add_argument("--screen_width", type=int, required=True)
    parser.add_argument("--screen_height", type=int, required=True)
    parser.add_argument("--snapshot_name", required=True)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--n_history_frames", type=int, default=4)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--server_max_model_len", type=int, required=True)
    parser.add_argument("--max_completion_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sampling_seed", type=int)
    parser.add_argument("--gate_absolute_manifest", type=Path)
    parser.add_argument("--gate_absolute_manifest_sha256")
    parser.add_argument("--gate_absolute_payload_sha256")
    parser.add_argument("--gate_relative_manifest", type=Path)
    parser.add_argument("--gate_relative_manifest_sha256")
    parser.add_argument("--gate_relative_payload_sha256")
    parser.add_argument("--continuation_start_index", type=int)
    parser.add_argument("--continuation_parent", type=Path)
    parser.add_argument("--continuation_parent_failure_sha256")
    parser.add_argument("--continuation_parent_result0_sha256")
    parser.add_argument("--continuation_parent_trace0_sha256")
    parser.add_argument("--continuation_parent_result1_sha256")
    parser.add_argument("--continuation_parent_trace1_sha256")
    values = vars(parser.parse_args(argv))
    values["action_format"] = values.pop("action_format")
    return RunConfig(**values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seal_payload(payload: dict[str, Any]) -> None:
    payload.pop("payload_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def _publish_runtime_invalid(
    config: RunConfig,
    payload: dict[str, Any],
    runtime_provenance: dict[str, Any],
    *,
    stage: str,
) -> None:
    (config.output / TRUSTED_MARKER).unlink(missing_ok=True)
    payload["status"] = "infra_invalid"
    payload["artifact_valid"] = False
    payload["runtime_provenance_verification"][stage] = runtime_provenance
    payload["runtime_provenance_verification"][f"unchanged_{stage}"] = False
    _seal_payload(payload)
    _atomic_json(config.output / "infra_invalid_manifest.json", payload)
    raise PilotError("runtime modules changed during artifact publication")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PilotError(f"expected JSON object: {path}")
    return payload


def _runtime_provenance(runtime_repo: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative_path in RUNTIME_PROVENANCE_FILES:
        path = runtime_repo / relative_path
        if not path.is_file():
            raise PilotError(f"runtime provenance file is missing: {path}")
        hashes[relative_path] = _sha256(path)
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return {
        "files": hashes,
        "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _git_checkout_provenance(root: Path) -> dict[str, Any]:
    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise PilotError(f"cannot inspect git checkout {root}: {exc}") from exc

    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    worktree_diff = git("diff", "--binary", "HEAD")
    index_diff = git("diff", "--cached", "--binary", "HEAD")
    return {
        "root": str(root.resolve()),
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "worktree_diff_sha256": hashlib.sha256(worktree_diff).hexdigest(),
        "index_diff_sha256": hashlib.sha256(index_diff).hexdigest(),
    }


def _file_stat_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _task_ids(path: Path) -> list[str]:
    try:
        task_ids = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotError(f"cannot read task list {path}: {exc}") from exc
    if not task_ids or any(not task_id.strip() for task_id in task_ids):
        raise PilotError(f"task list must be non-empty without blank lines: {path}")
    if len(task_ids) != len(set(task_ids)):
        raise PilotError(f"task list contains duplicate IDs: {path}")
    return task_ids


def _split_ids(split: dict[str, Any]) -> set[str]:
    return {
        task_id
        for values in split.values()
        if isinstance(values, list)
        for task_id in values
        if isinstance(task_id, str)
    }


def _validate_probe_gate(
    config: RunConfig, canonical_order: list[str]
) -> dict[str, Any]:
    required = (
        config.gate_absolute_manifest,
        config.gate_absolute_manifest_sha256,
        config.gate_absolute_payload_sha256,
        config.gate_relative_manifest,
        config.gate_relative_manifest_sha256,
        config.gate_relative_payload_sha256,
    )
    if any(value is None for value in required):
        raise PilotError("probe_seed requires both trusted pilot gate manifests")
    absolute_path = config.gate_absolute_manifest
    relative_path = config.gate_relative_manifest
    assert absolute_path is not None and relative_path is not None
    absolute = _json_object(absolute_path)
    relative = _json_object(relative_path)
    expected = (
        (
            absolute_path,
            absolute,
            config.gate_absolute_manifest_sha256,
            config.gate_absolute_payload_sha256,
            "absolute_r32",
        ),
        (
            relative_path,
            relative,
            config.gate_relative_manifest_sha256,
            config.gate_relative_payload_sha256,
            "relative_r256",
        ),
    )
    for path, payload, manifest_sha256, payload_sha256, arm in expected:
        sealed_payload = dict(payload)
        observed_payload_sha256 = sealed_payload.pop("payload_sha256", None)
        recomputed_payload_sha256 = hashlib.sha256(
            json.dumps(sealed_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            _sha256(path) != manifest_sha256
            or observed_payload_sha256 != recomputed_payload_sha256
            or payload.get("status") != "complete"
            or payload.get("artifact_valid") is not True
            or payload.get("all_infra_valid") is not True
            or payload.get("task_count_completed") != 12
            or payload.get("task_count_expected") != 12
            or payload.get("arm") != arm
            or payload.get("payload_sha256") != payload_sha256
            or set(payload.get("task_ids") or ()) != set(canonical_order)
        ):
            raise PilotError(f"probe gate pilot is not trusted and complete: {path}")
    absolute_rewards = {
        task["task_id"]: float(task["raw_reward"]) for task in absolute["tasks"]
    }
    relative_rewards = {
        task["task_id"]: float(task["raw_reward"]) for task in relative["tasks"]
    }
    absolute_successes = sum(value >= 0.999 for value in absolute_rewards.values())
    relative_successes = sum(value >= 0.999 for value in relative_rewards.values())
    paired_mean_gap = sum(
        absolute_rewards[task_id] - relative_rewards[task_id]
        for task_id in canonical_order
    ) / len(canonical_order)
    if absolute_successes - relative_successes < 2 and paired_mean_gap < 0.15:
        raise PilotError("preregistered probe materiality gate is closed")
    return {
        "absolute_manifest": str(absolute_path.resolve()),
        "absolute_manifest_sha256": _sha256(absolute_path),
        "absolute_payload_sha256": config.gate_absolute_payload_sha256,
        "relative_manifest": str(relative_path.resolve()),
        "relative_manifest_sha256": _sha256(relative_path),
        "relative_payload_sha256": config.gate_relative_payload_sha256,
        "absolute_successes": absolute_successes,
        "relative_successes": relative_successes,
        "paired_mean_absolute_minus_relative": paired_mean_gap,
        "materiality_gate_open": True,
    }


def _validate_probe_continuation_parent(
    config: RunConfig, canonical_order: list[str]
) -> dict[str, Any]:
    """Seal the failed seed-503 attempt and its two reusable valid cells."""
    parent = config.continuation_parent
    expected_hashes = (
        config.continuation_parent_failure_sha256,
        config.continuation_parent_result0_sha256,
        config.continuation_parent_trace0_sha256,
        config.continuation_parent_result1_sha256,
        config.continuation_parent_trace1_sha256,
    )
    if parent is None or any(value is None for value in expected_hashes):
        raise PilotError("probe_continuation requires the sealed failed parent")
    if config.sampling_seed != 503 or config.continuation_start_index != 2:
        raise PilotError("only the exact seed-503 continuation from index 2 is authorized")
    failure_path = parent / "failure.json"
    paths = [failure_path]
    for index in (0, 1):
        task_dir = parent / "tasks" / f"{index:02d}_{canonical_order[index]}"
        paths.extend((task_dir / "result.json", task_dir / "trace.json"))
    observed_hashes = [_sha256(path) for path in paths]
    if observed_hashes != list(expected_hashes):
        raise PilotError("seed-503 failed-parent evidence seal mismatch")
    failure = _json_object(failure_path)
    if (
        failure.get("status") != "failed"
        or failure.get("artifact_valid") is not False
        or failure.get("error_type") != "TaskError"
        or "infrastructure-invalid" not in str(failure.get("message"))
        or "no byte screenshot" not in str(failure.get("message"))
    ):
        raise PilotError("seed-503 parent is not the reviewed screenshot infrastructure failure")
    if (parent / TRUSTED_MARKER).exists():
        raise PilotError("seed-503 failed parent unexpectedly has a trusted marker")
    result_files = sorted((parent / "tasks").glob("*/result.json"))
    if result_files != [paths[1], paths[3]]:
        raise PilotError("seed-503 parent valid-result set changed")
    for index, result_path, trace_path in (
        (0, paths[1], paths[2]),
        (1, paths[3], paths[4]),
    ):
        result = _json_object(result_path)
        trace = _json_object(trace_path)
        if (
            result.get("mode") != "probe_seed"
            or result.get("arm") != "relative_r256"
            or result.get("action_format") != "move_rel"
            or result.get("order_index") != index
            or result.get("task_id") != canonical_order[index]
            or result.get("infra_valid") is not True
            or result.get("infra_error") is not None
            or result.get("trace_error") is not None
            or result.get("sampling")
            != {"temperature": 0.7, "top_p": 0.95, "seed": 503}
            or trace.get("errors") != []
            or trace.get("is_completed") is not True
            or trace.get("state", {}).get("infra_valid") is not True
        ):
            raise PilotError(f"seed-503 parent cell {index} is not reusable and valid")
    return {
        "path": str(parent.resolve()),
        "failure_sha256": observed_hashes[0],
        "failure_classification": "screenshot_transport_infrastructure_invalid",
        "reused_valid_canonical_indices": [0, 1],
        "excluded_infra_attempt_canonical_index": 2,
        "result_sha256": observed_hashes[1::2],
        "trace_sha256": observed_hashes[2::2],
    }


def _server_models(base_url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 — preflight must fail closed
        raise PilotError(f"OpenAI server /models preflight failed: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("data"):
        raise PilotError("OpenAI server /models returned no models")
    return payload


def validate_zmq_tmpdir(path: Path) -> None:
    # vLLM appends "/" plus a 36-character UUID and ZeroMQ's Unix socket
    # sockaddr path is capped at 107 bytes on Linux.
    if len(str(path)) + 1 + 36 >= 107:
        raise PilotError(f"TMPDIR is too long for vLLM ZeroMQ IPC sockets: {path}")


def validate_preflight(config: RunConfig) -> tuple[list[str], dict[str, Any]]:
    if (config.screen_width, config.screen_height) != (1920, 1080):
        raise PilotError("proper-task pilot is pinned to screen size 1920x1080")
    if config.snapshot_name != "osworld_ready":
        raise PilotError("proper-task pilot is pinned to snapshot name osworld_ready")
    if config.max_steps != 15 or config.n_history_frames != 4 or config.pause != 1.0:
        raise PilotError("rollout geometry/history/settle configuration drifted")
    if config.mode in {"probe_seed", "probe_continuation"}:
        if (
            config.temperature != 0.7
            or config.top_p != 0.95
            or config.sampling_seed not in PROBE_SEEDS
            or config.max_completion_tokens != 1024
        ):
            raise PilotError("probe sampling configuration drifted")
    elif (
        config.temperature != 0.0
        or config.top_p != 1.0
        or config.sampling_seed is not None
        or config.max_completion_tokens != 1024
    ):
        raise PilotError("sampling configuration drifted")
    if config.server_max_model_len != 16384:
        raise PilotError("OpenAI server max_model_len must be pinned to 16384")
    if config.port_lock_dir != Path("/tmp/osworld_port_locks"):
        raise PilotError("OSWorld port locks must use the pinned node-shared directory")
    if config.port_base != 30000 or config.port_base % 10:
        raise PilotError("OSWorld port base must be pinned and stride-aligned at 30000")
    if os.environ.get("OSWORLD_PORT_BASE") != str(config.port_base):
        raise PilotError("OSWORLD_PORT_BASE does not match the pinned recipe value")
    tmpdir = Path(os.environ.get("TMPDIR", ""))
    if not tmpdir.is_dir():
        raise PilotError(f"TMPDIR is not a directory: {tmpdir}")
    validate_zmq_tmpdir(tmpdir)
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise PilotError("/dev/kvm is not readable and writable")
    for label, path, executable in (
        ("checkpoint", config.checkpoint, False),
        ("runtime repo", config.runtime_repo, False),
        ("task base", config.task_base, False),
        ("OSWorld root", config.osworld_root, False),
        ("qcow", config.qcow_path, False),
        ("provider source", config.provider_source, False),
        ("Apptainer image", config.apptainer_image, False),
        ("qemu binary", config.qemu_bin, True),
        ("host Python", config.host_python, True),
    ):
        if not path.exists() or (executable and not os.access(path, os.X_OK)):
            raise PilotError(f"{label} missing or unusable: {path}")
    provider_text = config.provider_source.read_text(encoding="utf-8")
    if "snapshot=on" not in provider_text or '"-enable-kvm"' not in provider_text:
        raise PilotError("native provider lacks pinned snapshot=on/KVM launch contract")
    if _sha256(config.provider_source) != config.provider_sha256:
        raise PilotError("native provider SHA-256 mismatch")
    required_port_env = {
        "OSWORLD_APPTAINER_SERVER_PORT",
        "OSWORLD_APPTAINER_CHROMIUM_PORT",
        "OSWORLD_APPTAINER_VNC_PORT",
        "OSWORLD_APPTAINER_VLC_PORT",
    }
    if not required_port_env <= set(provider_text.split('"')):
        raise PilotError("native provider does not consume the four leased ports")
    if "_free_port" in provider_text:
        raise PilotError("native provider still contains racy free-port selection")
    if os.environ.get("OSWORLD_USE_KVM_PROVIDER") != "1":
        raise PilotError("OSWORLD_USE_KVM_PROVIDER=1 is not set")

    runtime_provenance = _runtime_provenance(config.runtime_repo)
    if runtime_provenance["tree_sha256"] != config.runtime_files_sha256:
        raise PilotError("runtime module tree SHA-256 mismatch")

    if _sha256(config.tasks_file) != config.tasks_sha256:
        raise PilotError("task-list SHA-256 mismatch")
    if _sha256(config.canonical_tasks_file) != CANONICAL_PILOT_SHA256:
        raise PilotError("canonical 12-task selection SHA-256 mismatch")
    task_ids = _task_ids(config.tasks_file)
    canonical_order = _task_ids(config.canonical_tasks_file)
    canonical_ids = set(canonical_order)
    if config.mode in {"pilot", "probe_seed", "probe_continuation"}:
        if config.tasks_sha256 != CANONICAL_PILOT_SHA256 or len(task_ids) != 12:
            raise PilotError("pilot/probe mode requires the exact frozen 12-task list")
        if config.shard_index is not None or config.shard_count is not None:
            raise PilotError("unsharded pilot/probe cannot declare shard metadata")
        if config.mode in {"probe_seed", "probe_continuation"} and config.reverse_tasks:
            raise PilotError("probe modes require the exact canonical forward task order")
    elif config.mode == "pilot_shard":
        if config.shard_count != 2 or config.shard_index not in {0, 1}:
            raise PilotError("pilot_shard requires shard_count=2 and shard_index=0 or 1")
        expected_shard = canonical_order[config.shard_index :: config.shard_count]
        if task_ids != expected_shard or len(task_ids) != 6:
            raise PilotError(
                "pilot shard does not match the deterministic canonical index assignment"
            )
    else:
        if config.shard_index is not None or config.shard_count is not None:
            raise PilotError("smoke cannot declare shard metadata")
        if len(task_ids) != 1 or set(task_ids) & canonical_ids:
            raise PilotError("smoke mode requires exactly one task outside the frozen 12")

    train_split = _json_object(config.train_split)
    heldout_split = _json_object(config.heldout_split)
    if _sha256(config.train_split) != config.train_split_sha256:
        raise PilotError("train split SHA-256 mismatch")
    if _sha256(config.heldout_split) != config.heldout_split_sha256:
        raise PilotError("held-out split SHA-256 mismatch")
    if not set(task_ids) <= _split_ids(train_split):
        raise PilotError("task list is not a subset of the OSWorld train split")
    if set(task_ids) & _split_ids(heldout_split):
        raise PilotError("held-out task ID found in task list")

    manifest = _json_object(config.checkpoint_manifest)
    expected_arm = "abstool_act" if config.action_format == "absolute" else "reltool_act"
    expected_fields = {
        "schema_version": 1,
        "artifact_type": "relative_factorial_hf_checkpoint",
        "status": "complete",
        "arm": expected_arm,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "step": 750,
        "lora_rank": config.expected_lora_rank,
        "lora_alpha": config.expected_lora_alpha,
        "hf_subdir": "hf",
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise PilotError(f"checkpoint manifest mismatch: {mismatches}")
    if _sha256(config.checkpoint_manifest) != config.checkpoint_manifest_sha256:
        raise PilotError("checkpoint manifest SHA-256 mismatch")
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (config.checkpoint / name).is_file():
            raise PilotError(f"checkpoint file missing: {config.checkpoint / name}")

    models = _server_models(config.base_url, config.api_key)
    model_entries = models.get("data")
    advertised_max_model_len = (
        model_entries[0].get("max_model_len")
        if isinstance(model_entries, list)
        and model_entries
        and isinstance(model_entries[0], dict)
        else None
    )
    if advertised_max_model_len != config.server_max_model_len:
        raise PilotError(
            "OpenAI server advertised max_model_len "
            f"{advertised_max_model_len!r}, expected {config.server_max_model_len}"
        )
    visible_gpu_ids = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible_gpu_ids) != 1:
        raise PilotError(
            "expected exactly one CUDA_VISIBLE_DEVICES allocation, found "
            f"{visible_gpu_ids}"
        )
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible_gpu_ids[0],
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(nvidia) != 1:
        raise PilotError(f"expected one allocated GPU record, found {len(nvidia)}")

    probe_gate = (
        _validate_probe_gate(config, canonical_order)
        if config.mode in {"probe_seed", "probe_continuation"}
        else None
    )
    if config.mode not in {"probe_seed", "probe_continuation"} and any(
        value is not None
        for value in (
            config.gate_absolute_manifest,
            config.gate_absolute_manifest_sha256,
            config.gate_absolute_payload_sha256,
            config.gate_relative_manifest,
            config.gate_relative_manifest_sha256,
            config.gate_relative_payload_sha256,
        )
    ):
        raise PilotError("pilot/smoke modes cannot declare probe gate manifests")
    continuation_fields = (
        config.continuation_start_index,
        config.continuation_parent,
        config.continuation_parent_failure_sha256,
        config.continuation_parent_result0_sha256,
        config.continuation_parent_trace0_sha256,
        config.continuation_parent_result1_sha256,
        config.continuation_parent_trace1_sha256,
    )
    continuation_parent = None
    if config.mode == "probe_continuation":
        continuation_parent = _validate_probe_continuation_parent(
            config, canonical_order
        )
        task_ids = task_ids[2:]
    elif any(value is not None for value in continuation_fields):
        raise PilotError("continuation metadata is forbidden outside probe_continuation")
    if config.reverse_tasks:
        task_ids.reverse()
    return task_ids, {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "kvm_read_write": True,
        "gpu": nvidia[0],
        "server_models": models,
        "server_advertised_max_model_len": advertised_max_model_len,
        "screen_size": [config.screen_width, config.screen_height],
        "snapshot": {
            "name": config.snapshot_name,
            "drive_mode": "snapshot=on",
            "provider_source": str(config.provider_source.resolve()),
            "provider_sha256": _sha256(config.provider_source),
            "expected_provider_sha256": config.provider_sha256,
        },
        "no_heldout_use": True,
        "tmpdir": str(tmpdir),
        "runtime_provenance": runtime_provenance,
        "osworld_checkout": _git_checkout_provenance(config.osworld_root),
        "qcow": _file_stat_provenance(config.qcow_path),
        "port_allocation": {
            "lock_dir": str(config.port_lock_dir),
            "base": config.port_base,
            "stride": 10,
            "shared_advisory_locks": True,
        },
        "probe_gate": probe_gate,
        "continuation_parent": continuation_parent,
    }


def _task_app(train_split: dict[str, Any], task_id: str) -> str:
    matches = [
        app
        for app, task_ids in train_split.items()
        if isinstance(task_ids, list) and task_id in task_ids
    ]
    if len(matches) != 1:
        raise PilotError(f"expected one train app for {task_id}, found {matches}")
    return matches[0]


def _trace_payload(trace: vf.Trace) -> dict[str, Any]:
    payload = trace.model_dump(mode="json")
    payload["state"] = trace.state.model_dump(mode="json")
    return payload


class _SeededCompletions:
    def __init__(self, completions: Any, *, top_p: float, seed: int) -> None:
        self._completions = completions
        self._top_p = top_p
        self._seed = seed

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        if "top_p" in kwargs or "seed" in kwargs:
            raise PilotError("probe sampling parameters were already supplied")
        return await self._completions.create(
            *args, top_p=self._top_p, seed=self._seed, **kwargs
        )


class _SeededChat:
    def __init__(self, chat: Any, *, top_p: float, seed: int) -> None:
        self._chat = chat
        self.completions = _SeededCompletions(
            chat.completions, top_p=top_p, seed=seed
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _SeededAsyncOpenAI:
    def __init__(
        self,
        client_type: Any,
        *args: Any,
        top_p: float,
        seed: int,
        **kwargs: Any,
    ) -> None:
        self._client = client_type(*args, **kwargs)
        self.chat = _SeededChat(self._client.chat, top_p=top_p, seed=seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def result_is_infra_valid(result: Any, *, trace_has_error: bool) -> bool:
    if not isinstance(result, dict) or result.get("validity") != "valid":
        return False
    reward = result.get("task_reward")
    return (
        isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
        and not trace_has_error
    )


def _desktop_pool_config(config: RunConfig) -> DesktopPoolConfig:
    return DesktopPoolConfig(
        min_ready_sessions=1,
        max_sessions=1,
        max_rollouts_per_session=1,
        checkout_timeout_s=900.0,
        lease_timeout_s=600.0,
        startup_timeout_s=840.0,
        port_lock_dir=config.port_lock_dir,
    )


async def execute(config: RunConfig) -> dict[str, Any]:
    task_ids, preflight = validate_preflight(config)
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / TRUSTED_MARKER).unlink(missing_ok=True)
    train_split = _json_object(config.train_split)

    pool_config = _desktop_pool_config(config)
    desktop_config = OSWorldDesktopRuntimeConfig(
        screen_width=config.screen_width,
        screen_height=config.screen_height,
        osworld_root=config.osworld_root,
        qcow_path=config.qcow_path,
        cache_dir=config.output / "asset_cache",
        output_dir=config.output / "runtime",
        apptainer_qemu_snapshot_name=config.snapshot_name,
        desktop_pool_config=pool_config,
    )
    if desktop_config.apptainer_image.resolve() != config.apptainer_image.resolve():
        raise PilotError("runtime Apptainer image does not match pinned recipe input")
    harness_config = OSWorldTaskHarnessConfig(
        max_steps=config.max_steps,
        n_history_frames=config.n_history_frames,
        pause=config.pause,
        rel_coord_grid=1000,
        absolute_coord_grid=1000,
        persist_instruction=True,
        action_format=config.action_format,
        require_fresh_session=True,
        validate_screen_size=True,
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
        desktop=desktop_config,
    )
    original_async_openai = harness_task_module.AsyncOpenAI
    if config.mode in {"probe_seed", "probe_continuation"}:
        assert config.sampling_seed is not None

        def seeded_client(*args: Any, **kwargs: Any) -> _SeededAsyncOpenAI:
            return _SeededAsyncOpenAI(
                original_async_openai,
                *args,
                top_p=config.top_p,
                seed=config.sampling_seed,
                **kwargs,
            )

        harness_task_module.AsyncOpenAI = seeded_client
    harness = OSWorldTaskHarness(harness_config)
    task_records: list[dict[str, Any]] = []
    artifact_invalid = False
    try:
        canonical_start = config.continuation_start_index or 0
        for local_index, task_id in enumerate(task_ids):
            order_index = canonical_start + local_index
            task_path = config.task_base / f"{task_id}.json"
            task_config = load_json(task_path)
            if task_config.get("id") != task_id:
                raise PilotError(f"task config ID mismatch: {task_path}")
            instruction = task_config.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise PilotError(f"task instruction missing: {task_path}")
            data = OSWorldTaskData(
                idx=order_index,
                name=task_id,
                prompt=instruction,
                task_id=task_id,
                instruction=instruction,
                path=str(task_path),
            )
            task = OSWorldRealTask(data)
            trace = vf.Trace(
                task=vf.TraceTask(type=type(task).__name__, data=data),
                state=OSWorldRealState(),
            )
            pool_before = harness._desktop_pool.snapshot()
            started = time.time()
            unexpected_error: dict[str, str] | None = None
            try:
                await harness.launch(
                    SimpleNamespace(model=config.model),
                    trace,
                    SimpleNamespace(),
                    config.base_url,
                    config.api_key,
                    {},
                )
            except Exception as exc:  # noqa: BLE001 — artifact records and fails
                unexpected_error = {
                    "stage": "harness",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            pool_after = harness._desktop_pool.snapshot()
            if unexpected_error is None:
                await task.score(trace)
            result = trace.info.get(OSWORLD_TASK_RESULT_KEY)
            result_valid = result_is_infra_valid(
                result, trace_has_error=trace.has_error
            )
            if not result_valid:
                artifact_invalid = True

            task_dir = config.output / "tasks" / f"{order_index:02d}_{task_id}"
            trace_path = task_dir / "trace.json"
            _atomic_json(trace_path, _trace_payload(trace))
            record = {
                "schema_version": SCHEMA_VERSION,
                "mode": config.mode,
                "arm": config.arm,
                "action_format": config.action_format,
                "order_index": order_index,
                "task_id": task_id,
                "app": _task_app(train_split, task_id),
                "instruction": instruction,
                "task_json": str(task_path.resolve()),
                "task_json_sha256": _sha256(task_path),
                "raw_reward": result.get("task_reward") if isinstance(result, dict) else None,
                "full_success": result.get("full_success") if isinstance(result, dict) else None,
                "full_success_threshold": OSWORLD_FULL_SUCCESS_THRESHOLD,
                "infra_valid": result_valid,
                "infra_error": (
                    result.get("infra_error") if isinstance(result, dict) else unexpected_error
                ),
                "parse_errors": (
                    result.get("parse_errors") if isinstance(result, dict) else None
                ),
                "sampling": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "seed": config.sampling_seed,
                },
                "runtime_status": (
                    result.get("runtime_status") if isinstance(result, dict) else None
                ),
                "session_id": trace.info.get("osworld_session_id"),
                "pool_before": pool_before,
                "pool_after": pool_after,
                "elapsed_s": round(time.time() - started, 6),
                "trace": str(trace_path.relative_to(config.output)),
                "trace_sha256": _sha256(trace_path),
                "trace_error": trace.error.model_dump(mode="json") if trace.error else None,
            }
            record_path = task_dir / "result.json"
            _atomic_json(record_path, record)
            record["result"] = str(record_path.relative_to(config.output))
            record["result_sha256"] = _sha256(record_path)
            task_records.append(record)
            print(
                f"task {order_index + 1}/{len(task_ids)} {task_id}: "
                f"infra_valid={result_valid} reward={record['raw_reward']} "
                f"full_success={record['full_success']}",
                flush=True,
            )
            if not result_valid:
                break
    finally:
        harness.close()
        harness_task_module.AsyncOpenAI = original_async_openai

    runtime_before_write = _runtime_provenance(config.runtime_repo)
    runtime_unchanged = runtime_before_write == preflight["runtime_provenance"]
    if not runtime_unchanged:
        artifact_invalid = True

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not artifact_invalid else "infra_invalid",
        "artifact_valid": not artifact_invalid,
        "mode": config.mode,
        "benchmark_data": config.mode != "smoke",
        "estimand": (
            PROBE_ESTIMAND
            if config.mode in {"probe_seed", "probe_continuation"}
            else EXPECTED_ESTIMAND
        ),
        "arm": config.arm,
        "action_format": config.action_format,
        "expected_lora_rank": config.expected_lora_rank,
        "task_selection_sha256": config.tasks_sha256,
        "canonical_pilot_selection_sha256": CANONICAL_PILOT_SHA256,
        "reverse_tasks": config.reverse_tasks,
        "shard": (
            {
                "index": config.shard_index,
                "count": config.shard_count,
                "assignment": "canonical_index_modulo_shard_count",
            }
            if config.mode == "pilot_shard"
            else None
        ),
        "task_ids": task_ids,
        "canonical_task_indices": list(
            range(
                config.continuation_start_index or 0,
                (config.continuation_start_index or 0) + len(task_ids),
            )
        ),
        "task_count_expected": len(task_ids),
        "task_count_completed": len(task_records),
        "all_infra_valid": all(record["infra_valid"] for record in task_records),
        "preflight": preflight,
        "runtime_provenance_verification": {
            "expected_tree_sha256": config.runtime_files_sha256,
            "preflight": preflight["runtime_provenance"],
            "immediately_before_artifact_write": runtime_before_write,
            "unchanged_before_write": runtime_unchanged,
            "policy": "preflight_equals_before_write_and_after_trusted_write",
        },
        "rollout_config": {
            "screen_size": [config.screen_width, config.screen_height],
            "max_steps": config.max_steps,
            "n_history_frames": config.n_history_frames,
            "pause": config.pause,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "sampling_seed": config.sampling_seed,
            "max_completion_tokens": config.max_completion_tokens,
            "server_max_model_len": config.server_max_model_len,
            "max_sessions": 1,
            "min_ready_sessions": 1,
            "max_rollouts_per_session": 1,
            "port_lock_dir": str(config.port_lock_dir),
            "port_base": config.port_base,
            "port_stride": 10,
            "fresh_vm_per_rollout": True,
            "snapshot_drive_mode": "snapshot=on",
        },
        "continuation_contract": (
            {
                "start_index": config.continuation_start_index,
                "parent": preflight["continuation_parent"],
                "policy": (
                    "reuse only sealed valid parent indices 0-1; treat the "
                    "infrastructure-invalid index-2 attempt as unobserved; run "
                    "exactly canonical indices 2-11"
                ),
            }
            if config.mode == "probe_continuation"
            else None
        ),
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "manifest": str(config.checkpoint_manifest.resolve()),
            "manifest_sha256": config.checkpoint_manifest_sha256,
            "config_sha256": _sha256(config.checkpoint / "config.json"),
            "model_safetensors_size": (config.checkpoint / "model.safetensors").stat().st_size,
        },
        "tasks": task_records,
        "labctl_context_sha256": (
            _sha256(Path(os.environ["LABCTL_CONTEXT"]))
            if os.environ.get("LABCTL_CONTEXT")
            and Path(os.environ["LABCTL_CONTEXT"]).is_file()
            else None
        ),
    }
    _seal_payload(manifest_payload)
    if artifact_invalid or len(task_records) != len(task_ids):
        _atomic_json(config.output / "infra_invalid_manifest.json", manifest_payload)
        raise PilotError("rollout artifact is infrastructure-invalid")
    _atomic_json(config.output / TRUSTED_MARKER, manifest_payload)
    runtime_after_candidate_write = _runtime_provenance(config.runtime_repo)
    if runtime_after_candidate_write != preflight["runtime_provenance"]:
        _publish_runtime_invalid(
            config,
            manifest_payload,
            runtime_after_candidate_write,
            stage="immediately_after_candidate_write",
        )
    manifest_payload["runtime_provenance_verification"][
        "immediately_after_candidate_write"
    ] = runtime_after_candidate_write
    manifest_payload["runtime_provenance_verification"][
        "unchanged_after_candidate_write"
    ] = True
    _seal_payload(manifest_payload)
    _atomic_json(config.output / TRUSTED_MARKER, manifest_payload)
    runtime_after_final_write = _runtime_provenance(config.runtime_repo)
    if runtime_after_final_write != preflight["runtime_provenance"]:
        _publish_runtime_invalid(
            config,
            manifest_payload,
            runtime_after_final_write,
            stage="immediately_after_final_write",
        )
    return manifest_payload


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / TRUSTED_MARKER).unlink(missing_ok=True)
    try:
        manifest = asyncio.run(execute(config))
    except Exception as exc:  # noqa: BLE001 — marker stays absent
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "artifact_valid": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _atomic_json(config.output / "failure.json", failure)
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        f"VALID {manifest['mode']} artifact: {config.output / TRUSTED_MARKER}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
