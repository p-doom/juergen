#!/usr/bin/env python3
"""Recover registration after the step-900 checkpoint sealed but finalizer clock failed."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import tensorstore as ts

import train_contract as contract


TRAIN_JOB = "135586"
EXPECTED_SOURCE = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
    "franz.srambical/phaseb_normalized_v2_A_to_A_r256_s900_"
    "production_control_v1_run_019fb5faf966715194bd16bfeee051cd"
)
EXPECTED_START = "2026-07-31T04:22:50+02:00"
EXPECTED_END = "2026-07-31T09:26:36+02:00"
EXPECTED_NODE = "hai003"
STEP_LINE = re.compile(
    r"time=(\S+\s+\S+) step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lifecycle() -> dict[str, Any]:
    proc = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", TRAIN_JOB, "-o",
         "JobIDRaw,State,ExitCode,Start,End,NodeList,ElapsedRaw"],
        text=True, capture_output=True, check=True,
    )
    rows = [line.split("|") for line in proc.stdout.splitlines()
            if line.startswith(f"{TRAIN_JOB}|")]
    expected = [TRAIN_JOB, "FAILED", "1:0", EXPECTED_START[:-6],
                EXPECTED_END[:-6], EXPECTED_NODE, "18226"]
    if rows != [expected]:
        raise RuntimeError(f"training lifecycle changed: {rows}")
    queued = subprocess.run(
        ["squeue", "-h", "-j", TRAIN_JOB, "-o", "%i|%T|%R"],
        text=True, capture_output=True, check=False,
    )
    if queued.stdout.strip():
        raise RuntimeError(f"training job is still queued: {queued.stdout.strip()}")
    return {
        "job_id": TRAIN_JOB,
        "state": "FAILED",
        "exit_code": "1:0",
        "start": EXPECTED_START,
        "end": EXPECTED_END,
        "node": EXPECTED_NODE,
        "elapsed_s": 18226,
    }


async def read_counters(checkpoint: Path) -> dict[str, int]:
    root = checkpoint / "train_state"
    kvstore = {"driver": "ocdbt", "base": {"driver": "file", "path": str(root)}}
    names = {
        "global_gradient_step": "optimizer.opt_state.gradient_step.value",
        "adam_count_0": "optimizer.opt_state.inner_opt_state.1.0.count.value",
        "adam_count_2": "optimizer.opt_state.inner_opt_state.1.2.count.value",
        "gradient_accumulation_remainder": "optimizer.opt_state.mini_step.value",
        "optimizer_micro_step": "optimizer.step.value",
    }
    result = {}
    for key, name in names.items():
        array = await ts.open(
            {"driver": "zarr", "kvstore": kvstore, "path": name}, open=True
        )
        if tuple(array.shape):
            raise RuntimeError(f"counter {name} is no longer scalar: {array.shape}")
        result[key] = int((await array.read()).item())
    return result


def hardlink_tree(source: Path, destination: Path) -> dict[str, Any]:
    subprocess.run(["cp", "-al", str(source), str(destination)], check=True)
    files = 0
    bytes_total = 0
    for source_file in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / source_file.relative_to(source)
        left, right = source_file.stat(), target.stat()
        if left.st_dev != right.st_dev or left.st_ino != right.st_ino:
            raise RuntimeError(f"hardlink parity failed: {source_file}")
        files += 1
        bytes_total += left.st_size
    if not files:
        raise RuntimeError("source Orbax tree is empty")
    return {"file_count": files, "logical_bytes": bytes_total, "inode_parity": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-model", "--source_model", type=Path, required=True)
    parser.add_argument("--failed-output", "--failed_output", type=Path, required=True)
    parser.add_argument("--training-log", "--training_log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.failed_output.resolve()
    if source != Path(EXPECTED_SOURCE).resolve() or source.is_symlink():
        raise RuntimeError(f"unexpected failed training output: {source}")
    if (source / "train_manifest.json").exists():
        raise RuntimeError("failed output unexpectedly has train_manifest.json")
    job = lifecycle()

    log = args.training_log.read_text(encoding="utf-8", errors="replace")
    rows = [(stamp, int(step), float(loss), float(grad))
            for stamp, step, loss, grad in STEP_LINE.findall(log)]
    if not rows or max(row[1] for row in rows) != 900:
        raise RuntimeError("training log does not terminate at optimizer step 900")
    step900 = [row for row in rows if row[1] == 900]
    if len(step900) != 1 or not all(math.isfinite(value) for value in step900[0][2:]):
        raise RuntimeError(f"step-900 metrics are not uniquely finite: {step900}")
    log_proof = (
        "Finished saving checkpoint (finalized tmp dir)" in log
        and f"{source}/orbax/000900`" in log
        and "finished step=900 loss=0.0030" in log
        and "FATAL normalized-v2 production-control contract: actual start" in log
    )
    if not log_proof:
        raise RuntimeError("training log lacks atomic-finalization/finalizer-failure proof")

    source_orbax = source / "orbax"
    if list(source_orbax.glob("*.orbax-checkpoint-tmp")):
        raise RuntimeError("temporary checkpoint remains")
    checkpoint = source_orbax / "000900"
    metadata_path = checkpoint / "_CHECKPOINT_METADATA"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("commit_timestamp_nsecs", 0)
            <= metadata.get("init_timestamp_nsecs", 0)):
        raise RuntimeError(f"step-900 commit metadata is invalid: {metadata}")
    counters = asyncio.run(read_counters(checkpoint))
    expected_counters = {
        "optimizer_micro_step": 7200,
        "global_gradient_step": 900,
        "gradient_accumulation_remainder": 0,
        "adam_count_0": 900,
        "adam_count_2": 900,
    }
    if counters != expected_counters:
        raise RuntimeError(f"step-900 optimizer counters changed: {counters}")

    preflight = contract.validate(
        args.dataset,
        args.source_model,
        attested_slurm_start=EXPECTED_START,
        attested_slurm_node=EXPECTED_NODE,
    )
    saved_preflight = contract.load(source / "training_preflight.json")
    if saved_preflight != preflight:
        raise RuntimeError("dataset/config/hash preflight differs from original training")
    if contract.load(source / "preregistration.json") != contract.PREREGISTRATION:
        raise RuntimeError("original preregistration changed")

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("recovery output is not fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "preregistration.json", args.output / "preregistration.json")
    shutil.copy2(source / "training_preflight.json", args.output / "training_preflight.json")
    clone = hardlink_tree(source_orbax, args.output / "orbax")
    contract.finalize(args.dataset, args.source_model, args.output, preflight)
    manifest_path = args.output / "train_manifest.json"
    manifest = contract.load(manifest_path)
    manifest["slurm_job_id"] = TRAIN_JOB
    manifest["node"] = EXPECTED_NODE
    manifest["manifest_recovery"] = {
        "status": "pass",
        "science_change": False,
        "training_job": job,
        "recovery_job_id": os.environ.get("SLURM_JOB_ID"),
        "recovery_node": os.environ.get("SLURMD_NODENAME"),
        "reason": (
            "original finalizer re-ran the actual-start gate using finalizer wall-clock "
            "09:26:36 instead of original Slurm start 04:22:50"
        ),
        "step900_metrics": {"loss": step900[0][2], "grad_norm": step900[0][3]},
        "step900_optimizer_counters": counters,
        "step900_checkpoint_metadata_sha256": sha256(metadata_path),
        "atomic_commit_verified": True,
        "dataset_config_hash_preflight_revalidated": True,
        "orbax_hardlink_clone": clone,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
