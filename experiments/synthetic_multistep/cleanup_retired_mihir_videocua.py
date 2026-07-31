#!/usr/bin/env python3
"""Delete one exact, retired Mihir/VideoCUA checkpoint tree after sealed audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


CHECKPOINT_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"
)
TARGET_ALIAS = (
    "bc_qwen3vl8b_lora_videocua_diffabs_mihiractions_"
    "syspromptaligned_16k_truncate_v1_run_019f947d2cd375d1893f115a00d43266"
)
TARGET = CHECKPOINT_ROOT / TARGET_ALIAS
EXPECTED_ALLOCATED_BYTES = 730_454_450_176
EXPECTED_LOGICAL_BYTES = 547_690_791_299
EXPECTED_FILE_COUNT = 1_213
EXPECTED_DIR_COUNT = 403
EXPECTED_INVENTORY_SHA256 = (
    "1814a1a22c095e65dc6f1641dca4440d436e5431267b00734a728003548743b3"
)
PRODUCER_JOB_ID = "131143"
PRODUCER_RUN_ID = "run_019f947d2cd375d1893f115a00d43266"
RUNS_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs/franz.srambical"
)
MEMORY_NOTE = Path(
    "/fast/home/franz.srambical/.claude/projects/-fast-home-franz-srambical/"
    "memory/project_videocua_format_defects.md"
)
MEMORY_NOTE_SHA256 = "bd0435528544f31f5681d4081f06c44259f2d80d9150b2db1daa7bc05262b918"
EXPECTED_STEPS = {
    f"{step:06d}"
    for step in range(1_125, 34_875 + 1, 1_125)
}
EXPECTED_TOP_FILES = {"config.json", "lora_metadata.json", "memory.log"}
EXPECTED_ROOT_FILE_HASHES = {
    "config.json": "431c3209d531f04bb383751c91c7477a8e7c66c717dfd1584464be7e35b0dbd3",
    "lora_metadata.json": "877c7be33455d87488588ca52742d5a5ae053d52a2b5b4ef083c04d5eb47cb3c",
    "memory.log": "5a68bef0929e3c28e346cead4f867b5f38119ae12d5f3ee44909de3e4d696fc3",
}
REFERENCE_RUNS = {
    PRODUCER_RUN_ID: {
        "job_id": PRODUCER_JOB_ID,
        "context_sha256": "f7b6646b08aebd41b1c3c331dcf40d7740411bd26e53e122a709b422b916d142",
        "log": "bc_qwen3vl8b_lora_videocua_diffabs_mihiractions_syspromptaligned_131143.log",
        "log_sha256": "aaf897cddc027fc68df5392d1e8c413041b6cbedfb1c120f40be5f76fb2f6167",
        "state": "FAILED",
    },
    "run_019f9987b49375e1bf67c37a31ba3ae3": {
        "job_id": "131661",
        "context_sha256": "b220389db06869545cc6de323fca27f24ec20155fe7275a57ec62ce9e8afc4e5",
        "log": "bc_export_hf_per_checkpoint_8b_lora_v1_131661.log",
        "log_sha256": "4a187997152947465ece86b5f9cb4beb70c3e24e73e5d1b1a5119debdac9c9da",
        "state": "COMPLETED",
    },
    "run_019f9987c21771e09a33b26adc4818f7": {
        "job_id": "131662",
        "context_sha256": "7c5b4fd42d6acdbd06a86b3600d39411c032c40e24bff1f9c61364cf6c38706f",
        "log": "bc_export_hf_per_checkpoint_8b_lora_v1_131662.log",
        "log_sha256": "0e644760e0422c2dd072303a31992708d5742d42fb7b5a80b6a12cf2b91e2f22",
        "state": "COMPLETED",
    },
    "run_019f9987ce1275e1a66c408c3ca29704": {
        "job_id": "131663",
        "context_sha256": "21008345dc619b184e184158824a1af66772edcaa150407e1cc785b475e154f8",
        "log": "bc_export_hf_per_checkpoint_8b_lora_v1_131663.log",
        "log_sha256": "b197bce2a095425428acfb96f91b62c57dbb946b8a199178ab797206b56ec1d9",
        "state": "COMPLETED",
    },
}
EXPORT_OUTPUTS = {
    "run_019f9987b49375e1bf67c37a31ba3ae3": CHECKPOINT_ROOT
    / "bc_export_hf_8b_lora_artifact_da080c310ee41df6",
    "run_019f9987c21771e09a33b26adc4818f7": CHECKPOINT_ROOT
    / "bc_export_hf_8b_lora_artifact_a1d6978442472076",
    "run_019f9987ce1275e1a66c408c3ca29704": CHECKPOINT_ROOT
    / "bc_export_hf_8b_lora_artifact_29051505182b85ff",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def run_checked(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def tree_inventory(root: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    logical = allocated = files = directories = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name, kind in [*((name, "d") for name in dirnames), *((name, "f") for name in filenames)]:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink inside exact deletion target: {path}")
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            digest.update(
                f"{kind}\0{relative}\0{stat.st_size}\0{stat.st_blocks}\0{stat.st_mode:o}\n".encode()
            )
            allocated += stat.st_blocks * 512
            if kind == "f":
                files += 1
                logical += stat.st_size
            else:
                directories += 1
    allocated += root.stat().st_blocks * 512
    return {
        "file_count": files,
        "directory_count": directories,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "path_size_inventory_sha256": digest.hexdigest(),
    }


def run_evidence_hashes() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for run_id, expected in REFERENCE_RUNS.items():
        lab = RUNS_ROOT / run_id / ".lab"
        context = lab / "context.json"
        log = lab / expected["log"]
        actual = {"context_sha256": sha256(context), "log_sha256": sha256(log)}
        for key, value in actual.items():
            if value != expected[key]:
                raise RuntimeError(f"retained {key} mismatch for {run_id}: {value}")
        result[run_id] = actual
    return result


def audit_context_lineage() -> dict[str, dict[str, object]]:
    references: dict[str, dict[str, object]] = {}
    for context_path in RUNS_ROOT.glob("run_*/.lab/context.json"):
        raw = context_path.read_text()
        if str(TARGET) not in raw:
            continue
        context = json.loads(raw)
        references[context["run_id"]] = {
            "recipe_name": context["recipe_name"],
            "inputs": context["inputs"],
            "outputs": context["outputs"],
        }
    if set(references) != set(REFERENCE_RUNS):
        raise RuntimeError(
            f"exact target lineage changed; expected {sorted(REFERENCE_RUNS)}, got {sorted(references)}"
        )
    return references


def audit_terminal_jobs() -> dict[str, str]:
    result: dict[str, str] = {}
    for run_id, expected in REFERENCE_RUNS.items():
        output = run_checked(
            ["sacct", "-X", "-n", "-j", expected["job_id"], "--format=State", "-P"]
        )
        states = [line.strip().split("+")[0] for line in output.splitlines() if line.strip()]
        if states != [expected["state"]]:
            raise RuntimeError(
                f"unexpected scheduler state for {run_id}/{expected['job_id']}: {states}"
            )
        result[expected["job_id"]] = states[0]
    active = run_checked(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%o"])
    protected_jobs = {record["job_id"] for record in REFERENCE_RUNS.values()}
    active_protected = [line for line in active.splitlines() if line.split("|", 1)[0] in protected_jobs]
    if active_protected:
        raise RuntimeError(f"producer/export job unexpectedly active: {active_protected}")
    return result


def output_inventory(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"unexpected export output type: {path}")
    inventory = tree_inventory(path)
    return {"path": str(path), "exists": True, **inventory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root.resolve()
    if checkpoint_root != CHECKPOINT_ROOT:
        raise SystemExit(f"refusing unexpected checkpoint root: {checkpoint_root}")
    if TARGET.parent != checkpoint_root or TARGET.name != TARGET_ALIAS:
        raise RuntimeError(f"internal exact-target invariant failed: {TARGET}")
    if not TARGET.is_dir() or TARGET.is_symlink():
        raise RuntimeError(f"exact target missing or unsafe: {TARGET}")
    output = args.out.resolve()
    if output == TARGET or TARGET in output.parents or output in TARGET.parents:
        raise RuntimeError(f"audit output overlaps deletion target: {output}")

    memory_sha = sha256(MEMORY_NOTE)
    memory_text = MEMORY_NOTE.read_text()
    required_retirement_text = (
        "Mihir capture is LOSSY",
        "6.7% of chars overall",
        "42% of typing episodes",
        "fully abandonable",
        "drop the mihir thread ENTIRELY",
        "mihir format retired",
    )
    if memory_sha != MEMORY_NOTE_SHA256 or any(
        phrase not in memory_text for phrase in required_retirement_text
    ):
        raise RuntimeError("retirement authority memory note changed")

    context_path = RUNS_ROOT / PRODUCER_RUN_ID / ".lab/context.json"
    producer = json.loads(context_path.read_text())
    producer_output = producer["outputs"]["checkpoint"]
    producer_input = producer["inputs"][0]
    if (
        producer["recipe_name"]
        != "bc_qwen3vl8b_lora_videocua_diffabs_mihiractions_syspromptaligned_16k_truncate_v1"
        or producer_output["path"] != str(TARGET)
        or producer_output["alias"] != TARGET_ALIAS
        or producer_input["resolved_path"].rsplit("/", 1)[-1]
        != "videocua_diffabs_mihir_syspromptaligned_v1"
        or "crowd" in TARGET_ALIAS.lower()
    ):
        raise RuntimeError("target is not the pinned retired Mihir producer")

    top_entries = {path.name for path in TARGET.iterdir()}
    if top_entries != EXPECTED_STEPS | EXPECTED_TOP_FILES:
        raise RuntimeError(f"unexpected exact-target top entries: {sorted(top_entries)}")
    metadata_hashes: dict[str, str] = {}
    for step in sorted(EXPECTED_STEPS):
        metadata = TARGET / step / "_CHECKPOINT_METADATA"
        if not metadata.is_file() or metadata.is_symlink():
            raise RuntimeError(f"missing checkpoint metadata: {metadata}")
        metadata_hashes[step] = sha256(metadata)
    root_file_hashes = {name: sha256(TARGET / name) for name in sorted(EXPECTED_TOP_FILES)}
    if root_file_hashes != EXPECTED_ROOT_FILE_HASHES:
        raise RuntimeError(f"target root evidence changed: {root_file_hashes}")

    inventory = tree_inventory(TARGET)
    expected_inventory = {
        "file_count": EXPECTED_FILE_COUNT,
        "directory_count": EXPECTED_DIR_COUNT,
        "logical_bytes": EXPECTED_LOGICAL_BYTES,
        "allocated_bytes": EXPECTED_ALLOCATED_BYTES,
        "path_size_inventory_sha256": EXPECTED_INVENTORY_SHA256,
    }
    if inventory != expected_inventory:
        raise RuntimeError(f"exact target inventory changed: {inventory}")
    du_bytes = int(run_checked(["du", "-s", "-B1", str(TARGET)]).split()[0])
    if du_bytes != EXPECTED_ALLOCATED_BYTES:
        raise RuntimeError(f"exact target du changed: {du_bytes}")

    terminal_states = audit_terminal_jobs()
    references = audit_context_lineage()
    retained_before = run_evidence_hashes()
    export_outputs_before = {
        run_id: output_inventory(path) for run_id, path in EXPORT_OUTPUTS.items()
    }

    output.mkdir(parents=True, exist_ok=True)
    preserved = output / "preserved_target_root_evidence"
    if preserved.exists():
        raise RuntimeError(f"fresh evidence output unexpectedly exists: {preserved}")
    preserved.mkdir()
    for name in sorted(EXPECTED_TOP_FILES):
        shutil.copy2(TARGET / name, preserved / name)
    if {name: sha256(preserved / name) for name in sorted(EXPECTED_TOP_FILES)} != root_file_hashes:
        raise RuntimeError("preserved target root evidence copy failed verification")

    available_before = available(checkpoint_root)
    shutil.rmtree(TARGET)
    available_after = available(checkpoint_root)
    if TARGET.exists():
        raise RuntimeError(f"exact target remained after cleanup: {TARGET}")
    if run_evidence_hashes() != retained_before:
        raise RuntimeError("retained producer/export context or log changed during cleanup")
    export_outputs_after = {
        run_id: output_inventory(path) for run_id, path in EXPORT_OUTPUTS.items()
    }
    if export_outputs_after != export_outputs_before:
        raise RuntimeError("separate HF export output state changed during cleanup")

    result = {
        "schema_version": 1,
        "artifact_type": "retired_mihir_videocua_exact_checkpoint_cleanup",
        "status": "complete",
        "cpu_only": True,
        "retirement_authority": {
            "path": str(MEMORY_NOTE),
            "sha256": memory_sha,
            "finding": "lossy Mihir capture; thread explicitly retired and fully abandonable",
        },
        "producer": {
            "run_id": PRODUCER_RUN_ID,
            "job_id": PRODUCER_JOB_ID,
            "terminal_state": terminal_states[PRODUCER_JOB_ID],
            "dataset": producer_input["resolved_path"],
        },
        "deleted_exact_path": str(TARGET),
        "deleted_inventory": inventory,
        "deleted_du_allocated_bytes": du_bytes,
        "checkpoint_metadata_sha256": metadata_hashes,
        "preserved_target_root_evidence": {
            "path": str(preserved),
            "sha256": root_file_hashes,
        },
        "exact_context_references": references,
        "referencing_job_terminal_states": terminal_states,
        "retained_run_context_and_log_hashes": retained_before,
        "separate_hf_export_outputs_before_and_after": {
            run_id: {"before": export_outputs_before[run_id], "after": export_outputs_after[run_id]}
            for run_id in sorted(EXPORT_OUTPUTS)
        },
        "registry_mutated_by_cleanup_script": False,
        "filesystem_available_bytes_before": available_before,
        "filesystem_available_bytes_after": available_after,
        "filesystem_available_delta_bytes": available_after - available_before,
        "preserved": (
            "labctl registry; producer/export contexts and logs; copies of target-root config, "
            "LoRA metadata, and memory log; all separate HF export/eval paths"
        ),
    }
    manifest = output / "cleanup_manifest.json"
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
