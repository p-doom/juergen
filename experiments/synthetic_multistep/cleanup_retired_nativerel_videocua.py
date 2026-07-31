#!/usr/bin/env python3
"""Delete one exact retired VideoCUA native_rel Orbax stream after sealed audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path


CHECKPOINT_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical"
)
TARGET_ALIAS = (
    "bc_qwen3vl8b_lora_videocua_sol_nativerel_v1_"
    "run_019f99176a807822b55b2850e01e1ccc"
)
TARGET = CHECKPOINT_ROOT / TARGET_ALIAS
RUNS_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs/franz.srambical"
)
MEMORY_NOTE = Path(
    "/fast/home/franz.srambical/.claude/projects/-fast-home-franz-srambical/"
    "memory/project_videocua_format_defects.md"
)
MEMORY_NOTE_SHA256 = "bd0435528544f31f5681d4081f06c44259f2d80d9150b2db1daa7bc05262b918"

PRODUCER_RUN_ID = "run_019f99176a807822b55b2850e01e1ccc"
PRODUCER_JOB_ID = "131654"
EXPECTED_STEPS = {
    "001125", "002250", "003375", "004500",
    "005625", "006750", "007875", "009000",
}
EXPECTED_TOP_FILES = {"config.json", "lora_metadata.json", "memory.log"}
EXPECTED_ROOT_FILE_HASHES = {
    "config.json": "431c3209d531f04bb383751c91c7477a8e7c66c717dfd1584464be7e35b0dbd3",
    "lora_metadata.json": "877c7be33455d87488588ca52742d5a5ae053d52a2b5b4ef083c04d5eb47cb3c",
    "memory.log": "3008a279dd720b1a747ea2d14390d872c14166468af9cbe64e2543d2c1eedf2f",
}
EXPECTED_INVENTORY = {
    "file_count": 333,
    "directory_count": 104,
    "logical_bytes": 141_350_120_427,
    "allocated_bytes": 188_641_075_200,
    "path_size_inventory_sha256": "26f5524f1d11387ab2dd86713b5d3acc1078cafc8ee9c996a9db2ad2e86dd19b",
}

# Every registered export from the eight source steps. Only the scientifically
# referenced step-1125 HF model remains on disk after earlier HF storage cleanup.
EXPORTS = {
    "001125": ("run_019f9989691c7ad1a4ad9725ea0bf1af", "131664", "6a26a4e2452eb6c5"),
    "002250": ("run_019f99d1c18978a2a6285683cb6ec4c1", "131701", "24af9733218b4c03"),
    "003375": ("run_019f9a2de7077db387d09fd5ae7edc88", "131726", "48ae1d555f8373c4"),
    "004500": ("run_019f9a85913776b292bb7cc94f22d88a", "131748", "6c9968ae73607556"),
    "005625": ("run_019f9ae161967222b6f3dec9bf98f365", "131774", "d58ec960edfee330"),
    "006750": ("run_019f9b3d22e27fe0bcf7a5527d3ed19e", "131799", "4a58149a7c7e2859"),
    "007875": ("run_019f9b944d8471439264c10b70d1eec5", "131856", "22d42d4e5c246d60"),
    "009000": ("run_019f9bf0086f79808e01edeb8d668550", "131871", "67ae8ae144b1b6d0"),
}
RETAINED_HF_MODEL = (
    CHECKPOINT_ROOT / "bc_export_hf_8b_nativerel_artifact_6a26a4e2452eb6c5"
    / "001125/model.safetensors"
)
RETAINED_HF_SHA256 = "b57c9a663f4368f92354368bf6cda85578d2967a436bd5ca4f8d59bd07dd1ae6"

EVAL_RESULTS = {
    "run_019f99a833ba7932b4daf78aafbf73df": "0b1a6e22d1387468161d9ce11cc551b1e854093082c1ca7b98a90556e5ca1659",
    "run_019f99a840197773a61c08065ba7649b": "2dcde038c55fa789a99a72b8f37455edab3c32f9674c389af9a79b8cc522106a",
    "run_019f99dafc727c919ead2df027781532": "77da90d8d25fab8be2c6567997e99d98b6a28d42f4544551f8b12a532a17faf2",
    "run_019f99db02577f8092b4a79ab46e9601": "aa4e0027ca84a50c6f15350c0c7ffafd713b5647a0b7567616a84bc7cf6c95e9",
    "run_019f9a52deaa7a819d4c264c2eb745d2": "31b92e17efd753493380f139232525e085b76a819618dc9c373041bdbb7e4223",
    "run_019f9a52ec2779019673d9efd8892d60": "8e20c2c7b7de3919a82fe0c1d2db3304a9a5f229531e9636fedb08200545a706",
    "run_019f9a935e6b7b13afbdccf5b5010860": "cd6702c23d17af11a2f312b025373765a1976fff6c29f598cc6af913920bcc8c",
    "run_019f9a9362eb74f392ceeaa7f9afe390": "beb7dca0925aa9b511ecc07b8b0aabae5c743de25cec5984f4951dab13b598b0",
    "run_019f9ae603b779909bdb8f8c1c2b28b3": "0b1a6e22d1387468161d9ce11cc551b1e854093082c1ca7b98a90556e5ca1659",
    "run_019f9ae6089a79239313622430b5d4e3": "5dc86daf67827c1ec843cec482adb27323336d2fc4adbe023191571ee93938b4",
    "run_019f9b41c73d71c0bd9252b36702200b": "f93eeecc8dc21600aa3f88bee545faca98430671e09af1cf8e2724e24dcb4255",
    "run_019f9b41cbf1747183c170dc960ae722": "0e756d161322ddee5d017d93cd098c9b0a38133b9cade295d841632cb2fef57d",
    "run_019f9b98f0ae7373bb96f0057ddbf867": "efaaea12a7e019fea48b8ade6b2dd770ad501bb1954d580e66c558f0755a2b8d",
    "run_019f9b98f5dd7a7088b32ae937a65b9d": "9b107c5f72ab551e389877eae2a2606426b9f166acb8702d2a56dfd4d6010948",
    "run_019f9bf943257b528ac0ed35559370ee": "0b1a6e22d1387468161d9ce11cc551b1e854093082c1ca7b98a90556e5ca1659",
    "run_019f9bf948d27e63b7fd8b4b26a4ec5a": "f5b090b438d752311c301f945f864e584fd1999945b6b56dc441bb51509eac36",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def available(path: Path) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def tree_inventory(root: Path) -> dict[str, int | str]:
    entries: list[str] = []
    logical = allocated = files = directories = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name, kind in [
            *((name, "d") for name in dirnames),
            *((name, "f") for name in filenames),
        ]:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink inside exact deletion target: {path}")
            value = path.stat()
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(value.st_mode)
            entries.append(
                f"{kind}\t{relative}\t{value.st_size}\t{value.st_blocks}\t{mode:o}\n"
            )
            allocated += value.st_blocks * 512
            if kind == "f":
                files += 1
                logical += value.st_size
            else:
                directories += 1
    allocated += root.stat().st_blocks * 512
    digest = hashlib.sha256("".join(sorted(entries)).encode()).hexdigest()
    return {
        "file_count": files,
        "directory_count": directories,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "path_size_inventory_sha256": digest,
    }


def terminal(job_id: str) -> str:
    output = run_checked(["sacct", "-X", "-n", "-j", job_id, "--format=State", "-P"])
    states = [line.strip().split("+")[0] for line in output.splitlines() if line.strip()]
    if states != ["COMPLETED"]:
        raise RuntimeError(f"job {job_id} is not uniquely COMPLETED: {states}")
    return states[0]


def context_and_log_hashes(run_ids: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for run_id in sorted(run_ids):
        lab = RUNS_ROOT / run_id / ".lab"
        context = lab / "context.json"
        logs = list(lab.glob("*.log"))
        if len(logs) != 1:
            raise RuntimeError(f"expected one retained log for {run_id}, got {logs}")
        result[run_id] = {
            "context_path": str(context),
            "context_sha256": sha256(context),
            "log_path": str(logs[0]),
            "log_sha256": sha256(logs[0]),
        }
    return result


def audit_context_lineage() -> dict[str, dict[str, object]]:
    references: dict[str, dict[str, object]] = {}
    for context_path in RUNS_ROOT.glob("run_*/.lab/context.json"):
        raw = context_path.read_text()
        if str(TARGET) not in raw:
            continue
        value = json.loads(raw)
        references[value["run_id"]] = {
            "recipe_name": value["recipe_name"],
            "inputs": value["inputs"],
            "outputs": value["outputs"],
        }
    expected = {PRODUCER_RUN_ID, *(item[0] for item in EXPORTS.values())}
    if set(references) != expected:
        raise RuntimeError(
            f"exact target lineage changed; expected {sorted(expected)}, got {sorted(references)}"
        )
    return references


def eval_evidence() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for run_id, expected_hash in sorted(EVAL_RESULTS.items()):
        context_path = RUNS_ROOT / run_id / ".lab/context.json"
        context = json.loads(context_path.read_text())
        recipe = context["recipe_name"]
        if "nativerel" not in recipe or not recipe.startswith("osworld_"):
            raise RuntimeError(f"unexpected eval recipe for {run_id}: {recipe}")
        result_path = Path(context["outputs"]["result"]["path"]) / "result.json"
        actual_hash = sha256(result_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"retained eval changed for {run_id}: {actual_hash}")
        result[run_id] = {
            "recipe_name": recipe,
            "result_path": str(result_path),
            "result_sha256": actual_hash,
        }
    if len(result) != 16:
        raise RuntimeError("expected sixteen retained grounding/typing evaluations")
    return result


def export_state() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for step, (run_id, job_id, artifact) in sorted(EXPORTS.items()):
        context = json.loads((RUNS_ROOT / run_id / ".lab/context.json").read_text())
        source = context["inputs"][0]["resolved_path"]
        expected_source = str(TARGET / step)
        output = CHECKPOINT_ROOT / f"bc_export_hf_8b_nativerel_artifact_{artifact}"
        if source != expected_source or Path(context["outputs"]["hf_checkpoint"]["path"]) != output:
            raise RuntimeError(f"export lineage mismatch at step {step}")
        state: dict[str, object] = {
            "run_id": run_id,
            "job_id": job_id,
            "terminal_state": terminal(job_id),
            "source": source,
            "output": str(output),
            "output_exists": output.exists(),
        }
        if step == "001125":
            if not RETAINED_HF_MODEL.is_file() or sha256(RETAINED_HF_MODEL) != RETAINED_HF_SHA256:
                raise RuntimeError("retained step-1125 HF export changed")
            state["model_safetensors_sha256"] = RETAINED_HF_SHA256
        elif output.exists():
            raise RuntimeError(f"unexpected previously-cleaned HF output reappeared: {output}")
        result[step] = state
    return result


def preserve_small_files(output: Path) -> dict[str, str]:
    preserved = output / "preserved_orbax_metadata"
    if preserved.exists():
        raise RuntimeError(f"fresh metadata snapshot unexpectedly exists: {preserved}")
    hashes: dict[str, str] = {}
    for directory, _dirnames, filenames in os.walk(TARGET, followlinks=False):
        for name in filenames:
            source = Path(directory) / name
            if source.stat().st_size > 2 * 1024 * 1024:
                continue
            relative = source.relative_to(TARGET)
            destination = preserved / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[relative.as_posix()] = sha256(destination)
    if len(hashes) < 80:
        raise RuntimeError(f"metadata preservation unexpectedly sparse: {len(hashes)} files")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    root = args.checkpoint_root.resolve()
    if root != CHECKPOINT_ROOT or TARGET.parent != root or TARGET.name != TARGET_ALIAS:
        raise SystemExit(f"refusing unexpected checkpoint root/target: {root} / {TARGET}")
    if not TARGET.is_dir() or TARGET.is_symlink():
        raise RuntimeError(f"exact target missing or unsafe: {TARGET}")
    output = args.out.resolve()
    if output == TARGET or TARGET in output.parents or output in TARGET.parents:
        raise RuntimeError(f"audit output overlaps exact deletion target: {output}")

    memory_hash = sha256(MEMORY_NOTE)
    memory = MEMORY_NOTE.read_text()
    required = (
        "native_rel = relative-delta in an absolute-semantics schema",
        "overloads an absolute schema with relative meaning",
        "rebuilding corrected videocua + crowd-cast datasets in the honest relative-pyautogui format",
    )
    if memory_hash != MEMORY_NOTE_SHA256 or any(phrase not in memory for phrase in required):
        raise RuntimeError("native_rel retirement/supersession authority changed")

    producer_context_path = RUNS_ROOT / PRODUCER_RUN_ID / ".lab/context.json"
    producer = json.loads(producer_context_path.read_text())
    producer_input = producer["inputs"][0]["resolved_path"]
    if (
        producer["recipe_name"] != "bc_qwen3vl8b_lora_videocua_sol_nativerel_v1"
        or producer["outputs"]["checkpoint"]["path"] != str(TARGET)
        or producer_input.rsplit("/", 1)[-1] != "videocua_nativerel_v1"
        or "crowd" in TARGET_ALIAS.lower()
    ):
        raise RuntimeError("target is not the pinned non-crowd-cast retired native_rel producer")

    top = {path.name for path in TARGET.iterdir()}
    if top != EXPECTED_STEPS | EXPECTED_TOP_FILES:
        raise RuntimeError(f"unexpected exact-target top entries: {sorted(top)}")
    root_hashes = {name: sha256(TARGET / name) for name in sorted(EXPECTED_TOP_FILES)}
    if root_hashes != EXPECTED_ROOT_FILE_HASHES:
        raise RuntimeError(f"target root evidence changed: {root_hashes}")
    metadata_hashes = {
        step: sha256(TARGET / step / "_CHECKPOINT_METADATA")
        for step in sorted(EXPECTED_STEPS)
    }
    inventory = tree_inventory(TARGET)
    if inventory != EXPECTED_INVENTORY:
        raise RuntimeError(f"exact target inventory changed: {inventory}")
    du_bytes = int(run_checked(["du", "-s", "-B1", str(TARGET)]).split()[0])
    if du_bytes != EXPECTED_INVENTORY["allocated_bytes"]:
        raise RuntimeError(f"exact target du changed: {du_bytes}")

    jobs = {PRODUCER_JOB_ID: terminal(PRODUCER_JOB_ID)}
    jobs.update({job: terminal(job) for _run, job, _artifact in EXPORTS.values()})
    protected_jobs = set(jobs)
    active = run_checked(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A|%T|%j"])
    if any(line.split("|", 1)[0] in protected_jobs for line in active.splitlines()):
        raise RuntimeError("producer/export job unexpectedly active")

    references = audit_context_lineage()
    run_ids = {PRODUCER_RUN_ID, *(value[0] for value in EXPORTS.values())}
    run_evidence_before = context_and_log_hashes(run_ids)
    evals_before = eval_evidence()
    exports_before = export_state()

    audit = {
        "schema_version": 1,
        "artifact_type": "retired_nativerel_videocua_exact_checkpoint_cleanup",
        "status": "audit_only" if args.audit_only else "complete",
        "cpu_only": True,
        "retirement_authority": {
            "path": str(MEMORY_NOTE),
            "sha256": memory_hash,
            "finding": "native_rel overloaded absolute-schema semantics and was superseded",
        },
        "producer": {
            "run_id": PRODUCER_RUN_ID,
            "job_id": PRODUCER_JOB_ID,
            "terminal_state": jobs[PRODUCER_JOB_ID],
            "dataset": producer_input,
            "contains_crowd_cast_training_data": False,
        },
        "exact_target": str(TARGET),
        "pre_inventory": inventory,
        "pre_du_allocated_bytes": du_bytes,
        "checkpoint_metadata_sha256": metadata_hashes,
        "root_file_sha256": root_hashes,
        "exact_context_references": references,
        "producer_export_context_log_hashes": run_evidence_before,
        "retained_eval_results": evals_before,
        "export_states": exports_before,
        "retained_hf_model": {
            "path": str(RETAINED_HF_MODEL),
            "sha256": RETAINED_HF_SHA256,
        },
    }
    if args.audit_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    output.mkdir(parents=True, exist_ok=True)
    preserved_hashes = preserve_small_files(output)
    available_before = available(root)
    shutil.rmtree(TARGET)
    available_after = available(root)
    if TARGET.exists():
        raise RuntimeError(f"exact target remained after cleanup: {TARGET}")
    if context_and_log_hashes(run_ids) != run_evidence_before:
        raise RuntimeError("retained producer/export context or log changed")
    if eval_evidence() != evals_before:
        raise RuntimeError("retained evaluation evidence changed")
    exports_after = export_state()
    if exports_after != exports_before:
        raise RuntimeError("retained/previously-cleaned HF export state changed")
    for relative, digest in preserved_hashes.items():
        if sha256(output / "preserved_orbax_metadata" / relative) != digest:
            raise RuntimeError(f"preserved metadata changed: {relative}")

    post_absence_sha256 = hashlib.sha256(
        f"absent\0{TARGET}\0{inventory['path_size_inventory_sha256']}".encode()
    ).hexdigest()
    audit.update({
        "deleted_exact_path": str(TARGET),
        "deleted_inventory": inventory,
        "preserved_orbax_metadata": {
            "path": str(output / "preserved_orbax_metadata"),
            "file_count": len(preserved_hashes),
            "content_sha256": hashlib.sha256(
                json.dumps(preserved_hashes, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "post_target_exists": False,
        "post_absence_sha256": post_absence_sha256,
        "post_producer_export_context_log_hashes": run_evidence_before,
        "post_retained_eval_results": evals_before,
        "post_export_states": exports_after,
        "filesystem_available_bytes_before": available_before,
        "filesystem_available_bytes_after": available_after,
        "filesystem_available_delta_bytes": available_after - available_before,
        "registry_mutated_by_cleanup_script": False,
        "preserved": (
            "labctl registry, producer/export/eval contexts and logs, all eval result manifests, "
            "the selected step-1125 HF export, and a content-hashed snapshot of small Orbax metadata"
        ),
    })
    manifest = output / "cleanup_manifest.json"
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
