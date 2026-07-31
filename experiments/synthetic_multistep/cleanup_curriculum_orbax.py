#!/usr/bin/env python3
"""Delete only the two sealed original-curriculum Orbax trees."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


class CleanupError(RuntimeError):
    pass


BASE = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical")
EXPECTED = {
    "A_to_B": {
        "original_alias": "synthetic_multistep_curriculum_A_to_B_raw_pre_r256_s750_v1_run_019fb53249dc7cf28e5d5ffe62151507",
        "recovered_alias": "synthetic_multistep_curriculum_A_to_B_raw_pre_r256_s750_recovered_v1_run_019fb56fb2f471118f1a9ed683def8b0",
        "job_id": 135464,
        "run_id": "run_019fb53249dc7cf28e5d5ffe62151507",
        "log_name": "synthetic_multistep_train_curriculum_A_to_B_r256_pinned_v1_135464.log",
        "file_count": 91,
        "logical_bytes": 106843104179,
        "allocated_bytes": 141289857024,
        "checkpoint_metadata_sha256": "cfa38ccdbad31f92c897743b22d2995a56131a987fc014b3edb0f565b539b838",
        "failed_run_log_sha256": "00da156d9d7f465e91715072c5e4ca3d9ca989cd3564070059632792ba28a93e",
    },
    "B_to_B": {
        "original_alias": "synthetic_multistep_curriculum_B_to_B_raw_pre_r256_s750_v1_run_019fb53249dd781085931aa776ff8bdc",
        "recovered_alias": "synthetic_multistep_curriculum_B_to_B_raw_pre_r256_s750_recovered_v1_run_019fb56fb3007ee2a1deba3f28b5e5cb",
        "job_id": 135463,
        "run_id": "run_019fb53249dd781085931aa776ff8bdc",
        "log_name": "synthetic_multistep_train_curriculum_B_to_B_r256_pinned_v1_135463.log",
        "file_count": 92,
        "logical_bytes": 106879779390,
        "allocated_bytes": 141593059328,
        "checkpoint_metadata_sha256": "36de30fceea5f6a9f79dc488a1774f1a5d95c5c0a1cbc1b143bffb7ad8bd964a",
        "failed_run_log_sha256": "18011489eeae24cc7cb67aa661d598061ed79d7fb0d1ac313c481f693b20e18e",
    },
}
LORA_METADATA_SHA256 = "3e491761873096823bb0a8cb296af699ab40c588b01c8472ef3e1ed0632211ec"
INPUT_ITERATOR_SHA256 = "2f33350e0f9148002563ef8cf6b628cf07889dac96298c664ce8c6df9d62ecf9"
TRAIN_STATE_METADATA_SHA256 = "729381d3b8f4ed8d96ba1733dba2a42adfb70117e6d8c5775cee73bec6e8298c"
HF_WEIGHT_BYTES = 35068587488


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CleanupError(f"sealed JSON is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanupError(message)


def files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: str(p.relative_to(root)))


def tree_record(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    paths = files(root)
    logical = sum(path.stat().st_size for path in paths)
    allocated = sum(path.stat().st_blocks * 512 for path in paths)
    require(len(paths) == expected["file_count"], f"unexpected file count in {root}")
    require(logical == expected["logical_bytes"], f"unexpected logical size in {root}: {logical}")
    require(allocated == expected["allocated_bytes"], f"unexpected allocated size in {root}: {allocated}")
    # This identifies the exact path/size tree without reading 213 GB of already
    # recovery-validated tensors again. Critical endpoint contents are SHA-gated below.
    inventory = "\n".join(f"{path.relative_to(root)}\t{path.stat().st_size}" for path in paths) + "\n"
    return {
        "file_count": len(paths),
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "path_size_inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
    }


def verify_recovered(branch: str, original: Path, recovered: Path, run_dir: Path) -> dict[str, Any]:
    expected = EXPECTED[branch]
    require(original.resolve() == (BASE / expected["original_alias"]).resolve(), f"refusing wrong original root for {branch}")
    require(original.parent.resolve() == BASE.resolve(), f"refusing broad original path for {branch}")
    require(recovered.name == expected["recovered_alias"], f"wrong recovered alias for {branch}")
    require(run_dir.name == expected["run_id"], f"wrong failed run directory for {branch}")
    orbax = original / "orbax"
    require(orbax.is_dir() and not orbax.is_symlink(), f"Orbax source is absent or symlinked: {orbax}")
    require({path.name for path in orbax.iterdir()} == {"000250", "000500", "000750", "config.json", "lora_metadata.json"},
            f"unexpected top-level entries in exact Orbax target: {orbax}")
    record = tree_record(orbax, expected)

    manifest_path = recovered / "curriculum_train_export_manifest.json"
    manifest = load_json(manifest_path)
    fixed = {
        "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
        "status": "complete", "branch": branch, "step": 750,
        "lora_rank": 256, "lora_alpha": 256, "hf_subdir": "hf",
        "recovered_from_failed_job_id": expected["job_id"],
        "recovered_from_failed_run_id": expected["run_id"],
        "recovered_from_failed_run_root": str(original.resolve()),
        "source_checkpoint": str((orbax / "000750").resolve()),
    }
    bad = {key: (manifest.get(key), wanted) for key, wanted in fixed.items() if manifest.get(key) != wanted}
    require(not bad, f"invalid recovered manifest for {branch}: {bad}")
    endpoints = manifest.get("endpoint_hashes", {})
    hashes = {
        "checkpoint_metadata_sha256": sha256(orbax / "000750" / "_CHECKPOINT_METADATA"),
        "lora_metadata_sha256": sha256(orbax / "lora_metadata.json"),
        "input_iterator_sha256": sha256(orbax / "000750" / "input_iter" / "process_0-of-1.json"),
        "train_state_metadata_sha256": sha256(orbax / "000750" / "train_state" / "_METADATA"),
        "failed_run_log_sha256": sha256(run_dir / ".lab" / expected["log_name"]),
    }
    pinned = {
        "checkpoint_metadata_sha256": expected["checkpoint_metadata_sha256"],
        "lora_metadata_sha256": LORA_METADATA_SHA256,
        "input_iterator_sha256": INPUT_ITERATOR_SHA256,
        "train_state_metadata_sha256": TRAIN_STATE_METADATA_SHA256,
        "failed_run_log_sha256": expected["failed_run_log_sha256"],
    }
    require(hashes == pinned and all(endpoints.get(key) == value for key, value in hashes.items()),
            f"source endpoint hash mismatch for {branch}")
    hf = recovered / "hf"
    for name in ("model.safetensors", "config.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json"):
        require((hf / name).is_file(), f"recovered HF file missing for {branch}: {name}")
    require((hf / "model.safetensors").stat().st_size == HF_WEIGHT_BYTES, f"wrong HF weight size for {branch}")
    record.update({"root": str(orbax), "endpoint_hashes": hashes,
                   "recovered_manifest_sha256": sha256(manifest_path),
                   "retained_hf_weight_bytes": HF_WEIGHT_BYTES})
    return record


def verify_teacher(branch: str, root: Path, recovered_manifest_sha: str) -> None:
    report = load_json(root / "teacher_forced_report.json")
    require(report.get("artifact_type") == "synthetic_multistep_curriculum_teacher_forced_eval"
            and report.get("status") == "complete" and report.get("branch") == branch
            and report.get("summary", {}).get("n_examples") == 200,
            f"invalid teacher-forced report for {branch}")
    require(sha256(root / "teacher_forced_rows.jsonl") == report.get("rows_sha256"),
            f"teacher-forced rows hash mismatch for {branch}")
    require(report.get("model_manifest", {}).get("evaluation_input_hashes", {}).get("model_manifest_sha256")
            == recovered_manifest_sha, f"teacher-forced model hash mismatch for {branch}")


def verify_multistep(branch: str, root: Path, recovered_alias: str, recovered_manifest_sha: str) -> str:
    manifest = load_json(root / "eval_manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("semantic") == "deltatype_raw"
            and manifest.get("preamble") is True and manifest.get("comparison_label") == "curriculum_transfer"
            and manifest.get("checkpoint_alias") == recovered_alias and manifest.get("n_episodes") == 80,
            f"invalid closed-loop artifact for {branch}")
    require(sha256(root / "rows.jsonl") == manifest.get("rows_sha256")
            and sha256(root / "report.json") == manifest.get("report_sha256"),
            f"closed-loop hash mismatch for {branch}")
    require(manifest.get("model_provenance", {}).get("export_manifest_sha256") == recovered_manifest_sha,
            f"closed-loop model hash mismatch for {branch}")
    return str(manifest.get("episode_manifest_sha256"))


def available_bytes(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    branches = {
        "A_to_B": (args.original_a.resolve(), args.recovered_a.resolve(), args.failed_run_a.resolve()),
        "B_to_B": (args.original_b.resolve(), args.recovered_b.resolve(), args.failed_run_b.resolve()),
    }
    source_records = {branch: verify_recovered(branch, *values) for branch, values in branches.items()}
    recovered_hashes = {branch: record["recovered_manifest_sha256"] for branch, record in source_records.items()}

    verify_teacher("A_to_B", args.teacher_a.resolve(), recovered_hashes["A_to_B"])
    verify_teacher("B_to_B", args.teacher_b.resolve(), recovered_hashes["B_to_B"])
    teacher_comparison = load_json(args.teacher_comparison / "teacher_forced_comparison.json")
    require(teacher_comparison.get("status") == "complete"
            and teacher_comparison.get("effect_direction") == "A_to_B_minus_B_to_B",
            "invalid teacher-forced comparison")

    episode_a = verify_multistep("A_to_B", args.multistep_a.resolve(), EXPECTED["A_to_B"]["recovered_alias"], recovered_hashes["A_to_B"])
    episode_b = verify_multistep("B_to_B", args.multistep_b.resolve(), EXPECTED["B_to_B"]["recovered_alias"], recovered_hashes["B_to_B"])
    require(episode_a == episode_b, "closed-loop episode hashes differ")
    comparison = load_json(args.multistep_comparison / "curriculum_comparison.json")
    require(comparison.get("status") == "complete"
            and comparison.get("effect_direction") == "A_to_B_minus_B_to_B"
            and comparison.get("episode_manifest_sha256") == episode_a,
            "invalid closed-loop comparison")

    before = available_bytes(BASE)
    # Destruction starts only after all retained models, rows, reports, and both
    # independently registered comparisons have passed every gate above.
    for original, _, _ in branches.values():
        shutil.rmtree(original / "orbax")
    after = available_bytes(BASE)
    for branch, (original, recovered, _) in branches.items():
        require(not (original / "orbax").exists(), f"Orbax tree remained for {branch}")
        require((recovered / "hf" / "model.safetensors").stat().st_size == HF_WEIGHT_BYTES,
                f"retained HF export disappeared for {branch}")

    result = {
        "schema_version": 2,
        "artifact_type": "synthetic_multistep_curriculum_original_orbax_cleanup",
        "status": "complete", "cpu_only": True,
        "exact_targets": source_records,
        "removed_logical_bytes": sum(record["logical_bytes"] for record in source_records.values()),
        "removed_allocated_bytes": sum(record["allocated_bytes"] for record in source_records.values()),
        "filesystem_available_bytes_before": before,
        "filesystem_available_bytes_after": after,
        "filesystem_available_delta_bytes": after - before,
        "validated_before_deletion": {
            "recovered_hf_exports": [EXPECTED[key]["recovered_alias"] for key in ("A_to_B", "B_to_B")],
            "teacher_forced_rows_reports_and_registered_comparison": True,
            "closed_loop_rows_reports_and_registered_comparison": True,
            "shared_episode_manifest_sha256": episode_a,
        },
        "retained": "both recovered HF exports; every run manifest, log, row, report, and comparison; all stage-1, low-LR, and typing artifacts",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "cleanup_manifest.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("original-a", "original-b", "recovered-a", "recovered-b",
                 "failed-run-a", "failed-run-b", "teacher-a", "teacher-b",
                 "teacher-comparison", "multistep-a", "multistep-b",
                 "multistep-comparison", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = cleanup(args)
    except CleanupError as exc:
        print(f"FATAL cleanup gate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
