#!/usr/bin/env python3
"""Delete only the two fully sealed low-LR curriculum Orbax trees."""

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
COMMON = {
    "train_state_metadata_sha256": "729381d3b8f4ed8d96ba1733dba2a42adfb70117e6d8c5775cee73bec6e8298c",
    "input_iterator_sha256": "2f33350e0f9148002563ef8cf6b628cf07889dac96298c664ce8c6df9d62ecf9",
    "lora_metadata_sha256": "3e491761873096823bb0a8cb296af699ab40c588b01c8472ef3e1ed0632211ec",
}
EXPECTED = {
    "A_to_B": {
        "source_alias": "synthetic_multistep_curriculum_A_to_B_raw_pre_r256_lr5e5_s750_rescue_v1_run_019fb56283317ff2abadc7db933bfc32",
        "recovered_alias": "synthetic_multistep_curriculum_A_to_B_raw_pre_r256_lr5e5_s750_recovered_v1_run_019fb59ca7927631b332d44a4adc27f6",
        "run_id": "run_019fb56283317ff2abadc7db933bfc32", "job_id": 135475,
        "checkpoint_metadata_sha256": "73ddb2c7659e2e8f6724c8f40c3cc29eff177bf8b7a26acb83572a0218a7bebf",
        "terminal_run_log_sha256": "a499511388bf27fd55e8846e73ce765c8c4b83c8a3c1489d7526c8e4a386a54d",
        "recovered_manifest_sha256": "44bb75ceb916e806d766233e402e1ef9ded2ce1a5716769dbe218fe333aca519",
        "file_count": 90, "logical_bytes": 106847932850, "allocated_bytes": 141726187520,
        "inventory_sha256": "0d8cb2a6ae97783f0fd241559fb177d1cb29549857ffdb5edfc21e195c219aa2",
        "teacher_report_sha256": "f66a039cd6d51034503f271644340b4a008ec9e0eba73a97c9d689eb44cba9d0",
        "teacher_rows_sha256": "737984f8e7ffeae32732f14d07c97206ecd5ae15ed525f861dc5fbaa01ecdeb4",
        "multistep_manifest_sha256": "11612a5dcc65e29006b24bd05bdbcf82d0bf871edd7c3c5b0e165d0b2b54d403",
        "multistep_report_sha256": "74ab9d5f5b91ee1205c7a83a1daea23a4c39ca5c411a7881ab1f47b706aff205",
        "multistep_rows_sha256": "5fedf537b96f3ecc1018577dfd4a7d14b5bce916945dfaf667f4bce4bab830a0",
    },
    "B_to_B": {
        "source_alias": "synthetic_multistep_curriculum_B_to_B_raw_pre_r256_lr5e5_s750_rescue_v1_run_019fb562832e7dc38b2ad0264c508a87",
        "recovered_alias": "synthetic_multistep_curriculum_B_to_B_raw_pre_r256_lr5e5_s750_recovered_v1_run_019fb59cbf727b028103427bea95ee88",
        "run_id": "run_019fb562832e7dc38b2ad0264c508a87", "job_id": 135476,
        "checkpoint_metadata_sha256": "186d95324d6bf1fa56447f4aba40bbd6d06388a1e67b7371a74e4f7f0d4b7ac8",
        "terminal_run_log_sha256": "345ea3796b23f45d1201b8e38598b8ff7d5032adac9644dd19e62f1f28893198",
        "recovered_manifest_sha256": "99b85c7b4781377f6d7bc01e3f8233b30a5b0f3d61237c5ab5384dc99ad11b42",
        "file_count": 92, "logical_bytes": 106881847924, "allocated_bytes": 142300045312,
        "inventory_sha256": "e1f47f74436ff87732bbd04e6d7cb0352ead6b7b29b5212edec2406be0bab940",
        "teacher_report_sha256": "675d7a6dc39e6bd5df36c8338d653581c9ff74057a6c471ba2f5a6696a5da002",
        "teacher_rows_sha256": "1d10350f452bfcb591f9a8a0acd175068a18108508c707f1b004b522cfba7a70",
        "multistep_manifest_sha256": "ccf2bf117334709bd895ab61b343e31aa9b3f204f835d809c6a97678799e336a",
        "multistep_report_sha256": "62f994f63e082ba33d0f39e90c41c42bafeb2760dffbd5eeb6734808bae0e220",
        "multistep_rows_sha256": "8f240d2e3bc4647168bc9a5e8a52ae54ccb2ad4eb24bd0c35fa7311427d88f0b",
    },
}
HF_WEIGHT_BYTES = 35068587488


def require(value: bool, message: str) -> None:
    if not value:
        raise CleanupError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def inventory(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    paths = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(root)))
    logical = sum(p.stat().st_size for p in paths)
    allocated = sum(p.stat().st_blocks * 512 for p in paths)
    text = "".join(f"{p.relative_to(root)}\t{p.stat().st_size}\n" for p in paths)
    record = {"file_count": len(paths), "logical_bytes": logical, "allocated_bytes": allocated,
              "path_size_inventory_sha256": hashlib.sha256(text.encode()).hexdigest()}
    pinned = {"file_count": expected["file_count"], "logical_bytes": expected["logical_bytes"],
              "allocated_bytes": expected["allocated_bytes"],
              "path_size_inventory_sha256": expected["inventory_sha256"]}
    require(record == pinned, f"exact Orbax inventory mismatch: {root}: {record}")
    return record


def verify_branch(branch: str, source: Path, recovered: Path, run: Path,
                  teacher: Path, multistep: Path) -> dict[str, Any]:
    e = EXPECTED[branch]
    require(source.resolve() == (BASE / e["source_alias"]).resolve()
            and source.parent.resolve() == BASE.resolve(), f"wrong exact source root: {branch}")
    require(recovered.name == e["recovered_alias"] and run.name == e["run_id"],
            f"wrong recovery lineage: {branch}")
    orbax = source / "orbax"
    require(orbax.is_dir() and not orbax.is_symlink(), f"missing/symlink Orbax: {orbax}")
    require({p.name for p in orbax.iterdir()} == {"000250", "000500", "000750", "config.json", "lora_metadata.json"},
            f"unexpected top-level Orbax entries: {orbax}")
    record = inventory(orbax, e)
    log = next((run / ".lab").glob(f"*_{e['job_id']}.log"))
    endpoint = {
        "checkpoint_metadata_sha256": sha(orbax / "000750/_CHECKPOINT_METADATA"),
        "train_state_metadata_sha256": sha(orbax / "000750/train_state/_METADATA"),
        "input_iterator_sha256": sha(orbax / "000750/input_iter/process_0-of-1.json"),
        "lora_metadata_sha256": sha(orbax / "lora_metadata.json"),
        "terminal_run_log_sha256": sha(log),
    }
    pinned = {**COMMON, "checkpoint_metadata_sha256": e["checkpoint_metadata_sha256"],
              "terminal_run_log_sha256": e["terminal_run_log_sha256"]}
    require(endpoint == pinned, f"endpoint hash mismatch: {branch}")
    manifest_path = recovered / "curriculum_train_export_manifest.json"
    manifest = obj(manifest_path)
    fixed = {"artifact_type": "synthetic_multistep_curriculum_hf_checkpoint", "status": "complete",
             "branch": branch, "step": 750, "learning_rate": 5e-5,
             "recovered_from_terminal_run_id": e["run_id"],
             "recovered_from_terminal_job_id": e["job_id"],
             "source_checkpoint": str((orbax / "000750").resolve())}
    require(not {k: (manifest.get(k), v) for k, v in fixed.items() if manifest.get(k) != v},
            f"recovered manifest mismatch: {branch}")
    require(sha(manifest_path) == e["recovered_manifest_sha256"]
            and all(manifest.get("endpoint_hashes", {}).get(k) == v for k, v in endpoint.items()),
            f"recovered manifest hash/endpoint mismatch: {branch}")
    require((recovered / "hf/model.safetensors").stat().st_size == HF_WEIGHT_BYTES,
            f"retained HF missing: {branch}")
    teacher_report = obj(teacher / "teacher_forced_report.json")
    require(sha(teacher / "teacher_forced_report.json") == e["teacher_report_sha256"]
            and sha(teacher / "teacher_forced_rows.jsonl") == e["teacher_rows_sha256"]
            and teacher_report.get("status") == "complete"
            and teacher_report.get("summary", {}).get("n_examples") == 200
            and teacher_report.get("model_manifest", {}).get("evaluation_input_hashes", {}).get("model_manifest_sha256") == e["recovered_manifest_sha256"],
            f"teacher artifact mismatch: {branch}")
    ms = obj(multistep / "eval_manifest.json")
    require(sha(multistep / "eval_manifest.json") == e["multistep_manifest_sha256"]
            and sha(multistep / "report.json") == e["multistep_report_sha256"]
            and sha(multistep / "rows.jsonl") == e["multistep_rows_sha256"]
            and ms.get("status") == "complete" and ms.get("checkpoint_alias") == e["recovered_alias"]
            and ms.get("comparison_label") == "curriculum_transfer_lr5e5"
            and ms.get("model_provenance", {}).get("export_manifest_sha256") == e["recovered_manifest_sha256"],
            f"multistep artifact mismatch: {branch}")
    return {**record, "root": str(orbax), "endpoint_hashes": endpoint,
            "recovered_manifest_sha256": e["recovered_manifest_sha256"],
            "teacher_report_sha256": e["teacher_report_sha256"],
            "teacher_rows_sha256": e["teacher_rows_sha256"],
            "multistep_manifest_sha256": e["multistep_manifest_sha256"],
            "multistep_report_sha256": e["multistep_report_sha256"],
            "multistep_rows_sha256": e["multistep_rows_sha256"]}


def available() -> int:
    stat = os.statvfs(BASE)
    return stat.f_bavail * stat.f_frsize


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("source-a", "source-b", "recovered-a", "recovered-b", "run-a", "run-b",
                 "teacher-a", "teacher-b", "multistep-a", "multistep-b",
                 "teacher-comparison", "multistep-comparison", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = {
            "A_to_B": verify_branch("A_to_B", args.source_a.resolve(), args.recovered_a.resolve(),
                                    args.run_a.resolve(), args.teacher_a.resolve(), args.multistep_a.resolve()),
            "B_to_B": verify_branch("B_to_B", args.source_b.resolve(), args.recovered_b.resolve(),
                                    args.run_b.resolve(), args.teacher_b.resolve(), args.multistep_b.resolve()),
        }
        teacher_compare = args.teacher_comparison / "teacher_forced_comparison.json"
        multistep_compare = args.multistep_comparison / "curriculum_comparison.json"
        tc, mc = obj(teacher_compare), obj(multistep_compare)
        require(sha(teacher_compare) == "c12f5cfe6da9e3595658dba4eacf3239dd46e2b8312a7212e50479f770237aa2"
                and tc.get("status") == "complete" and tc.get("effect_direction") == "A_to_B_minus_B_to_B",
                "teacher comparison mismatch")
        require(sha(multistep_compare) == "bc77b74c33314f93b467f076b86f9d4cfad0305d71eeecfbed440d0a75fc110e"
                and mc.get("status") == "complete" and mc.get("variant") == "lr5e5"
                and mc.get("effect_direction") == "A_to_B_minus_B_to_B",
                "multistep comparison mismatch")
        before = available()
        # The exact, non-symlinked targets above are the only destructive paths.
        shutil.rmtree(args.source_a.resolve() / "orbax")
        shutil.rmtree(args.source_b.resolve() / "orbax")
        after = available()
        require(not (args.source_a / "orbax").exists() and not (args.source_b / "orbax").exists(),
                "an exact Orbax target remained")
        for recovered in (args.recovered_a, args.recovered_b):
            require((recovered / "hf/model.safetensors").stat().st_size == HF_WEIGHT_BYTES,
                    "retained recovered HF disappeared")
        result = {"schema_version": 1,
                  "artifact_type": "synthetic_multistep_curriculum_lr5e5_orbax_cleanup",
                  "status": "complete", "cpu_only": True, "exact_targets": records,
                  "removed_logical_bytes": sum(x["logical_bytes"] for x in records.values()),
                  "removed_allocated_bytes": sum(x["allocated_bytes"] for x in records.values()),
                  "filesystem_available_bytes_before": before,
                  "filesystem_available_bytes_after": after,
                  "filesystem_available_delta_bytes": after - before,
                  "retained": "both low-LR HF exports; every manifest/log/row/report/comparison; all original-LR, stage-1, and typing artifacts",
                  "comparison_hashes": {"teacher_forced": sha(teacher_compare),
                                        "multistep": sha(multistep_compare)}}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cleanup_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CleanupError, OSError, json.JSONDecodeError, StopIteration) as exc:
        print(f"FATAL cleanup gate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
