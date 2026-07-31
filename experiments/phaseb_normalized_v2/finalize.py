#!/usr/bin/env python3
"""Seal the tokenized full-call move_rel control artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


EXPECTED_WARMSTART_MANIFEST = "d37db583163fddf85c40815417b77cafaba42bafaa1ab31382a1d427ab054e71"
EXPECTED_OUTPUT_SHA256 = {
    "train": "4cc72eb35c845ecd1aad5412ee0872f9be784675452677a88face644236c97aa",
    "val": "b51221df5f044f21092fec6a973c6d8164a7119f2b4971841fc5926df6e9ef7c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--omegalax", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    build_path = out / "build_manifest.json"
    build = load(build_path)
    expected = {
        "status": "complete", "format": "move_rel_full_v2", "records": 2616,
        "assistant_spans": 10721, "source_tool_calls": 11471,
        "source_multi_call_spans": 750, "source_drag_spans": 444,
        "output_tool_calls": 18483,
        "calls_0_collapse_forbidden": True,
    }
    bad = {key: (build.get(key), value) for key, value in expected.items()
           if build.get(key) != value}
    if bad:
        raise SystemExit(f"FATAL normalized-v2 build contract mismatch: {bad}")
    if build.get("output_file_sha256") != EXPECTED_OUTPUT_SHA256:
        raise SystemExit("FATAL deterministic normalized-v2 output seals changed")
    manifest_path = args.source_model / "train_export_manifest.json"
    if sha256(manifest_path) != EXPECTED_WARMSTART_MANIFEST:
        raise SystemExit("FATAL normalized Phase-A warm-start manifest changed")
    warm = load(manifest_path)
    if not (warm.get("status") == "complete" and warm.get("arm") == "reltool_pre"
            and warm.get("step") == 750 and warm.get("lora_rank") == 256
            and warm.get("lora_alpha") == 256):
        raise SystemExit(f"FATAL wrong normalized Phase-A warm-start: {warm}")
    tokenized = {}
    for split, count in (("train", 2383), ("val", 233)):
        root = out / "normalized_v2" / split
        metadata = load(root / "metadata.json")
        truncation = load(root / "truncation_stats.json")
        if metadata.get("num_records") != count or metadata.get("max_length") != 16384:
            raise SystemExit(f"FATAL {split} tokenized metadata mismatch: {metadata}")
        sessions = truncation.get("sessions", {})
        tokens = truncation.get("tokens", {})
        if (int(sessions.get("truncated_total", -1)) != 0
                or int(sessions.get("dropped_entirely", -1)) != 0
                or int(tokens.get("dropped", -1)) != 0):
            raise SystemExit(f"FATAL {split} contains truncated records: {truncation}")
        shards = sorted(root.glob("*.array_record"))
        if len(shards) != 1:
            raise SystemExit(f"FATAL {split} expected one compiled shard")
        tokenized[split] = {
            "records": count, "max_length": 16384,
            "metadata_sha256": sha256(root / "metadata.json"),
            "compiled_shard": shards[0].name,
            "compiled_shard_sha256": sha256(shards[0]),
        }
    vision_path = out / "vision_budget_preflight.json"
    vision = load(vision_path)
    if vision.get("status") != "pass" or vision.get("records_scanned") != 2616:
        raise SystemExit("FATAL normalized-v2 full-scan vision preflight failed")
    head = subprocess.check_output(
        ["git", "-C", str(args.omegalax), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(args.omegalax), "diff", "--binary"]
    )
    diff_hash = hashlib.sha256(diff).hexdigest()
    if head != "b3f32c002998a1134c78845847a53ca9cc17fb10" or diff_hash != (
        "cf71abb330177e1035d6aa8b134d9d7b4bd92e9bfb0ffd41ed9d485834b2ab14"
    ):
        raise SystemExit("FATAL Omegalax training code seal changed")
    manifest = {
        "artifact_type": "phaseb_normalized_move_rel_v2_tokenized",
        "schema_version": 1, "status": "complete", "format": "move_rel_full_v2",
        "build_manifest_sha256": sha256(build_path), "tokenized": tokenized,
        "vision_budget_preflight_sha256": sha256(vision_path),
        "warmstart_manifest_sha256": EXPECTED_WARMSTART_MANIFEST,
        "warmstart_artifact": str(args.source_model.resolve()),
        "omegalax_commit": head, "omegalax_tracked_diff_sha256": diff_hash,
        "training_contract": {
            "branch": "A_to_A", "fresh_lora": True, "fresh_optimizer": True,
            "lora_rank": 256, "lora_alpha": 256, "learning_rate": 0.0001,
            "num_steps": 900, "max_length": 16384, "in_loop_validation": False,
            "external_eval_only": True, "deadline": "2026-07-31T09:35:00+02:00",
            "comparison_estimand": "end_to_end_best_pipeline_control",
            "causal_action_semantics_only_claimed": False,
            "warmstart_exposure_matched_to_raw_v2": False,
            "warmstart_exposure_note": (
                "normalized A-to-A starts directly from Phase-A reltool_pre r256 step 750; "
                "raw-v2 A-to-B includes Phase-A step 750 plus a separate 750-step "
                "curriculum warm-start, so this comparison does not isolate action semantics"
            ),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    marker = out / "tokenization_manifest.json"
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    build_path.chmod(0o444)
    marker.chmod(0o444)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
