#!/usr/bin/env python3
"""Fail-closed aggregate for the four task-preserving Phase-B train shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))
import phaseb_oracle_eval as oracle  # noqa: E402


class AggregateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def cluster_bootstrap(rows: list[dict[str, Any]], *, seed: int = 20260803,
                      replicates: int = 5000) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["app"]), str(row["task_id"]))].append(row)
    keys = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        drawn = [grouped[rng.choice(keys)] for _ in keys]
        flat = [row for group in drawn for row in group]
        coord = [row for row in flat if row["is_coord_record"] and row["net_landing_err_px"] is not None]
        samples["canonical_exact_plan_agreement"].append(
            sum(row["canonical_exact_plan_match"] for row in flat) / len(flat)
        )
        samples["canonical_tolerant_50px_agreement"].append(
            sum(row["canonical_tolerant_50px_match"] for row in flat) / len(flat)
        )
        samples["action_sequence_agreement"].append(
            sum(row["action_sequence_match"] for row in flat) / len(flat)
        )
        samples["within_50px"].append(
            sum(row["net_landing_err_px"] <= 50 for row in coord) / len(coord)
        )
    return {
        "unit": "(app,task_id)", "seed": seed, "replicates": replicates,
        "interval": "percentile_95pct",
        "metrics": {
            key: {"lower": percentile(values, 0.025), "upper": percentile(values, 0.975)}
            for key, values in sorted(samples.items())
        },
    }


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["is_coord_record"] for row in rows):
        return oracle.summarize(rows)
    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows)
    return {
        "n_rows": len(rows), "n_coord_records": 0,
        "n_request_errors": sum(bool(row["request_error"]) for row in rows),
        "parse_rate": rate("schema_parse_ok"),
        "action_sequence_agreement": rate("action_sequence_match"),
        "non_motion_payload_order_agreement": rate("non_motion_payload_order_match"),
        "canonical_exact_plan_agreement": rate("canonical_exact_plan_match"),
        "canonical_tolerant_50px_agreement": rate("canonical_tolerant_50px_match"),
        "canonical_tolerant_100px_agreement": rate("canonical_tolerant_100px_match"),
        "motion_segment_count_agreement": rate("motion_segment_count_match"),
        "coord_row_comparable_rate": None, "median_err_px": None,
        "within_50px": None, "within_100px": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for index in range(4):
        parser.add_argument(f"--shard{index}", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--normalized-train", type=Path, required=True)
    parser.add_argument("--val-result", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        raw_records, _, _ = oracle.load_gold(args.raw_train, args.normalized_train)
        expected_order = [str(record["sample_id"]) for record in raw_records]
        require(len(expected_order) == 2383 and len(set(expected_order)) == 2383,
                "sealed train source IDs are not 2,383 unique records")
        expected_tasks = {(str(record["app"]), str(record["task_id"])) for record in raw_records}
        require(len(expected_tasks) == 215, "sealed train source is not 215 tasks")

        shard_roots = [getattr(args, f"shard{index}") for index in range(4)]
        all_rows: list[dict[str, Any]] = []
        seen_tasks: set[tuple[str, str]] = set()
        shard_evidence: list[dict[str, Any]] = []
        for expected_index, root in enumerate(shard_roots):
            manifest_path, report_path, rows_path = (
                root / "eval_manifest.json", root / "report.json", root / "rows.jsonl"
            )
            require(all(path.is_file() for path in (manifest_path, report_path, rows_path)),
                    f"shard {expected_index} is incomplete")
            manifest, report, rows = read_json(manifest_path), read_json(report_path), read_rows(rows_path)
            shard = manifest.get("shard", {})
            require(manifest.get("valid") is True and manifest.get("status") == "complete",
                    f"shard {expected_index} not valid/complete")
            require(manifest.get("dataset_kind") == "train", f"shard {expected_index} wrong split")
            require(manifest.get("dataset_rows") == 2383 and manifest.get("dataset_tasks") == 215,
                    f"shard {expected_index} wrong source cardinality")
            require(shard.get("index") == expected_index and shard.get("count") == 4,
                    f"shard {expected_index} wrong shard contract")
            require(manifest.get("raw_gold_sha256") == oracle.TRAIN_RAW_SHA256,
                    f"shard {expected_index} raw source mismatch")
            require(manifest.get("normalized_gold_sha256") == oracle.TRAIN_NORMALIZED_SHA256,
                    f"shard {expected_index} normalized source mismatch")
            require(manifest.get("canonical_gold_sha256") == oracle.TRAIN_CANONICAL_GOLD_SHA256,
                    f"shard {expected_index} canonical source mismatch")
            require(manifest.get("model_manifest_sha256") == sha256(args.model_manifest),
                    f"shard {expected_index} model mismatch")
            require(manifest.get("sampling") == {"temperature": 0.0, "max_tokens": 256},
                    f"shard {expected_index} sampling mismatch")
            require(manifest.get("estimand") == "oracle_history_single_turn_greedy_generation",
                    f"shard {expected_index} estimand mismatch")
            require(manifest.get("request_errors") == 0 and report["summary"]["n_request_errors"] == 0,
                    f"shard {expected_index} has request errors")
            require(sha256(report_path) == manifest.get("report_sha256"),
                    f"shard {expected_index} report digest mismatch")
            require(sha256(rows_path) == manifest.get("rows_sha256"),
                    f"shard {expected_index} rows digest mismatch")
            require(len(rows) == shard.get("rows") == report["summary"]["n_rows"],
                    f"shard {expected_index} row count mismatch")
            tasks = {(str(row["app"]), str(row["task_id"])) for row in rows}
            require(len(tasks) == shard.get("tasks"), f"shard {expected_index} task count mismatch")
            require(not (tasks & seen_tasks), f"shard {expected_index} task overlap")
            seen_tasks |= tasks
            all_rows.extend(rows)
            shard_evidence.append({
                "index": expected_index, "root": str(root.resolve()),
                "slurm_job_id": manifest.get("slurm_job_id"),
                "rows": len(rows), "tasks": len(tasks),
                "manifest_sha256": sha256(manifest_path),
                "report_sha256": sha256(report_path), "rows_sha256": sha256(rows_path),
            })

        ids = [str(row["sample_id"]) for row in all_rows]
        require(len(ids) == 2383 and len(set(ids)) == 2383, "aggregate rows are not 2,383 unique IDs")
        require(set(ids) == set(expected_order), "aggregate has source omission or foreign ID")
        require(seen_tasks == expected_tasks, "aggregate task coverage mismatch")
        by_id = {str(row["sample_id"]): row for row in all_rows}
        rows = [by_id[sample_id] for sample_id in expected_order]
        require(sum(bool(row["request_error"]) for row in rows) == 0, "aggregate request errors")

        overall = oracle.summarize(rows)
        val_report = read_json(args.val_result / "report.json")
        val = val_report["summary"]
        comparable = [
            "parse_rate", "action_sequence_agreement", "non_motion_payload_order_agreement",
            "canonical_exact_plan_agreement", "canonical_tolerant_50px_agreement",
            "canonical_tolerant_100px_agreement", "motion_segment_count_agreement",
            "coord_row_comparable_rate", "median_err_px", "within_50px", "within_100px",
        ]
        train_vs_val = {
            key: {"train": overall[key], "val": val[key], "delta_train_minus_val": overall[key] - val[key]}
            for key in comparable
        }

        by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
        movement_bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_app[str(row["app"])].append(row)
            by_action[str(row["source_sequence"])].append(row)
            by_step[str(row["step"])].append(row)
            if row["is_coord_record"]:
                dx = sum(item[1] for item in row["gold_plan"] if item[0] == "move_px")
                dy = sum(item[2] for item in row["gold_plan"] if item[0] == "move_px")
                magnitude = math.hypot(dx, dy)
                label = ("0" if magnitude == 0 else "(0,50]" if magnitude <= 50 else
                         "(50,100]" if magnitude <= 100 else "(100,250]" if magnitude <= 250 else
                         "(250,500]" if magnitude <= 500 else ">500")
                movement_bins[label].append(row)
        stratified = {
            "app": {key: subset_summary(value) for key, value in sorted(by_app.items())},
            "source_action_sequence": {
                key: subset_summary(value) for key, value in sorted(by_action.items())
            },
            "trajectory_step": {key: subset_summary(value) for key, value in sorted(by_step.items(), key=lambda x: int(x[0]))},
            "gold_movement_magnitude_px": {
                key: subset_summary(value) for key, value in movement_bins.items()
            },
        }

        args.out.mkdir(parents=True, exist_ok=True)
        rows_path = args.out / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        coverage = {
            "expected_rows": 2383, "observed_rows": len(rows), "unique_sample_ids": len(set(ids)),
            "missing_sample_ids": [], "foreign_sample_ids": [], "duplicate_sample_ids": 0,
            "expected_tasks": 215, "observed_tasks": len(seen_tasks), "task_overlap_across_shards": 0,
            "request_errors": 0, "order": "sealed_source_row_order", "valid": True,
        }
        (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        report = {
            "valid": True, "schema": "raw", "dataset_kind": "train",
            "result_classification": "non_rollout_teacher_forced_evaluation",
            "estimand": "oracle_history_single_turn_greedy_generation",
            "history_conditioning": "gold_prefix", "token_forced_nll": False,
            "sampling": {"temperature": 0.0, "max_tokens": 256},
            "summary": overall, "train_vs_val": train_vs_val,
            "task_cluster_bootstrap": cluster_bootstrap(rows), "stratified": stratified,
            "coverage": coverage,
        }
        report_path = args.out / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        evaluator_files = {
            str(path.relative_to(EXPERIMENTS)): sha256(path) for path in (
                EXPERIMENTS / "phaseb_oracle_eval.py",
                EXPERIMENTS / "phaseb_canonical_eval.py",
                EXPERIMENTS / "phaseb_deltatype_raw_v2" / "action_v2.py",
                EXPERIMENTS / "phaseb_relative" / "relative_eval.py",
                Path(__file__).resolve(),
            )
        }
        manifest = {
            "artifact_type": "phaseb_raw_train_canonical_oracle_eval_aggregate",
            "schema_version": 1, "status": "complete", "valid": True,
            "dataset_kind": "train", "dataset_rows": 2383, "dataset_tasks": 215,
            "raw_gold_sha256": oracle.TRAIN_RAW_SHA256,
            "normalized_gold_sha256": oracle.TRAIN_NORMALIZED_SHA256,
            "canonical_gold_sha256": oracle.TRAIN_CANONICAL_GOLD_SHA256,
            "model_manifest_sha256": sha256(args.model_manifest),
            "evaluator_files": evaluator_files, "shards": shard_evidence,
            "coverage_sha256": sha256(args.out / "coverage.json"),
            "rows_sha256": sha256(rows_path), "report_sha256": sha256(report_path),
            "sampling": {"temperature": 0.0, "max_tokens": 256},
            "estimand": "oracle_history_single_turn_greedy_generation",
            "history_conditioning": "gold_prefix", "token_forced_nll": False,
            "request_errors": 0, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (args.out / "eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (AggregateError, oracle.EvalError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"FATAL train eval aggregate: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
