#!/usr/bin/env python3
"""Sealed descriptive comparison of final normalized-v2 evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import train_contract


EXPECTED_NAMES = {
    "final": "phaseb_normalized_v2_eval_A_to_A_r256_s900_v1_run_019fb71df66776f383d630f5d5763095",
    "warmstart": "phaseb_normalized_v2_warmstart_reltool_pre_r256_eval_v1",
    "absolute": "phaseb_eval_prose_keep_recovery_135312_v1_run_019fb56b2f8379d1a98671befabd0eca",
}
CANONICAL_GOLD = "abe4dc7891662c1f325bf2d7e4d4b49c804ab2c9e75d034858663be0b0bd8412"


class ComparisonError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"expected object: {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 233 or any(not isinstance(row, dict) for row in rows):
        raise ComparisonError(f"expected exactly 233 object rows: {path}")
    return rows


def exact_root(label: str, root: Path) -> Path:
    resolved = root.resolve()
    if root.is_symlink() or resolved.name != EXPECTED_NAMES[label]:
        raise ComparisonError(f"unexpected {label} root: {root}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--absolute", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    final_root = exact_root("final", args.final)
    warm_root = exact_root("warmstart", args.warmstart)
    absolute_root = exact_root("absolute", args.absolute)
    final_manifest, final_report = (
        load(final_root / "eval_manifest.json"), load(final_root / "report.json")
    )
    warm_manifest, warm_report = (
        load(warm_root / "eval_manifest.json"), load(warm_root / "report.json")
    )
    absolute_manifest, absolute_report = (
        load(absolute_root / "eval_manifest.json"), load(absolute_root / "report.json")
    )
    final_rows = load_rows(final_root / "rows.jsonl")
    warm_rows = load_rows(warm_root / "rows.jsonl")
    absolute_rows = load_rows(absolute_root / "rows.jsonl")

    for label, root, manifest, report in (
        ("final", final_root, final_manifest, final_report),
        ("warmstart", warm_root, warm_manifest, warm_report),
    ):
        if (
            manifest.get("status") != "complete"
            or manifest.get("valid") is not True
            or manifest.get("schema") != "normalized"
            or manifest.get("canonical_gold_sha256") != CANONICAL_GOLD
            or manifest.get("request_errors") != 0
            or manifest.get("report_sha256") != sha256(root / "report.json")
            or manifest.get("rows_sha256") != sha256(root / "rows.jsonl")
            or report.get("valid") is not True
            or report.get("summary", {}).get("n_rows") != 233
        ):
            raise ComparisonError(f"{label} canonical evaluation seal failed")
    if (
        absolute_manifest.get("valid") is not True
        or absolute_manifest.get("arm") != "prose_keep"
        or absolute_manifest.get("evaluation", {}).get("request_errors") != 0
        or absolute_manifest.get("evaluation", {}).get("report_sha256")
        != sha256(absolute_root / "report.json")
        or absolute_manifest.get("evaluation", {}).get("rows_sha256")
        != sha256(absolute_root / "rows.jsonl")
        or absolute_manifest.get("own_val_contract", {}).get("n_rows") != 233
        or absolute_manifest.get("own_val_contract", {}).get("n_coordinate_rows") != 178
        or absolute_report.get("meta", {}).get("valid") is not True
        or absolute_report.get("summary", {}).get("n_rows") != 233
    ):
        raise ComparisonError("absolute prose_keep baseline seal failed")
    ids = [[row.get("sample_id") for row in rows]
           for rows in (final_rows, warm_rows, absolute_rows)]
    coord_flags = [[row.get("is_coord_record") for row in rows]
                   for rows in (final_rows, warm_rows, absolute_rows)]
    if ids[0] != ids[1] or ids[0] != ids[2] or coord_flags[0] != coord_flags[1] \
            or coord_flags[0] != coord_flags[2] or sum(coord_flags[0]) != 178:
        raise ComparisonError("row identity/order/coordinate flags differ across comparisons")
    prereg = load(args.preregistration)
    if prereg != train_contract.PREREGISTRATION:
        raise ComparisonError("normalized-v2 preregistration changed")

    final = final_report["summary"]
    warm = warm_report["summary"]
    absolute = absolute_report["summary"]
    canonical_metrics = sorted(
        key for key in final
        if key in warm and isinstance(final[key], (int, float))
    )
    final_vs_warm = {
        key: {"final": final[key], "warmstart": warm[key], "delta": final[key] - warm[key]}
        for key in canonical_metrics
    }
    shared_absolute_metrics = (
        "n_rows", "n_request_errors", "n_coord_records", "median_err_px",
        "within_50px", "within_100px",
    )
    final_vs_absolute = {
        key: {"final": final[key], "absolute": absolute[key],
              "delta": final[key] - absolute[key]}
        for key in shared_absolute_metrics
    }
    payload = {
        "artifact_type": "phaseb_normalized_v2_final_descriptive_comparison",
        "schema_version": 1,
        "status": "complete",
        "row_pairing": {
            "n_rows": 233,
            "n_coordinate_rows": 178,
            "sample_id_order_identical": True,
            "coordinate_flags_identical": True,
        },
        "final_vs_warmstart": {
            "comparison": "exact same normalized canonical evaluator/gold/estimand",
            "metrics": final_vs_warm,
        },
        "final_vs_absolute_baseline": {
            "comparison": "same ordered validation examples; exact shared reported metrics only",
            "metrics": final_vs_absolute,
            "excluded_as_nonshared": {
                "final_action_sequence_agreement": final["action_sequence_agreement"],
                "absolute_action_type_agreement": absolute["action_type_agreement"],
                "reason": "sequence agreement and single action-type agreement are different definitions",
            },
            "coordinate_metric_caveat": (
                "descriptive across the canonical normalized evaluator and legacy absolute "
                "evaluator; no paired uncertainty or inferential gate was preregistered"
            ),
        },
        "preregistered_noninferiority": {
            "exists_for_final_vs_warmstart": False,
            "exists_for_final_vs_absolute_baseline": False,
            "result": "not_applicable_no_preregistered_gate",
            "evidence": (
                "the exact saved normalized-v2 PREREGISTRATION specifies training, timing, "
                "and external-eval contracts but no performance margin, CI rule, or NI gate"
            ),
            "separate_stage1_5_gate_not_reused": True,
        },
        "seals": {
            "preregistration_sha256": sha256(args.preregistration),
            "final_manifest_sha256": sha256(final_root / "eval_manifest.json"),
            "final_report_sha256": sha256(final_root / "report.json"),
            "final_rows_sha256": sha256(final_root / "rows.jsonl"),
            "warmstart_manifest_sha256": sha256(warm_root / "eval_manifest.json"),
            "warmstart_report_sha256": sha256(warm_root / "report.json"),
            "warmstart_rows_sha256": sha256(warm_root / "rows.jsonl"),
            "absolute_manifest_sha256": sha256(absolute_root / "eval_manifest.json"),
            "absolute_report_sha256": sha256(absolute_root / "report.json"),
            "absolute_rows_sha256": sha256(absolute_root / "rows.jsonl"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
