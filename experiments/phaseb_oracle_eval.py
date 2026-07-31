#!/usr/bin/env python3
"""Shared 233-row oracle-history evaluator for Phase-B action grammars."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_EXPERIMENTS / "phaseb_relative"))
sys.path.insert(0, str(_EXPERIMENTS / "phaseb_deltatype_raw_v2"))
import relative_eval as wire_base  # noqa: E402
from action_v2 import format_deltatype_v2, parse_deltatype_v2  # noqa: E402
from phaseb_canonical_eval import (  # noqa: E402
    CanonicalError,
    Plan,
    canonical_normalized,
    canonical_raw,
    net_delta,
)


EXPECTED_ROWS = 233
EXPECTED_COORD = 178
EXPECTED_RAW_SHA256 = "a819011d5f8524cad1980d720fcdbc98a838a37b33de499c46eb4c13c94acadd"
EXPECTED_NORMALIZED_SHA256 = "b51221df5f044f21092fec6a973c6d8164a7119f2b4971841fc5926df6e9ef7c"
EXPECTED_CANONICAL_GOLD_SHA256 = "abe4dc7891662c1f325bf2d7e4d4b49c804ab2c9e75d034858663be0b0bd8412"
COORD_SOURCE_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "left_click_drag", "mouse_down",
}
EXPORT_TYPES = {
    "normalized": "phaseb_normalized_move_rel_v2_A_to_A_hf_checkpoint",
    "raw": "phaseb_raw_deltatype_v2_A_to_B_hf_checkpoint",
}
WARMSTART_MANIFEST_SHA256 = {
    "normalized": "d37db583163fddf85c40815417b77cafaba42bafaa1ab31382a1d427ab054e71",
    "raw": "f0b8d729e4dfd5f1352ec92e780b13ec6ed6bf938f0e0ee7a0389750ff66cd46",
}
WARMSTART_CONFIG_SHA256 = "306e36b825faad2e26a884add51223b67c8ef109521c69c4bb72ddd490e73efa"
WARMSTART_WEIGHT_BYTES = 35_068_587_488
_IMAGES: dict[str, str] = {}
_LOCK = threading.Lock()
_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class EvalError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(key) for key in ("sample_id", "recording_id", "app", "task_id", "step"))


def _raw_label(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise CanonicalError("empty raw-v2 response")
    label = lines[-1]
    action = parse_deltatype_v2(label)
    if format_deltatype_v2(action) != label:
        raise CanonicalError("raw-v2 response is not canonical")
    return label


def _textual_calls(text: str) -> list[dict[str, Any]]:
    matches = list(_TOOL_BLOCK.finditer(text))
    if (text.count("<tool_call>") != len(matches)
            or text.count("</tool_call>") != len(matches)):
        raise CanonicalError("malformed or unmatched tool-call block")
    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise CanonicalError(f"malformed tool-call JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise CanonicalError("tool-call payload must contain exactly name and arguments")
        if payload.get("name") != "computer_use" or not isinstance(payload.get("arguments"), dict):
            raise CanonicalError("non-computer_use or non-object tool arguments")
        calls.append(dict(payload["arguments"]))
    return calls


def _structured_calls(structured: Any) -> list[dict[str, Any]]:
    if structured is None:
        return []
    if not isinstance(structured, (list, tuple)):
        raise CanonicalError("structured tool_calls is not a list")
    calls: list[dict[str, Any]] = []
    for call in structured:
        function = (call.get("function", call) if isinstance(call, dict)
                    else getattr(call, "function", call))
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
        arguments = (function.get("arguments") if isinstance(function, dict)
                     else getattr(function, "arguments", None))
        if name != "computer_use":
            raise CanonicalError("structured call is not computer_use")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise CanonicalError(f"malformed structured arguments: {exc}") from exc
        if not isinstance(arguments, dict):
            raise CanonicalError("structured tool arguments is not an object")
        calls.append(dict(arguments))
    return calls


def normalized_calls(text: str, structured: Any = None) -> list[dict[str, Any]]:
    textual = _textual_calls(text)
    structured_values = _structured_calls(structured)
    if not textual and not structured_values:
        raise CanonicalError("no normalized computer-use calls")
    if textual:
        textual_plan = canonical_normalized(textual)
    if structured_values:
        structured_plan = canonical_normalized(structured_values)
    if textual and structured_values and textual_plan != structured_plan:
        raise CanonicalError("textual and structured tool-call channels disagree")
    return textual if textual else structured_values


def _gold(raw_record: dict[str, Any], normalized_record: dict[str, Any]) -> dict[str, Any]:
    if _identity(raw_record) != _identity(normalized_record):
        raise EvalError("raw/normalized held-out identity mismatch")
    audit = raw_record.get("raw_deltatype_v2_audit")
    if not isinstance(audit, list) or not audit:
        raise EvalError("raw held-out record lacks full-source audit")
    item = audit[-1]
    sequence = str(item.get("source_sequence", ""))
    raw_plan = canonical_raw(str(item.get("label", "")), sequence)
    calls = normalized_calls(wire_base.txt(normalized_record["messages"][-1]["content"]))
    normalized_plan = canonical_normalized(calls)
    if raw_plan != normalized_plan:
        raise EvalError(f"paired gold canonical plan mismatch: {_identity(raw_record)}")
    return {
        "plan": raw_plan,
        "source_sequence": sequence,
        "is_coord": bool(set(sequence.split("+")) & COORD_SOURCE_ACTIONS),
    }


def load_gold(raw_path: Path, normalized_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256(raw_path) != EXPECTED_RAW_SHA256:
        raise EvalError("sealed raw-v2 held-out file changed")
    if sha256(normalized_path) != EXPECTED_NORMALIZED_SHA256:
        raise EvalError("sealed normalized-v2 held-out file changed")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    normalized = [json.loads(line) for line in normalized_path.read_text().splitlines() if line.strip()]
    if len(raw) != EXPECTED_ROWS or len(normalized) != EXPECTED_ROWS:
        raise EvalError("held-out row count changed")
    gold = [_gold(r, n) for r, n in zip(raw, normalized, strict=True)]
    payload = [[r.get("sample_id"), item["plan"]] for r, item in zip(raw, gold, strict=True)]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != EXPECTED_CANONICAL_GOLD_SHA256:
        raise EvalError(f"canonical gold seal changed: {digest}")
    if sum(item["is_coord"] for item in gold) != EXPECTED_COORD:
        raise EvalError("coordinate-row count changed")
    return raw, normalized, gold


def _parse(schema: str, text: str, structured: Any = None,
           source_sequence: str = "") -> Plan:
    if schema == "raw":
        return canonical_raw(_raw_label(text), source_sequence)
    calls = normalized_calls(text, structured)
    return canonical_normalized(calls)


def _skeleton(plan: Plan) -> tuple[tuple[Any, ...], ...]:
    return tuple(("move_px",) if item[0] == "move_px" else item for item in plan)


def _moves(plan: Plan) -> list[tuple[int, int]]:
    return [(int(item[1]), int(item[2])) for item in plan if item[0] == "move_px"]


def score(schema: str, record: dict[str, Any], gold: dict[str, Any],
          raw: str, structured: Any = None) -> dict[str, Any]:
    gold_plan: Plan = gold["plan"]
    row = {
        "sample_id": record.get("sample_id"),
        "schema": schema,
        "source_sequence": gold["source_sequence"],
        "is_coord_record": gold["is_coord"],
        "schema_parse_ok": False,
        "action_sequence_match": False,
        "non_motion_payload_order_match": False,
        "canonical_exact_plan_match": False,
        "canonical_tolerant_50px_match": False,
        "canonical_tolerant_100px_match": False,
        "motion_segment_count_match": False,
        "gold_plan": [list(item) for item in gold_plan],
        "pred_plan": [],
        "motion_segment_errors_px": [],
        "net_landing_err_px": None,
        "request_error": False,
        "raw_output": raw[:1200],
    }
    try:
        pred = _parse(schema, raw, structured, gold["source_sequence"])
    except (CanonicalError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        return row
    gold_moves, pred_moves = _moves(gold_plan), _moves(pred)
    skeleton_match = _skeleton(pred) == _skeleton(gold_plan)
    errors = ([math.dist(actual, expected) for actual, expected in zip(pred_moves, gold_moves)]
              if len(pred_moves) == len(gold_moves) else [])
    gold_delta, pred_delta = net_delta(gold_plan), net_delta(pred)
    row.update({
        "schema_parse_ok": True,
        "action_sequence_match": [item[0] for item in pred] == [item[0] for item in gold_plan],
        "non_motion_payload_order_match": skeleton_match,
        "canonical_exact_plan_match": pred == gold_plan,
        "canonical_tolerant_50px_match": (
            skeleton_match and len(errors) == len(gold_moves) and all(value <= 50 for value in errors)
        ),
        "canonical_tolerant_100px_match": (
            skeleton_match and len(errors) == len(gold_moves) and all(value <= 100 for value in errors)
        ),
        "motion_segment_count_match": len(pred_moves) == len(gold_moves),
        "pred_plan": [list(item) for item in pred],
        "motion_segment_errors_px": errors,
        "net_landing_err_px": math.dist(pred_delta, gold_delta),
    })
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coord = [row for row in rows if row["is_coord_record"]]
    coord_errors = sorted(
        row["net_landing_err_px"] for row in coord if row["net_landing_err_px"] is not None
    )
    def rate(key: str, values: list[dict[str, Any]] = rows) -> float:
        return sum(bool(row[key]) for row in values) / len(values)
    return {
        "n_rows": len(rows),
        "n_coord_records": len(coord),
        "n_request_errors": sum(row["request_error"] for row in rows),
        "parse_rate": rate("schema_parse_ok"),
        "action_sequence_agreement": rate("action_sequence_match"),
        "non_motion_payload_order_agreement": rate("non_motion_payload_order_match"),
        "canonical_exact_plan_agreement": rate("canonical_exact_plan_match"),
        "canonical_tolerant_50px_agreement": rate("canonical_tolerant_50px_match"),
        "canonical_tolerant_100px_agreement": rate("canonical_tolerant_100px_match"),
        "motion_segment_count_agreement": rate("motion_segment_count_match"),
        "coord_row_comparable_rate": len(coord_errors) / len(coord),
        "median_err_px": coord_errors[len(coord_errors) // 2] if coord_errors else None,
        "within_50px": sum(value <= 50 for value in coord_errors) / len(coord),
        "within_100px": sum(value <= 100 for value in coord_errors) / len(coord),
    }


def selftest(raw_records: list[dict[str, Any]], normalized_records: list[dict[str, Any]],
             gold: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {"raw": [], "normalized": []}
    for schema, records in (("raw", raw_records), ("normalized", normalized_records)):
        for record, item in zip(records, gold, strict=True):
            text = wire_base.txt(record["messages"][-1]["content"])
            rows[schema].append(score(schema, record, item, text))
    reports = {schema: summarize(values) for schema, values in rows.items()}
    for schema, report in reports.items():
        required = (
            "parse_rate", "action_sequence_agreement", "non_motion_payload_order_agreement",
            "canonical_exact_plan_agreement", "canonical_tolerant_50px_agreement",
            "canonical_tolerant_100px_agreement", "motion_segment_count_agreement",
            "coord_row_comparable_rate", "within_50px", "within_100px",
        )
        if any(report[key] != 1 for key in required) or report["median_err_px"] != 0:
            raise EvalError(f"{schema} canonical gold selftest failed: {report}")
    return reports


def validate_model(schema: str, stage: str, model_dir: Path,
                   manifest_path: Path) -> dict[str, Any]:
    if not (model_dir / "config.json").is_file():
        raise EvalError("model config is missing")
    manifest = json.loads(manifest_path.read_text())
    if stage == "phaseb_final":
        valid = (
            manifest.get("artifact_type") == EXPORT_TYPES[schema]
            and manifest.get("arm") == ("raw_v2" if schema == "raw" else "normalized_v2")
            and manifest.get("status") == "complete" and manifest.get("step") == 900
            and manifest.get("config_sha256") == sha256(model_dir / "config.json")
        )
        expected_weights = {item.get("name"): item.get("size")
                            for item in manifest.get("weights", []) if isinstance(item, dict)}
        actual_weights = {path.name: path.stat().st_size for path in model_dir.glob("*.safetensors")}
        valid = valid and bool(actual_weights) and actual_weights == expected_weights
    elif schema == "normalized":
        weights = model_dir / "model.safetensors"
        valid = (
            sha256(manifest_path) == WARMSTART_MANIFEST_SHA256[schema]
            and manifest.get("artifact_type") == "relative_factorial_hf_checkpoint"
            and manifest.get("arm") == "reltool_pre" and manifest.get("status") == "complete"
            and manifest.get("step") == 750 and manifest.get("lora_rank") == 256
            and sha256(model_dir / "config.json") == WARMSTART_CONFIG_SHA256
            and weights.is_file() and weights.stat().st_size == WARMSTART_WEIGHT_BYTES
        )
    else:
        weights = model_dir / "model.safetensors"
        valid = (
            sha256(manifest_path) == WARMSTART_MANIFEST_SHA256[schema]
            and manifest.get("artifact_type") == "synthetic_multistep_curriculum_hf_checkpoint"
            and manifest.get("branch") == "A_to_B"
            and manifest.get("target_format") == "deltatype_raw_pre"
            and manifest.get("status") == "complete" and manifest.get("step") == 750
            and manifest.get("lora_rank") == 256
            and sha256(model_dir / "config.json") == WARMSTART_CONFIG_SHA256
            and weights.is_file() and weights.stat().st_size == WARMSTART_WEIGHT_BYTES
        )
    if not valid:
        raise EvalError("wrong or unsealed model manifest/endpoint")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=("normalized", "raw"), required=True)
    parser.add_argument("--val-chat", type=Path, required=True)
    parser.add_argument("--raw-gold", type=Path, required=True)
    parser.add_argument("--normalized-gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-manifest", "--export-manifest", dest="model_manifest", type=Path)
    parser.add_argument("--model-stage", choices=("warmstart", "phaseb_final"), required=True)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    try:
        if not args.model_dir or not args.model_manifest:
            raise EvalError("model directory and sealed model manifest are required")
        model_manifest = validate_model(
            args.schema, args.model_stage, args.model_dir, args.model_manifest
        )
        raw_records, normalized_records, gold = load_gold(args.raw_gold, args.normalized_gold)
        target = raw_records if args.schema == "raw" else normalized_records
        if sha256(args.val_chat) != sha256(args.raw_gold if args.schema == "raw" else args.normalized_gold):
            raise EvalError("target held-out file is not the sealed schema gold file")
        args.out.mkdir(parents=True, exist_ok=True)
        if args.selftest:
            reports = selftest(raw_records, normalized_records, gold)
            (args.out / "canonical_gold_selftest.json").write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n"
            )
            print(f"paired canonical gold selftest PASS: {reports}")
            if not args.base_url:
                return 0
        if not args.base_url:
            raise EvalError("inference requires base URL")
        client = wire_base.OpenAI(base_url=args.base_url, api_key="x", timeout=600, max_retries=3)
        def evaluate(pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            record, item = pair
            try:
                completion = client.chat.completions.create(
                    model=args.model, messages=wire_base.wire(record["messages"][:-1]),
                    max_tokens=256, temperature=0.0,
                )
                message = completion.choices[0].message
                return score(args.schema, record, item, message.content or "",
                             getattr(message, "tool_calls", None))
            except Exception as exc:
                row = score(args.schema, record, item, "")
                row.update(request_error=True, request_error_type=type(exc).__name__,
                           request_error_message=str(exc)[:400])
                return row
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(evaluate, zip(target, gold, strict=True)))
        summary = summarize(rows)
        if summary["n_request_errors"] != 0:
            raise EvalError(f"request errors invalidate evaluation: {summary}")
        rows_path = args.out / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = {
            "valid": True,
            "schema": args.schema,
            "summary": summary,
            "sampling": {"temperature": 0.0, "max_tokens": 256},
            "estimand": "oracle_history_single_turn_greedy_generation",
            "token_forced_nll": False,
            "cross_format_string_comparison": False,
            "runtime_timing_equivalent": False,
            "canonical_gold_sha256": EXPECTED_CANONICAL_GOLD_SHA256,
            "canonical_note": (
                "exact ordered non-motion payload/order plus pixel motion vectors; raw drag duration "
                "and normalized zero-move VM suppression are excluded from logical equivalence"
            ),
        }
        report_path = args.out / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest = {
            "artifact_type": f"phaseb_{args.schema}_canonical_oracle_eval",
            "schema_version": 1, "status": "complete", "valid": True,
            "schema": args.schema, "model_dir": str(args.model_dir.resolve()),
            "model_stage": args.model_stage,
            "model_manifest_sha256": sha256(args.model_manifest),
            "raw_gold_sha256": sha256(args.raw_gold),
            "normalized_gold_sha256": sha256(args.normalized_gold),
            "canonical_gold_sha256": EXPECTED_CANONICAL_GOLD_SHA256,
            "report_sha256": sha256(report_path), "rows_sha256": sha256(rows_path),
            "sampling": {"temperature": 0.0, "max_tokens": 256},
            "estimand": "oracle_history_single_turn_greedy_generation",
            "token_forced_nll": False, "cross_format_string_comparison": False,
            "runtime_timing_equivalent": False, "request_errors": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (args.out / "eval_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except (EvalError, CanonicalError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL Phase-B canonical oracle eval: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
