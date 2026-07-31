#!/usr/bin/env python3
"""Held-out oracle-history one-step evaluator for full-call normalized move_rel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phaseb_relative"))
import relative_eval as base  # noqa: E402


EXPECTED_ROWS = 233
EXPECTED_COORD = 178
COORD_SOURCE_ACTIONS = {
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "left_click_drag", "mouse_down",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher(record: dict[str, Any]) -> dict[str, Any]:
    audit = record.get("phaseb_normalized_v2_audit")
    if not isinstance(audit, list) or not audit:
        raise ValueError("missing normalized-v2 full-call audit metadata")
    state = audit[-1]
    text = base.txt(record["messages"][-1]["content"])
    calls = base.calls_from(text)
    if not calls:
        raise ValueError("teacher action does not parse")
    landing, actions = base.execute(calls, [0, 0])
    sequence = str(state.get("source_sequence", ""))
    is_coord = bool(set(sequence.split("+")) & COORD_SOURCE_ACTIONS)
    if int(state.get("output_call_count", -1)) != len(calls):
        raise ValueError("teacher ordered-call count disagrees with build audit")
    return {
        "calls": calls,
        "actions": actions,
        "landing": landing,
        "cursor_before": [0, 0],
        "is_coord": is_coord,
    }


base.teacher = teacher
_relative_score = base.score


def score(record: dict[str, Any], raw: str, structured: Any = None) -> dict[str, Any]:
    gold = teacher(record)
    parsed = base.calls_from(raw, structured)
    row = _relative_score(record, raw, structured)
    row.update({
        "teacher_calls": gold["calls"],
        "pred_calls": parsed,
        "ordered_call_payload_match": bool(parsed) and parsed == gold["calls"],
    })
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = base.summarize(rows)
    result["ordered_call_payload_agreement"] = (
        sum(row["ordered_call_payload_match"] for row in rows) / len(rows)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-chat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.val_chat.read_text().splitlines() if line.strip()]
    if len(records) != EXPECTED_ROWS:
        raise SystemExit(f"FATAL val rows={len(records)} expected={EXPECTED_ROWS}")
    gold = [teacher(record) for record in records]
    if sum(item["is_coord"] for item in gold) != EXPECTED_COORD:
        raise SystemExit("FATAL normalized-v2 coordinate-row count changed")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        self_rows = [score(record, base.txt(record["messages"][-1]["content"]))
                     for record in records]
        summary = summarize(self_rows)
        if (summary["parse_rate"] != 1 or summary["action_sequence_agreement"] != 1
                or summary["ordered_call_payload_agreement"] != 1
                or summary["median_err_px"] != 0
                or summary["n_coord_records"] != EXPECTED_COORD):
            raise SystemExit(f"FATAL normalized-v2 own-val selftest: {summary}")
        (args.out / "selftest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"normalized-v2 own-val selftest PASS: {summary}")
        if not args.base_url:
            return 0
    if not args.base_url or not args.model_dir or not args.export_manifest:
        raise SystemExit("FATAL inference requires base URL, model dir, and export manifest")
    client = base.OpenAI(base_url=args.base_url, api_key="x", timeout=600, max_retries=3)

    def evaluate(record: dict[str, Any]) -> dict[str, Any]:
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=base.wire(record["messages"][:-1]),
                max_tokens=256,
                temperature=0.0,
            )
            message = completion.choices[0].message
            return score(record, message.content or "", getattr(message, "tool_calls", None))
        except Exception as exc:
            row = score(record, "")
            row.update(request_error=True, request_error_type=type(exc).__name__,
                       request_error_message=str(exc)[:400])
            return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(evaluate, records))
    summary = summarize(rows)
    if summary["n_request_errors"] != 0:
        raise SystemExit(f"FATAL request errors: {summary}")
    rows_path = args.out / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        "valid": True,
        "summary": summary,
        "sampling": {"temperature": 0.0},
        "evaluation_design": "oracle_history_one_step_generation",
        "token_forced_nll": False,
        "cross_format_string_comparison": False,
        "metric_note": (
            "ordered_call_payload_agreement compares exact ordered tool arguments, including "
            "coordinates and type/key/scroll/button payloads; coordinate error separately compares "
            "predicted versus teacher net move_rel displacement, where a common origin is valid "
            "because the existing relative evaluator is translation-linear"
        ),
        "val_chat": str(args.val_chat.resolve()),
        "val_sha256": sha256(args.val_chat),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    export = json.loads(args.export_manifest.read_text())
    if (export.get("artifact_type") !=
            "phaseb_normalized_move_rel_v2_A_to_A_hf_checkpoint"
            or export.get("arm") != "normalized_v2" or export.get("step") != 900):
        raise SystemExit("FATAL wrong normalized-v2 exported model")
    manifest = {
        "artifact_type": "phaseb_normalized_move_rel_v2_A_to_A_eval",
        "schema_version": 1,
        "status": "complete",
        "arm": "normalized_v2",
        "valid": True,
        "model_dir": str(args.model_dir.resolve()),
        "export_manifest": str(args.export_manifest.resolve()),
        "export_manifest_sha256": sha256(args.export_manifest),
        "val_chat": str(args.val_chat.resolve()),
        "val_chat_sha256": sha256(args.val_chat),
        "report_sha256": sha256(report_path),
        "rows_sha256": sha256(rows_path),
        "sampling": {"temperature": 0.0},
        "evaluation_design": "oracle_history_one_step_generation",
        "token_forced_nll": False,
        "cross_format_string_comparison": False,
        "request_errors": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
