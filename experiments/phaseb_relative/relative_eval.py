#!/usr/bin/env python3
"""Teacher-forced own-val evaluator for the Phase-B move_rel prose_keep arm."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

SW, SH = 1920, 1080
EXPECTED_ROWS, EXPECTED_COORD = 233, 178
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_IMAGES: dict[str, str] = {}
_LOCK = threading.Lock()


def txt(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in (content or [])
                    if isinstance(part, dict) and part.get("type") == "text")


def calls_from(text: str, structured: Any = None) -> list[dict[str, Any]]:
    calls = []
    for match in TOOL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            args = payload.get("arguments", {})
            if payload.get("name") == "computer_use" and isinstance(args, dict):
                calls.append(args)
        except (json.JSONDecodeError, AttributeError):
            pass
    for call in structured or []:
        function = getattr(call, "function", call)
        if getattr(function, "name", None) != "computer_use":
            continue
        raw = getattr(function, "arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            calls.append(args)
    return calls


def execute(calls: list[dict[str, Any]], cursor_before: list[int]) -> tuple[tuple[int, int], list[str]]:
    cursor = [int(cursor_before[0]), int(cursor_before[1])]
    actions = []
    for args in calls:
        action = str(args.get("action", "")).lower()
        if not action:
            continue
        actions.append(action)
        if action == "move_rel":
            coord = args.get("coordinate")
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                raise ValueError(f"move_rel missing coordinate: {args}")
            cursor[0] += round(float(coord[0]) * SW / 1000.0)
            cursor[1] += round(float(coord[1]) * SH / 1000.0)
    return (cursor[0], cursor[1]), actions


def teacher(record: dict[str, Any]) -> dict[str, Any]:
    audit = record.get("phaseb_relative_audit")
    if not isinstance(audit, list) or not audit:
        raise ValueError("missing Phase-B relative audit metadata")
    state = audit[-1]
    text = txt(record["messages"][-1]["content"])
    calls = calls_from(text)
    if not calls:
        raise ValueError("teacher action does not parse")
    landing, actions = execute(calls, state["cursor_before_px"])
    expected = tuple(state["relative_landing_px"])
    if landing != expected:
        raise ValueError(f"teacher execution disagrees with build audit: {landing} != {expected}")
    return {"calls": calls, "actions": actions, "landing": landing,
            "cursor_before": state["cursor_before_px"],
            "is_coord": state.get("absolute_landing_px") is not None}


def image_url(path: str) -> str:
    with _LOCK:
        if path not in _IMAGES:
            _IMAGES[path] = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        return _IMAGES[path]


def wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        parts = message.get("content")
        if message.get("role") == "system":
            out.append({"role": "system", "content": txt(parts)})
            continue
        content = []
        for part in parts if isinstance(parts, list) else []:
            if part.get("type") == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                content.append({"type": "image_url", "image_url": {"url": image_url(part["image"])}})
        out.append({"role": message["role"], "content": content})
    return out


def score(record: dict[str, Any], raw: str, structured: Any = None) -> dict[str, Any]:
    gold = teacher(record)
    parsed = calls_from(raw, structured)
    row = {"sample_id": record.get("sample_id"), "teacher_actions": gold["actions"],
           "pred_actions": [], "action_match": False, "is_coord_record": gold["is_coord"],
           "teacher_landing_px": list(gold["landing"]), "pred_landing_px": None,
           "err_px": None, "parse_ok": False, "request_error": False,
           "raw_output": raw[:800]}
    if not parsed:
        return row
    landing, actions = execute(parsed, gold["cursor_before"])
    row.update({"pred_actions": actions, "action_match": actions == gold["actions"],
                "pred_landing_px": list(landing), "parse_ok": True})
    if gold["is_coord"]:
        row["err_px"] = math.dist(landing, gold["landing"])
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coord = [row for row in rows if row["is_coord_record"]]
    errors = sorted(row["err_px"] for row in coord if row["err_px"] is not None)
    return {"n_rows": len(rows), "n_coord_records": len(coord),
            "n_request_errors": sum(row["request_error"] for row in rows),
            "parse_rate": sum(row["parse_ok"] for row in rows) / len(rows),
            "action_sequence_agreement": sum(row["action_match"] for row in rows) / len(rows),
            "coord_emit_rate": len(errors) / len(coord),
            "median_err_px": errors[len(errors)//2] if errors else None,
            "within_50px": sum(x <= 50 for x in errors) / len(coord),
            "within_100px": sum(x <= 100 for x in errors) / len(coord)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    # Validate all gold actions before any request.
    gold = [teacher(record) for record in records]
    if sum(item["is_coord"] for item in gold) != EXPECTED_COORD:
        raise SystemExit("FATAL coordinate-row count changed")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        self_rows = [score(record, txt(record["messages"][-1]["content"])) for record in records]
        summary = summarize(self_rows)
        if (summary["parse_rate"] != 1 or summary["action_sequence_agreement"] != 1
                or summary["median_err_px"] != 0 or summary["n_coord_records"] != EXPECTED_COORD):
            raise SystemExit(f"FATAL own-val selftest: {summary}")
        (args.out / "selftest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"own-val selftest PASS: {summary}")
        if not args.base_url:
            return 0
    if not args.base_url or not args.model_dir or not args.export_manifest:
        raise SystemExit("FATAL inference requires base URL, model dir, and export manifest")
    client = OpenAI(base_url=args.base_url, api_key="x", timeout=600, max_retries=3)
    def evaluate(record):
        try:
            completion = client.chat.completions.create(
                model=args.model, messages=wire(record["messages"][:-1]),
                max_tokens=256, temperature=0.0)
            message = completion.choices[0].message
            return score(record, message.content or "", getattr(message, "tool_calls", None))
        except Exception as exc:  # transport errors remain explicit and invalidate publication
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
    report = {"valid": True, "summary": summary, "sampling": {"temperature": 0.0},
              "val_chat": str(args.val_chat.resolve()), "val_sha256": sha256(args.val_chat)}
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    export = json.loads(args.export_manifest.read_text())
    if export.get("artifact_type") != "phaseb_relative_hf_checkpoint" or export.get("arm") != "prose_keep":
        raise SystemExit("FATAL wrong exported model")
    manifest = {"artifact_type": "phaseb_relative_eval", "schema_version": 1,
                "status": "complete", "arm": "prose_keep", "valid": True,
                "model_dir": str(args.model_dir.resolve()),
                "export_manifest": str(args.export_manifest.resolve()),
                "export_manifest_sha256": sha256(args.export_manifest),
                "val_chat": str(args.val_chat.resolve()), "val_chat_sha256": sha256(args.val_chat),
                "report_sha256": sha256(report_path), "rows_sha256": sha256(rows_path),
                "sampling": {"temperature": 0.0}, "request_errors": 0,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (args.out / "eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
