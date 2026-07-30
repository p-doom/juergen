#!/usr/bin/env python3
"""Run one matched greedy rung-2 scene evaluation and fail on instrument errors."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


GRAMMAR_LEVELS = {
    "move_rel": {"relativity": "relative", "grammar_wrapper": "tool"},
    "deltatype_raw": {"relativity": "relative", "grammar_wrapper": "bare"},
    "absolute_toolcall": {"relativity": "absolute", "grammar_wrapper": "tool"},
    "absolute_raw": {"relativity": "absolute", "grammar_wrapper": "bare"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--grammar", choices=sorted(GRAMMAR_LEVELS), required=True)
    parser.add_argument("--preamble", action="store_true")
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    script = (args.audit_dir / "rung2_scene.py").resolve()
    if not script.is_file():
        print(f"FATAL missing canonical evaluator: {script}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--base_url", args.base_url,
        "--model", args.model,
        "--out", str(args.out),
        "--scene_dir", str(args.out / "_scenes_seed0"),
        "--grammars", args.grammar,
        "--n_long", "40",
        "--n_short", "40",
        "--k", "1",
        "--temperature", "0.0",
        "--max_tokens", "192",
        "--concurrency", str(args.concurrency),
        "--seed", "0",
        "--selftest",
        "--tag", f"relative_factorial/{args.grammar}/{'pre' if args.preamble else 'act'}",
    ]
    if args.preamble:
        command.append("--preamble")
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return completed.returncode

    report_path = args.out / "report.json"
    rows_path = args.out / "rows.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        print("FATAL evaluator returned 0 without report.json and rows.jsonl", file=sys.stderr)
        return 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    errors = [row for row in rows if row.get("request_error")]
    summary_key = f"{args.grammar}/all"
    meta = report.get("meta", {})
    if len(rows) != 80 or meta.get("k") != 1 or meta.get("seed") != 0:
        print(f"FATAL unexpected eval shape: rows={len(rows)} meta={meta}", file=sys.stderr)
        return 3
    if meta.get("sampling", {}).get("temperature") != 0.0:
        print(f"FATAL evaluation was not greedy: {meta.get('sampling')}", file=sys.stderr)
        return 3
    if summary_key not in report.get("summary", {}):
        print(f"FATAL missing summary key {summary_key}", file=sys.stderr)
        return 3
    if errors:
        print(f"FATAL {len(errors)}/80 chat-completion requests failed", file=sys.stderr)
        for row in errors[:5]:
            print(row.get("raw_output"), file=sys.stderr)
        return 4

    manifest = {
        "artifact_type": "synthetic_factorial_eval",
        "schema_version": 1,
        "status": "complete",
        "grammar_name": args.grammar,
        "preamble": args.preamble,
        **GRAMMAR_LEVELS[args.grammar],
        "sampling": {"temperature": 0.0, "k": 1},
        "known_answer_selftest": {"passing": 80, "total": 80},
        "request_errors": {"count": 0, "total": 80},
        "report": "report.json",
    }
    (args.out / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
