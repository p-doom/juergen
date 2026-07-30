#!/usr/bin/env python3
"""Run one matched greedy rung-2 evaluation and publish only validated artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_SCHEMA_VERSION = 2
EXPECTED_ROWS = 80
EXPECTED_SCENE_IDS = {
    *(f"long_{index:04d}" for index in range(40)),
    *(f"short_{index:04d}" for index in range(40, 80)),
}
GRAMMAR_LEVELS = {
    "move_rel": {
        "relativity": "relative", "grammar_wrapper": "tool", "space": "rel_norm",
        "expected_action": "move_rel", "arms": {False: "reltool_act", True: "reltool_pre"},
    },
    "deltatype_raw": {
        "relativity": "relative", "grammar_wrapper": "bare", "space": "rel_px",
        "expected_action": "delta", "arms": {False: "relraw_act", True: "relraw_pre"},
    },
    "absolute_toolcall": {
        "relativity": "absolute", "grammar_wrapper": "tool", "space": "abs_norm",
        "expected_action": "left_click", "arms": {False: "abstool_act", True: "abstool_pre"},
    },
    "absolute_raw": {
        "relativity": "absolute", "grammar_wrapper": "bare", "space": "abs_px",
        "expected_action": "delta", "arms": {False: "absraw_act", True: "absraw_pre"},
    },
}
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_ACTION_RE = re.compile(
    r"^-?\d+\s+-?\d+\s+0\s*;\s*\+LMB\s+-LMB$"
)
_TRUSTED_ARTIFACTS = ("report.json", "rows.jsonl", "eval_manifest.json")


class EvalError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _clear_trusted_artifacts(out: Path) -> None:
    for name in _TRUSTED_ARTIFACTS:
        (out / name).unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError(f"{label} is not a JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalError(f"cannot read evaluator rows {path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise EvalError(f"rows must be non-empty JSONL without blank lines: {path}")
    rows = []
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"malformed row {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise EvalError(f"non-object row {path}:{line_no}")
        rows.append(row)
    return rows


def _model_provenance(model_dir: Path, grammar: str, preamble: bool) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise EvalError(f"model config missing: {config_path}")
    expected_arm = GRAMMAR_LEVELS[grammar]["arms"][preamble]
    candidates = [
        model_dir.parent / "train_export_manifest.json",
        model_dir.parent / "export_manifest.json",
    ]
    manifests = [path for path in candidates if path.is_file()]
    if len(manifests) != 1:
        raise EvalError(
            f"expected exactly one export manifest beside {model_dir}, found {manifests}"
        )
    manifest_path = manifests[0]
    manifest = _load_json(manifest_path, "model export manifest")
    expected_fields = {
        "artifact_type": "relative_factorial_hf_checkpoint",
        "schema_version": 1,
        "status": "complete",
        "arm": expected_arm,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "step": 750,
        "lora_rank": 32,
        "lora_alpha": 32,
        "max_length": 4096,
        "hf_subdir": "hf",
    }
    mismatches = {
        key: (manifest.get(key), expected) for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise EvalError(f"wrong model artifact for {grammar}/preamble={preamble}: {mismatches}")
    if model_dir != (manifest_path.parent / manifest["hf_subdir"]).resolve():
        raise EvalError(f"model directory is not the HF directory named by {manifest_path}")
    source_checkpoint = Path(str(manifest.get("source_checkpoint", "")))
    if (source_checkpoint.name != "000750" or not source_checkpoint.is_dir()
            or not (source_checkpoint / "_CHECKPOINT_METADATA").is_file()):
        raise EvalError(f"source step-750 checkpoint is missing or incomplete: {source_checkpoint}")

    index_path = model_dir / "model.safetensors.index.json"
    weights = []
    if index_path.is_file():
        index = _load_json(index_path, "safetensors index")
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise EvalError(f"empty safetensors index: {index_path}")
        for name in names:
            path = model_dir / name
            if not path.is_file():
                raise EvalError(f"indexed weight shard missing: {path}")
            weights.append({"name": name, "size": path.stat().st_size})
        weight_index_sha256 = _sha256(index_path)
    else:
        path = model_dir / "model.safetensors"
        if not path.is_file():
            raise EvalError(f"model weights missing in {model_dir}")
        weights.append({"name": path.name, "size": path.stat().st_size})
        weight_index_sha256 = None

    return {
        "arm": expected_arm,
        "model_id": manifest.get("model_id"),
        "step": manifest["step"],
        "source_checkpoint": manifest.get("source_checkpoint"),
        "model_dir": str(model_dir),
        "artifact_manifest": str(manifest_path.resolve()),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "weight_index_sha256": weight_index_sha256,
        "weights": weights,
    }


def _payload_coord(arguments: Any) -> tuple[int, int] | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    coord = arguments.get("coordinate")
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    try:
        return int(round(float(coord[0]))), int(round(float(coord[1])))
    except (TypeError, ValueError):
        return None


def _strict_tool_schema(row: dict[str, Any], expected_action: str) -> bool:
    raw = row.get("raw_output")
    coord = row.get("coord")
    if not isinstance(raw, str) or not isinstance(coord, list) or len(coord) != 2:
        return False
    payloads = []
    for match in _TOOL_CALL_RE.finditer(raw):
        try:
            payloads.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    marker = " | tool_calls="
    if marker in raw:
        try:
            tool_calls = json.loads(raw.split(marker, 1)[1])
        except json.JSONDecodeError:
            tool_calls = []
        if isinstance(tool_calls, list):
            payloads.extend(tool_calls)
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("name") != "computer_use":
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict) or arguments.get("action") != expected_action:
            continue
        parsed_coord = _payload_coord(arguments)
        if parsed_coord == tuple(coord):
            return row.get("action") == expected_action
    return False


def _strict_bare_schema(row: dict[str, Any]) -> bool:
    raw = row.get("raw_output")
    action = row.get("action")
    coord = row.get("coord")
    if (not isinstance(raw, str) or "<tool_call>" in raw or " | tool_calls=" in raw
            or not isinstance(coord, list) or len(coord) != 2):
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False
    return action == "delta" and bool(_BARE_ACTION_RE.fullmatch(lines[-1]))


def _schema_ok(row: dict[str, Any], grammar: str) -> bool:
    spec = GRAMMAR_LEVELS[grammar]
    if spec["grammar_wrapper"] == "tool":
        return _strict_tool_schema(row, spec["expected_action"])
    return _strict_bare_schema(row)


def _summarize(rows: list[dict[str, Any]], grammar: str) -> dict[str, Any]:
    output = {}
    for kind in ("all", "long", "short"):
        selected = [row for row in rows if kind == "all" or row.get("kind") == kind]
        parsed = [row for row in selected if row.get("schema_ok") and row.get("coord") is not None]
        preds = [tuple(row["coord"]) for row in parsed]
        errors = sorted(row["endpoint_err_px"] for row in parsed
                        if isinstance(row.get("endpoint_err_px"), (int, float)))
        ratios = []
        for pred, row in zip(preds, parsed, strict=True):
            ideal = row.get("ideal_coord")
            if isinstance(ideal, list) and len(ideal) == 2 and math.hypot(*ideal) > 0:
                ratios.append(math.hypot(*pred) / math.hypot(*ideal))
        ratios.sort()
        n = len(selected)
        output[f"{grammar}/{kind}"] = {
            "n": n,
            "parse_rate": len(parsed) / n,
            "schema_emit_rate": sum(bool(row.get("schema_ok")) for row in selected) / n,
            "request_error_rate": sum(bool(row.get("request_error")) for row in selected) / n,
            "in_box": sum(bool(row.get("in_box")) for row in selected) / n,
            "median_endpoint_err_px": errors[len(errors) // 2] if errors else None,
            "on_lattice": (sum(bool(row.get("on_lattice")) for row in parsed) / len(parsed)
                           if parsed else None),
            "n_distinct_preds": len(set(preds)),
            "median_mag_ratio": ratios[len(ratios) // 2] if ratios else None,
            "n_scenes": len({row.get("scene_id") for row in selected}),
            "top5_preds": Counter(preds).most_common(5),
        }
    return output


def _validate_and_harden(
    rows: list[dict[str, Any]], report: dict[str, Any], *, grammar: str,
    preamble: bool, model: str, tag: str, model_provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise EvalError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    meta = report.get("meta")
    expected_meta = {
        "model": model, "tag": tag, "state_cursor": False, "preamble": preamble,
        "n_scenes": 80, "k": 1, "n_long": 40, "n_short": 40, "seed": 0,
    }
    if not isinstance(meta, dict):
        raise EvalError("report meta is missing")
    mismatches = {key: (meta.get(key), value) for key, value in expected_meta.items()
                  if meta.get(key) != value}
    if meta.get("sampling", {}).get("temperature") != 0.0:
        mismatches["sampling.temperature"] = (meta.get("sampling"), 0.0)
    if mismatches:
        raise EvalError(f"unexpected report metadata: {mismatches}")

    seen = set()
    kinds = Counter()
    request_errors = []
    expected_space = GRAMMAR_LEVELS[grammar]["space"]
    for index, row in enumerate(rows):
        if row.get("grammar") != grammar or row.get("k") != 0:
            raise EvalError(f"row {index} has wrong grammar/k: {row.get('grammar')}/{row.get('k')}")
        if row.get("space") != expected_space:
            raise EvalError(f"row {index} has wrong coordinate space {row.get('space')!r}")
        scene_id = row.get("scene_id")
        key = (scene_id, row.get("k"))
        if not isinstance(scene_id, str) or key in seen:
            raise EvalError(f"duplicate/missing scene identity at row {index}: {key}")
        seen.add(key)
        if row.get("kind") not in ("long", "short"):
            raise EvalError(f"row {index} has invalid kind {row.get('kind')!r}")
        kinds[row["kind"]] += 1
        if row.get("request_error") is not False:
            request_errors.append(row)
        schema_ok = _schema_ok(row, grammar)
        row["schema_ok"] = schema_ok
        row["raw_in_box"] = bool(row.get("in_box"))
        row["in_box"] = bool(row.get("in_box") and schema_ok)
    if kinds != {"long": 40, "short": 40}:
        raise EvalError(f"expected 40 long/40 short scenes, got {dict(kinds)}")
    if {scene_id for scene_id, _k in seen} != EXPECTED_SCENE_IDS:
        raise EvalError("rows are not the exact deterministic seed-0 scene set")
    if request_errors:
        raise EvalError(f"REQUEST_ERRORS:{len(request_errors)}")

    hardened_summary = _summarize(rows, grammar)
    hardened_meta = dict(meta)
    hardened_meta.update({
        "valid": True,
        "grammar_name": grammar,
        "row_count": EXPECTED_ROWS,
        "schema_scoring": "strict_action_and_wrapper_v1",
        "request_errors": 0,
        "model_provenance": model_provenance,
    })
    return rows, {"meta": hardened_meta, "summary": hardened_summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--grammar", choices=sorted(GRAMMAR_LEVELS), required=True)
    parser.add_argument("--preamble", action="store_true")
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    script = (args.audit_dir / "rung2_scene.py").resolve()
    if not script.is_file():
        print(f"FATAL missing canonical evaluator: {script}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    _clear_trusted_artifacts(args.out)
    try:
        model_provenance = _model_provenance(args.model_dir, args.grammar, args.preamble)
    except EvalError as exc:
        print(f"FATAL model provenance: {exc}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix=".eval_work_", dir=args.out))
    tag = f"relative_factorial/{args.grammar}/{'pre' if args.preamble else 'act'}"
    command = [
        sys.executable, str(script),
        "--base_url", args.base_url,
        "--model", args.model,
        "--out", str(work),
        "--scene_dir", str(args.out / "_scenes_seed0"),
        "--grammars", args.grammar,
        "--n_long", "40", "--n_short", "40", "--k", "1",
        "--temperature", "0.0", "--max_tokens", "192",
        "--concurrency", str(args.concurrency), "--seed", "0", "--selftest",
        "--tag", tag,
    ]
    if args.preamble:
        command.append("--preamble")
    try:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
        report_path = work / "report.json"
        rows_path = work / "rows.jsonl"
        if not report_path.is_file() or not rows_path.is_file():
            print("FATAL evaluator returned 0 without report.json and rows.jsonl", file=sys.stderr)
            return 3
        try:
            rows = _load_rows(rows_path)
            report = _load_json(report_path, "evaluator report")
            rows, report = _validate_and_harden(
                rows, report, grammar=args.grammar, preamble=args.preamble,
                model=args.model, tag=tag, model_provenance=model_provenance,
            )
        except EvalError as exc:
            if str(exc).startswith("REQUEST_ERRORS:"):
                count = int(str(exc).split(":", 1)[1])
                print(f"FATAL {count}/{EXPECTED_ROWS} chat-completion requests failed", file=sys.stderr)
                return 4
            print(f"FATAL invalid evaluator output: {exc}", file=sys.stderr)
            return 3

        rows_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        _atomic_write(args.out / "rows.jsonl", rows_text)
        _atomic_write(args.out / "report.json", report_text)
        if (work / "scenes.jsonl").is_file():
            _atomic_write(args.out / "scenes.jsonl", (work / "scenes.jsonl").read_text())

        spec = GRAMMAR_LEVELS[args.grammar]
        manifest = {
            "artifact_type": "synthetic_factorial_eval",
            "schema_version": EVAL_SCHEMA_VERSION,
            "status": "complete",
            "grammar_name": args.grammar,
            "preamble": args.preamble,
            "relativity": spec["relativity"],
            "grammar_wrapper": spec["grammar_wrapper"],
            "expected_action": spec["expected_action"],
            "sampling": {"temperature": 0.0, "k": 1},
            "known_answer_selftest": {"passing": EXPECTED_ROWS, "total": EXPECTED_ROWS},
            "request_errors": {"count": 0, "total": EXPECTED_ROWS},
            "row_contract": {
                "count": EXPECTED_ROWS, "unique_scenes": EXPECTED_ROWS,
                "long": 40, "short": 40, "k_values": [0],
            },
            "model_provenance": model_provenance,
            "report": "report.json",
            "report_sha256": _sha256(args.out / "report.json"),
            "rows": "rows.jsonl",
            "rows_sha256": _sha256(args.out / "rows.jsonl"),
        }
        _atomic_write(
            args.out / "eval_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
