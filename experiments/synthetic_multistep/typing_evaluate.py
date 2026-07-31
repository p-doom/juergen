#!/usr/bin/env python3
"""Greedy production-parser execution evaluation for one typing cell."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class TypingEvalError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypingEvalError(f"expected JSON object: {path}")
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def validate(model_root: Path, dataset: Path, lineage: str, fmt: str, parser_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    manifest_path = model_root / "typing_train_export_manifest.json"
    manifest = load_json(manifest_path)
    expected_arm = {"A": "reltool_pre", "B": "relraw_pre"}[lineage]
    fixed = {
        "artifact_type": "synthetic_typing_factorial_hf_checkpoint",
        "status": "complete", "lineage": lineage,
        "source_stage1_arm": expected_arm, "target_format": fmt,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct", "step": 750,
        "lora_rank": 256, "lora_alpha": 256,
        "learning_rate": 5e-5, "max_length": 4096, "hf_subdir": "hf",
    }
    bad = {key: (manifest.get(key), wanted) for key, wanted in fixed.items() if manifest.get(key) != wanted}
    if bad:
        raise TypingEvalError(f"wrong typing model: {bad}")
    if manifest.get("schema_version") == 1:
        if manifest.get("fresh_optimizer") is not True:
            raise TypingEvalError("schema-1 typing model must use a fresh optimizer")
    elif manifest.get("schema_version") == 2:
        topology = manifest.get("training_topology", {})
        if (manifest.get("fresh_optimizer") is not False
                or manifest.get("exact_resume_from_step") != 250
                or manifest.get("recovery_change") != "in_loop_validation_disabled_only"
                or topology != {
                    "tp_size": 1, "fsdp_size": 1, "dp_size": 1,
                    "global_batch_size": 1, "gradient_accumulation_steps": 8,
                    "unchanged_from_parent": True,
                }
                or manifest.get("sealed_parent_orbax_unchanged") is not True):
            raise TypingEvalError("schema-2 exact-resume provenance is incomplete")
    else:
        raise TypingEvalError("unsupported typing model manifest schema")
    dataset_manifest_path = dataset / "typing_dataset_manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    if (dataset_manifest.get("artifact_type") != "synthetic_typing_factorial_tokenized"
            or dataset_manifest.get("status") != "complete"
            or dataset_manifest.get("validation_records_per_format") != 200):
        raise TypingEvalError("wrong typing evaluation dataset")
    if (Path(manifest.get("dataset", "")).resolve() != dataset.resolve()
            or manifest.get("dataset_manifest_sha256") != sha256(dataset_manifest_path)):
        raise TypingEvalError("model/dataset identity mismatch")
    hf = model_root / "hf"
    for name in ("model.safetensors", "config.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json"):
        if not (hf / name).is_file():
            raise TypingEvalError(f"HF export missing: {name}")
    rows = jsonl(dataset / fmt / "_normalized" / "val" / "chat.jsonl")
    pairs = jsonl(dataset / "pairs_val.jsonl")
    if len(rows) != 200 or len(pairs) != 200:
        raise TypingEvalError("typing validation count mismatch")
    for row, pair in zip(rows, pairs):
        if (row.get("sample_id") != pair.get("sample_id")
                or row.get("target_text") != pair.get("target_text")):
            raise TypingEvalError("typing pair order mismatch")
        image = Path(row["messages"][1]["content"][0]["image"])
        if sha256(image) != pair.get("image_sha256"):
            raise TypingEvalError(f"typing image hash mismatch: {row.get('sample_id')}")
    expected_parser = (parser_dir / "action_parser.py").resolve()
    if sha256(expected_parser) != "f916757d17e4a5f53627510616ffff411e9109e8737d1309067c6338caae4a9a":
        raise TypingEvalError("production action parser hash mismatch")
    sys.path.insert(0, str(parser_dir.resolve()))
    import action_parser  # type: ignore
    if Path(action_parser.__file__).resolve() != expected_parser:
        raise TypingEvalError("loaded wrong production action parser")
    manifest["evaluation_input_hashes"] = {
        "model_manifest_sha256": sha256(manifest_path),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "parser_sha256": sha256(expected_parser),
        "config_sha256": sha256(hf / "config.json"),
        "weights": [{"name": path.name, "size": path.stat().st_size}
                    for path in sorted(hf.glob("*.safetensors"))],
    }
    return manifest, rows, action_parser


def decode(elements: tuple[Any, ...]) -> str:
    inverse = {"Space": " ", "Minus": "-", "Dot": "."}
    output: list[str] = []
    shift = False
    for kind, value in elements:
        if kind == "type":
            output.append(value)
            continue
        if value.what == "ShiftLeft":
            shift = value.kind == "press"
            continue
        if value.kind != "press":
            continue
        key = value.what
        if key.startswith("Key") and len(key) == 4 and key[-1] in string.ascii_uppercase:
            char = key[-1].lower()
        elif key.startswith("Num") and len(key) == 4 and key[-1] in string.digits:
            char = key[-1]
        elif key in inverse:
            char = inverse[key]
        else:
            raise ValueError(f"undecodable key: {key}")
        output.append(char.upper() if shift else char)
    return "".join(output)


def schema_ok(parsed: Any, fmt: str) -> bool:
    if parsed.no_op or parsed.terminate or parsed.fail:
        return False
    if fmt == "coalesced":
        return len(parsed.elements) == 1 and parsed.elements[0][0] == "type"
    if not parsed.elements or any(kind != "event" for kind, _ in parsed.elements):
        return False
    events = [value for _, value in parsed.elements]
    return (len(events) % 2 == 0 and all(
        events[index].kind == "press"
        and events[index + 1].kind == "release"
        and events[index].what == events[index + 1].what
        and events[index].mouse_button is None
        for index in range(0, len(events), 2)
    ))


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_root = args.model_root.resolve()
    dataset = args.dataset.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("typing_generation_rows.jsonl", "typing_generation_report.json", "typing_generation_manifest.json"):
        (out / name).unlink(missing_ok=True)
    manifest, examples, parser = validate(model_root, dataset, args.lineage, args.format, args.parser_dir.resolve())
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="x", timeout=600.0, max_retries=3)
    lock = threading.Lock()
    completed = 0

    def work(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        sample_id = row["sample_id"]
        system = row["messages"][0]["content"][0]["text"]
        user = row["messages"][1]["content"][1]["text"]
        image = Path(row["messages"][1]["content"][0]["image"])
        raw = ""
        request_error = None
        completion_tokens = None
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url(image)}},
                        {"type": "text", "text": user},
                    ]},
                ],
                temperature=0.0, max_tokens=256,
                seed=int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) & 0x7fffffff,
            )
            raw = response.choices[0].message.content or ""
            completion_tokens = getattr(response.usage, "completion_tokens", None)
        except Exception as exc:  # retained and made fatal after every row is written
            request_error = f"{type(exc).__name__}: {exc}"
        parsed = None
        parse_error = None
        canonical = False
        strict = False
        zero = False
        executed = None
        if request_error is None:
            try:
                parsed = parser.parse_deltatype(raw)
                final_lines = [line.strip() for line in raw.splitlines() if line.strip()]
                canonical = bool(final_lines) and parser.format_deltatype(parsed) == final_lines[-1]
                strict = schema_ok(parsed, args.format)
                zero = parsed.dx == 0 and parsed.dy == 0 and parsed.scroll == 0
                executed = decode(parsed.elements)
            except (TypeError, ValueError) as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
        success = bool(parsed is not None and canonical and strict and zero
                       and executed == row["target_text"])
        result = {
            "sample_id": sample_id, "lineage": args.lineage, "format": args.format,
            "target_text": row["target_text"], "raw_output": raw,
            "completion_tokens": completion_tokens, "request_error": request_error,
            "parse_error": parse_error, "parse_ok": parsed is not None,
            "canonical_action_line": canonical, "strict_schema_ok": strict,
            "zero_mouse_delta_ok": zero, "executed_text": executed,
            "exact_typed_string_success": success,
        }
        with lock:
            completed += 1
            if completed % 20 == 0:
                print(f"[typing-generation] {args.lineage}/{args.format} {completed}/200", flush=True)
        return result

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(work, examples))
    rows_path = out / "typing_generation_rows.jsonl"
    atomic(rows_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    n = len(rows)
    rate = lambda key: sum(bool(row[key]) for row in rows) / n
    report = {
        "schema_version": 1, "artifact_type": "synthetic_typing_generation_report",
        "status": "complete", "lineage": args.lineage, "target_format": args.format,
        "n_examples": n, "sampling": {"temperature": 0.0, "max_tokens": 256},
        "metrics": {
            "exact_typed_string_success_rate": rate("exact_typed_string_success"),
            "parse_rate": rate("parse_ok"), "canonical_action_line_rate": rate("canonical_action_line"),
            "strict_schema_rate": rate("strict_schema_ok"),
            "zero_mouse_delta_rate": rate("zero_mouse_delta_ok"),
            "request_error_count": sum(row["request_error"] is not None for row in rows),
        },
    }
    report_path = out / "typing_generation_report.json"
    atomic(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if report["metrics"]["request_error_count"]:
        raise TypingEvalError("request errors invalidate typing generation evaluation")
    result = {
        "schema_version": 1, "artifact_type": "synthetic_typing_generation_eval",
        "status": "complete", "lineage": args.lineage, "target_format": args.format,
        "model_manifest": manifest, "dataset": str(dataset),
        "n_examples": n, "elapsed_seconds": time.time() - started,
        "rows_sha256": sha256(rows_path), "report_sha256": sha256(report_path),
    }
    atomic(out / "typing_generation_manifest.json", json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--parser-dir", required=True, type=Path)
    parser.add_argument("--lineage", required=True, choices=("A", "B"))
    parser.add_argument("--format", required=True, choices=("coalesced", "perkey"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args), indent=2))
    except TypingEvalError as exc:
        print(f"FATAL typing generation eval: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
