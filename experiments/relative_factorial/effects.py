#!/usr/bin/env python3
"""Compute a validated 2x2x2 factorial from eight complete eval artifacts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_SCHEMA_VERSION = 2
EXPECTED_ROWS = 80
EXPECTED_SCENE_IDS = {
    *(f"long_{index:04d}" for index in range(40)),
    *(f"short_{index:04d}" for index in range(40, 80)),
}
CELLS = {
    "abs_tool_act": (-1, +1, -1, "absolute_toolcall", "abstool_act", "abs_norm"),
    "abs_bare_act": (-1, -1, -1, "absolute_raw", "absraw_act", "abs_px"),
    "abs_tool_pre": (-1, +1, +1, "absolute_toolcall", "abstool_pre", "abs_norm"),
    "abs_bare_pre": (-1, -1, +1, "absolute_raw", "absraw_pre", "abs_px"),
    "rel_tool_act": (+1, +1, -1, "move_rel", "reltool_act", "rel_norm"),
    "rel_bare_act": (+1, -1, -1, "deltatype_raw", "relraw_act", "rel_px"),
    "rel_tool_pre": (+1, +1, +1, "move_rel", "reltool_pre", "rel_norm"),
    "rel_bare_pre": (+1, -1, +1, "deltatype_raw", "relraw_pre", "rel_px"),
}
FACTOR_NAMES = ("relativity", "grammar", "preamble")
LEVEL_NAMES = {
    "relativity": {+1: "relative", -1: "absolute"},
    "grammar": {+1: "tool_call", -1: "bare_token"},
    "preamble": {+1: "preamble", -1: "action_only"},
}
EXPECTED_ACTIONS = {
    "absolute_toolcall": "left_click",
    "absolute_raw": "delta",
    "move_rel": "move_rel",
    "deltatype_raw": "delta",
}


class EffectError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EffectError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EffectError(f"{label} is not an object: {path}")
    return value


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EffectError(f"cannot read rows {path}: {exc}") from exc
    if len(lines) != EXPECTED_ROWS or any(not line.strip() for line in lines):
        raise EffectError(f"expected exactly {EXPECTED_ROWS} nonblank rows in {path}")
    rows = []
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EffectError(f"malformed row {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise EffectError(f"non-object row {path}:{line_no}")
        rows.append(row)
    return rows


def _require_model_provenance(
    cell: str, provenance: Any, expected_arm: str,
) -> dict[str, Any]:
    if not isinstance(provenance, dict) or provenance.get("arm") != expected_arm:
        raise EffectError(f"{cell}: model provenance does not identify arm {expected_arm}")
    for path_key, sha_key in (
        ("artifact_manifest", "artifact_manifest_sha256"),
        ("config", "config_sha256"),
    ):
        path_value = provenance.get(path_key)
        digest = provenance.get(sha_key)
        if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
            raise EffectError(f"{cell}: incomplete model provenance {path_key}/{sha_key}")
        path = Path(path_value)
        if not path.is_file() or _sha256(path) != digest:
            raise EffectError(f"{cell}: model provenance checksum mismatch for {path}")
    model_dir = provenance.get("model_dir")
    weights = provenance.get("weights")
    if not isinstance(model_dir, str) or not isinstance(weights, list) or not weights:
        raise EffectError(f"{cell}: model directory/weight inventory missing")
    model_root = Path(model_dir).resolve()
    artifact_path = Path(provenance["artifact_manifest"]).resolve()
    artifact = _json_object(artifact_path, "model artifact manifest")
    artifact_required = {
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
    artifact_mismatches = {
        key: (artifact.get(key), expected) for key, expected in artifact_required.items()
        if artifact.get(key) != expected
    }
    if artifact_mismatches or model_root != (artifact_path.parent / "hf").resolve():
        raise EffectError(f"{cell}: wrong model artifact mapping: {artifact_mismatches}")
    source_checkpoint = Path(str(artifact.get("source_checkpoint", "")))
    if (provenance.get("source_checkpoint") != artifact.get("source_checkpoint")
            or source_checkpoint.name != "000750" or not source_checkpoint.is_dir()
            or not (source_checkpoint / "_CHECKPOINT_METADATA").is_file()):
        raise EffectError(f"{cell}: source checkpoint provenance mismatch")
    if Path(provenance["config"]).resolve() != model_root / "config.json":
        raise EffectError(f"{cell}: config provenance is outside the model directory")
    for item in weights:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise EffectError(f"{cell}: malformed weight inventory")
        weight = model_root / item["name"]
        if not weight.is_file() or weight.stat().st_size != item.get("size"):
            raise EffectError(f"{cell}: model weight inventory mismatch for {weight}")
    index_digest = provenance.get("weight_index_sha256")
    if index_digest is not None:
        index_path = model_root / "model.safetensors.index.json"
        if (not isinstance(index_digest, str) or len(index_digest) != 64
                or not index_path.is_file() or _sha256(index_path) != index_digest):
            raise EffectError(f"{cell}: weight index checksum mismatch")
    return provenance


def _load_cell(cell: str, directory: Path, metric: str) -> tuple[float, dict[str, Any]]:
    if metric != "in_box":
        raise EffectError(f"{cell}: only the bounded row-verifiable metric 'in_box' is supported")
    r, g, p, grammar, expected_arm, expected_space = CELLS[cell]
    directory = directory.resolve()
    report_path = directory / "report.json"
    rows_path = directory / "rows.jsonl"
    manifest_path = directory / "eval_manifest.json"
    if not all(path.is_file() for path in (report_path, rows_path, manifest_path)):
        raise EffectError(f"{cell}: missing report.json/rows.jsonl/eval_manifest.json in {directory}")
    report = _json_object(report_path, "report")
    manifest = _json_object(manifest_path, "eval manifest")
    rows = _jsonl_rows(rows_path)

    required_manifest = {
        "artifact_type": "synthetic_factorial_eval",
        "schema_version": EVAL_SCHEMA_VERSION,
        "status": "complete",
        "relativity": LEVEL_NAMES["relativity"][r],
        "grammar_wrapper": "tool" if g == 1 else "bare",
        "preamble": p == 1,
        "grammar_name": grammar,
        "expected_action": EXPECTED_ACTIONS[grammar],
        "sampling": {"k": 1, "temperature": 0.0},
        "known_answer_selftest": {"passing": EXPECTED_ROWS, "total": EXPECTED_ROWS},
        "request_errors": {"count": 0, "total": EXPECTED_ROWS},
        "row_contract": {
            "count": EXPECTED_ROWS, "unique_scenes": EXPECTED_ROWS,
            "long": 40, "short": 40, "k_values": [0],
        },
        "report": "report.json",
        "rows": "rows.jsonl",
    }
    mismatches = {key: (manifest.get(key), expected) for key, expected in required_manifest.items()
                  if manifest.get(key) != expected}
    if mismatches:
        raise EffectError(f"{cell}: eval manifest invariant mismatch: {mismatches}")
    if manifest.get("report_sha256") != _sha256(report_path):
        raise EffectError(f"{cell}: report checksum does not match manifest")
    if manifest.get("rows_sha256") != _sha256(rows_path):
        raise EffectError(f"{cell}: rows checksum does not match manifest")
    model_provenance = _require_model_provenance(
        cell, manifest.get("model_provenance"), expected_arm,
    )

    meta = report.get("meta")
    required_meta = {
        "valid": True,
        "model": "policy",
        "tag": f"relative_factorial/{grammar}/{'pre' if p == 1 else 'act'}",
        "grammar_name": grammar,
        "preamble": p == 1,
        "state_cursor": False,
        "n_scenes": EXPECTED_ROWS,
        "row_count": EXPECTED_ROWS,
        "n_long": 40,
        "n_short": 40,
        "k": 1,
        "seed": 0,
        "request_errors": 0,
        "schema_scoring": "strict_action_and_wrapper_v1",
        "model_provenance": model_provenance,
    }
    if not isinstance(meta, dict):
        raise EffectError(f"{cell}: report meta missing")
    meta_mismatches = {key: (meta.get(key), expected) for key, expected in required_meta.items()
                       if meta.get(key) != expected}
    if meta_mismatches:
        raise EffectError(f"{cell}: report metadata mismatch: {meta_mismatches}")
    if not isinstance(meta.get("sampling"), dict) or meta["sampling"].get("temperature") != 0.0:
        raise EffectError(f"{cell}: report sampling is not greedy: {meta.get('sampling')}")

    seen = set()
    kinds = Counter()
    for index, row in enumerate(rows):
        key = (row.get("scene_id"), row.get("k"))
        if (row.get("grammar") != grammar or row.get("space") != expected_space
                or row.get("k") != 0 or not isinstance(row.get("scene_id"), str)):
            raise EffectError(f"{cell}: row {index} has wrong grammar/space/scene/k")
        if key in seen:
            raise EffectError(f"{cell}: duplicate scene row {key}")
        seen.add(key)
        if row.get("kind") not in ("long", "short"):
            raise EffectError(f"{cell}: row {index} has invalid kind")
        kinds[row["kind"]] += 1
        if row.get("request_error") is not False:
            raise EffectError(f"{cell}: row {index} is a request error or lacks explicit status")
        if not isinstance(row.get("schema_ok"), bool) or not isinstance(row.get("in_box"), bool):
            raise EffectError(f"{cell}: row {index} lacks hardened schema/score fields")
        if row["in_box"] and not row["schema_ok"]:
            raise EffectError(f"{cell}: row {index} scores an off-schema action as in-box")
    if (len(seen) != EXPECTED_ROWS or kinds != {"long": 40, "short": 40}
            or {scene_id for scene_id, _k in seen} != EXPECTED_SCENE_IDS):
        raise EffectError(f"{cell}: incomplete scene balance {dict(kinds)}")

    summary = report.get("summary", {}).get(f"{grammar}/all")
    if not isinstance(summary, dict) or metric not in summary:
        raise EffectError(f"{cell}: missing {grammar}/all metric {metric!r}")
    if summary.get("n") != EXPECTED_ROWS or summary.get("n_scenes") != EXPECTED_ROWS:
        raise EffectError(f"{cell}: summary is not based on exactly {EXPECTED_ROWS} scenes")
    if summary.get("request_error_rate") != 0.0:
        raise EffectError(f"{cell}: summary reports request errors")
    value = summary[metric]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise EffectError(f"{cell}: metric {metric} is not finite and bounded: {value!r}")
    row_value = sum(row["in_box"] for row in rows) / EXPECTED_ROWS
    if not math.isclose(float(value), row_value, rel_tol=0.0, abs_tol=1e-12):
        raise EffectError(f"{cell}: summary metric {value} != row-derived metric {row_value}")
    return float(value), {
        "directory": str(directory),
        "grammar": grammar,
        "levels": [r, g, p],
        "model_provenance": model_provenance,
        "report_sha256": manifest["report_sha256"],
        "rows_sha256": manifest["rows_sha256"],
    }


def calculate(values: dict[str, float]) -> dict[str, Any]:
    if set(values) != set(CELLS):
        raise EffectError(f"expected exactly eight cells, got {sorted(values)}")
    if any(not math.isfinite(value) for value in values.values()):
        raise EffectError("all factorial values must be finite")
    rows = [(CELLS[cell][:3], values[cell]) for cell in CELLS]
    terms: dict[str, Any] = {}
    for width in (1, 2, 3):
        for axes in itertools.combinations(range(3), width):
            name = "×".join(FACTOR_NAMES[i] for i in axes)
            positive = [y for codes, y in rows if _product(codes[i] for i in axes) == 1]
            negative = [y for codes, y in rows if _product(codes[i] for i in axes) == -1]
            effect = sum(positive) / len(positive) - sum(negative) / len(negative)
            terms[name] = {
                "effect": effect,
                "positive_product_mean": sum(positive) / len(positive),
                "negative_product_mean": sum(negative) / len(negative),
                "axes": [FACTOR_NAMES[i] for i in axes],
            }
    return {"grand_mean": sum(values.values()) / 8, "effects": terms}


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(
            f"--{cell.replace('_', '-')}", f"--{cell}", dest=cell, type=Path, required=True
        )
    parser.add_argument("--metric", default="in_box", choices=("in_box",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.unlink(missing_ok=True)
    try:
        values = {}
        provenance = {}
        for cell in CELLS:
            values[cell], provenance[cell] = _load_cell(cell, getattr(args, cell), args.metric)
        result = calculate(values)
    except EffectError as exc:
        print(f"FATAL factorial input invariant: {exc}", file=sys.stderr)
        return 2
    payload = {
        "artifact_type": "synthetic_relative_factorial_effects",
        "schema_version": 2,
        "status": "complete",
        "metric": args.metric,
        "coding": {
            "relativity": {"+1": "relative", "-1": "absolute"},
            "grammar": {"+1": "tool_call", "-1": "bare_token"},
            "preamble": {"+1": "preamble", "-1": "action_only"},
            "effect_definition": (
                "mean(metric for cells where the product of a term's factor codes is +1) "
                "minus the corresponding -1 mean; positive two-/three-way terms therefore "
                "mean the named high-level effects reinforce one another"
            ),
        },
        "cells": values,
        "provenance": provenance,
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(args.out, text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
