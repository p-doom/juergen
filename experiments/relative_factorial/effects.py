#!/usr/bin/env python3
"""Compute the complete 2x2x2 factorial from eight rung2 report directories."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any


CELLS = {
    "abs_tool_act": (-1, +1, -1, "absolute_toolcall"),
    "abs_bare_act": (-1, -1, -1, "absolute_raw"),
    "abs_tool_pre": (-1, +1, +1, "absolute_toolcall"),
    "abs_bare_pre": (-1, -1, +1, "absolute_raw"),
    "rel_tool_act": (+1, +1, -1, "move_rel"),
    "rel_bare_act": (+1, -1, -1, "deltatype_raw"),
    "rel_tool_pre": (+1, +1, +1, "move_rel"),
    "rel_bare_pre": (+1, -1, +1, "deltatype_raw"),
}
FACTOR_NAMES = ("relativity", "grammar", "preamble")
LEVEL_NAMES = {
    "relativity": {+1: "relative", -1: "absolute"},
    "grammar": {+1: "tool_call", -1: "bare_token"},
    "preamble": {+1: "preamble", -1: "action_only"},
}


class EffectError(RuntimeError):
    pass


def _load_cell(cell: str, directory: Path, metric: str) -> tuple[float, dict[str, Any]]:
    r, g, p, grammar = CELLS[cell]
    report_path = directory / "report.json"
    manifest_path = directory / "eval_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise EffectError(f"{cell}: missing report.json/eval_manifest.json in {directory}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "relativity": LEVEL_NAMES["relativity"][r],
        "grammar": "tool" if g == 1 else "bare",
        "preamble": p == 1,
        "grammar_name": grammar,
    }
    actual = {
        "relativity": manifest.get("relativity"),
        "grammar": manifest.get("grammar_wrapper"),
        "preamble": manifest.get("preamble"),
        "grammar_name": manifest.get("grammar_name"),
    }
    # eval_manifest grammar is both the rung2 grammar name and, separately, the
    # tool/bare level in the `grammar` key only in older manifests. Accept the
    # current explicit fields while checking every factor independently.
    if manifest.get("relativity") != expected["relativity"]:
        raise EffectError(f"{cell}: relativity mismatch in {manifest_path}")
    if manifest.get("preamble") != expected["preamble"]:
        raise EffectError(f"{cell}: preamble mismatch in {manifest_path}")
    if manifest.get("grammar_wrapper") != expected["grammar"]:
        raise EffectError(f"{cell}: grammar-wrapper mismatch in {manifest_path}: {actual}")
    if manifest.get("grammar_name") != grammar:
        raise EffectError(f"{cell}: grammar mismatch in {manifest_path}: {actual}")
    if manifest.get("sampling") != {"k": 1, "temperature": 0.0}:
        raise EffectError(f"{cell}: eval is not matched greedy k=1")
    summary = report.get("summary", {}).get(f"{grammar}/all")
    if not isinstance(summary, dict) or metric not in summary:
        raise EffectError(f"{cell}: missing {grammar}/all metric {metric!r}")
    value = summary[metric]
    if not isinstance(value, (int, float)):
        raise EffectError(f"{cell}: non-numeric metric {metric}: {value!r}")
    return float(value), {"directory": str(directory), "grammar": grammar, "levels": [r, g, p]}


def calculate(values: dict[str, float]) -> dict[str, Any]:
    if set(values) != set(CELLS):
        raise EffectError(f"expected exactly eight cells, got {sorted(values)}")
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
    return {
        "grand_mean": sum(values.values()) / 8,
        "effects": terms,
    }


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(
            f"--{cell.replace('_', '-')}", f"--{cell}", dest=cell, type=Path, required=True
        )
    parser.add_argument("--metric", default="in_box")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
        "schema_version": 1,
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
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
