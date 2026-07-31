#!/usr/bin/env python3
"""Aggregate roadmap stage-1.5 endpoint-actuation conformance arms."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from .gate import (
        PROTOCOL_PATH,
        GateError,
        actuation_plan,
        load_cells,
        load_protocol,
        paired_noninferiority,
        rgb_sha256,
        sha256_file,
        validate_protocol,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate import (  # type: ignore
        PROTOCOL_PATH,
        GateError,
        actuation_plan,
        load_cells,
        load_protocol,
        paired_noninferiority,
        rgb_sha256,
        sha256_file,
        validate_protocol,
    )

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Contract, strict_schema_ok, unit_range_ok  # type: ignore  # noqa: E402


ARM_NAMES = ("absolute_phase_a", "normalized_phase_a", "raw_a_to_b")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GateError(f"cannot load rows {path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise GateError(f"empty/blank row in {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"bad row {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise GateError(f"non-object row {path}:{line_no}")
        rows.append(value)
    return rows


def _require_bool(row: dict[str, Any], key: str, cell_id: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise GateError(f"{cell_id}: {key} is not boolean")
    return value


def _validate_replay(
    value: Any,
    *,
    cell: Any,
    semantic: str,
    operation: str,
    endpoint: tuple[int, int],
    expected_hit: bool,
) -> None:
    if not isinstance(value, dict):
        raise GateError(f"{cell.cell_id}: missing {operation} replay")
    if value.get("success") is not expected_hit:
        raise GateError(f"{cell.cell_id}: {operation} outcome/geometry mismatch")
    if value.get("cursor_after") != list(endpoint):
        raise GateError(f"{cell.cell_id}: {operation} cursor mismatch")
    expected_plan = [
        list(command) for command in actuation_plan(semantic, operation, cell.cursor, endpoint)
    ]
    if value.get("plan") != expected_plan:
        raise GateError(f"{cell.cell_id}: {operation} actuation-plan drift")
    state = value.get("state")
    success_key = "click_success" if operation == "click" else "drag_success"
    if (
        not isinstance(state, dict)
        or state.get(success_key) is not expected_hit
        or state.get("down") is not False
        or state.get("button_presses") != 1
        or state.get("button_releases") != 1
    ):
        raise GateError(f"{cell.cell_id}: {operation} guest-state mismatch")


def _validate_arm(
    root: Path,
    arm_name: str,
    protocol: dict[str, Any],
    protocol_hash: str,
    cells: list[Any],
    contract: Contract,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "arm_manifest.json"
    rows_path = root / "rows.jsonl"
    if (root / "rows.partial.jsonl").exists():
        raise GateError(f"{arm_name}: partial rows coexist with trusted output")
    manifest = _load_object(manifest_path)
    arm = protocol["arms"][arm_name]
    required = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_stage1_5_endpoint_actuation_arm",
        "status": "complete",
        "scope_classification": protocol["scope_classification"],
        "arm": arm_name,
        "semantic": arm["semantic"],
        "preamble": arm["preamble"],
        "checkpoint_alias": arm["checkpoint_alias"],
        "checkpoint_manifest_sha256": arm["checkpoint_manifest_sha256"],
        "model_weights_sha256": arm["model_weights_sha256"],
        "model_dir": str((Path(arm["checkpoint_root"]) / "hf").resolve()),
        "protocol_sha256": protocol_hash,
        "live_smoke_manifest_sha256": protocol["live_smoke_evidence"]["manifest_sha256"],
        "provider_sha256": protocol["vm"]["provider_sha256"],
        "n_cells": protocol["geometry"]["paired_cells"],
        "sampling": protocol["sampling"],
        "context": protocol["context"],
        "request_errors": 0,
        "infrastructure_mismatches": 0,
    }
    mismatch = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatch:
        raise GateError(f"{arm_name}: manifest mismatch: {mismatch}")
    if sha256_file(rows_path) != manifest.get("rows_sha256"):
        raise GateError(f"{arm_name}: row hash drift")
    rows = _load_rows(rows_path)
    if len(rows) != len(cells):
        raise GateError(f"{arm_name}: row count drift")
    semantic = arm["semantic"]
    counters = {
        "n_compound_success": 0,
        "n_parse_ok": 0,
        "n_schema_ok": 0,
        "n_unit_range_ok": 0,
        "n_endpoint_in_bbox": 0,
    }
    for cell, row in zip(cells, rows, strict=True):
        fixed = {
            "cell_id": cell.cell_id,
            "episode_id": cell.episode_id,
            "episode_index": cell.episode_index,
            "target_index": cell.target_index,
            "cursor_before": list(cell.cursor),
            "bbox": list(cell.bbox),
            "target_center": list(cell.target),
            "canonical_png_sha256": cell.image_sha256,
            "vm_observation_rgb_sha256": rgb_sha256(cell.image_path.read_bytes()),
            "request_seed": cell.request_seed,
        }
        if any(row.get(key) != expected for key, expected in fixed.items()):
            raise GateError(f"{arm_name}/{cell.cell_id}: paired cell identity drift")
        parse_ok = _require_bool(row, "parse_ok", cell.cell_id)
        schema_ok = _require_bool(row, "schema_ok", cell.cell_id)
        units_ok = _require_bool(row, "unit_range_ok", cell.cell_id)
        endpoint_hit = _require_bool(row, "endpoint_in_bbox", cell.cell_id)
        compound = _require_bool(row, "compound_success", cell.cell_id)
        raw = row.get("raw_output")
        if not isinstance(raw, str):
            raise GateError(f"{arm_name}/{cell.cell_id}: missing raw output")
        coord_value = row.get("coord")
        endpoint_value = row.get("endpoint")
        if coord_value is None:
            if parse_ok or endpoint_value is not None or endpoint_hit:
                raise GateError(f"{arm_name}/{cell.cell_id}: null-coordinate contract drift")
            if row.get("click") is not None or row.get("drag") is not None:
                raise GateError(f"{arm_name}/{cell.cell_id}: replay exists without coordinate")
            if row.get("vm_drag_rgb_sha256") is not None:
                raise GateError(f"{arm_name}/{cell.cell_id}: drag pixels exist without replay")
            replay_success = False
        else:
            if (
                not isinstance(coord_value, list)
                or len(coord_value) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in coord_value)
            ):
                raise GateError(f"{arm_name}/{cell.cell_id}: invalid coordinate")
            coord = (coord_value[0], coord_value[1])
            endpoint = contract.apply_coord(semantic, cell.cursor, coord)
            if endpoint_value != list(endpoint):
                raise GateError(f"{arm_name}/{cell.cell_id}: endpoint conversion drift")
            expected_hit = contract.in_bbox(endpoint, cell.bbox)
            if endpoint_hit is not expected_hit:
                raise GateError(f"{arm_name}/{cell.cell_id}: endpoint geometry drift")
            if row.get("vm_drag_rgb_sha256") != fixed["vm_observation_rgb_sha256"]:
                raise GateError(f"{arm_name}/{cell.cell_id}: reset drag pixels drift")
            _validate_replay(
                row.get("click"),
                cell=cell,
                semantic=semantic,
                operation="click",
                endpoint=endpoint,
                expected_hit=expected_hit,
            )
            _validate_replay(
                row.get("drag"),
                cell=cell,
                semantic=semantic,
                operation="drag",
                endpoint=endpoint,
                expected_hit=expected_hit,
            )
            replay_success = bool(expected_hit)
            parse_text = raw.split(" | tool_calls=", 1)[0]
            if schema_ok != strict_schema_ok(semantic, parse_text, coord):
                raise GateError(f"{arm_name}/{cell.cell_id}: schema recomputation mismatch")
            if units_ok != unit_range_ok(semantic, coord):
                raise GateError(f"{arm_name}/{cell.cell_id}: unit recomputation mismatch")
        expected_compound = bool(
            parse_ok and schema_ok and units_ok and endpoint_hit and replay_success
        )
        if compound is not expected_compound:
            raise GateError(f"{arm_name}/{cell.cell_id}: compound endpoint drift")
        counters["n_compound_success"] += int(compound)
        counters["n_parse_ok"] += int(parse_ok)
        counters["n_schema_ok"] += int(schema_ok)
        counters["n_unit_range_ok"] += int(units_ok)
        counters["n_endpoint_in_bbox"] += int(endpoint_hit)
    if any(manifest.get(key) != value for key, value in counters.items()):
        raise GateError(f"{arm_name}: manifest counters do not reproduce")
    return manifest, rows


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint_errors = []
    for row in rows:
        if row["endpoint"] is not None:
            endpoint_errors.append(
                math.hypot(
                    row["endpoint"][0] - row["target_center"][0],
                    row["endpoint"][1] - row["target_center"][1],
                )
            )
    return {
        "n": len(rows),
        "compound_success_rate": _rate(rows, "compound_success"),
        "endpoint_in_bbox_rate": _rate(rows, "endpoint_in_bbox"),
        "parse_rate": _rate(rows, "parse_ok"),
        "strict_schema_rate": _rate(rows, "schema_ok"),
        "unit_range_rate": _rate(rows, "unit_range_ok"),
        "click_success_rate": sum(bool(row.get("click") and row["click"]["success"]) for row in rows)
        / len(rows),
        "drag_success_rate": sum(bool(row.get("drag") and row["drag"]["success"]) for row in rows)
        / len(rows),
        "mean_cursor_endpoint_error_px": (
            sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else None
        ),
    }


def _stratified_contrast(
    absolute: list[dict[str, Any]], treatment: list[dict[str, Any]], cells: list[Any]
) -> dict[str, Any]:
    by_target: dict[str, Any] = {}
    for target_index in range(4):
        indices = [i for i, cell in enumerate(cells) if cell.target_index == target_index]
        a = {cells[i].cell_id: absolute[i]["compound_success"] for i in indices}
        t = {cells[i].cell_id: treatment[i]["compound_success"] for i in indices}
        by_target[str(target_index)] = paired_noninferiority(a, t)
    ranked = sorted(
        range(len(cells)),
        key=lambda i: (
            math.hypot(cells[i].target[0] - cells[i].cursor[0], cells[i].target[1] - cells[i].cursor[1]),
            cells[i].cell_id,
        ),
    )
    by_distance_quartile: dict[str, Any] = {}
    for quartile in range(4):
        indices = ranked[quartile * 80 : (quartile + 1) * 80]
        a = {cells[i].cell_id: absolute[i]["compound_success"] for i in indices}
        t = {cells[i].cell_id: treatment[i]["compound_success"] for i in indices}
        by_distance_quartile[str(quartile + 1)] = paired_noninferiority(a, t)
    episode_differences = []
    for episode_index in range(80):
        indices = [i for i, cell in enumerate(cells) if cell.episode_index == episode_index]
        episode_differences.append(
            (sum(treatment[i]["compound_success"] for i in indices)
             - sum(absolute[i]["compound_success"] for i in indices))
            / len(indices)
        )
    return {
        "by_target_index": by_target,
        "by_distance_quartile": by_distance_quartile,
        "episode_cluster_difference": {
            "clusters": len(episode_differences),
            "mean": sum(episode_differences) / len(episode_differences),
            "minimum": min(episode_differences),
            "maximum": max(episode_differences),
            "note": "descriptive only; the frozen-cell gate is primary",
        },
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol, require_launch_authorized=None)
    validate_protocol(protocol)
    if protocol["launch_gate"]["authorized"] is not True:
        raise GateError("cannot aggregate model arms under an unauthorized protocol")
    contract = Contract()
    cells = load_cells(protocol, contract)
    protocol_hash = sha256_file(args.protocol)
    roots = {
        "absolute_phase_a": args.absolute,
        "normalized_phase_a": args.normalized,
        "raw_a_to_b": args.raw_a_to_b,
    }
    manifests: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for arm_name in ARM_NAMES:
        manifests[arm_name], rows[arm_name] = _validate_arm(
            roots[arm_name], arm_name, protocol, protocol_hash, cells, contract
        )
    absolute = {
        row["cell_id"]: row["compound_success"] for row in rows["absolute_phase_a"]
    }
    contrasts: dict[str, Any] = {}
    for treatment_name in ("normalized_phase_a", "raw_a_to_b"):
        treatment = {
            row["cell_id"]: row["compound_success"] for row in rows[treatment_name]
        }
        primary = paired_noninferiority(
            absolute,
            treatment,
            margin=float(protocol["primary_endpoint"]["noninferiority_margin"]),
            alpha=0.05,
        )
        contrasts[f"{treatment_name}_minus_absolute_phase_a"] = {
            "primary": primary,
            "secondary_strata": _stratified_contrast(
                rows["absolute_phase_a"], rows[treatment_name], cells
            ),
        }
    global_pass = all(value["primary"]["pass"] for value in contrasts.values())
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_stage1_5_endpoint_actuation_paired_report",
        "status": "pass" if global_pass else "noninferiority_fail",
        "scope_classification": protocol["scope_classification"],
        "scope": protocol["evidence_scope"],
        "estimand": protocol["estimand"],
        "protocol_sha256": sha256_file(args.protocol),
        "live_smoke_manifest_sha256": protocol["live_smoke_evidence"]["manifest_sha256"],
        "n_paired_cells": len(cells),
        "arm_manifest_sha256": {
            arm_name: sha256_file(roots[arm_name] / "arm_manifest.json")
            for arm_name in ARM_NAMES
        },
        "arms": {arm_name: _arm_summary(rows[arm_name]) for arm_name in ARM_NAMES},
        "contrasts": contrasts,
        "global_intersection_union_noninferior": global_pass,
        "limitations": [
            protocol["actuation"]["important_limitation"],
            "Inference is for the finite frozen 320-cell benchmark; descriptive episode clusters do not establish population generalization.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "paired_report.json"
    if marker.exists():
        raise GateError(f"refusing to overwrite {marker}")
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--raw-a-to-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(aggregate(parse_args()), indent=2, sort_keys=True))
    except BaseException as exc:
        print(f"FATAL proper-VM aggregate: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
