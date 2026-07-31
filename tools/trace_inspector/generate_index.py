#!/usr/bin/env python3
"""Build a compact, provenance-attested index for the static trace inspector."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import difflib
import fnmatch
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_EVAL_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/eval_logs/franz.srambical"
)
DEFAULT_RUNS_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs/franz.srambical"
)
STATIC_FILES = ("index.html", "app.js", "styles.css")
RUN_RE = re.compile(r"^run_[A-Za-z0-9]+$")
JOB_LOG_RE = re.compile(r"_(\d+)\.log$")
RANK_RE = re.compile(r"(?:^|_)r(\d+)(?:_|$)")
STEP_RE = re.compile(r"(?:^|_)s(?:tep)?(\d+)(?:_|$)")


class AuditError(RuntimeError):
    """A sealed input failed an integrity or provenance assertion."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditError(f"non-object row {line_number} in {path}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read rows {path}: {exc}") from exc
    if not rows:
        raise AuditError(f"row file is empty: {path}")
    return rows


def require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AuditError(f"{label} has no valid SHA-256 seal")
    if not path.is_file():
        raise AuditError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def png_size(path: Path) -> list[int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
        return [width, height]
    return None


def match_number(pattern: re.Pattern[str], *values: Any) -> int | None:
    for value in values:
        if not isinstance(value, str):
            continue
        match = pattern.search(value)
        if match:
            return int(match.group(1))
    return None


def safe_bool(value: Any) -> bool:
    return value is True


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def char_metrics(target: str, executed: str) -> dict[str, Any]:
    ratio = difflib.SequenceMatcher(a=target, b=executed, autojunk=False).ratio()
    return {
        "requested_chars": len(target),
        "executed_chars": len(executed),
        "character_similarity": round(ratio, 6),
    }


def event_tokens(raw: str, target_format: str) -> list[str]:
    action_line = next((line for line in reversed(raw.splitlines()) if line.strip()), "")
    if target_format == "perkey":
        return re.findall(r"[+-][A-Za-z0-9_]+", action_line)
    call = re.search(r"type\((.*)\)", action_line)
    return [f"type({call.group(1)})"] if call else ([action_line] if action_line else [])


def seal_artifact(
    artifact: Path,
    adapter: str,
    runs_root: Path,
    expected_user: str,
) -> dict[str, Any]:
    meta_path = artifact / ".meta.json"
    meta = load_json(meta_path)
    if meta.get("kind") != "eval_result":
        raise AuditError(f"{artifact.name}: artifact kind is not eval_result")
    if meta.get("user") != expected_user:
        raise AuditError(f"{artifact.name}: owner mismatch in .meta.json")
    if meta.get("alias") != artifact.name:
        raise AuditError(f"{artifact.name}: alias/path mismatch in .meta.json")
    run_id = meta.get("producer_run_id")
    if not isinstance(run_id, str) or not RUN_RE.fullmatch(run_id):
        raise AuditError(f"{artifact.name}: invalid producer run ID")
    metadata = meta.get("metadata")
    if not isinstance(metadata, dict):
        raise AuditError(f"{artifact.name}: metadata object missing")
    marker = metadata.get("marker")
    if not isinstance(marker, str) or Path(marker).name != marker:
        raise AuditError(f"{artifact.name}: invalid manifest marker")
    manifest_path = artifact / marker
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("valid") is False:
        raise AuditError(f"{artifact.name}: manifest is not complete and valid")
    if metadata.get("result") != manifest:
        raise AuditError(f"{artifact.name}: registry result and manifest differ")
    recipe = metadata.get("producer_recipe")
    if not isinstance(recipe, str) or not recipe:
        raise AuditError(f"{artifact.name}: producer recipe missing")

    run_dir = runs_root / run_id
    context_path = run_dir / ".lab" / "context.json"
    context = load_json(context_path)
    if context.get("run_id") != run_id or context.get("recipe_name") != recipe:
        raise AuditError(f"{artifact.name}: labctl context run/recipe mismatch")
    outputs = context.get("outputs")
    if not isinstance(outputs, dict):
        raise AuditError(f"{artifact.name}: labctl context outputs missing")
    output_paths = {
        Path(item["path"]).resolve()
        for item in outputs.values()
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if artifact.resolve() not in output_paths:
        raise AuditError(f"{artifact.name}: artifact is not a declared labctl output")
    logs = sorted((run_dir / ".lab").glob("*.log"))
    job_ids = {
        match.group(1)
        for path in logs
        if (match := JOB_LOG_RE.search(path.name)) is not None
    }
    manifest_job = manifest.get("slurm_job_id")
    if manifest_job is not None:
        job_ids.add(str(manifest_job))
    if len(job_ids) != 1:
        raise AuditError(f"{artifact.name}: expected one attested Slurm job ID, got {sorted(job_ids)}")

    if adapter == "auto":
        artifact_type = str(manifest.get("artifact_type", ""))
        if marker == "typing_eval_manifest.json" or "typing" in artifact_type:
            adapter = "typing"
        elif "closed_loop" in artifact_type:
            adapter = "closed_loop_mouse"
        elif "phaseb" in artifact_type:
            adapter = "phaseb_mouse"
        elif "factorial_eval" in artifact_type:
            adapter = "relative_mouse"
        else:
            raise AuditError(f"{artifact.name}: no adapter for artifact type {artifact_type!r}")

    if adapter == "typing":
        files = {
            "rows": artifact / "typing_generation_rows.jsonl",
            "report": artifact / "typing_generation_report.json",
            "generation_manifest": artifact / "typing_generation_manifest.json",
            "teacher_rows": artifact / "typing_teacher_forced_rows.jsonl",
            "teacher_report": artifact / "typing_teacher_forced_report.json",
        }
        require_hash(files["rows"], manifest.get("generation_rows_sha256"), "generation rows")
        require_hash(files["report"], manifest.get("generation_report_sha256"), "generation report")
        require_hash(
            files["generation_manifest"],
            manifest.get("generation_manifest_sha256"),
            "generation manifest",
        )
        require_hash(files["teacher_rows"], manifest.get("teacher_forced_rows_sha256"), "teacher rows")
        require_hash(files["teacher_report"], manifest.get("teacher_forced_report_sha256"), "teacher report")
        generation_manifest = load_json(files["generation_manifest"])
        if generation_manifest.get("status") != "complete":
            raise AuditError(f"{artifact.name}: generation manifest is not complete")
        if generation_manifest.get("rows_sha256") != manifest.get("generation_rows_sha256"):
            raise AuditError(f"{artifact.name}: nested generation row seal mismatch")
        rows = load_jsonl(files["rows"])
        if len(rows) != manifest.get("n_examples"):
            raise AuditError(f"{artifact.name}: n_examples does not match rows")
    else:
        files = {"rows": artifact / "rows.jsonl"}
        require_hash(files["rows"], manifest.get("rows_sha256"), "rows")
        rows = load_jsonl(files["rows"])
        report_path = artifact / str(manifest.get("report", "report.json"))
        if manifest.get("report_sha256") is not None:
            require_hash(report_path, manifest.get("report_sha256"), "report")
            files["report"] = report_path
        chunk = manifest.get("chunk_manifest")
        if isinstance(chunk, dict):
            chunk_path = artifact / str(chunk.get("path"))
            require_hash(chunk_path, chunk.get("sha256"), "chunk manifest")
        expected_count = manifest.get("row_contract", {}).get("count")
        if expected_count is None and isinstance(manifest.get("rows"), int):
            expected_count = manifest["rows"]
        if expected_count is not None and len(rows) != expected_count:
            raise AuditError(f"{artifact.name}: manifest row count does not match rows")

    return {
        "adapter": adapter,
        "artifact": artifact,
        "artifact_id": meta.get("id"),
        "alias": artifact.name,
        "created_at": meta.get("created_at"),
        "run_id": run_id,
        "job_id": next(iter(job_ids)),
        "recipe": recipe,
        "manifest": manifest,
        "manifest_file": marker,
        "manifest_sha256": sha256_file(manifest_path),
        "rows": rows,
        "files": files,
    }


def asset_path(sealed: dict[str, Any], absolute: Any) -> tuple[str | None, list[int] | None]:
    if not isinstance(absolute, str):
        return None, None
    artifact = sealed["artifact"].resolve()
    image = Path(absolute).resolve()
    try:
        relative = image.relative_to(artifact)
    except ValueError as exc:
        raise AuditError(f"image path escapes sealed artifact: {image}") from exc
    if not image.is_file():
        raise AuditError(f"sealed screenshot is missing: {image}")
    return f"data/assets/{sealed['artifact_id']}/{relative.as_posix()}", png_size(image)


def common_trace(sealed: dict[str, Any], trace_id: str) -> dict[str, Any]:
    return {
        "id": trace_id,
        "run_id": sealed["run_id"],
        "artifact_id": sealed["artifact_id"],
        "job_id": sealed["job_id"],
        "recipe": sealed["recipe"],
    }


def normalize_relative(sealed: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = sealed["manifest"]
    scenes_path = sealed["artifact"] / "scenes.jsonl"
    scenes = {row["scene_id"]: row for row in load_jsonl(scenes_path)}
    provenance = manifest.get("model_provenance", {})
    arm = str(provenance.get("arm", manifest.get("arm", "unknown")))
    model_dir = str(provenance.get("model_dir", ""))
    source_checkpoint = str(provenance.get("source_checkpoint", ""))
    artifact_manifest = str(provenance.get("artifact_manifest", ""))
    rank = match_number(RANK_RE, model_dir, source_checkpoint, artifact_manifest, sealed["alias"])
    step = provenance.get("step") or match_number(
        STEP_RE,
        model_dir,
        source_checkpoint,
        sealed["alias"],
    )
    experiment = "relative_factorial_capacity" if "capacity" in sealed["alias"] else "relative_factorial"
    traces: list[dict[str, Any]] = []
    for row in sealed["rows"]:
        scene_id = str(row.get("scene_id"))
        if scene_id not in scenes:
            raise AuditError(f"{sealed['alias']}: row references absent scene {scene_id}")
        scene = scenes[scene_id]
        screenshot, image_size = asset_path(sealed, scene.get("image_path"))
        cursor = scene.get("cursor")
        target = scene.get("target_center")
        ideal = row.get("ideal_coord")
        landing = row.get("landing")
        overlay = {
            "cursor": cursor,
            "bbox": scene.get("bbox"),
            "target": target,
            "ideal_landing": target,
            "predicted_landing": landing,
            "ideal_vector": ideal,
            "predicted_vector": row.get("coord"),
            "error_px": row.get("endpoint_err_px"),
        }
        trace = common_trace(sealed, f"{sealed['run_id']}:{scene_id}")
        trace.update(
            {
                "experiment": experiment,
                "model": f"{arm} · r{rank or '?'} · s{step or '?'}",
                "modality": "mouse",
                "arm": arm,
                "checkpoint": f"step-{step}" if step is not None else "unknown",
                "rank": rank,
                "seed": row.get("k", 0),
                "grammar": manifest.get("grammar_name") or row.get("grammar"),
                "scale": row.get("kind", "unknown"),
                "prose": "preamble" if manifest.get("preamble") else "none",
                "typing_format": None,
                "success": safe_bool(row.get("in_box")),
                "parse_ok": safe_bool(row.get("parse_ok")),
                "format_ok": safe_bool(row.get("schema_ok")),
                "metrics": {
                    "in_box": row.get("in_box"),
                    "endpoint_error_px": row.get("endpoint_err_px"),
                    "distance_px": row.get("distance_px"),
                },
                "steps": [
                    {
                        "index": 0,
                        "instruction": "Move the cursor into the green target and click.",
                        "screenshot": screenshot,
                        "image_size": image_size,
                        "raw_output": row.get("raw_output", ""),
                        "parsed_action": {"action": row.get("action"), "coordinate": row.get("coord")},
                        "gold_action": {"action": row.get("action"), "coordinate": ideal},
                        "outcome": {"in_box": row.get("in_box"), "request_error": row.get("request_error")},
                        "metrics": trace_metric(row),
                        "overlay": overlay,
                    }
                ],
            }
        )
        traces.append(trace)
    return traces


def trace_metric(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        "endpoint_error_px": row.get("endpoint_err_px"),
        "distance_px": row.get("distance_px"),
        "in_box": row.get("in_box"),
        "guest_hit": row.get("guest_hit"),
        "action_span": row.get("completion_tokens"),
    }
    return {key: value for key, value in values.items() if value is not None}


def normalize_phaseb(sealed: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = sealed["manifest"]
    normalized = manifest.get("artifact_type") == "phaseb_normalized_canonical_oracle_eval"
    arm = str(manifest.get("arm", manifest.get("model_stage", "normalized")))
    model_dir = str(manifest.get("model_dir", ""))
    rank = match_number(RANK_RE, model_dir, sealed["alias"])
    step = match_number(STEP_RE, model_dir, sealed["alias"])
    traces: list[dict[str, Any]] = []
    for row in sealed["rows"]:
        error = row.get("net_landing_err_px") if normalized else row.get("err_px")
        coord = safe_bool(row.get("is_coord_record"))
        success = (isinstance(error, (int, float)) and error <= 50) if coord else safe_bool(
            row.get("action_sequence_match", row.get("action_match"))
        )
        parsed = row.get("pred_plan") if normalized else row.get("pred_actions")
        gold = row.get("gold_plan") if normalized else row.get("teacher_actions")
        trace = common_trace(sealed, f"{sealed['run_id']}:{row.get('sample_id')}")
        trace.update(
            {
                "experiment": "phaseb_normalized" if normalized else "phaseb_relative",
                "model": f"{arm} · r{rank or '?'} · s{step or '?'}",
                "modality": "mouse",
                "arm": arm,
                "checkpoint": f"step-{step}" if step is not None else "unknown",
                "rank": rank,
                "seed": None,
                "grammar": "move_rel",
                "scale": "osworld",
                "prose": "keep" if "prose_keep" in arm else "canonical",
                "typing_format": None,
                "success": success,
                "parse_ok": safe_bool(row.get("schema_parse_ok", row.get("parse_ok"))),
                "format_ok": safe_bool(row.get("schema_parse_ok", row.get("parse_ok"))),
                "metrics": {
                    "endpoint_error_px": error,
                    "action_sequence_match": row.get("action_sequence_match", row.get("action_match")),
                    "tolerant_50px": row.get("canonical_tolerant_50px_match", success),
                },
                "steps": [
                    {
                        "index": 0,
                        "instruction": f"Inspect OSWorld action sample {row.get('sample_id')}",
                        "screenshot": None,
                        "image_size": None,
                        "raw_output": row.get("raw_output", ""),
                        "parsed_action": parsed,
                        "gold_action": gold,
                        "outcome": {"success": success, "request_error": row.get("request_error")},
                        "metrics": {"endpoint_error_px": error} if error is not None else {},
                        "overlay": None,
                    }
                ],
            }
        )
        traces.append(trace)
    return traces


def normalize_typing(sealed: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = sealed["manifest"]
    generation = load_json(sealed["files"]["generation_manifest"])
    model_manifest = generation.get("model_manifest", {})
    if not isinstance(model_manifest, dict):
        raise AuditError(f"{sealed['alias']}: model manifest missing from typing seal")
    lineage = str(manifest.get("lineage"))
    target_format = str(manifest.get("target_format"))
    rank = model_manifest.get("lora_rank")
    checkpoint = "tp1-v10"
    traces: list[dict[str, Any]] = []
    for row in sealed["rows"]:
        target = str(row.get("target_text", ""))
        executed = str(row.get("executed_text", ""))
        events = event_tokens(str(row.get("raw_output", "")), target_format)
        trace = common_trace(sealed, f"{sealed['run_id']}:{row.get('sample_id')}")
        metrics = char_metrics(target, executed)
        metrics["action_span"] = len(events)
        trace.update(
            {
                "experiment": "typing_factorial",
                "model": f"lineage {lineage} · {target_format} · r{rank or '?'} · {checkpoint}",
                "modality": "typing",
                "arm": f"{lineage}_{target_format}",
                "checkpoint": checkpoint,
                "rank": rank,
                "seed": "frozen-200",
                "grammar": "deltatype_raw",
                "scale": "typing",
                "prose": "instruction-preamble" if "\n" in str(row.get("raw_output", "")) else "none",
                "typing_format": target_format,
                "success": safe_bool(row.get("exact_typed_string_success")),
                "parse_ok": safe_bool(row.get("parse_ok")),
                "format_ok": safe_bool(row.get("strict_schema_ok")),
                "metrics": metrics,
                "steps": [
                    {
                        "index": 0,
                        "instruction": f"Type exactly: {target}",
                        "screenshot": None,
                        "image_size": None,
                        "raw_output": row.get("raw_output", ""),
                        "parsed_action": {"executed_text": executed, "events": events},
                        "gold_action": {"requested_text": target},
                        "outcome": {
                            "exact": row.get("exact_typed_string_success"),
                            "parse_error": row.get("parse_error"),
                            "request_error": row.get("request_error"),
                        },
                        "metrics": metrics,
                        "typing": {
                            "requested": target,
                            "generated": row.get("raw_output", ""),
                            "executed": executed,
                            "events": events,
                        },
                        "overlay": None,
                    }
                ],
            }
        )
        traces.append(trace)
    return traces


def normalize_closed_loop(sealed: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = sealed["manifest"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sealed["rows"]:
        groups[(str(row.get("condition")), str(row.get("episode_id")))].append(row)
    rank = match_number(RANK_RE, str(manifest.get("checkpoint_alias", "")))
    traces: list[dict[str, Any]] = []
    for (condition, episode_id), rows in sorted(groups.items()):
        rows.sort(key=lambda row: (int(row.get("target_index", 0)), int(row.get("attempt", 0))))
        steps = []
        errors: list[float] = []
        for index, row in enumerate(rows):
            bbox = row.get("active_bbox")
            cursor = row.get("cursor_before")
            endpoint = row.get("endpoint")
            target = None
            error = None
            ideal_vector = None
            if isinstance(bbox, list) and len(bbox) == 4:
                target = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            if target and isinstance(endpoint, list) and len(endpoint) == 2:
                error = math.hypot(endpoint[0] - target[0], endpoint[1] - target[1])
                errors.append(error)
            if target and isinstance(cursor, list) and len(cursor) == 2:
                ideal_vector = [target[0] - cursor[0], target[1] - cursor[1]]
            steps.append(
                {
                    "index": index,
                    "instruction": f"Click target {row.get('target_index')} in closed-loop episode {episode_id}.",
                    "screenshot": None,
                    "image_size": [1440, 900],
                    "raw_output": row.get("raw_output", ""),
                    "parsed_action": {"coordinate": row.get("coord"), "dispatched": row.get("dispatched")},
                    "gold_action": {"target_bbox": bbox, "ideal_vector": ideal_vector},
                    "outcome": {
                        "guest_hit": row.get("guest_hit"),
                        "target_advanced": row.get("target_advanced"),
                        "terminated": row.get("terminated"),
                        "terminal_reason": row.get("terminal_reason"),
                    },
                    "metrics": {"endpoint_error_px": rounded(error), **trace_metric(row)},
                    "overlay": {
                        "cursor": cursor,
                        "bbox": bbox,
                        "target": target,
                        "ideal_landing": target,
                        "predicted_landing": endpoint,
                        "ideal_vector": ideal_vector,
                        "predicted_vector": row.get("coord"),
                        "error_px": rounded(error),
                    },
                }
            )
        success = all(safe_bool(row.get("guest_hit")) for row in rows)
        trace = common_trace(sealed, f"{sealed['run_id']}:{condition}:{episode_id}")
        trace.update(
            {
                "experiment": "proper_vm_closed_loop",
                "model": f"{manifest.get('arm')} · r{rank or '?'} · recovered-s750",
                "modality": "mouse",
                "arm": manifest.get("arm"),
                "checkpoint": "recovered-step-750",
                "rank": rank,
                "seed": "matched",
                "grammar": "deltatype_raw",
                "scale": condition,
                "prose": "model-history" if condition == "multi_step_closed_loop" else "none",
                "typing_format": None,
                "success": success,
                "parse_ok": all(safe_bool(row.get("parse_ok")) for row in rows),
                "format_ok": all(safe_bool(row.get("schema_ok")) for row in rows),
                "metrics": {
                    "in_box": success,
                    "endpoint_error_px": rounded(mean(errors)),
                    "steps": len(rows),
                },
                "steps": steps,
            }
        )
        traces.append(trace)
    return traces


ADAPTERS = {
    "relative_mouse": normalize_relative,
    "phaseb_mouse": normalize_phaseb,
    "typing": normalize_typing,
    "closed_loop_mouse": normalize_closed_loop,
}


def run_summary(sealed: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for trace in traces for step in trace["steps"]]
    success = [1.0 if trace["success"] else 0.0 for trace in traces]
    parse = [1.0 if trace["parse_ok"] else 0.0 for trace in traces]
    format_ok = [1.0 if trace["format_ok"] else 0.0 for trace in traces]
    in_box = [
        1.0 if value else 0.0
        for trace in traces
        for value in (trace.get("metrics", {}).get("in_box"),)
        if value is not None
    ]
    errors = [
        float(value)
        for step in steps
        for value in (step.get("metrics", {}).get("endpoint_error_px"),)
        if isinstance(value, (int, float))
    ]
    action_spans = [
        float(value)
        for trace in traces
        for value in (trace.get("metrics", {}).get("action_span"),)
        if isinstance(value, (int, float))
    ]
    first = traces[0]
    return {
        "run_id": sealed["run_id"],
        "job_id": sealed["job_id"],
        "artifact_id": sealed["artifact_id"],
        "artifact_alias": sealed["alias"],
        "recipe": sealed["recipe"],
        "manifest": sealed["manifest_file"],
        "manifest_sha256": sealed["manifest_sha256"],
        "created_at": sealed["created_at"],
        "experiment": first["experiment"],
        "model": first["model"],
        "modality": first["modality"],
        "arm": first["arm"],
        "checkpoint": first["checkpoint"],
        "rank": first["rank"],
        "seed": first["seed"],
        "grammar": first["grammar"],
        "prose": first["prose"],
        "typing_format": first["typing_format"],
        "n": len(steps),
        "n_traces": len(traces),
        "metrics": {
            "task_success_rate": rounded(mean(success)),
            "in_box_rate": rounded(mean(in_box)),
            "exact_typing_rate": rounded(mean(success)) if first["modality"] == "typing" else None,
            "parse_rate": rounded(mean(parse)),
            "format_validity_rate": rounded(mean(format_ok)),
            "mean_endpoint_error_px": rounded(mean(errors)),
            "median_endpoint_error_px": rounded(statistics.median(errors)) if errors else None,
            "mean_action_span": rounded(mean(action_spans)),
        },
    }


def discover(
    eval_root: Path,
    config: dict[str, Any],
) -> tuple[list[tuple[Path, str]], list[str], list[dict[str, str]]]:
    selected: dict[Path, str] = {}
    errors: list[str] = []
    directories = [path for path in eval_root.iterdir() if path.is_dir()]
    excluded: list[dict[str, str]] = []
    exclusions = config.get("exclusions", [])
    exclusion_matches: dict[str, int] = {str(item.get("glob")): 0 for item in exclusions}

    def exclusion_for(path: Path) -> dict[str, Any] | None:
        for item in exclusions:
            pattern = str(item.get("glob"))
            if fnmatch.fnmatch(path.name, pattern):
                exclusion_matches[pattern] += 1
                return item
        return None

    for rule in config.get("rules", []):
        pattern = rule.get("glob")
        adapter = rule.get("adapter")
        matches = sorted(path for path in directories if fnmatch.fnmatch(path.name, pattern))
        eligible: list[Path] = []
        for path in matches:
            exclusion = exclusion_for(path)
            if exclusion is None:
                eligible.append(path)
            elif not any(item["artifact_alias"] == path.name for item in excluded):
                excluded.append(
                    {
                        "artifact_alias": path.name,
                        "reason": str(exclusion.get("reason", "explicitly excluded")),
                    }
                )
        minimum = int(rule.get("min_matches", 0))
        if len(eligible) < minimum:
            errors.append(
                f"source rule {rule.get('name')!r} expected at least {minimum} eligible artifacts, "
                f"found {len(eligible)}"
            )
        for path in eligible:
            if path in selected and selected[path] != adapter:
                errors.append(f"artifact {path.name} matched incompatible adapters")
            selected[path] = adapter
    for pattern, count in exclusion_matches.items():
        if count == 0:
            errors.append(f"explicit exclusion matched no artifact: {pattern}")
    return sorted(selected.items()), errors, sorted(excluded, key=lambda item: item["artifact_alias"])


def write_bundle(
    output_dir: Path,
    index: dict[str, Any],
    sealed_artifacts: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    assets_dir = data_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copy2(HERE / filename, output_dir / filename)
    for sealed in sealed_artifacts:
        link = assets_dir / str(sealed["artifact_id"])
        target = sealed["artifact"].resolve()
        if link.is_symlink():
            if link.resolve() != target:
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
        elif link.exists():
            raise AuditError(f"asset link path exists but is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    temporary = data_dir / "index.json.tmp"
    temporary.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(data_dir / "index.json")


def build(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(args.rules)
    if config.get("schema_version") != 1:
        raise AuditError("unsupported source_rules schema")
    if not args.eval_root.is_dir() or not args.runs_root.is_dir():
        raise AuditError("eval or labctl-runs root is absent")
    discovered, errors, exclusions = discover(args.eval_root, config)
    sealed_artifacts: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for artifact, adapter in discovered:
        try:
            sealed = seal_artifact(
                artifact,
                adapter,
                args.runs_root,
                str(config.get("expected_user")),
            )
            normalizer = ADAPTERS.get(sealed["adapter"])
            if normalizer is None:
                raise AuditError(f"unsupported normalized adapter {sealed['adapter']}")
            artifact_traces = normalizer(sealed)
            if not artifact_traces:
                raise AuditError(f"{artifact.name}: adapter emitted no traces")
            sealed_artifacts.append(sealed)
            traces.extend(artifact_traces)
            runs.append(run_summary(sealed, artifact_traces))
        except AuditError as exc:
            errors.append(str(exc))
    status = "error" if errors else "complete"
    index = {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_rules_sha256": sha256_file(args.rules),
        "errors": errors,
        "excluded_sources": exclusions,
        "runs": sorted(runs, key=lambda run: (run["experiment"], run["arm"], run["rank"] or 0)),
        "traces": traces,
    }
    if errors:
        index["runs"] = []
        index["traces"] = []
        sealed_artifacts = []
    return index, sealed_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--rules", type=Path, default=HERE / "source_rules.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/fast/home/franz.srambical/tmp/relative_mouse_trace_inspector"),
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index, sealed = build(args)
        write_bundle(args.output_dir, index, sealed)
    except AuditError as exc:
        error_index = {
            "schema_version": 1,
            "status": "error",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "errors": [str(exc)],
            "runs": [],
            "traces": [],
        }
        try:
            write_bundle(args.output_dir, error_index, [])
        except AuditError:
            pass
        print(f"trace-index audit failed: {exc}", file=sys.stderr)
        return 2
    if index["status"] != "complete":
        for error in index["errors"]:
            print(f"trace-index audit failed: {error}", file=sys.stderr)
        return 2
    print(
        f"indexed {len(index['runs'])} sealed runs / {len(index['traces'])} traces -> "
        f"{args.output_dir / 'data/index.json'}"
    )
    if args.serve:
        os.chdir(args.output_dir)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler)
        print(f"serving http://127.0.0.1:{args.port}/ (Ctrl-C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
