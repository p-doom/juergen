#!/usr/bin/env python3
"""Matched Phase-B comparison of natural-prose absolute and move_rel arms.

The comparison is deliberately narrower than the Phase-B training matrix: it
accepts only the prose_keep source arm and its audited move_rel twin.  Rows are
paired by sample_id and uncertainty is a paired task-cluster bootstrap, because
several consecutive validation rows come from the same OSWorld task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


EXPECTED_ROWS = 233
EXPECTED_COORD_ROWS = 178
SCREEN_DIAGONAL_PX = math.hypot(1920, 1080)
BOOTSTRAP_SEED = 20260730


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnalysisError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise AnalysisError(f"blank JSONL line {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid JSON {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AnalysisError(f"non-object JSONL row {path}:{line_number}")
        rows.append(row)
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def validate_invariants(report: dict[str, Any]) -> None:
    require(report.get("status") == "pass", "relative twin invariant report did not pass")
    for key in ("assistant_outside_action_identity", "task_split_order_identity",
                "user_image_identity"):
        item = report.get(key, {})
        require(item.get("passing") == item.get("total") and item.get("total", 0) > 0,
                f"relative twin invariant failed: {key}")
    prose = report.get("natural_teacher_reasoning_retained", {})
    require(prose.get("passing") == prose.get("total_prose_turns")
            and prose.get("passing", 0) > 0,
            "natural teacher prose was not proved byte-retained")
    require(report.get("new_numeric_tokens_outside_action", {}).get("leaking") == 0,
            "relative twin introduced numeric text outside action spans")
    require(report.get("fallback_turns") == 0,
            "relative twin contains fallback action conversions")
    landing = report.get("common_pixel_landing", {})
    require(landing.get("within_2px") == landing.get("total_coordinate_turns")
            and landing.get("total_coordinate_turns", 0) > 0,
            "absolute/relative teacher landing parity failed")


def validate_eval_manifests(*, absolute: dict[str, Any], relative: dict[str, Any],
                            absolute_rows: Path, relative_rows: Path,
                            absolute_val: Path, relative_val: Path) -> None:
    require(absolute.get("valid") is True and absolute.get("arm") == "prose_keep",
            "absolute eval manifest is not a valid prose_keep evaluation")
    require(absolute.get("own_val_contract", {}).get("cross_arm_prompt_reuse") is False,
            "absolute evaluation reused another arm's prompts")
    require(absolute.get("own_val_contract", {}).get("n_rows") == EXPECTED_ROWS,
            "absolute eval manifest row count changed")
    require(absolute.get("own_val_contract", {}).get("sha256") == sha256(absolute_val),
            "absolute validation data hash disagrees with eval manifest")
    require(absolute.get("evaluation", {}).get("request_errors") == 0,
            "absolute evaluation contains request errors")
    require(absolute.get("evaluation", {}).get("rows_sha256") == sha256(absolute_rows),
            "absolute rows hash disagrees with eval manifest")

    require(relative.get("artifact_type") == "phaseb_relative_eval"
            and relative.get("status") == "complete"
            and relative.get("valid") is True
            and relative.get("arm") == "prose_keep",
            "relative eval manifest is not a complete prose_keep evaluation")
    require(relative.get("request_errors") == 0,
            "relative evaluation contains request errors")
    require(relative.get("rows_sha256") == sha256(relative_rows),
            "relative rows hash disagrees with eval manifest")
    require(relative.get("val_chat_sha256") == sha256(relative_val),
            "relative validation data hash disagrees with eval manifest")
    require(absolute.get("evaluation", {}).get("sampling") == {"temperature": 0.0}
            and relative.get("sampling") == {"temperature": 0.0},
            "sampling contract is not matched greedy decoding")


def validate_relative_lineage(relative: dict[str, Any]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    export_path = Path(str(relative.get("export_manifest", ""))).resolve()
    require(export_path.is_file(), "relative export manifest is missing")
    require(relative.get("export_manifest_sha256") == sha256(export_path),
            "relative eval/export manifest hash linkage failed")
    export = load_json(export_path)
    require(export.get("artifact_type") == "phaseb_relative_hf_checkpoint"
            and export.get("status") == "complete"
            and export.get("arm") == "prose_keep"
            and export.get("step") == 900
            and export.get("lora_rank") == 32
            and export.get("max_length") == 16384,
            "relative export is not the requested complete step-900 endpoint")
    source_checkpoint = Path(str(export.get("source_checkpoint", ""))).resolve()
    require(source_checkpoint.name == "000900"
            and (source_checkpoint / "_CHECKPOINT_METADATA").is_file(),
            "relative source step-900 checkpoint linkage failed")
    train_path = source_checkpoint.parents[1] / "train_manifest.json"
    require(train_path.is_file(), "relative training manifest is missing")
    require(export.get("train_manifest_sha256") == sha256(train_path),
            "relative train/export manifest hash linkage failed")
    train = load_json(train_path)
    require(train.get("artifact_type") == "phaseb_relative_orbax"
            and train.get("status") == "complete"
            and train.get("arm") == "prose_keep"
            and train.get("step") == 900
            and train.get("slurm_job_id") == "135403",
            "relative training manifest is not the requested source job")
    return export_path, export, train_path, train


def distance_regime(is_coord: bool, distance: float | None) -> str:
    if not is_coord:
        return "non_coordinate"
    assert distance is not None
    if distance <= 2:
        return "stationary_0_2px"
    if distance < 150:
        return "short_gt2_lt150px"
    if distance < 500:
        return "medium_150_lt500px"
    return "far_ge500px"


def pair_rows(*, absolute_rows: list[dict[str, Any]],
              relative_rows: list[dict[str, Any]],
              absolute_val: list[dict[str, Any]],
              relative_val: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for name, rows in (("absolute rows", absolute_rows), ("relative rows", relative_rows),
                       ("absolute val", absolute_val), ("relative val", relative_val)):
        require(len(rows) == EXPECTED_ROWS, f"{name}: expected {EXPECTED_ROWS}, got {len(rows)}")
        ids = [row.get("sample_id") for row in rows]
        require(all(isinstance(value, str) and value for value in ids),
                f"{name}: missing sample_id")
        require(len(set(ids)) == len(ids), f"{name}: duplicate sample_id")

    abs_eval = {row["sample_id"]: row for row in absolute_rows}
    rel_eval = {row["sample_id"]: row for row in relative_rows}
    abs_data = {row["sample_id"]: row for row in absolute_val}
    rel_data = {row["sample_id"]: row for row in relative_val}
    id_sets = [set(value) for value in (abs_eval, rel_eval, abs_data, rel_data)]
    require(all(value == id_sets[0] for value in id_sets[1:]),
            "sample_id sets differ across absolute/relative data or evaluations")
    require([row["sample_id"] for row in absolute_val]
            == [row["sample_id"] for row in relative_val],
            "absolute/relative validation row order changed")

    paired = []
    for sample_id in [row["sample_id"] for row in absolute_val]:
        ad, rd = abs_data[sample_id], rel_data[sample_id]
        ar, rr = abs_eval[sample_id], rel_eval[sample_id]
        for field in ("recording_id", "app", "task_id", "step"):
            require(ad.get(field) == rd.get(field),
                    f"{sample_id}: paired dataset field changed: {field}")
        require(ar.get("request_error") is False and rr.get("request_error") is False,
                f"{sample_id}: request error in matched comparison")
        is_coord = bool(ar.get("is_coord_record"))
        require(is_coord == bool(rr.get("is_coord_record")),
                f"{sample_id}: coordinate-record classification differs")
        audit = rd.get("phaseb_relative_audit")
        require(isinstance(audit, list) and audit,
                f"{sample_id}: relative geometry audit missing")
        state = audit[-1]
        if is_coord:
            require(state.get("absolute_landing_px") is not None,
                    f"{sample_id}: coordinate row lacks absolute landing audit")
            target_delta = math.dist(state["absolute_landing_px"],
                                     state["relative_landing_px"])
            require(target_delta <= math.sqrt(2) + 1e-9,
                    f"{sample_id}: paired teacher landings differ by {target_delta:.3f}px")
            distance = math.dist(state["cursor_before_px"], state["relative_landing_px"])
        else:
            require(state.get("absolute_landing_px") is None,
                    f"{sample_id}: non-coordinate row has an absolute landing")
            target_delta = None
            distance = None

        def error(row: dict[str, Any]) -> float | None:
            value = row.get("err_px")
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

        ae, re = error(ar), error(rr)
        paired.append({
            "sample_id": sample_id,
            "recording_id": ad["recording_id"],
            "task_id": ad["task_id"],
            "app": ad["app"],
            "step": ad["step"],
            "is_coord_record": is_coord,
            "distance_px": distance,
            "distance_regime": distance_regime(is_coord, distance),
            "paired_teacher_landing_delta_px": target_delta,
            "absolute": {
                "parse_ok": bool(ar.get("pred_parse_ok")),
                "action_match": bool(ar.get("action_match")),
                "coord_emitted": ae is not None if is_coord else None,
                "err_px": ae,
            },
            "relative": {
                "parse_ok": bool(rr.get("parse_ok")),
                "action_match": bool(rr.get("action_match")),
                "coord_emitted": re is not None if is_coord else None,
                "err_px": re,
            },
        })
    require(sum(row["is_coord_record"] for row in paired) == EXPECTED_COORD_ROWS,
            "matched coordinate-row count changed")
    return paired


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    coord = [row for row in rows if row["is_coord_record"]]
    errors = [row[arm]["err_px"] for row in coord if row[arm]["err_px"] is not None]
    integrated = []
    for row in rows:
        if row["is_coord_record"]:
            err = row[arm]["err_px"]
            integrated.append(bool(row[arm]["action_match"] and err is not None and err <= 100))
        else:
            integrated.append(bool(row[arm]["action_match"]))
    capped = [min(row[arm]["err_px"], SCREEN_DIAGONAL_PX)
              if row[arm]["err_px"] is not None else SCREEN_DIAGONAL_PX
              for row in coord]
    return {
        "n_rows": len(rows),
        "n_coord_rows": len(coord),
        "parse_rate": rate([row[arm]["parse_ok"] for row in rows]),
        "own_grammar_action_match_rate": rate([row[arm]["action_match"] for row in rows]),
        "integrated_action_and_landing_success_rate": rate(integrated),
        "coord_emit_rate": rate([row[arm]["coord_emitted"] for row in coord]),
        "within_50px_rate": rate([row[arm]["err_px"] is not None
                                   and row[arm]["err_px"] <= 50 for row in coord]),
        "within_100px_rate": rate([row[arm]["err_px"] is not None
                                    and row[arm]["err_px"] <= 100 for row in coord]),
        "median_err_px_emitted": statistics.median(errors) if errors else None,
        "p90_err_px_emitted": percentile(errors, 0.90),
        "mean_capped_err_px_missing_as_screen_diagonal": statistics.fmean(capped) if capped else None,
    }


Metric = Callable[[list[dict[str, Any]], str], float | None]


def _metric(name: str) -> Metric:
    def value(rows: list[dict[str, Any]], arm: str) -> float | None:
        return arm_summary(rows, arm)[name]
    return value


PAIRED_METRICS: dict[str, Metric] = {
    name: _metric(name) for name in (
        "parse_rate", "own_grammar_action_match_rate",
        "integrated_action_and_landing_success_rate", "coord_emit_rate",
        "within_50px_rate", "within_100px_rate",
        "mean_capped_err_px_missing_as_screen_diagonal",
    )
}


def bootstrap_ci(rows: list[dict[str, Any]], metric: Metric, *, n_boot: int,
                 seed: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task)
    require(len(tasks) >= 2, "paired task-cluster bootstrap needs at least two tasks")
    point_a, point_r = metric(rows, "absolute"), metric(rows, "relative")
    require(point_a is not None and point_r is not None, "undefined paired point estimate")
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = []
        for _task in tasks:
            sampled = tasks[rng.randrange(len(tasks))]
            sample.extend(by_task[sampled])
        a, r = metric(sample, "absolute"), metric(sample, "relative")
        if a is not None and r is not None:
            draws.append(r - a)
    require(len(draws) == n_boot, "undefined bootstrap replicate")
    draws.sort()
    lo = draws[math.floor(0.025 * (len(draws) - 1))]
    hi = draws[math.ceil(0.975 * (len(draws) - 1))]
    return {
        "absolute": point_a,
        "relative": point_r,
        "relative_minus_absolute": point_r - point_a,
        "paired_task_cluster_bootstrap_95ci": [lo, hi],
        "n_boot": n_boot,
        "n_task_clusters": len(tasks),
    }


def discordance(rows: list[dict[str, Any]], key: str, *, coord_only: bool = False,
                threshold: float | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if coord_only and not row["is_coord_record"]:
            continue
        if threshold is None:
            a, r = bool(row["absolute"][key]), bool(row["relative"][key])
        else:
            ae, re = row["absolute"]["err_px"], row["relative"]["err_px"]
            a, r = ae is not None and ae <= threshold, re is not None and re <= threshold
        counts[f"absolute_{int(a)}_relative_{int(r)}"] += 1
    return dict(sorted(counts.items()))


def analyze(rows: list[dict[str, Any]], *, n_boot: int) -> dict[str, Any]:
    overall = {name: bootstrap_ci(rows, metric, n_boot=n_boot,
                                  seed=BOOTSTRAP_SEED + index)
               for index, (name, metric) in enumerate(PAIRED_METRICS.items())}
    regimes = {}
    for regime_index, regime in enumerate((
            "non_coordinate", "stationary_0_2px", "short_gt2_lt150px",
            "medium_150_lt500px", "far_ge500px")):
        subset = [row for row in rows if row["distance_regime"] == regime]
        valid_metrics = {
            "parse_rate": PAIRED_METRICS["parse_rate"],
            "own_grammar_action_match_rate": PAIRED_METRICS["own_grammar_action_match_rate"],
            "integrated_action_and_landing_success_rate":
                PAIRED_METRICS["integrated_action_and_landing_success_rate"],
        }
        if regime != "non_coordinate":
            valid_metrics.update({key: PAIRED_METRICS[key] for key in (
                "coord_emit_rate", "within_50px_rate", "within_100px_rate",
                "mean_capped_err_px_missing_as_screen_diagonal")})
        regimes[regime] = {
            "n_rows": len(subset),
            "n_task_clusters": len({row["task_id"] for row in subset}),
            "absolute": arm_summary(subset, "absolute"),
            "relative": arm_summary(subset, "relative"),
            "paired": {name: bootstrap_ci(
                subset, metric, n_boot=n_boot,
                seed=BOOTSTRAP_SEED + 100 + i + 20 * regime_index
            ) for i, (name, metric) in enumerate(valid_metrics.items())},
        }
    return {
        "absolute": arm_summary(rows, "absolute"),
        "relative": arm_summary(rows, "relative"),
        "paired_effects": overall,
        "discordance": {
            "parse": discordance(rows, "parse_ok"),
            "own_grammar_action_match": discordance(rows, "action_match"),
            "within_50px": discordance(rows, "err_px", coord_only=True, threshold=50),
            "within_100px": discordance(rows, "err_px", coord_only=True, threshold=100),
        },
        "by_distance_regime": regimes,
    }


def markdown(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}%"

    def num(value: float | None) -> str:
        return "NA" if value is None else f"{value:.1f}"

    analysis = report["analysis"]
    lines = [
        "# Phase-B matched natural-prose comparison", "",
        "Only the paired prose_keep absolute and move_rel arms are inferential. "
        "The historical prose_strip arm is excluded.", "",
        "| metric | absolute | relative | relative − absolute (95% paired task-cluster CI) |",
        "|---|---:|---:|---:|",
    ]
    for key, label, is_rate in (
        ("parse_rate", "parse", True),
        ("own_grammar_action_match_rate", "own-grammar action match", True),
        ("integrated_action_and_landing_success_rate", "integrated success", True),
        ("coord_emit_rate", "coordinate emitted", True),
        ("within_50px_rate", "landing within 50 px", True),
        ("within_100px_rate", "landing within 100 px", True),
        ("mean_capped_err_px_missing_as_screen_diagonal", "capped mean landing error (px)", False),
    ):
        value = analysis["paired_effects"][key]
        formatter = pct if is_rate else num
        delta = (f"{100 * value['relative_minus_absolute']:+.1f} pp"
                 if is_rate else f"{value['relative_minus_absolute']:+.1f} px")
        ci = value["paired_task_cluster_bootstrap_95ci"]
        ci_text = (f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp"
                   if is_rate else f"[{ci[0]:+.1f}, {ci[1]:+.1f}] px")
        lines.append(f"| {label} | {formatter(value['absolute'])} | "
                     f"{formatter(value['relative'])} | {delta} ({ci_text}) |")
    lines += ["", "## Distance regimes", "",
              "Regimes use the audited teacher movement from cursor start to landing: "
              "stationary ≤2 px, short >2 and <150 px, medium 150–<500 px, far ≥500 px.", "",
              "| regime | n | absolute ≤100 px | relative ≤100 px | Δ |",
              "|---|---:|---:|---:|---:|"]
    for name, item in analysis["by_distance_regime"].items():
        if name == "non_coordinate":
            continue
        effect = item["paired"]["within_100px_rate"]
        lines.append(f"| {name} | {item['n_rows']} | {pct(effect['absolute'])} | "
                     f"{pct(effect['relative'])} | "
                     f"{100 * effect['relative_minus_absolute']:+.1f} pp |")
    lines += ["", "Bootstrap unit: OSWorld task_id; pairs remain joined within every replicate. "
              "Missing coordinate emissions count as landing failures and as one screen diagonal "
              "in the capped-error metric.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute-eval", type=Path, required=True)
    parser.add_argument("--relative-eval", type=Path, required=True)
    parser.add_argument("--absolute-val", type=Path, required=True)
    parser.add_argument("--relative-val", type=Path, required=True)
    parser.add_argument("--invariant-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=20_000)
    args = parser.parse_args()
    try:
        require(args.n_boot >= 1000, "n_boot must be at least 1000")
        abs_manifest = load_json(args.absolute_eval / "eval_manifest.json")
        rel_manifest = load_json(args.relative_eval / "eval_manifest.json")
        invariant = load_json(args.invariant_report)
        validate_invariants(invariant)
        abs_rows_path, rel_rows_path = (args.absolute_eval / "rows.jsonl",
                                        args.relative_eval / "rows.jsonl")
        validate_eval_manifests(
            absolute=abs_manifest, relative=rel_manifest,
            absolute_rows=abs_rows_path, relative_rows=rel_rows_path,
            absolute_val=args.absolute_val, relative_val=args.relative_val,
        )
        rel_export_path, rel_export, rel_train_path, rel_train = validate_relative_lineage(
            rel_manifest
        )
        rows = pair_rows(
            absolute_rows=load_jsonl(abs_rows_path), relative_rows=load_jsonl(rel_rows_path),
            absolute_val=load_jsonl(args.absolute_val), relative_val=load_jsonl(args.relative_val),
        )
        args.out.mkdir(parents=True, exist_ok=True)
        rows_out = args.out / "matched_rows.jsonl"
        rows_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                            encoding="utf-8")
        report = {
            "artifact_type": "phaseb_matched_natural_prose_analysis",
            "schema_version": 1,
            "status": "complete",
            "comparison": "absolute prose_keep vs relative prose_keep only",
            "historical_prose_strip_role": "diagnostic_only_excluded_from_inference",
            "pairing": {"unit": "sample_id", "n_pairs": len(rows),
                        "bootstrap_unit": "task_id", "seed": BOOTSTRAP_SEED,
                        "n_boot": args.n_boot},
            "preservation_evidence": {
                "invariant_report": str(args.invariant_report.resolve()),
                "invariant_report_sha256": sha256(args.invariant_report),
                "assistant_prose_outside_action_byte_identical": True,
                "user_and_image_content_identical": True,
                "task_split_and_order_identical": True,
                "goals_preserved_by_user_content_identity": True,
            },
            "sources": {
                "absolute_training_job_id": abs_manifest["source_training"]["slurm_job_id"],
                "absolute_export_job_id": abs_manifest["model"]["export_slurm_job_id"],
                "absolute_eval_job_id": abs_manifest["evaluation"]["slurm_job_id"],
                "relative_training_job_id": rel_train["slurm_job_id"],
                "relative_export_job_id": rel_export["slurm_job_id"],
                "relative_eval_job_id": rel_manifest["slurm_job_id"],
                "absolute_eval_manifest_sha256": sha256(args.absolute_eval / "eval_manifest.json"),
                "relative_eval_manifest_sha256": sha256(args.relative_eval / "eval_manifest.json"),
                "relative_train_manifest": str(rel_train_path),
                "relative_train_manifest_sha256": sha256(rel_train_path),
                "relative_export_manifest": str(rel_export_path),
                "relative_export_manifest_sha256": sha256(rel_export_path),
                "absolute_rows_sha256": sha256(abs_rows_path),
                "relative_rows_sha256": sha256(rel_rows_path),
                "absolute_val_sha256": sha256(args.absolute_val),
                "relative_val_sha256": sha256(args.relative_val),
            },
            "matched_rows_sha256": sha256(rows_out),
            "analysis": analyze(rows, n_boot=args.n_boot),
        }
        report_path = args.out / "matched_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        (args.out / "matched_report.md").write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"status": "complete", "report": str(report_path),
                          "n_pairs": len(rows)}, sort_keys=True))
        return 0
    except AnalysisError as exc:
        print(f"FATAL matched Phase-B analysis: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
