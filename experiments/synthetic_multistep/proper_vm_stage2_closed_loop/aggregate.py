#!/usr/bin/env python3
"""Fail-closed episode-cluster aggregation for true roadmap stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from contract import (  # type: ignore  # noqa: E402
    Contract,
    strict_schema_ok,
    unit_range_ok,
)

sys.path.insert(0, str(HERE.parent / "proper_vm_stage2"))
from gate import clopper_pearson_upper, load_cells, rgb_sha256, sha256_file  # type: ignore  # noqa: E402

from closed_loop_contract import (  # noqa: E402
    AttemptEvidence,
    ClosedLoopState,
    TransitionError,
    advance,
    initial_state,
    reference_png,
    request_seed,
)
from runner import (  # noqa: E402
    ARM_NAMES,
    PROTOCOL_PATH,
    GateError,
    load_protocol,
    validate_protocol,
)


ARM_SEMANTICS = {
    "absolute_matched_control": "absolute_toolcall",
    "normalized_relative": "move_rel",
    "raw_relative": "deltatype_raw",
}


def _tool_call_objects(value: Any) -> list[Any]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise GateError("attempt tool-call audit drift")
    calls = []
    for item in value:
        if set(item) != {"name", "arguments"}:
            raise GateError("attempt tool-call object drift")
        calls.append(
            SimpleNamespace(
                function=SimpleNamespace(
                    name=item["name"], arguments=item["arguments"]
                )
            )
        )
    return calls


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def _targets_by_episode(contract: Contract) -> dict[int, list[Any]]:
    stage1_protocol = _load_object(HERE.parent / "proper_vm_stage2" / "protocol.json")
    cells = load_cells(stage1_protocol, contract)
    episodes: dict[int, list[Any]] = {}
    for cell in cells:
        episodes.setdefault(cell.episode_index, []).append(cell)
    for episode_cells in episodes.values():
        episode_cells.sort(key=lambda cell: cell.target_index)
    return episodes


def _replay_unit(
    unit: dict[str, Any],
    *,
    arm: str,
    protocol_hash: str,
    contract: Contract,
    episode_cells: list[Any],
) -> dict[str, Any]:
    condition = unit.get("condition")
    if condition not in {"single_step_sentinel", "multi_step_closed_loop"}:
        raise GateError("unknown complete-unit condition")
    required = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_complete_unit",
        "status": "complete",
        "arm": arm,
        "protocol_sha256": protocol_hash,
    }
    if any(unit.get(key) != value for key, value in required.items()):
        raise GateError("complete-unit provenance drift")
    if condition == "single_step_sentinel":
        target_index = unit.get("sentinel_target_index")
        if not isinstance(target_index, int) or not (0 <= target_index < 4):
            raise GateError("sentinel target index drift")
        cell = episode_cells[target_index]
        targets = [cell.bbox]
        initial_cursor = cell.cursor
        expected_episode_id = f"{cell.episode_id}:t{target_index:02d}"
        maximum = 1
    else:
        targets = [cell.bbox for cell in episode_cells]
        initial_cursor = episode_cells[0].cursor
        expected_episode_id = episode_cells[0].episode_id
        maximum = 3
    if unit.get("episode_id") != expected_episode_id:
        raise GateError("complete-unit episode id drift")
    state = initial_state(expected_episode_id, initial_cursor)
    semantic = ARM_SEMANTICS[arm]
    episode_revision = hashlib.sha256(
        f"{arm}|{condition}|{expected_episode_id}".encode("utf-8")
    ).hexdigest()
    render_revision = "initial"
    command_sequence = 0
    releases = 0
    rows = unit.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GateError("complete unit has no attempt rows")
    for row in rows:
        if not isinstance(row, dict):
            raise GateError("non-object attempt row")
        before = state
        target_index = state.target_index
        expected_seed = request_seed(
            condition, expected_episode_id, target_index, state.attempts_on_target + 1
        )
        expected = {
            "condition": condition,
            "episode_id": expected_episode_id,
            "episode_index": episode_cells[0].episode_index,
            "target_index": target_index,
            "attempt": state.attempts_on_target + 1,
            "request_seed": expected_seed,
            "cursor_before": list(state.cursor),
            "active_bbox": list(targets[target_index]),
            "observation_rgb_sha256": rgb_sha256(reference_png(contract, state, targets)),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise GateError(f"attempt identity/dynamic observation drift: {expected_episode_id}")
        if row.get("render_revision_before") != render_revision:
            raise GateError("attempt render revision drift")
        raw = row.get("raw_output")
        if not isinstance(raw, str) or not raw:
            raise GateError("attempt raw output missing")
        tool_calls = _tool_call_objects(row.get("tool_calls"))
        parse_text = raw.split(" | tool_calls=", 1)[0]
        move = contract.parse(semantic, parse_text, tool_calls or None)
        replay_schema = strict_schema_ok(semantic, parse_text, move.coord)
        replay_units = unit_range_ok(semantic, move.coord)
        replay_endpoint = (
            contract.apply_coord(semantic, state.cursor, move.coord)
            if move.coord is not None
            else None
        )
        replay_parse = bool(move.parse_ok and move.coord is not None)
        replay_action = {
            "parse_ok": replay_parse,
            "schema_ok": replay_schema,
            "unit_range_ok": replay_units,
            "coord": list(move.coord) if move.coord is not None else None,
            "endpoint": list(replay_endpoint) if replay_endpoint is not None else None,
            "dispatched": bool(replay_parse and replay_schema and replay_units),
        }
        if any(row.get(key) != value for key, value in replay_action.items()):
            raise GateError("attempt parse/schema/endpoint replay drift")
        for key in (
            "parse_ok",
            "schema_ok",
            "unit_range_ok",
            "dispatched",
            "guest_hit",
            "target_advanced",
            "terminated",
            "success",
        ):
            if not isinstance(row.get(key), bool):
                raise GateError(f"attempt boolean drift: {key}")
        endpoint_value = row.get("endpoint")
        endpoint = tuple(endpoint_value) if endpoint_value is not None else None
        if row["dispatched"]:
            releases += 1
        actual_cursor = tuple(row["cursor_after"]) if row["dispatched"] else None
        evidence = AttemptEvidence(
            raw_output=str(row.get("raw_output", "")),
            parse_ok=row["parse_ok"],
            schema_ok=row["schema_ok"],
            unit_range_ok=row["unit_range_ok"],
            dispatched=row["dispatched"],
            endpoint=endpoint,
            actual_cursor_after=actual_cursor,
            guest_hit=(row["guest_hit"] if row["dispatched"] else None),
        )
        try:
            transition = advance(
                state, evidence, targets, max_attempts_per_target=maximum
            )
        except TransitionError as exc:
            raise GateError(f"attempt transition failed replay: {exc}") from exc
        state = transition.after
        reproduced = {
            "cursor_after": list(state.cursor),
            "guest_hit": transition.hit,
            "target_advanced": transition.target_advanced,
            "attempts_on_target_after": state.attempts_on_target,
            "terminated": state.terminated,
            "success": state.success,
            "terminal_reason": transition.terminal_reason,
        }
        if any(row.get(key) != value for key, value in reproduced.items()):
            raise GateError("attempt transition result drift")
        guest = row.get("guest_state")
        if not isinstance(guest, dict) or guest.get("down") is not False:
            raise GateError("attempt guest button state drift")
        if guest.get("target_index") != state.target_index:
            raise GateError("attempt guest target state drift")
        if guest.get("completed") is not state.success:
            raise GateError("attempt guest completion state drift")
        if int(guest.get("button_presses", -1)) != releases or int(
            guest.get("button_releases", -1)
        ) != releases:
            raise GateError("attempt guest button-count drift")
        if guest.get("render_revision") != render_revision:
            raise GateError("attempt guest render revision drift")
        if guest.get("rendered_cursor") != list(before.cursor):
            raise GateError("attempt guest rendered-cursor drift")
        if guest.get("image_sha256") != hashlib.sha256(
            reference_png(contract, before, targets)
        ).hexdigest():
            raise GateError("attempt guest PNG byte-hash drift")
        if row["dispatched"]:
            if guest.get("last_release_position") != row.get("endpoint"):
                raise GateError("attempt guest release endpoint drift")
            if guest.get("last_hit") is not transition.hit:
                raise GateError("attempt guest hit readback drift")
        if state.terminated:
            if row.get("next_observation_rgb_sha256") is not None or row.get(
                "next_render_revision"
            ) is not None:
                raise GateError("terminal attempt exposes a next observation")
        else:
            command_sequence += 1
            next_png = reference_png(contract, state, targets)
            expected_next_revision = hashlib.sha256(
                f"{episode_revision}|{command_sequence}|{state.target_index}|{state.cursor}".encode()
            ).hexdigest()
            if row.get("next_observation_rgb_sha256") != rgb_sha256(next_png):
                raise GateError("attempt next-observation hash drift")
            if row.get("next_render_revision") != expected_next_revision:
                raise GateError("attempt next-render revision drift")
            render_revision = expected_next_revision
        if before.terminated:
            raise GateError("attempt recorded after termination")
    if not state.terminated:
        raise GateError("complete unit did not terminate")
    summary = unit.get("summary")
    expected_summary = {
        "success": state.success,
        "terminated": state.terminated,
        "attempts_total": state.attempts_total,
        "target_hit_attempts": list(state.target_hit_attempts),
        "targets_reached": len(state.target_hit_attempts),
        "final_target_index": state.target_index,
        "final_cursor": list(state.cursor),
    }
    if summary != expected_summary:
        raise GateError("complete-unit summary does not replay")
    return unit


def _load_arm(
    root: Path,
    arm: str,
    protocol: dict[str, Any],
    protocol_hash: str,
    contract: Contract,
    episodes: dict[int, list[Any]],
) -> dict[str, Any]:
    manifest = _load_object(root / "arm_manifest.json")
    expected_manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_closed_loop_arm",
        "status": "complete",
        "arm": arm,
        "protocol_sha256": protocol_hash,
        "checkpoint_alias": protocol["arms_draft"][arm]["checkpoint_alias"],
        "model_weights_sha256": protocol["arms_draft"][arm]["model_weights_sha256"],
        "atomic_units": 400,
        "sentinel_units": 320,
        "multi_step_units": 80,
        "request_errors": 0,
        "infrastructure_mismatches": 0,
        "guest_teardown_proven": True,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise GateError(f"{arm}: arm manifest drift")
    rows_path = root / "rows.jsonl"
    if sha256_file(rows_path) != manifest.get("rows_sha256"):
        raise GateError(f"{arm}: merged rows hash drift")
    unit_paths = sorted((root / "units").rglob("*.json"))
    if len(unit_paths) != 400:
        raise GateError(f"{arm}: complete-unit cardinality drift")
    sentinel: dict[tuple[int, int], dict[str, Any]] = {}
    multi: dict[int, dict[str, Any]] = {}
    merged_rows: list[dict[str, Any]] = []
    for path in unit_paths:
        value = _load_object(path)
        episode_index = value.get("episode_index")
        if not isinstance(episode_index, int) or episode_index not in episodes:
            raise GateError(f"{arm}: unit episode index drift")
        value = _replay_unit(
            value,
            arm=arm,
            protocol_hash=protocol_hash,
            contract=contract,
            episode_cells=episodes[episode_index],
        )
        if value["condition"] == "single_step_sentinel":
            key = (episode_index, value["sentinel_target_index"])
            if key in sentinel:
                raise GateError(f"{arm}: duplicate sentinel unit")
            sentinel[key] = value
        else:
            if episode_index in multi:
                raise GateError(f"{arm}: duplicate multi-step episode")
            multi[episode_index] = value
        merged_rows.extend(value["rows"])
    if set(sentinel) != {(episode, target) for episode in range(80) for target in range(4)}:
        raise GateError(f"{arm}: incomplete sentinel grid")
    if set(multi) != set(range(80)):
        raise GateError(f"{arm}: incomplete multi-step episodes")
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged_rows)
    if rows_path.read_text(encoding="utf-8") != rendered:
        raise GateError(f"{arm}: merged rows do not reproduce atomic units")
    if manifest.get("rows") != len(merged_rows):
        raise GateError(f"{arm}: merged row count drift")
    return {"manifest": manifest, "sentinel": sentinel, "multi": multi}


def _curves(
    multi: dict[int, dict[str, Any]],
    keys: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    selected = keys or {(episode, target) for episode in range(80) for target in range(4)}
    hit_attempts: list[int | None] = []
    attempted: list[int | None] = []
    for episode_index in range(80):
        unit = multi[episode_index]
        summary_attempts = unit["summary"]["target_hit_attempts"]
        attempts_by_target = {
            target: [row for row in unit["rows"] if row["target_index"] == target]
            for target in range(4)
        }
        for target in range(4):
            if (episode_index, target) not in selected:
                continue
            rows = attempts_by_target[target]
            attempted.append(len(rows) if rows else None)
            hit_attempts.append(
                summary_attempts[target] if target < len(summary_attempts) else None
            )
    result: dict[str, Any] = {}
    risk_count = sum(value is not None for value in attempted)
    unconditional_count = len(selected)
    for attempt in (1, 2, 3):
        successes = sum(value is not None and value <= attempt for value in hit_attempts)
        result[f"cumulative_by_{attempt}"] = {
            "unconditional_numerator": successes,
            "unconditional_denominator": unconditional_count,
            "unconditional_rate": successes / unconditional_count,
            "risk_set_numerator": successes,
            "risk_set_denominator": risk_count,
            "risk_set_rate": successes / risk_count if risk_count else None,
        }
    return result


def _distance_quartiles(
    episodes: dict[int, list[Any]], contract: Contract
) -> dict[int, set[tuple[int, int]]]:
    ranked = sorted(
        (
            contract.distance_to_box(cell.cursor, cell.bbox),
            episode_index,
            cell.target_index,
        )
        for episode_index, cells in episodes.items()
        for cell in cells
    )
    if len(ranked) != 320:
        raise GateError("distance-quartile population drift")
    result = {quartile: set() for quartile in range(1, 5)}
    for rank, (_, episode_index, target_index) in enumerate(ranked):
        result[rank // 80 + 1].add((episode_index, target_index))
    if any(len(value) != 80 for value in result.values()):
        raise GateError("distance quartiles are not exactly balanced")
    return result


def _sentinel_rate(
    sentinel: dict[tuple[int, int], dict[str, Any]], keys: set[tuple[int, int]]
) -> dict[str, Any]:
    successes = sum(bool(sentinel[key]["summary"]["success"]) for key in keys)
    return {
        "successes": successes,
        "n": len(keys),
        "rate": successes / len(keys),
    }


def _contrast(
    absolute: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    absolute_success = {key: value["summary"]["success"] for key, value in absolute.items()}
    treatment_success = {key: value["summary"]["success"] for key, value in treatment.items()}
    absolute_only = sum(absolute_success[key] and not treatment_success[key] for key in absolute)
    treatment_only = sum(treatment_success[key] and not absolute_success[key] for key in absolute)
    both_success = sum(absolute_success[key] and treatment_success[key] for key in absolute)
    both_failure = sum(not absolute_success[key] and not treatment_success[key] for key in absolute)
    difference = (treatment_only - absolute_only) / 80
    finite_pass = difference > -0.05
    upper = clopper_pearson_upper(absolute_only, 80, 0.05)
    inferential_pass = upper < 0.05
    if finite_pass and inferential_pass:
        conclusion = "finite parity with inferential support"
    elif finite_pass:
        conclusion = "finite parity but inferentially unresolved"
    else:
        conclusion = "finite noninferiority failed"
    return {
        "n_paired_episode_clusters": 80,
        "absolute_only": absolute_only,
        "treatment_only": treatment_only,
        "both_success": both_success,
        "both_failure": both_failure,
        "treatment_minus_absolute": difference,
        "margin": 0.05,
        "finite_benchmark_noninferior": finite_pass,
        "absolute_only_rate_one_sided_95_upper": upper,
        "inferential_support": inferential_pass,
        "conclusion": conclusion,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol, require_authorized=True)
    validate_protocol(protocol)
    protocol_hash = sha256_file(args.protocol)
    contract = Contract()
    episodes = _targets_by_episode(contract)
    quartiles = _distance_quartiles(episodes, contract)
    target_keys = {
        target: {(episode, target) for episode in range(80)} for target in range(4)
    }
    roots = {
        "absolute_matched_control": args.absolute,
        "normalized_relative": args.normalized,
        "raw_relative": args.raw,
    }
    arms = {
        arm: _load_arm(root, arm, protocol, protocol_hash, contract, episodes)
        for arm, root in roots.items()
    }
    contrasts = {
        f"{arm}_minus_absolute_matched_control": _contrast(
            arms["absolute_matched_control"]["multi"], arms[arm]["multi"]
        )
        for arm in ("normalized_relative", "raw_relative")
    }
    finite_global = all(value["finite_benchmark_noninferior"] for value in contrasts.values())
    inferential_global = all(value["inferential_support"] for value in contrasts.values())
    result = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_paired_report",
        "status": "complete",
        "protocol_sha256": protocol_hash,
        "estimand": protocol["estimand"],
        "arms": {
            arm: {
                "episode_successes": sum(
                    unit["summary"]["success"] for unit in value["multi"].values()
                ),
                "episode_success_rate": sum(
                    unit["summary"]["success"] for unit in value["multi"].values()
                )
                / 80,
                "single_step_first_attempt_success_rate": sum(
                    unit["summary"]["success"] for unit in value["sentinel"].values()
                )
                / 320,
                "multi_step_curves": _curves(value["multi"]),
                "multi_step_curves_by_target_index": {
                    str(target + 1): _curves(value["multi"], keys)
                    for target, keys in target_keys.items()
                },
                "multi_step_curves_by_frozen_distance_quartile": {
                    str(quartile): _curves(value["multi"], keys)
                    for quartile, keys in quartiles.items()
                },
                "single_step_by_target_index": {
                    str(target + 1): _sentinel_rate(value["sentinel"], keys)
                    for target, keys in target_keys.items()
                },
                "single_step_by_frozen_distance_quartile": {
                    str(quartile): _sentinel_rate(value["sentinel"], keys)
                    for quartile, keys in quartiles.items()
                },
                "manifest_sha256": sha256_file(roots[arm] / "arm_manifest.json"),
            }
            for arm, value in arms.items()
        },
        "contrasts": contrasts,
        "global_finite_benchmark_parity": finite_global,
        "global_inferential_support": inferential_global,
        "global_conclusion": (
            "finite parity with inferential support"
            if finite_global and inferential_global
            else "finite parity but inferentially unresolved"
            if finite_global
            else "finite noninferiority failed"
        ),
        "single_step_and_multi_step_pooled": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "paired_report.json"
    if marker.exists():
        raise GateError("refusing to overwrite stage-2 paired report")
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(aggregate(parse_args()), indent=2, sort_keys=True))
