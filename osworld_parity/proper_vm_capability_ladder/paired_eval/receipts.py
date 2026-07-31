"""Pure validators for the approved curriculum's serialized runtime receipts.

The scored evaluator and the offline aggregator both use these functions.  They
operate on serialized dictionaries so aggregation never needs a live VM or a
mutable curriculum object.
"""

from __future__ import annotations

import time
from typing import Any

from ..rung1.executor import parse_compact_raw
from .contracts import ACTION_INTERFACES, sha256_json


BINDING_KEYS = {
    "schema_version",
    "task_id",
    "fixture_sha256",
    "binding_revision",
    "binding_sha256",
    "parent_binding_sha256",
    "refresh_evidence_sha256",
    "evidence_fresh_until_monotonic_ns",
    "reset_cycles",
    "resolved_initial_cursor",
    "initial_geometry",
    "initial_geometry_sha256",
    "refresh_transitions",
    "binding_receipt_sha256",
}
RESET_KEYS = {
    "session_id",
    "reset_id",
    "generation_id",
    "sequence",
    "provider_reset_sequence",
    "provider_session_id",
    "prior_provider_generation_id",
    "provider_reset_receipt_sha256",
    "provider_state_before_sha256",
    "provider_state_after_sha256",
    "provider_path_sha256",
    "prior_provider_transition_index",
    "new_provider_transition_index",
    "provider_transition_labels",
    "provider_transition_records_sha256",
    "guest_sentinel_path_sha256",
    "guest_sentinel_nonce_sha256",
    "reset_started_monotonic_ns",
    "provider_reset_completed_monotonic_ns",
    "probe_completed_monotonic_ns",
    "captured_wall_time_ns",
    "vm_snapshot_id",
    "setup_commit",
    "reset_provider",
    "transport_endpoint_sha256",
    "probe_sha256",
    "evidence_sha256",
}
REFRESH_KEYS = {
    "session_id",
    "refresh_id",
    "sequence",
    "task_id",
    "fixture_sha256",
    "reset_generation_id",
    "completed_step",
    "prior_binding_sha256",
    "executed_segment_sha256",
    "action_started_monotonic_ns",
    "action_completed_monotonic_ns",
    "probe_started_monotonic_ns",
    "probe_completed_monotonic_ns",
    "captured_wall_time_ns",
    "before_scroll_y",
    "after_scroll_y",
    "observed_scroll_delta",
    "required_minimum_delta",
    "expected_scroll_direction",
    "probe_sha256",
    "issuer_mac",
    "evidence_sha256",
}
TRANSITION_KEYS = {
    "pre_binding_revision",
    "post_binding_revision",
    "pre_binding_sha256",
    "post_binding_sha256",
    "refresh_evidence",
    "transition_receipt_sha256",
}
SEGMENT_KEYS = {
    "task_id",
    "fixture_sha256",
    "action_schema",
    "semantic_step_index",
    "actions",
    "resolved_primitive_actions",
    "resolved_primitive_events",
    "resolved_budget_sha256",
    "binding_revision",
    "binding_sha256",
    "expected_cursor_before",
    "expected_cursor_after",
}
EXECUTED_KEYS = {
    "schema_version",
    "task_id",
    "fixture_sha256",
    "action_schema",
    "semantic_step_index",
    "resolved_primitive_actions",
    "resolved_primitive_events",
    "resolved_budget_sha256",
    "binding_revision",
    "binding_sha256",
    "dispatch_receipt_sha256",
    "execution_started_monotonic_ns",
    "execution_completed_monotonic_ns",
    "executed_receipt_sha256",
}


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} field set mismatch")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _cursor(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} must be [int, int]")
    return [int(value[0]), int(value[1])]


def _sealed(value: dict[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    seal = payload.pop(field, None)
    if not is_sha256(seal) or sha256_json(payload) != seal:
        raise ValueError(f"{label} seal mismatch")


def validate_binding_receipt(
    receipt: Any,
    *,
    task_id: str,
    fixture_sha256: str,
    snapshot_id: str,
    setup_commit: str,
    require_fresh: bool,
    now_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    value = _exact_keys(receipt, BINDING_KEYS, "runtime binding receipt")
    _sealed(value, "binding_receipt_sha256", "runtime binding receipt")
    if value["schema_version"] != 1 or (
        value["task_id"], value["fixture_sha256"]
    ) != (task_id, fixture_sha256):
        raise ValueError("runtime binding identity/schema mismatch")
    for field in ("fixture_sha256", "binding_sha256", "initial_geometry_sha256"):
        if not is_sha256(value[field]):
            raise ValueError(f"runtime binding {field} is invalid")
    revision = _integer(value["binding_revision"], "binding revision", minimum=1)
    fresh_until = _integer(
        value["evidence_fresh_until_monotonic_ns"],
        "binding freshness deadline",
        minimum=1,
    )
    if require_fresh and (now_monotonic_ns or time.monotonic_ns()) > fresh_until:
        raise ValueError("runtime binding evidence is stale")
    cursor = _cursor(value["resolved_initial_cursor"], "resolved initial cursor")
    geometry = value["initial_geometry"]
    if not isinstance(geometry, dict) or not geometry:
        raise ValueError("runtime binding initial geometry is missing")
    for name, point in geometry.items():
        if not isinstance(name, str) or not name:
            raise ValueError("runtime binding geometry target is invalid")
        _cursor(point, f"runtime binding geometry {name}")
    if sha256_json(geometry) != value["initial_geometry_sha256"]:
        raise ValueError("runtime binding geometry seal mismatch")

    cycles = value["reset_cycles"]
    if not isinstance(cycles, list) or len(cycles) < 2:
        raise ValueError("runtime binding needs at least two reset cycles")
    reset_ids: set[str] = set()
    generations: set[str] = set()
    evidence_hashes: set[str] = set()
    previous: dict[str, Any] | None = None
    for index, raw in enumerate(cycles):
        row = _exact_keys(raw, RESET_KEYS, "reset-cycle receipt")
        for field in (
            "generation_id",
            "prior_provider_generation_id",
            "provider_reset_receipt_sha256",
            "provider_state_before_sha256",
            "provider_state_after_sha256",
            "provider_path_sha256",
            "provider_transition_records_sha256",
            "guest_sentinel_path_sha256",
            "guest_sentinel_nonce_sha256",
            "transport_endpoint_sha256",
            "probe_sha256",
            "evidence_sha256",
        ):
            if not is_sha256(row[field]):
                raise ValueError(f"reset-cycle {field} is invalid")
        for field in (
            "sequence",
            "provider_reset_sequence",
            "new_provider_transition_index",
            "reset_started_monotonic_ns",
            "provider_reset_completed_monotonic_ns",
            "probe_completed_monotonic_ns",
            "captured_wall_time_ns",
        ):
            _integer(row[field], f"reset-cycle {field}", minimum=1)
        _integer(
            row["prior_provider_transition_index"],
            "reset-cycle prior_provider_transition_index",
            minimum=0,
        )
        labels = row["provider_transition_labels"]
        if not isinstance(labels, (list, tuple)) or tuple(labels[:2]) != (
            f"loadvm[{snapshot_id}]",
            "loadvm_guest_ready",
        ):
            raise ValueError("reset-cycle native provider transition is missing")
        if (
            row["vm_snapshot_id"] != snapshot_id
            or row["setup_commit"] != setup_commit
            or not isinstance(row["reset_provider"], str)
            or not row["reset_provider"]
            or row["generation_id"] != row["provider_state_after_sha256"]
            or row["prior_provider_generation_id"]
            != row["provider_state_before_sha256"]
            or row["provider_state_before_sha256"]
            == row["provider_state_after_sha256"]
            or row["new_provider_transition_index"]
            <= row["prior_provider_transition_index"]
            or not (
                row["reset_started_monotonic_ns"]
                < row["provider_reset_completed_monotonic_ns"]
                < row["probe_completed_monotonic_ns"]
                <= fresh_until
            )
        ):
            raise ValueError("reset-cycle provider generation contract mismatch")
        if (
            not isinstance(row["session_id"], str)
            or not row["session_id"]
            or not isinstance(row["provider_session_id"], str)
            or not row["provider_session_id"]
            or not isinstance(row["reset_id"], str)
            or not row["reset_id"]
        ):
            raise ValueError("reset-cycle identity is invalid")
        if (
            row["reset_id"] in reset_ids
            or row["generation_id"] in generations
            or row["evidence_sha256"] in evidence_hashes
        ):
            raise ValueError("duplicate reset-cycle evidence")
        reset_ids.add(row["reset_id"])
        generations.add(row["generation_id"])
        evidence_hashes.add(row["evidence_sha256"])
        if previous is not None and (
            row["sequence"] != previous["sequence"] + 1
            or row["provider_reset_sequence"]
            != previous["provider_reset_sequence"] + 1
            or row["prior_provider_generation_id"] != previous["generation_id"]
            or row["provider_session_id"] != previous["provider_session_id"]
            or row["provider_path_sha256"] != previous["provider_path_sha256"]
            or row["prior_provider_transition_index"]
            != previous["new_provider_transition_index"]
            or row["reset_started_monotonic_ns"]
            <= previous["probe_completed_monotonic_ns"]
        ):
            raise ValueError("provider reset generation chain is discontinuous")
        previous = row

    transitions = value["refresh_transitions"]
    if revision == 1:
        if (
            value["parent_binding_sha256"] is not None
            or value["refresh_evidence_sha256"] is not None
            or transitions != []
        ):
            raise ValueError("initial binding revision lineage mismatch")
    elif revision == 2:
        if (
            not is_sha256(value["parent_binding_sha256"])
            or not is_sha256(value["refresh_evidence_sha256"])
            or not isinstance(transitions, list)
            or len(transitions) != 1
        ):
            raise ValueError("refreshed binding revision lineage mismatch")
        transition = _exact_keys(
            transitions[0], TRANSITION_KEYS, "binding refresh transition"
        )
        _sealed(
            transition,
            "transition_receipt_sha256",
            "binding refresh transition",
        )
        refresh = _exact_keys(
            transition["refresh_evidence"], REFRESH_KEYS, "refresh evidence"
        )
        _sealed(refresh, "evidence_sha256", "refresh evidence")
        for field in (
            "prior_binding_sha256",
            "executed_segment_sha256",
            "probe_sha256",
            "issuer_mac",
            "evidence_sha256",
        ):
            if not is_sha256(refresh[field]):
                raise ValueError(f"refresh evidence {field} is invalid")
        if (
            transition["pre_binding_revision"] != 1
            or transition["post_binding_revision"] != 2
            or transition["pre_binding_sha256"] != value["parent_binding_sha256"]
            or transition["post_binding_sha256"] != value["binding_sha256"]
            or refresh["task_id"] != task_id
            or refresh["fixture_sha256"] != fixture_sha256
            or refresh["completed_step"] != 2
            or refresh["prior_binding_sha256"] != value["parent_binding_sha256"]
            or refresh["evidence_sha256"] != value["refresh_evidence_sha256"]
            or refresh["reset_generation_id"] != cycles[-1]["generation_id"]
        ):
            raise ValueError("binding refresh identity/lineage mismatch")
        for field in (
            "sequence",
            "action_started_monotonic_ns",
            "action_completed_monotonic_ns",
            "probe_started_monotonic_ns",
            "probe_completed_monotonic_ns",
            "captured_wall_time_ns",
            "required_minimum_delta",
        ):
            _integer(refresh[field], f"refresh evidence {field}", minimum=1)
        delta = refresh["after_scroll_y"] - refresh["before_scroll_y"]
        direction = refresh["expected_scroll_direction"]
        if (
            refresh["observed_scroll_delta"] != delta
            or abs(delta) < refresh["required_minimum_delta"]
            or direction not in {"up", "down"}
            or (direction == "down" and delta <= 0)
            or (direction == "up" and delta >= 0)
            or not (
                cycles[-1]["probe_completed_monotonic_ns"]
                < refresh["action_started_monotonic_ns"]
                <= refresh["action_completed_monotonic_ns"]
                < refresh["probe_started_monotonic_ns"]
                <= refresh["probe_completed_monotonic_ns"]
                <= fresh_until
            )
        ):
            raise ValueError("binding refresh scroll/causality proof mismatch")
    else:
        raise ValueError("unsupported binding revision")
    return value


def validate_binding_successor(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    completed_step_2_receipt_sha256: str | None,
) -> None:
    stable = (
        "task_id",
        "fixture_sha256",
        "evidence_fresh_until_monotonic_ns",
        "reset_cycles",
        "resolved_initial_cursor",
        "initial_geometry",
        "initial_geometry_sha256",
    )
    if any(previous[field] != current[field] for field in stable):
        raise ValueError("runtime binding reset/geometry identity changed")
    if current["binding_sha256"] == previous["binding_sha256"]:
        if current != previous:
            raise ValueError("same runtime binding hash has different receipt content")
        return
    if (
        previous["binding_revision"] != 1
        or current["binding_revision"] != 2
        or current["parent_binding_sha256"] != previous["binding_sha256"]
        or completed_step_2_receipt_sha256 is None
        or current["refresh_transitions"][0]["refresh_evidence"][
            "executed_segment_sha256"
        ]
        != completed_step_2_receipt_sha256
    ):
        raise ValueError("runtime binding A/A/B successor chain mismatch")


def validate_prefix_replay(
    *,
    replay: Any,
    prefix_length: int,
    start_binding: dict[str, Any],
    task_id: str,
    fixture_sha256: str,
    snapshot_id: str,
    setup_commit: str,
    app: str,
    action_schema: str,
    require_fresh: bool,
) -> tuple[dict[str, Any], list[int], str | None]:
    if not isinstance(replay, (list, tuple)) or len(replay) != prefix_length:
        raise ValueError("gold-prefix replay coverage mismatch")
    current_binding: dict[str, Any] | None = None
    current_cursor: list[int] | None = None
    completed_step_2_receipt: str | None = None
    ordinary_keys = {
        "semantic_step",
        "binding_receipt",
        "binding_sha256",
        "compiled_segment",
        "executed_receipt",
        "actions",
    }
    refreshed_keys = ordinary_keys | {
        "post_scroll_refresh",
        "refreshed_binding_sha256",
        "refreshed_binding_receipt",
    }
    for semantic_step, row in enumerate(replay, start=1):
        expected_keys = (
            refreshed_keys if app == "chrome" and semantic_step == 2 else ordinary_keys
        )
        if not isinstance(row, dict) or set(row) != expected_keys or row.get(
            "semantic_step"
        ) != semantic_step:
            raise ValueError("gold-prefix replay journal schema/order mismatch")
        binding = validate_binding_receipt(
            row["binding_receipt"],
            task_id=task_id,
            fixture_sha256=fixture_sha256,
            snapshot_id=snapshot_id,
            setup_commit=setup_commit,
            require_fresh=require_fresh,
        )
        if row["binding_sha256"] != binding["binding_sha256"]:
            raise ValueError("gold-prefix binding hash mismatch")
        if current_binding is not None:
            validate_binding_successor(
                current_binding,
                binding,
                completed_step_2_receipt_sha256=completed_step_2_receipt,
            )
        else:
            current_cursor = list(binding["resolved_initial_cursor"])
        action_rows = row["actions"]
        compiled = row["compiled_segment"]
        compiled_actions = compiled.get("actions") if isinstance(compiled, dict) else None
        if (
            not isinstance(action_rows, list)
            or not isinstance(compiled_actions, (list, tuple))
            or len(action_rows) != len(compiled_actions)
        ):
            raise ValueError("gold-prefix action journal coverage mismatch")
        dispatches = []
        for action_index, (action_row, compiled_action) in enumerate(
            zip(action_rows, compiled_actions, strict=True)
        ):
            if (
                not isinstance(action_row, dict)
                or set(action_row)
                != {"action_index", "screenshot", "action", "dispatch"}
                or action_row["action_index"] != action_index
                or action_row["action"] != compiled_action
                or not isinstance(action_row["dispatch"], list)
            ):
                raise ValueError("gold-prefix action journal mismatch")
            dispatches.append(action_row["dispatch"])
        executed = validate_executed_segment(
            compiled_segment=compiled,
            dispatches=dispatches,
            executed_receipt=row["executed_receipt"],
            binding_receipt=binding,
            task_id=task_id,
            fixture_sha256=fixture_sha256,
            action_schema=action_schema,
            expected_semantic_step=semantic_step,
            expected_cursor_before=current_cursor,
        )
        current_cursor = list(compiled["expected_cursor_after"])
        current_binding = binding
        if semantic_step == 2:
            completed_step_2_receipt = executed["executed_receipt_sha256"]
        if app == "chrome" and semantic_step == 2:
            refreshed = validate_binding_receipt(
                row["refreshed_binding_receipt"],
                task_id=task_id,
                fixture_sha256=fixture_sha256,
                snapshot_id=snapshot_id,
                setup_commit=setup_commit,
                require_fresh=require_fresh,
            )
            validate_binding_successor(
                binding,
                refreshed,
                completed_step_2_receipt_sha256=completed_step_2_receipt,
            )
            transition = refreshed["refresh_transitions"][0]["refresh_evidence"]
            if (
                row["post_scroll_refresh"] != transition
                or row["refreshed_binding_sha256"] != refreshed["binding_sha256"]
            ):
                raise ValueError("gold-prefix Chrome refresh journal mismatch")
            current_binding = refreshed
    if current_binding is None:
        current_binding = start_binding
        current_cursor = list(start_binding["resolved_initial_cursor"])
    elif current_binding != start_binding:
        raise ValueError("session start binding is not the prefix terminal binding")
    assert current_cursor is not None
    return current_binding, current_cursor, completed_step_2_receipt


def _validate_dispatch_seal(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("executor dispatch result must be an object")
    _sealed(result, "dispatch_result_sha256", "executor dispatch result")
    atomic = result.get("atomic_state")
    expected_atomic_sha = sha256_json(atomic) if isinstance(atomic, dict) else None
    if result.get("atomic_state_sha256") != expected_atomic_sha:
        raise ValueError("executor atomic result seal mismatch")
    if result.get("parse_status") != "ok" or result.get(
        "executor_dispatch_status"
    ) != "ok":
        raise ValueError("executor dispatch did not complete successfully")
    return result


def _operations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("executor operations are missing")
    rows = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"kind", "args"} or not isinstance(
            item["args"], (list, tuple)
        ):
            raise ValueError("executor operation schema mismatch")
        rows.append({"kind": item["kind"], "args": list(item["args"])})
    return rows


def _native_expected(operation: dict[str, Any]) -> tuple[str, list[dict[str, Any]], Any]:
    kind = operation.get("action")
    coordinate = operation.get("coordinate")
    rows: list[dict[str, Any]] = []
    if coordinate is not None and kind in {"click", "mouse_down", "mouse_move", "mouse_up"}:
        rows.append({"kind": "move_to", "args": list(coordinate)})
    atomic = None
    if kind == "click":
        atomic = [
            {"kind": "mouse_down", "args": ["left"]},
            {"kind": "mouse_up", "args": ["left"]},
        ]
        rows.extend(atomic)
        action_class = "click"
    elif kind == "mouse_down":
        rows.append({"kind": "mouse_down", "args": [operation.get("button", "left")]})
        action_class = "button_hold"
    elif kind == "mouse_up":
        rows.append({"kind": "mouse_up", "args": [operation.get("button", "left")]})
        action_class = "button_release"
    elif kind == "mouse_move":
        action_class = "mouse_move"
    elif kind == "scroll":
        rows.append({"kind": "scroll", "args": [operation["clicks"]]})
        action_class = "scroll"
    elif kind == "key_chord":
        rows.append({"kind": "key_chord", "args": list(operation["keys"])})
        action_class = "key_chord"
    elif kind == "type":
        rows.append({"kind": "coalesced_type", "args": [operation["text"]]})
        action_class = "coalesced_type"
    else:
        raise ValueError("unsupported native compiled operation")
    return action_class, rows, atomic


def _compact_expected(
    action: str, cursor_before: tuple[int, int]
) -> tuple[list[dict[str, Any]], tuple[int, int], str]:
    parsed = parse_compact_raw(action)
    rows: list[dict[str, Any]] = []
    classes: set[str] = set()
    cursor_after = cursor_before
    if parsed.dx or parsed.dy:
        cursor_after = (cursor_before[0] + parsed.dx, cursor_before[1] + parsed.dy)
        rows.append({"kind": "move_to", "args": list(cursor_after)})
        classes.add("mouse_move")
    if parsed.scroll:
        rows.append({"kind": "scroll", "args": [parsed.scroll]})
        classes.add("scroll")
    for element in parsed.elements:
        if element.kind == "type":
            rows.append({"kind": "coalesced_type", "args": [element.value]})
            classes.add("coalesced_type")
        elif element.value in {"LMB", "RMB", "MMB"}:
            button = {"LMB": "left", "RMB": "right", "MMB": "middle"}[
                element.value
            ]
            rows.append(
                {
                    "kind": "mouse_down" if element.pressed else "mouse_up",
                    "args": [button],
                }
            )
            classes.add("button_hold" if element.pressed else "button_release")
        else:
            rows.append(
                {
                    "kind": "key_down" if element.pressed else "key_up",
                    "args": [element.value],
                }
            )
            classes.add("key_chord")
    return rows, cursor_after, "+".join(sorted(classes)) if classes else "no_op"


def validate_executed_segment(
    *,
    compiled_segment: Any,
    dispatches: Any,
    executed_receipt: Any,
    binding_receipt: dict[str, Any],
    task_id: str,
    fixture_sha256: str,
    action_schema: str,
    expected_semantic_step: int | None,
    expected_cursor_before: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    segment = _exact_keys(compiled_segment, SEGMENT_KEYS, "compiled segment")
    receipt = _exact_keys(executed_receipt, EXECUTED_KEYS, "executed segment receipt")
    if action_schema not in ACTION_INTERFACES.values():
        raise ValueError("unsupported action schema")
    identity = (task_id, fixture_sha256, action_schema)
    if (
        (segment["task_id"], segment["fixture_sha256"], segment["action_schema"])
        != identity
        or (receipt["task_id"], receipt["fixture_sha256"], receipt["action_schema"])
        != identity
        or (expected_semantic_step is not None and segment["semantic_step_index"] != expected_semantic_step)
        or receipt["semantic_step_index"] != segment["semantic_step_index"]
    ):
        raise ValueError("executed segment identity/step mismatch")
    actions = segment["actions"]
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("compiled segment actions are missing")
    resolved_actions = _integer(
        segment["resolved_primitive_actions"], "resolved primitive actions", minimum=1
    )
    resolved_events = _integer(
        segment["resolved_primitive_events"], "resolved primitive events", minimum=1
    )
    if len(actions) != resolved_actions:
        raise ValueError("compiled segment action count mismatch")
    before = _cursor(segment["expected_cursor_before"], "segment cursor before")
    after = _cursor(segment["expected_cursor_after"], "segment cursor after")
    if expected_cursor_before is not None and before != list(expected_cursor_before):
        raise ValueError("compiled segment cursor-before chain mismatch")
    if (
        segment["binding_revision"] != binding_receipt["binding_revision"]
        or segment["binding_sha256"] != binding_receipt["binding_sha256"]
        or not is_sha256(segment["resolved_budget_sha256"])
    ):
        raise ValueError("compiled segment binding mismatch")
    budget_payload = {
        "schema_version": 1,
        "task_id": task_id,
        "fixture_sha256": fixture_sha256,
        "action_schema": action_schema,
        "semantic_step_index": segment["semantic_step_index"],
        "resolved_primitive_actions": resolved_actions,
        "resolved_primitive_events": resolved_events,
        "binding_revision": segment["binding_revision"],
        "binding_sha256": segment["binding_sha256"],
        "expected_cursor_before": before,
        "expected_cursor_after": after,
        "actions": actions,
    }
    if sha256_json(budget_payload) != segment["resolved_budget_sha256"]:
        raise ValueError("compiled segment resolved-budget seal mismatch")
    if not isinstance(dispatches, (list, tuple)) or len(dispatches) != len(actions):
        raise ValueError("dispatch evidence does not cover every compiled action")
    observed = before
    normalized_dispatches: list[list[dict[str, Any]]] = []
    for action, raw_results in zip(actions, dispatches, strict=True):
        if not isinstance(raw_results, (list, tuple)) or not raw_results:
            raise ValueError("compiled action has no executor dispatch receipt")
        results = list(raw_results)
        normalized_dispatches.append(results)
        if action_schema == "native_absolute_sequence_v1":
            if not isinstance(action, dict) or not isinstance(action.get("operations"), list) or (
                len(results) != len(action["operations"])
            ):
                raise ValueError("native dispatch cardinality mismatch")
            for operation_index, (operation, raw_result) in enumerate(
                zip(action["operations"], results, strict=True)
            ):
                result = _validate_dispatch_seal(raw_result)
                cursor_before = _cursor(result.get("cursor_before"), "native dispatch cursor before")
                cursor_after = _cursor(result.get("cursor_after"), "native dispatch cursor after")
                coordinate = operation.get("coordinate")
                expected_after = (
                    [int(round(coordinate[0])), int(round(coordinate[1]))]
                    if coordinate is not None
                    and operation.get("action") in {"click", "mouse_down", "mouse_move", "mouse_up"}
                    else cursor_before
                )
                expected_class, expected_operations, expected_atomic = _native_expected(operation)
                if (
                    result.get("compiled_operation_index") != operation_index
                    or result.get("compiled_payload_sha256") != sha256_json(operation)
                    or result.get("adapter") != "native_absolute_control"
                    or cursor_before != observed
                    or cursor_after != expected_after
                    or result.get("action_class") != expected_class
                    or _operations(result.get("operations")) != expected_operations
                ):
                    raise ValueError("native compiled operation/dispatch mismatch")
                atomic = result.get("atomic_state")
                if expected_atomic is None:
                    if atomic is not None:
                        raise ValueError("unexpected native atomic result")
                elif (
                    not isinstance(atomic, dict)
                    or atomic.get("ok") is not True
                    or _operations(atomic.get("operations")) != expected_atomic
                ):
                    raise ValueError("native atomic result mismatch")
                observed = cursor_after
        else:
            if not isinstance(action, str) or len(results) != 1:
                raise ValueError("compact dispatch cardinality mismatch")
            result = _validate_dispatch_seal(results[0])
            cursor_before = _cursor(result.get("cursor_before"), "compact dispatch cursor before")
            cursor_after = _cursor(result.get("cursor_after"), "compact dispatch cursor after")
            expected_operations, expected_after, expected_class = _compact_expected(
                action, tuple(cursor_before)
            )
            atomic = result.get("atomic_state")
            if (
                result.get("compiled_operation_index") != 0
                or result.get("compiled_payload_sha256") != sha256_json(action)
                or result.get("adapter") != "compact_raw_phaseb"
                or cursor_before != observed
                or cursor_after != list(expected_after)
                or result.get("action_class") != expected_class
                or _operations(result.get("operations")) != expected_operations
                or not isinstance(atomic, dict)
                or atomic.get("ok") is not True
                or _operations(atomic.get("operations")) != expected_operations
            ):
                raise ValueError("compact compiled action/dispatch mismatch")
            observed = cursor_after
    if observed != after:
        raise ValueError("executed segment final cursor mismatch")

    dispatch_payload = {
        "schema_version": 1,
        "task_id": task_id,
        "semantic_step_index": segment["semantic_step_index"],
        "compiled_actions": actions,
        "dispatches": dispatches,
    }
    if receipt["dispatch_receipt_sha256"] != sha256_json(dispatch_payload):
        raise ValueError("executed segment dispatch seal mismatch")
    for field in (
        "resolved_primitive_actions",
        "resolved_primitive_events",
        "resolved_budget_sha256",
        "binding_revision",
        "binding_sha256",
    ):
        if receipt[field] != segment[field]:
            raise ValueError("executed receipt does not match compiled segment")
    started = _integer(
        receipt["execution_started_monotonic_ns"], "execution start", minimum=1
    )
    completed = _integer(
        receipt["execution_completed_monotonic_ns"], "execution completion", minimum=1
    )
    if not (
        binding_receipt["reset_cycles"][-1]["probe_completed_monotonic_ns"]
        < started
        < completed
        <= binding_receipt["evidence_fresh_until_monotonic_ns"]
    ):
        raise ValueError("executed segment is outside binding freshness/causality window")
    _sealed(receipt, "executed_receipt_sha256", "executed segment receipt")
    return receipt


def _validate_aggregate_receipt(
    item: Any, *, task_id: str, fixture_sha256: str, action_schema: str
) -> dict[str, Any]:
    value = _exact_keys(item, EXECUTED_KEYS, "aggregate executed-segment receipt")
    if (
        value["schema_version"] != 1
        or action_schema not in ACTION_INTERFACES.values()
        or (value["task_id"], value["fixture_sha256"], value["action_schema"])
        != (task_id, fixture_sha256, action_schema)
    ):
        raise ValueError("aggregate executed-segment identity mismatch")
    _integer(value["semantic_step_index"], "aggregate semantic step", minimum=1)
    _integer(
        value["resolved_primitive_actions"],
        "aggregate primitive actions",
        minimum=1,
    )
    _integer(
        value["resolved_primitive_events"],
        "aggregate primitive events",
        minimum=1,
    )
    _integer(value["binding_revision"], "aggregate binding revision", minimum=1)
    for field in (
        "fixture_sha256",
        "resolved_budget_sha256",
        "binding_sha256",
        "dispatch_receipt_sha256",
        "executed_receipt_sha256",
    ):
        if not is_sha256(value[field]):
            raise ValueError(f"aggregate executed-segment {field} is invalid")
    started = _integer(
        value["execution_started_monotonic_ns"], "aggregate execution start", minimum=1
    )
    completed = _integer(
        value["execution_completed_monotonic_ns"],
        "aggregate execution completion",
        minimum=1,
    )
    if completed <= started:
        raise ValueError("aggregate execution timestamps are not monotonic")
    _sealed(value, "executed_receipt_sha256", "aggregate executed-segment receipt")
    return value


def ordered_trace_aggregate(
    *, task_id: str, fixture_sha256: str, action_schema: str, receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    values = [
        _validate_aggregate_receipt(
            item,
            task_id=task_id,
            fixture_sha256=fixture_sha256,
            action_schema=action_schema,
        )
        for item in receipts
    ]
    payload = {
        "schema_version": 1,
        "schema_id": "paired_policy_turn_receipt_trace_v1",
        "task_id": task_id,
        "fixture_sha256": fixture_sha256,
        "action_schema": action_schema,
        "executed_segment_receipt_sha256": [
            item["executed_receipt_sha256"] for item in values
        ],
        "segment_semantic_step_indices": [item["semantic_step_index"] for item in values],
        "segment_budget_sha256": [item["resolved_budget_sha256"] for item in values],
        "segment_binding_sha256": [item["binding_sha256"] for item in values],
        "segment_binding_revisions": [item["binding_revision"] for item in values],
        "resolved_primitive_actions": sum(
            item["resolved_primitive_actions"] for item in values
        ),
        "resolved_primitive_events": sum(
            item["resolved_primitive_events"] for item in values
        ),
    }
    payload["binding_chain_sha256"] = sha256_json(payload["segment_binding_sha256"])
    result = dict(payload)
    result["trace_receipt_sha256"] = sha256_json(payload)
    return result


def executed_aggregate(
    *,
    task_id: str,
    fixture_sha256: str,
    action_schema: str,
    app: str,
    semantic_step_count: int,
    primitive_action_cap: int,
    primitive_event_cap: int,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replicate c603 ``aggregate_executed_segments`` over serialized receipts."""

    values = [
        _validate_aggregate_receipt(
            item,
            task_id=task_id,
            fixture_sha256=fixture_sha256,
            action_schema=action_schema,
        )
        for item in receipts
    ]
    if [item["semantic_step_index"] for item in values] != list(
        range(1, semantic_step_count + 1)
    ):
        raise ValueError("executed segment coverage/order mismatch")
    revisions = [item["binding_revision"] for item in values]
    binding_hashes = [item["binding_sha256"] for item in values]
    if app == "chrome":
        if (
            revisions != [1, 1, 2]
            or len(binding_hashes) != 3
            or binding_hashes[0] != binding_hashes[1]
            or binding_hashes[2] == binding_hashes[1]
        ):
            raise ValueError("Chrome binding transition mismatch")
    elif len(set(revisions)) != 1 or len(set(binding_hashes)) != 1:
        raise ValueError("unexpected mid-trajectory binding change")
    resolved_actions = sum(item["resolved_primitive_actions"] for item in values)
    resolved_events = sum(item["resolved_primitive_events"] for item in values)
    if resolved_actions > primitive_action_cap:
        raise ValueError("aggregate primitive actions exceed cap")
    if resolved_events > primitive_event_cap:
        raise ValueError("aggregate primitive events exceed cap")
    payload = {
        "schema_version": 1,
        "task_id": task_id,
        "fixture_sha256": fixture_sha256,
        "action_schema": action_schema,
        "executed_segment_receipt_sha256": [
            item["executed_receipt_sha256"] for item in values
        ],
        "segment_budget_sha256": [item["resolved_budget_sha256"] for item in values],
        "segment_binding_sha256": binding_hashes,
        "segment_binding_revisions": revisions,
        "resolved_primitive_actions": resolved_actions,
        "resolved_primitive_events": resolved_events,
    }
    payload["binding_sha256"] = sha256_json(payload["segment_binding_sha256"])
    result = dict(payload)
    result["resolved_budget_sha256"] = sha256_json(payload)
    return result
