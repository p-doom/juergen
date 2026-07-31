"""Runtime bridge from semantic tasks to the existing symbolic action layer.

Task records remain independent of an action encoding.  This module is the only
place that lowers a semantic gold/near-miss program to rung-2 ``ActionTurn``
objects, which the existing native and compact compilers can then encode.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from ..actions import (
    ACTION_SCHEMAS,
    ActionTurn,
    ScriptedTrajectory,
    SymbolicOperation,
    compile_compact,
    compile_native,
)
from ...rung1.executor import parse_compact_raw
from ..fixtures import canonical_json
from .schema import SemanticTask


@dataclass(frozen=True)
class CompiledSegment:
    """One live-bound semantic segment and its resolved budget receipt."""

    task_id: str
    fixture_sha256: str
    action_schema: str
    semantic_step_index: int
    actions: tuple[dict[str, Any] | str, ...]
    resolved_primitive_actions: int
    resolved_primitive_events: int
    resolved_budget_sha256: str
    binding_revision: int
    binding_sha256: str
    expected_cursor_before: tuple[int, int]
    expected_cursor_after: tuple[int, int]


@dataclass(frozen=True)
class CompiledProgram:
    """Aggregate receipt obtained from already-executed segment receipts."""

    task_id: str
    fixture_sha256: str
    action_schema: str
    segments: tuple["ExecutedSegmentReceipt", ...]
    resolved_primitive_actions: int
    resolved_primitive_events: int
    resolved_budget_sha256: str
    binding_sha256: str


@dataclass(frozen=True)
class ExecutedSegmentReceipt:
    """Immutable evidence tying one compiled segment to executor dispatches."""

    schema_version: int
    task_id: str
    fixture_sha256: str
    action_schema: str
    semantic_step_index: int
    resolved_primitive_actions: int
    resolved_primitive_events: int
    resolved_budget_sha256: str
    binding_revision: int
    binding_sha256: str
    dispatch_receipt_sha256: str
    execution_started_monotonic_ns: int
    execution_completed_monotonic_ns: int
    executed_receipt_sha256: str


def _op(kind: str, **kwargs: object) -> SymbolicOperation:
    return SymbolicOperation(kind, **kwargs)


def build_program(task: SemanticTask, *, near_miss: bool = False) -> ScriptedTrajectory:
    if task.app == "writer":
        text = str(task.near_miss["text"] if near_miss else task.expected["text"])
        formatting = () if near_miss else (
            _op("key_chord", keys=("ControlLeft", "KeyB")),
        )
        turns = (
            ActionTurn(1, (_op("click", target="editor"), _op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=text))),
            ActionTurn(2, (_op("key_chord", keys=("ControlLeft", "KeyA")),) + formatting),
            ActionTurn(3, (_op("key_chord", keys=("ControlLeft", "KeyS")),)),
        )
    elif task.app == "calc":
        formula = str(task.near_miss["formula"] if near_miss else task.expected["formula"]).removeprefix("of:")
        turns = (
            ActionTurn(1, (_op("click", target="cell"), _op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=str(task.params["cell"])), _op("key_chord", keys=("Return",)))),
            ActionTurn(2, (_op("type", text=formula),)),
            ActionTurn(3, (_op("key_chord", keys=("Return",)),)),
            ActionTurn(4, (_op("key_chord", keys=("ControlLeft", "KeyS")),)),
        )
    elif task.app == "files":
        destination = "decoy" if near_miss else "destination"
        final_name = str(task.near_miss["final_name"] if near_miss else task.expected["final_name"])
        turns = (
            ActionTurn(1, (_op("click", target="source"),)),
            ActionTurn(2, (_op("mouse_down", target="source", button="left"),)),
            ActionTurn(2, (_op("mouse_move", target=destination),)),
            ActionTurn(2, (_op("mouse_up", target=destination, button="left"),)),
            ActionTurn(3, (_op("click", target=destination), _op("key_chord", keys=("Return",)))),
            ActionTurn(3, (_op("click", target="moved"),)),
            ActionTurn(3, (_op("key_chord", keys=("F2",)),)),
            ActionTurn(3, (_op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=final_name), _op("key_chord", keys=("Return",)))),
        )
    elif task.app == "chrome":
        nav = "decoy_nav" if near_miss else "nav"
        toggle = "decoy_toggle" if near_miss else "toggle"
        turns = (
            ActionTurn(1, (_op("click", target=nav),)),
            ActionTurn(2, (_op("scroll", target="scroll_surface", clicks=int(task.params["scroll_clicks"])),)),
            ActionTurn(3, (_op("click", target=toggle),)),
        )
    elif task.app == "vscode":
        text = str(task.near_miss["text"] if near_miss else task.expected["text"])
        turns = (
            ActionTurn(1, (_op("click", target="editor"),)),
            ActionTurn(2, (_op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=text))),
            ActionTurn(3, (_op("key_chord", keys=("ControlLeft", "KeyS")),)),
        )
    else:  # pragma: no cover - SemanticTask validation owns this boundary.
        raise ValueError(f"unsupported curriculum app: {task.app}")
    result = ScriptedTrajectory(task.task_id, near_miss, turns)
    validate_program(task, result)
    return result


def validate_program(task: SemanticTask, program: ScriptedTrajectory) -> None:
    if {turn.semantic_step for turn in program.turns} != set(
        range(1, task.semantic_step_count + 1)
    ):
        raise ValueError(f"{task.task_id}: program/semantic-step mismatch")
    held: set[str] = set()
    for turn in program.turns:
        for operation in turn.operations:
            if operation.kind == "mouse_down":
                button = operation.button or "left"
                if button in held:
                    raise ValueError(f"{task.task_id}: redundant held input")
                held.add(button)
            elif operation.kind == "mouse_up":
                button = operation.button or "left"
                if button not in held:
                    raise ValueError(f"{task.task_id}: dangling input release")
                held.remove(button)
    if held:
        raise ValueError(f"{task.task_id}: program leaves held inputs: {sorted(held)}")
    observed = program_budget_upper_bounds(program)
    contract = task.budget_contract
    for observed_field, cap_field in (
        ("primitive_actions", "primitive_action_caps"),
        ("primitive_events", "primitive_event_caps"),
    ):
        for interface, count in observed[observed_field].items():
            if count > contract[cap_field][interface]:
                raise ValueError(
                    f"{task.task_id}: program exceeds {cap_field}/{interface} cap"
                )


def program_budget_upper_bounds(program: ScriptedTrajectory) -> dict[str, Any]:
    """Count emitted dispatch lines and their lowered executor events.

    A target-bearing native operation always lowers an absolute move before its
    mouse event. Compact relative movement is emitted at most once per line and
    only when the symbolic target reference changes. Keyboard chords count the
    ordered down/up events that the executor emits. These are caps, independent
    of the live coordinate values bound later.
    """

    native_events = 0
    compact_events = 0
    compact_cursor_ref = "runtime.initial_cursor"
    for turn in program.turns:
        target_refs = [operation.target for operation in turn.operations if operation.target]
        compact_target = target_refs[0] if target_refs else None
        if compact_target is not None and compact_target != compact_cursor_ref:
            compact_events += 1
            compact_cursor_ref = compact_target
        for operation in turn.operations:
            if operation.target is not None and operation.kind in {
                "click", "mouse_down", "mouse_move", "mouse_up"
            }:
                native_events += 1
            if operation.kind == "click":
                native_events += 2
                compact_events += 2
            elif operation.kind in {"mouse_down", "mouse_up"}:
                native_events += 1
                compact_events += 1
            elif operation.kind == "scroll":
                native_events += 1
                compact_events += 1
            elif operation.kind == "key_chord":
                events = 2 * len(operation.keys)
                native_events += events
                compact_events += events
            elif operation.kind == "type":
                native_events += 1
                compact_events += 1
    action_count = len(program.turns)
    return {
        "semantic_steps": len({turn.semantic_step for turn in program.turns}),
        "primitive_actions": {
            "native_absolute_sequence_v1": action_count,
            "compact_raw_phaseb_v1": action_count,
        },
        "primitive_events": {
            "native_absolute_sequence_v1": native_events,
            "compact_raw_phaseb_v1": compact_events,
        },
    }


# Compatibility for scaffold consumers. These values are conservative upper
# bounds, not coordinate-independent exact budgets.
program_budgets = program_budget_upper_bounds


def _resolved_event_count(
    action_schema: str, actions: tuple[dict[str, Any] | str, ...]
) -> int:
    if action_schema == "compact_raw_phaseb_v1":
        result = 0
        for value in actions:
            if not isinstance(value, str):  # pragma: no cover - compiler invariant
                raise TypeError("compact action must be text")
            parsed = parse_compact_raw(value)
            result += int(bool(parsed.dx or parsed.dy))
            result += int(bool(parsed.scroll))
            result += len(parsed.elements)
        return result

    result = 0
    for value in actions:
        if not isinstance(value, dict):  # pragma: no cover - compiler invariant
            raise TypeError("native action must be an object")
        for operation in value["operations"]:
            kind = operation["action"]
            if kind == "click":
                result += int("coordinate" in operation) + 2
            elif kind in {"mouse_down", "mouse_up"}:
                result += int("coordinate" in operation) + 1
            elif kind == "mouse_move":
                result += 1
            elif kind == "scroll":
                result += 1
            elif kind == "key_chord":
                result += 2 * len(operation["keys"])
            elif kind == "type":
                result += 1
            else:  # pragma: no cover - SymbolicOperation validates this boundary.
                raise ValueError(f"unsupported native operation: {kind!r}")
    return result


def _cursor_before(
    task: SemanticTask,
    program: ScriptedTrajectory,
    semantic_step_index: int,
    geometry: dict[str, tuple[int, int]],
    initial_cursor: tuple[int, int],
) -> tuple[int, int]:
    cursor = initial_cursor
    for turn in program.turns:
        if turn.semantic_step >= semantic_step_index:
            break
        targets = [operation.target for operation in turn.operations if operation.target]
        if targets:
            try:
                cursor = geometry[targets[0]]
            except KeyError as exc:
                raise ValueError(f"unresolved cursor target: {targets[0]}") from exc
    return cursor


def _advance_native_cursor(
    cursor: tuple[int, int], actions: tuple[dict[str, Any] | str, ...]
) -> tuple[int, int]:
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("native action must be an object")
        for operation in action["operations"]:
            coordinate = operation.get("coordinate")
            if coordinate is not None and operation["action"] in {
                "click",
                "mouse_down",
                "mouse_move",
                "mouse_up",
            }:
                cursor = (int(round(coordinate[0])), int(round(coordinate[1])))
    return cursor


def _native_cursor_before(
    program: ScriptedTrajectory,
    semantic_step_index: int,
    geometry: dict[str, tuple[int, int]],
    initial_cursor: tuple[int, int],
) -> tuple[int, int]:
    cursor = initial_cursor
    for turn in program.turns:
        if turn.semantic_step >= semantic_step_index:
            break
        cursor = _advance_native_cursor(cursor, (compile_native(turn, geometry),))
    return cursor


def compile_semantic_step(
    task: SemanticTask,
    action_schema: str,
    *,
    binding: Any,
    semantic_step_index: int,
    near_miss: bool = False,
) -> CompiledSegment:
    """Compile one segment only from a validated repeated-reset binding."""

    from .runtime import ValidatedRuntimeBinding

    if action_schema not in ACTION_SCHEMAS:
        raise ValueError(f"unsupported action schema: {action_schema!r}")
    if not isinstance(binding, ValidatedRuntimeBinding):
        raise TypeError("compilation requires a ValidatedRuntimeBinding")
    if not 1 <= semantic_step_index <= task.semantic_step_count:
        raise ValueError(f"invalid semantic step: {semantic_step_index}")
    probe = binding.probe_for_step(task, semantic_step_index)
    program = build_program(task, near_miss=near_miss)
    turns = tuple(
        turn for turn in program.turns if turn.semantic_step == semantic_step_index
    )
    geometry = probe.geometry
    if action_schema == "native_absolute_sequence_v1":
        expected_cursor_before = (
            probe.initial_cursor
            if task.app == "chrome" and semantic_step_index > 2
            else _native_cursor_before(
                program,
                semantic_step_index,
                geometry,
                binding.resolved_initial_cursor,
            )
        )
        actions: tuple[dict[str, Any] | str, ...] = tuple(
            compile_native(turn, geometry) for turn in turns
        )
        expected_cursor_after = _advance_native_cursor(
            expected_cursor_before, actions
        )
    else:
        # A refreshed Chrome probe reports the actual cursor after scrolling;
        # other segments resolve their declared prior semantic milestone.
        cursor = (
            probe.initial_cursor
            if task.app == "chrome" and semantic_step_index > 2
            else _cursor_before(
                task,
                program,
                semantic_step_index,
                geometry,
                binding.resolved_initial_cursor,
            )
        )
        expected_cursor_before = cursor
        compact: list[str] = []
        for turn in turns:
            action, cursor = compile_compact(turn, geometry, cursor)
            compact.append(action)
        actions = tuple(compact)
        expected_cursor_after = cursor

    resolved_actions = len(actions)
    resolved_events = _resolved_event_count(action_schema, actions)
    contract = task.budget_contract
    if resolved_actions > contract["primitive_action_caps"][action_schema]:
        raise ValueError(f"{task.task_id}: resolved primitive actions exceed cap")
    if resolved_events > contract["primitive_event_caps"][action_schema]:
        raise ValueError(f"{task.task_id}: resolved primitive events exceed cap")
    receipt = {
        "schema_version": 1,
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "action_schema": action_schema,
        "semantic_step_index": semantic_step_index,
        "resolved_primitive_actions": resolved_actions,
        "resolved_primitive_events": resolved_events,
        "binding_revision": binding.binding_revision,
        "binding_sha256": binding.binding_sha256,
        "expected_cursor_before": list(expected_cursor_before),
        "expected_cursor_after": list(expected_cursor_after),
        "actions": actions,
    }
    return CompiledSegment(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        action_schema=action_schema,
        semantic_step_index=semantic_step_index,
        actions=actions,
        resolved_primitive_actions=resolved_actions,
        resolved_primitive_events=resolved_events,
        resolved_budget_sha256=hashlib.sha256(canonical_json(receipt)).hexdigest(),
        binding_revision=binding.binding_revision,
        binding_sha256=binding.binding_sha256,
        expected_cursor_before=expected_cursor_before,
        expected_cursor_after=expected_cursor_after,
    )


def record_executed_segment(
    segment: CompiledSegment,
    dispatches: tuple[tuple[dict[str, Any], ...], ...],
    *,
    execution_started_monotonic_ns: int,
    execution_completed_monotonic_ns: int,
) -> ExecutedSegmentReceipt:
    if len(dispatches) != len(segment.actions) or not dispatches:
        raise ValueError("dispatch evidence does not cover every compiled action")
    if execution_completed_monotonic_ns <= execution_started_monotonic_ns:
        raise ValueError("segment execution timestamps are not monotonic")
    observed_cursor = list(segment.expected_cursor_before)
    for action, action_results in zip(segment.actions, dispatches, strict=True):
        if not action_results:
            raise ValueError("compiled action has no executor dispatch receipt")
        if segment.action_schema == "native_absolute_sequence_v1":
            if not isinstance(action, dict):
                raise ValueError("native compiled action type mismatch")
            operations = action.get("operations")
            if not isinstance(operations, list) or len(action_results) != len(operations):
                raise ValueError("native dispatch cardinality does not cover compiled operations")
            for operation_index, (compiled_operation, result) in enumerate(zip(
                operations, action_results, strict=True
            )):
                _validate_dispatch_result_seal(result)
                if result.get("compiled_operation_index") != operation_index or (
                    result.get("compiled_payload_sha256")
                    != hashlib.sha256(canonical_json(compiled_operation)).hexdigest()
                ):
                    raise ValueError("native dispatch compiled-operation seal mismatch")
                if result.get("adapter") != "native_absolute_control":
                    raise ValueError("native dispatch adapter mismatch")
                cursor_before = result.get("cursor_before")
                cursor_after = result.get("cursor_after")
                if not (
                    isinstance(cursor_before, list)
                    and isinstance(cursor_after, list)
                    and len(cursor_before) == len(cursor_after) == 2
                    and all(
                        isinstance(value, int)
                        for value in (*cursor_before, *cursor_after)
                    )
                ):
                    raise ValueError("native dispatch cursor evidence mismatch")
                if cursor_before != observed_cursor:
                    raise ValueError("native dispatch cursor chain mismatch")
                coordinate = compiled_operation.get("coordinate")
                expected_cursor_after = (
                    [int(round(coordinate[0])), int(round(coordinate[1]))]
                    if coordinate is not None
                    and compiled_operation.get("action")
                    in {"click", "mouse_down", "mouse_move", "mouse_up"}
                    else cursor_before
                )
                if cursor_after != expected_cursor_after:
                    raise ValueError("native dispatch cursor result mismatch")
                observed_cursor = cursor_after
                expected_class, expected_operations, expected_atomic = (
                    _expected_native_dispatch(compiled_operation)
                )
                if result.get("action_class") != expected_class or (
                    _normalized_operations(result.get("operations"))
                    != expected_operations
                ):
                    raise ValueError("native dispatch order/content mismatch")
                atomic = result.get("atomic_state")
                if expected_atomic is None:
                    if atomic is not None:
                        raise ValueError("unexpected native atomic result")
                elif not isinstance(atomic, dict) or atomic.get("ok") is not True or (
                    _normalized_operations(atomic.get("operations")) != expected_atomic
                ):
                    raise ValueError("native atomic result does not match compiled operation")
        else:
            if not isinstance(action, str) or len(action_results) != 1:
                raise ValueError("compact dispatch cardinality mismatch")
            result = action_results[0]
            _validate_dispatch_result_seal(result)
            if result.get("compiled_operation_index") != 0 or (
                result.get("compiled_payload_sha256")
                != hashlib.sha256(canonical_json(action)).hexdigest()
            ):
                raise ValueError("compact dispatch compiled-action seal mismatch")
            if result.get("adapter") != "compact_raw_phaseb":
                raise ValueError("compact dispatch adapter mismatch")
            cursor_before = result.get("cursor_before")
            cursor_after = result.get("cursor_after")
            if not (
                isinstance(cursor_before, list)
                and isinstance(cursor_after, list)
                and len(cursor_before) == len(cursor_after) == 2
                and all(
                    isinstance(value, int)
                    for value in (*cursor_before, *cursor_after)
                )
            ):
                raise ValueError("compact dispatch cursor evidence mismatch")
            if cursor_before != observed_cursor:
                raise ValueError("compact dispatch cursor chain mismatch")
            expected, expected_after, expected_class = _expected_compact_dispatch(
                action, (int(cursor_before[0]), int(cursor_before[1]))
            )
            atomic = result.get("atomic_state")
            if result.get("action_class") != expected_class or (
                cursor_after != list(expected_after)
            ) or (
                _normalized_operations(result.get("operations")) != expected
            ) or (
                not isinstance(atomic, dict)
                or atomic.get("ok") is not True
                or _normalized_operations(atomic.get("operations")) != expected
            ):
                raise ValueError("compact dispatch order/content mismatch")
            observed_cursor = cursor_after
    if observed_cursor != list(segment.expected_cursor_after):
        raise ValueError("segment final cursor does not match compiled binding")
    dispatch_payload = {
        "schema_version": 1,
        "task_id": segment.task_id,
        "semantic_step_index": segment.semantic_step_index,
        "compiled_actions": segment.actions,
        "dispatches": dispatches,
    }
    dispatch_sha = hashlib.sha256(canonical_json(dispatch_payload)).hexdigest()
    receipt_payload = {
        "schema_version": 1,
        "task_id": segment.task_id,
        "fixture_sha256": segment.fixture_sha256,
        "action_schema": segment.action_schema,
        "semantic_step_index": segment.semantic_step_index,
        "resolved_primitive_actions": segment.resolved_primitive_actions,
        "resolved_primitive_events": segment.resolved_primitive_events,
        "resolved_budget_sha256": segment.resolved_budget_sha256,
        "binding_revision": segment.binding_revision,
        "binding_sha256": segment.binding_sha256,
        "dispatch_receipt_sha256": dispatch_sha,
        "execution_started_monotonic_ns": execution_started_monotonic_ns,
        "execution_completed_monotonic_ns": execution_completed_monotonic_ns,
    }
    return ExecutedSegmentReceipt(
        **receipt_payload,
        executed_receipt_sha256=hashlib.sha256(
            canonical_json(receipt_payload)
        ).hexdigest(),
    )


def _normalized_operations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("executor result operations are missing")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"kind", "args"}:
            raise ValueError("executor result operation schema mismatch")
        args = item["args"]
        if not isinstance(args, (list, tuple)):
            raise ValueError("executor result operation args mismatch")
        rows.append({"kind": item["kind"], "args": list(args)})
    return rows


def _validate_dispatch_result_seal(result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise ValueError("executor dispatch result must be an object")
    payload = dict(result)
    result_sha = payload.pop("dispatch_result_sha256", None)
    if not isinstance(result_sha, str) or len(result_sha) != 64 or (
        hashlib.sha256(canonical_json(payload)).hexdigest() != result_sha
    ):
        raise ValueError("executor dispatch result seal mismatch")
    atomic = result.get("atomic_state")
    atomic_sha = result.get("atomic_state_sha256")
    expected_atomic_sha = (
        hashlib.sha256(canonical_json(atomic)).hexdigest()
        if isinstance(atomic, dict)
        else None
    )
    if atomic_sha != expected_atomic_sha:
        raise ValueError("executor atomic result seal mismatch")
    if result.get("parse_status") != "ok" or (
        result.get("executor_dispatch_status") != "ok"
    ):
        raise ValueError("executor dispatch did not complete successfully")


def _expected_native_dispatch(
    operation: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]:
    kind = operation["action"]
    coordinate = operation.get("coordinate")
    rows: list[dict[str, Any]] = []
    if coordinate is not None and kind in {"click", "mouse_down", "mouse_move", "mouse_up"}:
        rows.append({"kind": "move_to", "args": list(coordinate)})
    atomic: list[dict[str, Any]] | None = None
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
        raise ValueError(f"unsupported native compiled operation: {kind!r}")
    return action_class, rows, atomic


def _expected_compact_dispatch(
    action: str, cursor_before: tuple[int, int]
) -> tuple[list[dict[str, Any]], tuple[int, int], str]:
    parsed = parse_compact_raw(action)
    rows: list[dict[str, Any]] = []
    classes: set[str] = set()
    cursor_after = cursor_before
    if parsed.dx or parsed.dy:
        cursor_after = (
            cursor_before[0] + parsed.dx,
            cursor_before[1] + parsed.dy,
        )
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


def aggregate_executed_segments(
    task: SemanticTask,
    action_schema: str,
    *,
    segments: tuple[ExecutedSegmentReceipt, ...] | list[ExecutedSegmentReceipt],
) -> CompiledProgram:
    """Sum and hash receipts from execution; never recompile earlier segments."""

    values = tuple(segments)
    if tuple(item.semantic_step_index for item in values) != tuple(
        range(1, task.semantic_step_count + 1)
    ):
        raise ValueError(f"{task.task_id}: executed segment coverage/order mismatch")
    if any(
        (item.task_id, item.fixture_sha256, item.action_schema)
        != (task.task_id, task.fixture_sha256, action_schema)
        for item in values
    ):
        raise ValueError(f"{task.task_id}: executed segment identity mismatch")
    revisions = tuple(item.binding_revision for item in values)
    binding_hashes = tuple(item.binding_sha256 for item in values)
    if task.app == "chrome":
        if revisions != (1, 1, 2) or binding_hashes[0] != binding_hashes[1] or (
            binding_hashes[2] == binding_hashes[1]
        ):
            raise ValueError(f"{task.task_id}: Chrome binding transition mismatch")
    elif len(set(revisions)) != 1 or len(set(binding_hashes)) != 1:
        raise ValueError(f"{task.task_id}: unexpected mid-trajectory binding change")
    for item in values:
        validate_executed_segment_receipt(item)
    resolved_actions = sum(item.resolved_primitive_actions for item in values)
    resolved_events = sum(item.resolved_primitive_events for item in values)
    if resolved_actions > task.budget_contract["primitive_action_caps"][action_schema]:
        raise ValueError(f"{task.task_id}: aggregate primitive actions exceed cap")
    if resolved_events > task.budget_contract["primitive_event_caps"][action_schema]:
        raise ValueError(f"{task.task_id}: aggregate primitive events exceed cap")
    payload = {
        "schema_version": 1,
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "action_schema": action_schema,
        "executed_segment_receipt_sha256": [
            item.executed_receipt_sha256 for item in values
        ],
        "segment_budget_sha256": [item.resolved_budget_sha256 for item in values],
        "segment_binding_sha256": [item.binding_sha256 for item in values],
        "segment_binding_revisions": [item.binding_revision for item in values],
        "resolved_primitive_actions": resolved_actions,
        "resolved_primitive_events": resolved_events,
    }
    binding_chain_sha256 = hashlib.sha256(
        canonical_json(payload["segment_binding_sha256"])
    ).hexdigest()
    payload["binding_sha256"] = binding_chain_sha256
    return CompiledProgram(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        action_schema=action_schema,
        segments=values,
        resolved_primitive_actions=resolved_actions,
        resolved_primitive_events=resolved_events,
        resolved_budget_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
        binding_sha256=binding_chain_sha256,
    )


def validate_executed_segment_receipt(item: ExecutedSegmentReceipt) -> None:
    if not isinstance(item, ExecutedSegmentReceipt) or item.schema_version != 1:
        raise ValueError("executed segment receipt schema/type mismatch")
    if (
        not item.task_id
        or not item.fixture_sha256
        or item.action_schema not in ACTION_SCHEMAS
        or item.semantic_step_index < 1
        or item.resolved_primitive_actions < 1
        or item.resolved_primitive_events < 1
        or item.binding_revision < 1
        or item.execution_started_monotonic_ns < 1
        or item.execution_completed_monotonic_ns
        <= item.execution_started_monotonic_ns
    ):
        raise ValueError("executed segment receipt field contract mismatch")
    lowercase_hex = set("0123456789abcdef")
    for name in (
        "fixture_sha256",
        "resolved_budget_sha256",
        "binding_sha256",
        "dispatch_receipt_sha256",
        "executed_receipt_sha256",
    ):
        value = getattr(item, name)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in lowercase_hex for char in value
        ):
            raise ValueError(f"executed segment receipt {name} is not lowercase SHA-256")
    payload = {
        key: value
        for key, value in asdict(item).items()
        if key != "executed_receipt_sha256"
    }
    if hashlib.sha256(canonical_json(payload)).hexdigest() != item.executed_receipt_sha256:
        raise ValueError(f"{item.task_id}: executed segment receipt seal mismatch")


def compile_program(*args: Any, **kwargs: Any) -> None:
    """Bulk precompilation is forbidden: production compiles after each probe."""

    raise RuntimeError(
        "bulk compile_program is disabled; compile, execute, and receipt semantic segments sequentially"
    )
