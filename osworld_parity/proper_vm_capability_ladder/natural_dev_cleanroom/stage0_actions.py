from __future__ import annotations

from typing import Any

from ..rung2_sameapp.actions import (
    ActionTurn,
    ScriptedTrajectory,
    SymbolicOperation,
    compile_compact,
    compile_native,
)
from .stage0_loader import Stage0SourceTask


MAX_MULTI_PRIMITIVES = 8
MAX_MULTI_EMITTED_EVENTS = 25


def _op(kind: str, **kwargs: object) -> SymbolicOperation:
    return SymbolicOperation(kind, **kwargs)


def visible_app_switch_turn() -> ActionTurn:
    return ActionTurn(
        2,
        (_op("key_chord", keys=("AltLeft", "Tab")),),
    )


def compile_visible_app_switch_native() -> dict[str, Any]:
    return compile_native(visible_app_switch_turn(), {})


def compile_visible_app_switch_compact() -> str:
    value, _ = compile_compact(visible_app_switch_turn(), {}, (0, 0))
    return value


def build_multi_component_program(
    task: Stage0SourceTask, *, near_miss: bool = False
) -> ScriptedTrajectory:
    if "-multi-" not in task.id or task.semantic_steps != 1:
        raise ValueError(f"not a short multi-app source task: {task.id}")
    if task.app == "writer":
        text = str(task.near_miss["text"] if near_miss else task.expected["text"])
        operations = (
            _op("key_chord", keys=("ControlLeft", "KeyA")),
            _op("type", text=text),
            _op("key_chord", keys=("ControlLeft", "KeyS")),
        )
        turns = (ActionTurn(1, operations),)
    elif task.app == "calc":
        expected = task.near_miss if near_miss else task.expected
        formula = str(expected["formula"]).removeprefix("of:")
        operations = (
            _op("type", text=formula),
            _op("key_chord", keys=("Return",)),
            _op("key_chord", keys=("ControlLeft", "KeyS")),
        )
        turns = (ActionTurn(1, operations),)
    elif task.app == "files":
        operations = (
            _op("key_chord", keys=("F2",)),
            _op(
                "type",
                text=str(
                    task.near_miss["final_name"]
                    if near_miss
                    else task.expected["final_name"]
                ),
            ),
            _op("key_chord", keys=("Return",)),
        )
        turns = (ActionTurn(1, operations),)
    elif task.app == "chrome":
        turns = (
            ActionTurn(
                1,
                (_op("click", target="decoy_nav" if near_miss else "nav"),),
            ),
            ActionTurn(
                1,
                (_op("click", target="decoy_toggle" if near_miss else "toggle"),),
            ),
        )
    elif task.app == "vscode":
        operations = (
            _op("key_chord", keys=("ControlLeft", "KeyA")),
            _op(
                "type",
                text=str(task.near_miss["text"] if near_miss else task.expected["text"]),
            ),
            _op("key_chord", keys=("ControlLeft", "KeyS")),
        )
        turns = (ActionTurn(1, operations),)
    else:  # pragma: no cover - loader fixes the app set
        raise ValueError(f"unsupported multi-app source: {task.app}")
    trajectory = ScriptedTrajectory(task.id, False, turns)
    if sum(len(turn.operations) for turn in turns) != task.horizon:
        raise ValueError(f"{task.id}: symbolic program/horizon drift")
    return trajectory


def compile_multi_native(
    task: Stage0SourceTask,
    geometry: dict[str, tuple[int, int]],
    *,
    near_miss: bool = False,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        compile_native(turn, geometry)
        for turn in build_multi_component_program(task, near_miss=near_miss).turns
    )


def compile_multi_compact(
    task: Stage0SourceTask,
    geometry: dict[str, tuple[int, int]],
    cursor: tuple[int, int],
    *,
    near_miss: bool = False,
) -> tuple[str, ...]:
    rows: list[str] = []
    current = cursor
    for turn in build_multi_component_program(task, near_miss=near_miss).turns:
        value, current = compile_compact(turn, geometry, current)
        rows.append(value)
    return tuple(rows)


def component_program_counts(task: Stage0SourceTask) -> dict[str, int]:
    operations = [
        operation
        for turn in build_multi_component_program(task).turns
        for operation in turn.operations
    ]
    events = 0
    for operation in operations:
        if operation.kind == "click":
            events += 3  # pointer motion plus press/release
        elif operation.kind == "key_chord":
            events += 2 * len(operation.keys)
        else:
            events += 1
    return {"primitive_actions": len(operations), "emitted_events": events}


def record_program_counts(source_tasks: tuple[Stage0SourceTask, ...]) -> dict[str, int]:
    if len(source_tasks) != 2:
        raise ValueError("multi-app record count requires two ordered sources")
    rows = [component_program_counts(task) for task in source_tasks]
    switch = visible_app_switch_turn().operations[0]
    result = {
        "primitive_actions": sum(row["primitive_actions"] for row in rows) + 1,
        "emitted_events": sum(row["emitted_events"] for row in rows)
        + 2 * len(switch.keys),
    }
    if result["primitive_actions"] > MAX_MULTI_PRIMITIVES:
        raise ValueError(f"multi-app primitive ceiling exceeded: {result}")
    if result["emitted_events"] > MAX_MULTI_EMITTED_EVENTS:
        raise ValueError(f"multi-app emitted-event ceiling exceeded: {result}")
    return result
