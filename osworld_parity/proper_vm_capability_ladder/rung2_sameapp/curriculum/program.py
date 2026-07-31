"""Runtime bridge from semantic tasks to the existing symbolic action layer.

Task records remain independent of an action encoding.  This module is the only
place that lowers a semantic gold/near-miss program to rung-2 ``ActionTurn``
objects, which the existing native and compact compilers can then encode.
"""

from __future__ import annotations

from typing import Any

from ..actions import (
    ACTION_SCHEMAS,
    ActionTurn,
    ScriptedTrajectory,
    SymbolicOperation,
    compile_compact,
    compile_native,
)
from .schema import SemanticTask


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
            ActionTurn(1, (_op("click", target="name_box"), _op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=str(task.params["cell"])), _op("key_chord", keys=("Return",)))),
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
    if len(program.turns) > task.max_action_turns:
        raise ValueError(f"{task.task_id}: program exceeds the frozen horizon")
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


def compile_program(
    task: SemanticTask, action_schema: str, *, near_miss: bool = False
) -> list[dict[str, Any] | str]:
    """Compile after task selection; action encoding is never task identity."""

    if action_schema not in ACTION_SCHEMAS:
        raise ValueError(f"unsupported action schema: {action_schema!r}")
    program = build_program(task, near_miss=near_miss)
    geometry = {
        key: tuple(value) for key, value in task.params["geometry"].items()
    }
    if action_schema == "native_absolute_sequence_v1":
        return [compile_native(turn, geometry) for turn in program.turns]
    cursor = tuple(task.initial_cursor)
    compiled: list[str] = []
    for turn in program.turns:
        action, cursor = compile_compact(turn, geometry, cursor)
        compiled.append(action)
    return compiled
