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
    observed = program_budgets(program)
    if not program.near_miss and observed != task.budgets:
        raise ValueError(
            f"{task.task_id}: declared/observed program budgets differ: "
            f"{task.budgets!r} != {observed!r}"
        )
    if program.near_miss:
        for field in ("primitive_actions", "primitive_events"):
            for interface, count in observed[field].items():
                if count > task.budgets[field][interface]:
                    raise ValueError(
                        f"{task.task_id}: near miss exceeds {field}/{interface} budget"
                    )


def program_budgets(program: ScriptedTrajectory) -> dict[str, Any]:
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


def compile_program(
    task: SemanticTask,
    action_schema: str,
    *,
    geometry: dict[str, tuple[int, int]],
    initial_cursor: tuple[int, int],
    near_miss: bool = False,
) -> list[dict[str, Any] | str]:
    """Compile after task selection; action encoding is never task identity."""

    if action_schema not in ACTION_SCHEMAS:
        raise ValueError(f"unsupported action schema: {action_schema!r}")
    program = build_program(task, near_miss=near_miss)
    required = set(task.geometry_contract["required_targets"])
    if set(geometry) != required:
        raise ValueError(
            f"{task.task_id}: live geometry keys drifted: "
            f"{sorted(geometry)} != {sorted(required)}"
        )
    if action_schema == "native_absolute_sequence_v1":
        return [compile_native(turn, geometry) for turn in program.turns]
    cursor = tuple(initial_cursor)
    compiled: list[str] = []
    for turn in program.turns:
        action, cursor = compile_compact(turn, geometry, cursor)
        compiled.append(action)
    return compiled
