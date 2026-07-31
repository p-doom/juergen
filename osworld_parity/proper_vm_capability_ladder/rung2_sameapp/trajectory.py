from __future__ import annotations

from .actions import ActionTurn, ScriptedTrajectory, SymbolicOperation
from .fixtures import Fixture


def _op(kind: str, **kwargs: object) -> SymbolicOperation:
    return SymbolicOperation(kind, **kwargs)


def build_trajectory(fixture: Fixture, *, near_miss: bool = False) -> ScriptedTrajectory:
    if fixture.app == "writer":
        text = str(
            fixture.near_miss["text"] if near_miss else fixture.expected["text"]
        )
        formatting = () if near_miss else (_op("key_chord", keys=("ControlLeft", "KeyB")),)
        turns = (
            ActionTurn(1, (_op("click", target="editor"), _op("key_chord", keys=("ControlLeft", "KeyA")), _op("type", text=text))),
            ActionTurn(2, (_op("key_chord", keys=("ControlLeft", "KeyA")),) + formatting),
            ActionTurn(3, (_op("key_chord", keys=("ControlLeft", "KeyS")),)),
        )
    elif fixture.app == "calc":
        formula = str(
            fixture.near_miss["formula"] if near_miss else fixture.expected["formula"]
        ).removeprefix("of:")
        turns = (
            ActionTurn(
                1,
                (
                    _op("click", target="cell"),
                    _op("key_chord", keys=("ControlLeft", "KeyA")),
                    _op("type", text=str(fixture.params["cell"])),
                    _op("key_chord", keys=("Return",)),
                ),
            ),
            ActionTurn(2, (_op("type", text=formula),)),
            ActionTurn(3, (_op("key_chord", keys=("Return",)),)),
            ActionTurn(4, (_op("key_chord", keys=("ControlLeft", "KeyS")),)),
        )
    elif fixture.app == "files":
        destination = "decoy" if near_miss else "destination"
        final_name = str(
            fixture.near_miss["final_name"]
            if near_miss
            else fixture.expected["final_name"]
        )
        turns = (
            ActionTurn(1, (_op("click", target="source"),)),
            ActionTurn(2, (_op("mouse_down", target="source", button="left"),)),
            ActionTurn(2, (_op("mouse_move", target=destination),)),
            ActionTurn(2, (_op("mouse_up", target=destination, button="left"),)),
            ActionTurn(
                3,
                (
                    _op("click", target=destination),
                    _op("key_chord", keys=("Return",)),
                ),
            ),
            ActionTurn(3, (_op("click", target="moved"),)),
            ActionTurn(3, (_op("key_chord", keys=("F2",)),)),
            ActionTurn(
                3,
                (
                    _op("key_chord", keys=("ControlLeft", "KeyA")),
                    _op("type", text=final_name),
                    _op("key_chord", keys=("Return",)),
                ),
            ),
        )
    elif fixture.app == "chrome":
        nav = "decoy_nav" if near_miss else "nav"
        toggle = "decoy_toggle" if near_miss else "toggle"
        turns = (
            ActionTurn(1, (_op("click", target=nav),)),
            ActionTurn(2, (_op("scroll", target="scroll_surface", clicks=-7),)),
            ActionTurn(3, (_op("click", target=toggle),)),
        )
    else:
        raise ValueError(f"unsupported same-app fixture: {fixture.app}")
    if len(turns) > fixture.horizon:
        raise ValueError(f"script exceeds horizon for {fixture.id}")
    if len({turn.semantic_step for turn in turns}) != fixture.semantic_steps:
        raise ValueError(f"semantic step mismatch for {fixture.id}")
    return ScriptedTrajectory(fixture.id, near_miss, turns)
