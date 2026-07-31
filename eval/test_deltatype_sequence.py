from __future__ import annotations

import json

from action_parser import format_deltatype, parse_deltatype_sequence
from osworld_vm_client import OSWorldClient


class FakeClient(OSWorldClient):
    def __init__(self) -> None:
        self.position = (100, 100)
        self.commands: list[str] = []

    def cursor_position(self) -> tuple[int, int]:
        return self.position

    def screen_size(self) -> tuple[int, int]:
        return 400, 300

    def execute(self, command: str) -> None:
        self.commands.append(command)
        if command.startswith("pyautogui.moveTo("):
            coordinates = command.removeprefix("pyautogui.moveTo(").removesuffix(")")
            x, y = coordinates.split(", ")
            self.position = int(x), int(y)


def test_causal_drag_sequence_round_trip() -> None:
    text = "0 0 0 ; +LMB\n120 -4 0\n0 0 0 ; -LMB"
    parsed = parse_deltatype_sequence(text)
    assert "\n".join(format_deltatype(action) for action in parsed) == text


def test_coalesced_type_round_trip() -> None:
    value = 'hello; +world "quoted"'
    text = f"0 0 0 ; type({json.dumps(value)})"
    parsed = parse_deltatype_sequence(text)
    assert parsed[0].type_texts == (value,)
    assert format_deltatype(parsed[0]) == text


def test_sequence_executes_drag_and_coalesced_type_in_order() -> None:
    client = FakeClient()
    sequence = parse_deltatype_sequence('0 0 0 ; +LMB\n20 -10 0\n0 0 0 ; -LMB type("done")')
    for action in sequence:
        client.dispatch_deltatype(action)
    assert client.position == (120, 90)
    assert client.commands == [
        "pyautogui.mouseDown(button='left')",
        "pyautogui.moveTo(120, 90)",
        "pyautogui.mouseUp(button='left')",
        "pyautogui.write('done', interval=0)",
    ]
