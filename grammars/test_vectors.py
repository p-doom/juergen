"""Conformance tests for every production grammar vector."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
from desktop import ir
from desktop.geometry import DisplayGeometry

import grammars

from . import _support

NAMES = tuple(grammars.available())
_MOVE_KINDS = ("move_to", "glide_to")


def _vectors(name: str) -> dict:
    module = importlib.import_module(f"grammars.{name}.codec")
    path = Path(module.__file__).parent / "vectors" / f"{name}.json"
    payload = json.loads(path.read_text())
    assert payload["grammar"] == name
    return payload


def _geometry(value: dict) -> DisplayGeometry:
    return DisplayGeometry(
        desktop_width=int(value["desktop_width"]),
        desktop_height=int(value["desktop_height"]),
    )


def _context(payload: dict, case: dict) -> tuple[DisplayGeometry, tuple[int, int]]:
    return (
        _geometry(case.get("geometry", payload["geometry"])),
        tuple(case.get("cursor", payload["default_cursor"])),
    )


def _operations(rows: list) -> tuple[ir.Operation, ...]:
    return tuple(ir.Operation(kind, tuple(args)) for kind, args in rows)


def _rows(operations) -> list:
    return [[item.kind, list(item.args)] for item in operations]


def _cases(section: str):
    return [
        pytest.param(name, payload, case, id=f"{name}-{case['name']}")
        for name in NAMES
        for payload in (_vectors(name),)
        for case in payload.get(section, ())
    ]


def _codec(name: str):
    return grammars.load(name)


def _action_from_dict(name: str, value: dict):
    module = importlib.import_module(f"grammars.{name}.codec")
    return module.action_from_dict(value)


@pytest.mark.parametrize(("name", "payload", "case"), _cases("cases"))
def test_case(name, payload, case):
    codec = _codec(name)
    geometry, cursor = _context(payload, case)
    expected = _action_from_dict(name, case["action"])
    assert codec.parse(case["text"]) == expected
    assert codec.format(expected) == case["canonical"]
    assert _rows(codec.compile(case["text"], geometry, cursor)) == case["operations"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("cases"))
def test_canonical_is_a_fixpoint(name, payload, case):
    codec = _codec(name)
    reparsed = codec.parse(case["canonical"])
    assert codec.format(reparsed) == case["canonical"]
    assert reparsed == codec.parse(case["text"])


@pytest.mark.parametrize(("name", "payload", "case"), _cases("cases"))
def test_dict_round_trip(name, payload, case):
    parsed = _codec(name).parse(case["text"])
    assert _action_from_dict(name, parsed.to_dict()) == parsed


@pytest.mark.parametrize(("name", "payload", "case"), _cases("format_only"))
def test_format_only(name, payload, case):
    assert _codec(name).format(_action_from_dict(name, case["action"])) == case["canonical"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_parse"))
def test_invalid_parse(name, payload, case):
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        _codec(name).parse(case["text"])


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_compile"))
def test_invalid_compile(name, payload, case):
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        _codec(name).compile(case["text"], geometry, cursor)


@pytest.mark.parametrize(("name", "payload", "case"), _cases("lift"))
def test_lift(name, payload, case):
    codec = _codec(name)
    geometry, cursor = _context(payload, case)
    lifted = codec.action_from_operations(
        _operations(case["operations"]),
        geometry=geometry,
        cursor=cursor,
        terminate=case.get("terminate"),
    )
    assert lifted == _action_from_dict(name, case["action"])
    assert codec.format(lifted) == case["canonical"]
    assert _rows(codec.compile(case["canonical"], geometry, cursor)) == case["recompiled"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("lift_invalid"))
def test_lift_invalid(name, payload, case):
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        _codec(name).action_from_operations(
            _operations(case["operations"]),
            geometry=geometry,
            cursor=cursor,
            terminate=case.get("terminate"),
        )


def test_every_vector_section_is_executed():
    executed = {"cases", "format_only", "invalid_parse", "invalid_compile", "lift", "lift_invalid"}
    for name in NAMES:
        for section, value in _vectors(name).items():
            if isinstance(value, list) and section != "default_cursor":
                assert section in executed, f"{name}: nothing executes {section!r}"


@pytest.mark.parametrize("name", NAMES)
def test_codec_contract_and_prompt_pin(name):
    codec = _codec(name)
    assert codec is importlib.import_module(f"grammars.{name}.codec").CODEC
    assert codec.name == name
    assert codec.stop_sequences == ()
    described = codec.describe()
    assert codec.describe() == described
    assert codec.digest == _vectors(name)["prompt_sha256"]
    assert _support.CONTROL_SPEC in described


@pytest.mark.parametrize("status", ["success", "failure"])
@pytest.mark.parametrize("name", NAMES)
def test_control_channel_round_trip(name, status):
    codec = _codec(name)
    payload = _vectors(name)
    geometry = _geometry(payload["geometry"])
    cursor = tuple(payload["default_cursor"])
    action = codec.action_from_operations(
        (ir.mouse_down("left"), ir.mouse_up("left")),
        geometry=geometry,
        cursor=cursor,
        terminate=status,
    )
    text = codec.format(action)
    control = _support.split_control(text)
    assert control.status == status
    assert control.ignored == 0
    assert _rows(codec.compile(control.body, geometry, cursor)) == [
        ["mouse_down", ["left"]],
        ["mouse_up", ["left"]],
    ]


def test_control_line_is_exact_and_final():
    assert _support.split_control("NO_OP\nTERMINATE: failure").status == "failure"
    assert _support.split_control("TERMINATE: success\n").status == "success"
    for text in (
        "terminate: success",
        '<tool_call>{"name":"computer_use","arguments":{"action":"terminate"}}</tool_call>',
        '{"action":"terminate","status":"success"}',
    ):
        control = _support.split_control(text)
        assert control.status is None
        assert control.body == text
    for text in (
        "TERMINATE",
        "TERMINATE: success\nNO_OP",
        "I will TERMINATE: success",
        "TERMINATE:success",
        "TERMINATE:  success",
        "TERMINATE:\tsuccess",
        " TERMINATE: success",
        "TERMINATE: success ",
        "TERMINATE: success\n\n",
        "NO_OP\nTERMINATE: success\nNO_OP\nTERMINATE: failure",
    ):
        with pytest.raises(ValueError, match="TERMINATE must occur exactly once"):
            _support.split_control(text)


@pytest.mark.parametrize("name", NAMES)
def test_intended_cursor_matches_compile(name):
    codec = _codec(name)
    payload = _vectors(name)
    for case in payload["cases"]:
        geometry, cursor = _context(payload, case)
        intent = codec.intended_cursor(codec.parse(case["text"]), geometry, cursor)
        moves = [
            operation
            for operation in codec.compile(case["text"], geometry, cursor)
            if operation.kind in _MOVE_KINDS
        ]
        if intent is None:
            assert not moves
        elif moves:
            assert tuple(moves[-1].args[:2]) == _support.clamp((intent.x, intent.y), geometry)


CANONICAL_PROBES = {
    "move_to": ir.move_to(700, 400),
    "glide_to": ir.glide_to(700, 400, 0.5),
    "drag": ir.drag(700, 400, 800, 450),
    "click": ir.click("left"),
    "mouse_down": ir.mouse_down("left"),
    "mouse_up": ir.mouse_up("left"),
    "key_down": ir.key_down("KeyA"),
    "key_up": ir.key_up("KeyA"),
    "scroll": ir.scroll(0, 3),
    "coalesced_type": ir.coalesced_type("x"),
    "ascii_type": ir.ascii_type("x"),
    "wait": ir.wait(1.0),
}


def test_every_canonical_operation_kind_is_groupable():
    assert set(CANONICAL_PROBES) == set(ir.CANONICAL_KINDS) - {"raise_for_test"}
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    for operation in CANONICAL_PROBES.values():
        assert _support.group_operations((operation,), geometry=geometry, cursor=(960, 540))


def test_peer_directories_are_the_sole_registry():
    assert NAMES == ("deltatype_v2", "ordered_events_v3_relative_1000_grid_v1")
    for name in NAMES:
        assert _codec(name) is importlib.import_module(f"grammars.{name}.codec").CODEC
    for retired in ("compact_raw", "ordered_events_v3"):
        with pytest.raises(KeyError, match=retired):
            grammars.load(retired)


def test_no_handler_table_exists():
    root = Path(grammars.__file__).parent
    assert not list(root.glob("*/handlers.py"))
    assert not hasattr(grammars, "handlers")
    assert not hasattr(_support, "core_handlers")
