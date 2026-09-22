"""Conformance tests for every production grammar vector."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from desktop.geometry import DisplayGeometry

import grammars
from grammars import _support

NAMES = tuple(grammars.available())


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
    if name == "ordered_events_v3_relative_1000_grid_v1":
        return module.action_from_dict({**value, "terminate": None})
    elements = []
    for element in value["elements"]:
        kind = element["kind"]
        if kind == "event":
            elements.append(
                _support.Element(
                    "event", name=element["name"], pressed=element["pressed"]
                )
            )
        elif kind == "type":
            elements.append(_support.Element("type", text=element["text"]))
        elif kind == "move":
            elements.append(_support.Element("move", delta=tuple(element["delta"])))
        else:
            raise AssertionError(f"unknown fixture element kind: {kind!r}")
    return module.DeltatypeV2Action(
        dx=value["dx"],
        dy=value["dy"],
        scroll=value["scroll"],
        elements=tuple(elements),
        no_op=value["no_op"],
    )


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


@pytest.mark.parametrize(("name", "payload", "case"), _cases("format_only"))
def test_format_only(name, payload, case):
    assert (
        _codec(name).format(_action_from_dict(name, case["action"]))
        == case["canonical"]
    )


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_parse"))
def test_invalid_parse(name, payload, case):
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        _codec(name).parse(case["text"])


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_compile"))
def test_invalid_compile(name, payload, case):
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        _codec(name).compile(case["text"], geometry, cursor)


def test_every_vector_section_is_executed():
    executed = {
        "cases",
        "format_only",
        "invalid_parse",
        "invalid_compile",
    }
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
    action = replace(
        codec.parse(
            {
                "deltatype_v2": "0 0 0 ; +LMB -LMB",
                "ordered_events_v3_relative_1000_grid_v1": "down(LMB); up(LMB)",
            }[name]
        ),
        terminate=status,
    )
    text = codec.format(action)
    control = _support.split_control(text)
    assert control.status == status
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


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"kind": "other"}, "unknown element kind"),
        ({"kind": "event", "name": "KeyA", "pressed": 1}, "only name and pressed"),
        ({"kind": "type", "text": "x", "name": None}, "require only text"),
        ({"kind": "move", "delta": [1, 2]}, "two integer deltas"),
    ],
)
def test_element_rejects_noncanonical_state(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _support.Element(**kwargs)


def test_ordered_action_dict_requires_exact_schema():
    module = importlib.import_module(
        "grammars.ordered_events_v3_relative_1000_grid_v1.codec"
    )
    canonical = {
        "primitives": [{"kind": "move", "dx": 1, "dy": -2}],
        "no_op": False,
        "terminate": None,
    }
    assert module.action_from_dict(canonical).to_dict() == canonical
    invalid = [
        (
            {key: value for key, value in canonical.items() if key != "terminate"},
            "exactly",
        ),
        ({**canonical, "extra": None}, "exactly"),
        ({**canonical, "primitives": tuple(canonical["primitives"])}, "must be a list"),
        ({**canonical, "no_op": 0}, "must be a boolean"),
        ({**canonical, "primitives": []}, "exactly identify"),
        (
            {
                **canonical,
                "primitives": [{"kind": "move", "dx": True, "dy": -2}],
            },
            "two integer deltas",
        ),
        (
            {
                **canonical,
                "primitives": [{"kind": "move", "dx": 1, "dy": -2, "name": "ignored"}],
            },
            "invalid move primitive fields",
        ),
    ]
    for value, error in invalid:
        with pytest.raises(ValueError, match=error):
            module.action_from_dict(value)


def test_render_spec_requires_every_docstring():
    class MissingProductionDoc:
        """Preamble."""

        @_support.production("ACTION")
        def action(self):
            pass

        def notes(self):
            """Notes."""

    with pytest.raises(ValueError, match="action must have a docstring"):
        _support.render_spec(MissingProductionDoc())


def test_registry_contains_exactly_the_two_training_grammars():
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
