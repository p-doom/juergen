from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from action_v2 import (
    DeltaTypeV2Error,
    dispatch_deltatype_v2,
    format_deltatype_v2,
    ordered_plan,
    parse_deltatype_v2,
)
from readiness import validate_response


from conftest import repo_relative

# The legacy production parser is the repo's own eval/action_parser.py. The
# sealed build hash-pinned it at f916757d…; test_pinned_contract.py asserts that.
PRODUCTION_PARSER = repo_relative("eval", "action_parser.py")


def load_production_parser():
    name = "raw_v2_test_production_parser"
    spec = importlib.util.spec_from_file_location(name, PRODUCTION_PARSER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "label",
    (
        "NO_OP",
        "TERMINATE",
        "FAIL",
        "12 -7 0",
        "12 -7 0 ; +LMB -LMB",
        "0 0 -3 ; +_Private -_Private",
        '0 0 0 ; type("a; + b \\"quoted\\"") +Return -Return',
    ),
)
def test_legacy_labels_are_byte_and_plan_invariant(label: str):
    production = load_production_parser()
    old = production.parse_deltatype(label)
    new = parse_deltatype_v2(label)
    assert production.format_deltatype(old) == label
    assert format_deltatype_v2(new) == label

    cursor = (205, 315)
    old_events = []
    if not (old.no_op or old.terminate or old.fail):
        target = (
            max(0, min(1919, cursor[0] + old.dx)),
            max(0, min(1079, cursor[1] + old.dy)),
        )
        if target != cursor:
            old_events.append(("moveTo", *target))
        if old.scroll:
            old_events.append(("scroll", old.scroll))
        for kind, value in old.elements:
            old_events.append(
                ("type", value) if kind == "type" else (value.kind, value.what)
            )
    assert ordered_plan(new, cursor) == tuple(old_events)


@pytest.mark.parametrize(
    ("label", "cursor", "expected"),
    (
        (
            "0 0 0 ; +LMB MOVE(1051,254) -LMB",
            (205, 315),
            (
                ("press", "LMB"),
                ("moveTo", 1256, 569, 0.5),
                ("release", "LMB"),
            ),
        ),
        (
            "-793 -229 0 ; +LMB MOVE(547,321) -LMB",
            (960, 540),
            (
                ("moveTo", 167, 311),
                ("press", "LMB"),
                ("moveTo", 714, 632, 0.5),
                ("release", "LMB"),
            ),
        ),
        (
            "0 0 0 ; +LMB MOVE(0,0) -LMB",
            (770, 589),
            (
                ("press", "LMB"),
                ("moveTo", 770, 589, 0.5),
                ("release", "LMB"),
            ),
        ),
        (
            "100 100 0 ; +LMB MOVE(100,100) -LMB",
            (1900, 1070),
            (
                ("moveTo", 1919, 1079),
                ("press", "LMB"),
                ("moveTo", 1919, 1079, 0.5),
                ("release", "LMB"),
            ),
        ),
    ),
)
def test_drag_roundtrip_and_ordered_dispatch_are_exact(label, cursor, expected):
    action = parse_deltatype_v2(label)
    assert format_deltatype_v2(action) == label
    assert ordered_plan(action, cursor) == expected

    class Recorder:
        def __init__(self):
            self.commands = []

        def execute_ordered(self, command):
            self.commands.append(command)

    recorder = Recorder()
    assert dispatch_deltatype_v2(recorder, action, cursor) == expected
    assert tuple(recorder.commands) == expected


@pytest.mark.parametrize(
    "label",
    (
        "1 2 0 ; MOVE(3,4)",
        "1 2 0 ; +LMB MOVE(3,4)",
        "1 2 0 ; +LMB MOVE(3,4) -LMB +LMB",
        "1 2 1 ; +LMB MOVE(3,4) -LMB",
        "1 2 0 ; +RMB MOVE(3,4) -RMB",
        "1 2 0 ; +LMB MOVE(3,4) MOVE(5,6) -LMB",
        '1 2 0 ; +LMB MOVE(3,4) type("x") -LMB',
        "1 2 0 ; +LMB MOVE -LMB",
        "1 2 0 ; +LMB MOVE(3) -LMB",
        "1 2 0 ; +LMB MOVE(3,4,5) -LMB",
        "1 2 0 ; +LMB MOVE(3.5,4) -LMB",
        "1 2 0 ; +LMB MOVE(3,4)x -LMB",
        "1 2 0 ; +LMB move(3,4) -LMB",
    ),
)
def test_unreviewed_or_malformed_move_fails_closed(label: str):
    with pytest.raises(DeltaTypeV2Error):
        parse_deltatype_v2(label)


def test_readiness_requires_canonical_final_action_line():
    validate_response("reasoning\n0 0 0 ; +LMB MOVE(3,4) -LMB")
    with pytest.raises(ValueError, match="not canonical"):
        validate_response("0 0 0; +LMB MOVE(3, 4) -LMB")
