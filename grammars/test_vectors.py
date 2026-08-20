"""The conformance gate: every vector in every ``grammars/*/vectors/*.json``.

Each case below is one parametrised test, so a regression names the grammar, the
section and the case.

Run it with ``pytest grammars/`` (``pip install -e '.[dev]'`` for pytest).

Five invariants that are not expressible as vectors are asserted here as code:

* The control channel's round trip, in every grammar. A vector case hands
  ``codec.parse`` its whole ``text``, and the codec is never given the control
  line, so a terminating turn cannot be a vector at all — it is
  lift -> ``format`` -> ``split_control`` -> ``parse`` -> ``compile``, which is
  what the episode driver does.
* ``isinstance(codec, Codec)`` for all seven. The protocol is
  ``@runtime_checkable``, so a caller may write that gate; it returned False for
  every grammar while ``Codec`` demanded a ``handlers`` table no codec had and a
  ``stop_sequences`` method no codec had.
* The matched pair's shared prose is byte-identical. A line-wrap alone is a
  different token sequence in the two arms.
* Every canonical Operation kind can be grouped. A kind that falls through to
  "unknown Operation kind" makes any recorded trajectory containing it unliftable
  in all seven grammars at once.
* The normalized grammar's quantisation ceiling. The vectors elsewhere use deltas
  that round-trip exactly, leaving the lossy region — where a relative label is
  silently wrong — uncovered.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

import grammars
from desktop import ir
from desktop.codec_protocol import Codec
from desktop.geometry import DisplayGeometry

from . import _support

NAMES = tuple(grammars.available())

#: Sections whose value is a list of cases but which are not case tables.
_NOT_CASES = {"default_cursor"}


def _vectors(name: str) -> dict:
    module = importlib.import_module(f"grammars.{name}.codec")
    path = Path(module.__file__).parent / "vectors" / f"{name}.json"
    payload = json.loads(path.read_text())
    assert payload["grammar"] == name, f"{path} declares {payload['grammar']!r}"
    return payload


def _geometry(payload: dict) -> DisplayGeometry:
    """Vector geometry -> ``DisplayGeometry``, using desktop's own field names.

    One spelling throughout: a vector saying ``width``/``height`` at the top
    level and ``desktop_width``/``desktop_height`` in a per-case override would
    drift silently.
    """
    return DisplayGeometry(
        desktop_width=int(payload["desktop_width"]),
        desktop_height=int(payload["desktop_height"]),
    )


def _context(payload: dict, case: dict) -> tuple[DisplayGeometry, tuple[int, int]]:
    geometry = _geometry(case.get("geometry", payload["geometry"]))
    cursor = tuple(case.get("cursor", payload["default_cursor"]))
    return geometry, cursor


def _operations(rows: list) -> tuple[ir.Operation, ...]:
    return tuple(ir.Operation(kind, tuple(args)) for kind, args in rows)


def _rows(operations) -> list:
    return [[item.kind, list(item.args)] for item in operations]


def _elements(case: dict) -> tuple[_support.Element, ...]:
    return tuple(_support.element_from_dict(item) for item in case.get("elements", ()))


def _cases(section: str):
    """Every case in one section across every grammar, as pytest parameters."""
    collected = []
    for name in NAMES:
        payload = _vectors(name)
        for case in payload.get(section, ()):
            collected.append(
                pytest.param(name, payload, case, id=f"{name}-{case['name']}")
            )
    return collected


#: One codec's source file, as it was when this process imported it.
#:
#: A codec's rendered prompt comes from docstrings, which are fixed when the
#: module is imported; its pin is read from disk when the assertion runs. Editing
#: a grammar during a ten-minute suite therefore reported ``rendered`` from the
#: old text against a pin from the new one -- a mismatch describing two different
#: trees rather than a bad pin. That inversion was read as a real regression
#: twice. Comparing this against the file at assert time separates the two.
_SOURCE_WHEN_IMPORTED: dict[str, bytes] = {}


def _codec_source(name: str) -> Path:
    return Path(importlib.import_module(f"grammars.{name}.codec").__file__)


def _codec(name: str):
    codec = grammars.load(name)
    _SOURCE_WHEN_IMPORTED.setdefault(name, _codec_source(name).read_bytes())
    return codec


def _message(case: dict) -> str:
    """A vector's ``error`` is a literal substring, not a regex.

    ``pytest.raises(match=...)`` searches with ``re``, and several of these
    messages contain ``type()`` — which as a pattern matches the bare word
    ``type`` and then demands the following text immediately, so the assertion
    passed or failed for reasons unrelated to the message.
    """
    return re.escape(case["error"])


def _action_from_dict(name: str, value: dict):
    return importlib.import_module(f"grammars.{name}.codec").action_from_dict(value)


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
    """``format`` writes exactly one spelling, and it parses back to itself."""
    codec = _codec(name)
    reparsed = codec.parse(case["canonical"])
    assert codec.format(reparsed) == case["canonical"]
    assert reparsed == codec.parse(case["text"])


@pytest.mark.parametrize(("name", "payload", "case"), _cases("cases"))
def test_to_dict_round_trips_through_action_from_dict(name, payload, case):
    """``to_dict`` is the eval record, so it is executed here.

    ``agent._action_record`` serialises every parsed action by probing for a
    ``to_dict``, so this method's output is the ``parsed_action`` field of every
    trajectory row. Reached only through that dynamic probe, it — and
    ``Element.to_dict`` and ``Primitive.to_dict`` under it — never ran.
    """
    codec = _codec(name)
    parsed = codec.parse(case["text"])
    assert _action_from_dict(name, parsed.to_dict()) == parsed


@pytest.mark.parametrize(("name", "payload", "case"), _cases("format_only"))
def test_format_only(name, payload, case):
    codec = _codec(name)
    assert codec.format(_action_from_dict(name, case["action"])) == case["canonical"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_parse"))
def test_invalid_parse(name, payload, case):
    with pytest.raises(ValueError, match=_message(case)):
        _codec(name).parse(case["text"])


@pytest.mark.parametrize(("name", "payload", "case"), _cases("invalid_compile"))
def test_invalid_compile(name, payload, case):
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=_message(case)):
        _codec(name).compile(case["text"], geometry, cursor)


_MOVE_KINDS = ("move_to", "glide_to")

#: Per grammar: a move the display cannot honour, the same turn asking to stay
#: where it is, and whether ``compile`` still emits a move operation for the
#: first. The cursor is placed on the right edge, so both turns dispatch the
#: same thing in the six grammars that drop a zero-extent move — which is the
#: blind spot ``IntendedCursor`` exists to remove. Declared per grammar rather
#: than discovered at runtime, so a grammar that starts or stops dropping the
#: move fails here instead of quietly changing what an idle turn means.
_EDGE_OF_DISPLAY = {
    "compact_raw": ("5000 0 0", "0 0 0", False),
    "deltatype_v2": ("5000 0 0", "0 0 0", False),
    "diffabs": ("5000 0 0", "0 0 0", False),
    "compact_absolute": ("99999 540 0", "1919 540 0", False),
    "ordered_events_v3": ("move(5000,0)", "move(0,0)", False),
    "move_rel": (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "move_rel", "coordinate": [999, 0]}}\n</tool_call>',
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "move_rel", "coordinate": [0, 0]}}\n</tool_call>',
        False,
    ),
    "native_absolute": (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "mouse_move", "coordinate": [99999, 540]}}\n</tool_call>',
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "mouse_move", "coordinate": [1919, 540]}}\n</tool_call>',
        True,
    ),
}


@pytest.mark.parametrize(("name", "payload", "case"), _cases("cases"))
def test_intended_cursor_agrees_with_what_compile_dispatched(name, payload, case):
    """The anti-drift half: two functions resolve a coordinate, so they are tied.

    ``intended_cursor`` folds the same requests ``compile_action`` does, and its
    clamped result must therefore be the position the dispatched stream ends on.
    Nothing else would stop the two from drifting apart, and a drifted intent is
    worse than none: it would be read as the model's request.
    """
    codec = _codec(name)
    geometry, cursor = _context(payload, case)
    intent = codec.intended_cursor(codec.parse(case["text"]), geometry, cursor)
    moves = [
        operation
        for operation in codec.compile(case["text"], geometry, cursor)
        if operation.kind in _MOVE_KINDS
    ]
    if intent is None:
        assert not moves, "reported no cursor request, yet moved the pointer"
        return
    landed = _support.clamp((intent.x, intent.y), geometry)
    if moves:
        assert tuple(moves[-1].args)[:2] == landed
    else:
        assert landed == _support.clamp(cursor, geometry)
    if landed != (intent.x, intent.y):
        assert intent.clamped, "the target left the display and was not flagged"


@pytest.mark.parametrize("name", NAMES)
def test_a_clamped_move_is_distinguishable_from_asking_to_stay(name):
    """The measurement this exists for.

    Six of the seven grammars emit no operation when a resolved move does not
    change the position, so at the edge of the display a large delta and a delta
    of zero dispatch the same empty stream and land as the same ``no_op``. Every
    closed-loop "the policy stopped moving the mouse" reading is taken off that
    stream. ``clamped`` is what separates them.
    """
    clamped_text, stay_text, expect_move = _EDGE_OF_DISPLAY[name]
    codec = _codec(name)
    payload = _vectors(name)
    geometry = _geometry(payload["geometry"])
    edge = (geometry.desktop_width - 1, 540)

    refused = codec.intended_cursor(codec.parse(clamped_text), geometry, edge)
    stayed = codec.intended_cursor(codec.parse(stay_text), geometry, edge)
    assert refused is not None and stayed is not None
    assert refused.clamped is True
    assert stayed.clamped is False
    assert (refused.x, refused.y) != (stayed.x, stayed.y)
    assert refused.x > geometry.desktop_width - 1, "the request must be off-display"

    dispatched = codec.compile(clamped_text, geometry, edge)
    assert bool([o for o in dispatched if o.kind in _MOVE_KINDS]) is expect_move
    if not expect_move:
        assert dispatched == codec.compile(stay_text, geometry, edge), (
            "the two turns must be indistinguishable in the operation stream; "
            "that is the blind spot, and IntendedCursor is the only way out of it"
        )


@pytest.mark.parametrize(("name", "payload", "case"), _cases("lift"))
def test_lift(name, payload, case):
    """The full triangle. ``recompiled`` closes it; where it differs, the case says why."""
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
    if case["recompiled"] != case["operations"]:
        assert case.get("documents"), (
            "a lift that does not close byte-exactly must say what it loses"
        )


@pytest.mark.parametrize(("name", "payload", "case"), _cases("lift"))
def test_lift_is_a_fixpoint(name, payload, case):
    """Lifting the recompiled stream again reproduces the same text.

    A lossy lift is allowed; a lift that loses something different on the second
    pass is not, because a converter run over its own output would keep drifting.
    """
    codec = _codec(name)
    geometry, cursor = _context(payload, case)
    again = codec.action_from_operations(
        _operations(case["recompiled"]),
        geometry=geometry,
        cursor=cursor,
        terminate=case.get("terminate"),
    )
    assert codec.format(again) == case["canonical"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("lift_invalid"))
def test_lift_invalid(name, payload, case):
    """An expressiveness ceiling must raise, never flatten silently."""
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=_message(case)):
        _codec(name).action_from_operations(
            _operations(case["operations"]),
            geometry=geometry,
            cursor=cursor,
            terminate=case.get("terminate"),
        )


@pytest.mark.parametrize(("name", "payload", "case"), _cases("from_target"))
def test_from_target(name, payload, case):
    codec = _codec(name)
    geometry = _geometry(payload["geometry"])
    cursor = tuple(case["cursor"])
    target = tuple(case["target"])
    # The signatures differ: the relative arm needs a fresh cursor read, the
    # absolute arm does not.
    if name == "compact_raw":
        action = codec.from_target(cursor, target, elements=_elements(case))
    else:
        action = codec.from_target(target, elements=_elements(case))
    assert codec.format(action) == case["canonical"]
    if "operations" in case:
        assert (
            _rows(codec.compile(case["canonical"], geometry, cursor))
            == case["operations"]
        )


@pytest.mark.parametrize(("name", "payload", "case"), _cases("from_absolute"))
def test_from_absolute(name, payload, case):
    codec = _codec(name)
    geometry = _geometry(payload["geometry"])
    cursor = tuple(case["cursor"])
    action = codec.from_absolute(cursor, tuple(case["target"]), elements=_elements(case))
    assert codec.format(action) == case["canonical"]
    assert _rows(codec.compile(case["canonical"], geometry, cursor)) == case["operations"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("from_pixel_delta"))
def test_from_pixel_delta(name, payload, case):
    codec = _codec(name)
    geometry = _geometry(payload["geometry"])
    then = case.get("then")
    calls = codec.from_pixel_delta(
        tuple(case["delta"]),
        geometry,
        then=None if then is None else codec.validate_call(then),
    )
    assert [call.arguments() for call in calls] == case["calls"]


@pytest.mark.parametrize(("name", "payload", "case"), _cases("matched_pair"))
def test_matched_pair(name, payload, case):
    """One intent, two encodings, one operation sequence."""
    codec = _codec(name)
    twin = _codec(payload["paired_with"])
    geometry = _geometry(payload["geometry"])
    cursor = tuple(case["cursor"])
    target = tuple(case["target"])
    elements = (
        _support.Element("event", name="LMB", pressed=True),
        _support.Element("event", name="LMB", pressed=False),
    )
    assert codec.format(codec.from_target(target, elements=elements)) == (
        case["this_canonical"]
    )
    assert twin.format(twin.from_target(cursor, target, elements=elements)) == (
        case["compact_raw_canonical"]
    )
    absolute = _rows(codec.compile(case["this_canonical"], geometry, cursor))
    relative = _rows(twin.compile(case["compact_raw_canonical"], geometry, cursor))
    assert absolute == relative == case["operations"]


def test_every_section_is_executed():
    """No vector section may exist that no test above runs.

    Without this, adding a section to a JSON file silently adds nothing: it looks
    like coverage in the diff and is never executed.
    """
    executed = {
        "cases",
        "format_only",
        "invalid_parse",
        "invalid_compile",
        "lift",
        "lift_invalid",
        "from_target",
        "from_absolute",
        "from_pixel_delta",
        "matched_pair",
    }
    for name in NAMES:
        for section, value in _vectors(name).items():
            if not isinstance(value, list) or section in _NOT_CASES:
                continue
            assert section in executed, f"{name}: nothing executes section {section!r}"


@pytest.mark.parametrize("name", NAMES)
def test_codec_satisfies_the_protocol(name):
    """``isinstance(codec, Codec)`` for every registered grammar.

    ``Codec`` is ``@runtime_checkable``; a caller writing this gate got a false
    negative on every grammar while the protocol required a ``handlers`` table
    that described a dispatch engine desktop does not have.
    """
    codec = _codec(name)
    assert isinstance(codec, Codec)
    assert codec.name == name
    assert isinstance(codec.stop_sequences, tuple)


@pytest.mark.parametrize("name", NAMES)
def test_describe_is_deterministic(name):
    """``describe()`` IS the system prompt, so it may not vary between calls."""
    codec = _codec(name)
    first = codec.describe()
    assert all(codec.describe() == first for _ in range(5))
    for production in _support.productions(codec):
        assert production.syntax in first


@pytest.mark.parametrize("name", NAMES)
def test_prompt_digest_matches_its_pin(name):
    """``describe()`` is what a trained model saw, so its digest is pinned here.

    ``codec.digest`` is computed from ``describe()``, so comparing the two in
    code cannot fail. The pin lives in the vectors file — outside the module
    whose docstrings it measures — so editing any docstring that reaches the
    prompt turns this red. Rewrite the pin in the same commit as the edit, and
    the digest change is then reviewable as a line of the diff.

    The two sides are sampled at different instants — the rendering at import,
    the pin here — so a grammar edited while this suite runs used to be reported
    as a digest regression. A moving tree is refused instead, because a check
    that fails wrongly costs more than one that does not run.
    """
    codec = _codec(name)
    rendered = codec.digest
    source = _codec_source(name)
    assert source.read_bytes() == _SOURCE_WHEN_IMPORTED[name], (
        f"{source} was edited after this process imported it, so `rendered` "
        "measures the text as it was and the pin measures the text as it is. "
        "Neither side is wrong and this is not a digest regression: the "
        "comparison is void. Re-run against a tree nobody is editing."
    )
    assert rendered == _vectors(name)["prompt_sha256"]


@pytest.mark.parametrize("name", NAMES)
def test_report_never_raises(name):
    """A prompt digest is data; ``report()`` must not raise on drift."""
    report = _codec(name).report()
    assert report["grammar"] == name
    assert report["prompt_sha256"] == _codec(name).digest
    recorded = getattr(
        importlib.import_module(f"grammars.{name}.codec"), "PRODUCER", {}
    ).get("prompt_sha256")
    # None, never False, when there is nothing to compare against.
    assert report["matches_producer"] is (
        None if recorded is None else report["prompt_sha256"] == recorded
    )


#: A click where the cursor already is: the one operation stream all seven
#: grammars express exactly, so a terminating turn's work can be asserted
#: byte-for-byte in every grammar from one stream.
TERMINATING_WORK = (ir.mouse_down("left"), ir.mouse_up("left"))


@pytest.mark.parametrize("status", ["success", "failure"])
@pytest.mark.parametrize("carries_work", [True, False])
@pytest.mark.parametrize("name", NAMES)
def test_a_terminating_turn_round_trips_through_the_control_channel(
    name, carries_work, status
):
    """The label direction, the channel and the dispatch direction, in one pass.

    Both statuses in all seven grammars: ``diffabs`` and ``ordered_events_v3`` had
    TERMINATE and no FAIL, and ``compact_raw`` and ``compact_absolute``
    could not terminate at all, so four of the seven could not have passed this.
    """
    codec = _codec(name)
    payload = _vectors(name)
    geometry = _geometry(payload["geometry"])
    cursor = tuple(payload["default_cursor"])
    operations = TERMINATING_WORK if carries_work else ()

    action = codec.action_from_operations(
        operations, geometry=geometry, cursor=cursor, terminate=status
    )
    assert action.terminate == status
    # `to_dict` is the published `parsed_action`, so the status has to survive it.
    assert _action_from_dict(name, action.to_dict()) == action

    text = codec.format(action)
    assert text.splitlines()[-1] == f"{_support.CONTROL_TOKEN}: {status}"
    control = _support.split_control(text)
    assert control.status == status and control.ignored == 0
    assert _support.CONTROL_TOKEN not in control.body

    if not control.body:
        assert not operations, "work was dropped rather than spelled"
        return
    # Exactly the operations preceding the termination, and a parse that carries no
    # control of its own -- the channel is the only source.
    assert _rows(codec.compile(control.body, geometry, cursor)) == _rows(operations)
    assert codec.parse(control.body).terminate is None


@pytest.mark.parametrize("legacy", ["TERMINATE", "FAIL", "NO_OP\nTERMINATE"])
@pytest.mark.parametrize("name", NAMES)
def test_the_retired_control_tokens_are_loud_in_every_grammar(name, legacy):
    """What a checkpoint trained before the channel emits, and why it must be loud.

    A bare ``TERMINATE`` is neither the channel's line nor an action, so such a
    turn is a scored parse error. The alternative — keeping the old token as a
    second accepted spelling — is what made termination a per-grammar surface.
    """
    assert _support.split_control(legacy).status is None
    with pytest.raises(ValueError):
        _codec(name).parse(legacy)


@pytest.mark.parametrize("name", NAMES)
def test_every_prompt_carries_the_channel_and_no_grammar_of_its_own(name):
    described = _codec(name).describe()
    assert _support.CONTROL_SPEC in described
    # Twice: the two lines of CONTROL_SPEC and nowhere else.
    assert described.count(_support.CONTROL_TOKEN) == 2
    assert "FAIL" not in described


def _terminate_call(**arguments) -> str:
    return _support.render_tool_calls([{"action": "terminate", **arguments}])


def test_the_vendor_terminate_is_read_and_the_calls_after_it_are_counted():
    """``native_absolute`` conforms to a schema we do not own.

    An off-the-shelf Qwen3-VL emits the vendor's ``terminate`` call whatever the
    prompt says, and that model in that grammar is the only calibrated reference
    this program has, so the channel reads it. The codec does not: ``terminate``
    is not one of its actions, and the channel has already cut it out.
    """
    click = _support.render_tool_calls([{"action": "left_click"}])
    control = _support.split_control(
        "\n".join([click, _terminate_call(status="success"), click, click])
    )
    assert control.status == "success" and control.ignored == 2
    assert control.body == click
    assert "terminate" not in control.body

    codec = _codec("native_absolute")
    with pytest.raises(ValueError, match="terminate"):
        codec.parse(_terminate_call(status="success"))

    # Untagged, which is how the RL rollout path sees vLLM-parsed output.
    untagged = json.dumps(
        [
            {"name": "computer_use", "arguments": {"action": "left_click"}},
            {"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}},
        ]
    )
    control = _support.split_control(untagged)
    assert control.status == "failure" and control.ignored == 0
    assert codec.parse(control.body).calls[0].action == "left_click"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"status": "success"}, "success"),
        ({"status": "failure"}, "failure"),
        # The vendor's own default, and the one the 33.9% reference was measured
        # under. Not invented here, so it is not ours to change.
        ({}, "success"),
        # Present and unrecognised is refused rather than guessed: a status that
        # decays to success is how a lost failure becomes a claimed success.
        ({"status": "done"}, None),
        ({"status": "SUCCESS"}, None),
    ],
)
def test_the_vendor_status_is_adopted_but_never_guessed(arguments, expected):
    assert _support.split_control(_terminate_call(**arguments)).status == expected


def test_the_control_line_must_be_last_and_exact():
    """Strict by design: near-misses are parse errors, not terminations.

    A permissive channel is a regex over free text, which is what this whole
    overhaul removed. The status rides inside the token, so it cannot be lost.
    """
    assert _support.split_control("0 0 0\nTERMINATE: success").status == "success"
    assert _support.split_control("0 0 0\nTERMINATE: success\n\n").status == "success"
    for near_miss in (
        "TERMINATE: success\n0 0 0",
        "terminate: success",
        "TERMINATE: succes",
        "TERMINATE",
        "I will TERMINATE: success",
    ):
        control = _support.split_control(near_miss)
        assert control.status is None and control.body == near_miss


MATCHED_ARMS = ("compact_raw", "compact_absolute")


def test_matched_arms_share_their_prose_byte_for_byte():
    """Everything except the two mouse-triple productions must be identical.

    Both arms take this text from ``_support.MATCHED_ARM_*``; this guards against
    a reintroduced local copy, which is how the two came to differ by a line-wrap
    the first time.
    """
    first, second = (_codec(name) for name in MATCHED_ARMS)
    assert type(first).__doc__ == type(second).__doc__ == _support.MATCHED_ARM_PREAMBLE
    shared = {"_press", "_release", "_type", "notes"}
    for member in shared:
        assert getattr(type(first), member).__doc__ == (
            getattr(type(second), member).__doc__
        ), member
    differing = {
        production.member
        for arm in (first, second)
        for production in _support.productions(arm)
    } - shared
    assert differing == {"_mouse", "_with_events"}, differing


def test_matched_arms_agree_on_everything_but_the_coordinate():
    """The surface rules, mechanically: separator, vocabulary, control tokens."""
    first, second = (_codec(name) for name in MATCHED_ARMS)
    assert first.stop_sequences == second.stop_sequences
    probe = '0 0 0 ; +ControlLeft +KeyA -KeyA -ControlLeft type("x")'
    assert first.parse(probe).elements == second.parse(probe).elements
    assert _support.render_elements(
        (_support.Element("event", name="LMB", pressed=True),)
    ) == " ; +LMB"
    for codec in (first, second):
        for token in ("NO_OP", "TERMINATE", "FAIL"):
            with pytest.raises(ValueError):
                codec.parse(token)


@pytest.mark.parametrize("name", MATCHED_ARMS)
def test_matched_arms_declare_each_other(name):
    module = importlib.import_module(f"grammars.{name}.codec")
    other = module.PAIRED_WITH
    assert other in NAMES and other != name
    assert importlib.import_module(f"grammars.{other}.codec").PAIRED_WITH == name
    assert _codec(name).report()["paired_with"] == other


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


def test_every_canonical_kind_is_groupable():
    """Every kind in ``ir.CANONICAL_KINDS`` must reach a group.

    ``drag``, ``click`` and ``ascii_type`` are all kinds desktop's own executor
    handles — it synthesises ``click`` — so a stream containing one must not fall
    through to "unknown Operation kind", which would make it unliftable in every
    grammar simultaneously.
    """
    expected = set(ir.CANONICAL_KINDS) - {"raise_for_test"}
    assert set(CANONICAL_PROBES) == expected, "probe table is out of date"
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    for kind, probe in CANONICAL_PROBES.items():
        groups = _support.group_operations(
            (probe,), geometry=geometry, cursor=(960, 540)
        )
        assert groups, kind


@pytest.mark.parametrize("kind", ["drag", "click", "ascii_type"])
def test_the_recovered_kinds_reach_a_grammar_that_can_express_them(kind):
    """Grouping is not enough: some grammar must lift each kind to real text."""
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    able = {}
    for name in NAMES:
        codec = _codec(name)
        try:
            action = codec.action_from_operations(
                (CANONICAL_PROBES[kind],), geometry=geometry, cursor=(960, 540)
            )
        except ValueError:
            continue  # a genuine ceiling; it raised, which is the contract
        able[name] = codec.format(action)
    assert able, f"no grammar can lift {kind!r}"
    if kind == "drag":
        # The press and the release survive and the stroke stays inside them,
        # rather than being degraded into a stationary click. Asserted on the
        # parsed primitives, not on substring positions -- the approach move to
        # the drag's start point also spells `move(`, and it correctly precedes
        # the press.
        assert "deltatype_v2" in able and "ordered_events_v3" in able
        assert "MOVE(" in able["deltatype_v2"]
        assert "left_click_drag" in able["native_absolute"]
        primitives = _codec("ordered_events_v3").parse(
            able["ordered_events_v3"]
        ).primitives
        kinds = [item.kind for item in primitives]
        press, release = kinds.index("down"), kinds.index("up")
        assert any(
            item.kind == "move" for item in primitives[press + 1 : release]
        ), f"the stroke is not held inside the button: {able['ordered_events_v3']}"


def test_a_zero_extent_drag_keeps_its_press_and_release():
    """``ir.drag(x, y, x, y)`` exists precisely so this cannot be optimised away."""
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    for name in ("deltatype_v2", "ordered_events_v3", "native_absolute"):
        codec = _codec(name)
        action = codec.action_from_operations(
            (ir.drag(900, 500, 900, 500),), geometry=geometry, cursor=(960, 540)
        )
        recompiled = codec.compile(codec.format(action), geometry, (960, 540))
        kinds = [item.kind for item in recompiled]
        assert "mouse_down" in kinds and "mouse_up" in kinds, (name, kinds)


SCREEN_SIZES = [(1920, 1080), (1280, 720), (2560, 1440), (4000, 2000), (3840, 2160)]


@pytest.mark.parametrize(("width", "height"), SCREEN_SIZES)
@pytest.mark.parametrize("axis", [0, 1])
def test_move_rel_quantisation_ceiling(width, height, axis):
    """Pin the region the vectors avoid.

    ``move_rel`` encodes a delta as thousandths of the screen, so most pixel
    deltas do not round-trip, and the vectors were chosen from the ones that do.

    Two things are asserted for every delta on every axis:

    * a delta whose normalized value is zero while its pixel value is not raises,
      per axis and not only when both axes vanish. At 4000 wide a ``(1, 100)`` px
      move became ``[0, 50]``.
    * otherwise the recompiled pixel lies inside the grid's own tolerance,
      ``dimension / 2000 + 0.5`` — half a grid step from rounding the encode plus
      half a pixel from rounding the decode.
    """
    from .move_rel.codec import GRID, MoveRelError, norm_from_pixels

    codec = _codec("move_rel")
    geometry = DisplayGeometry(desktop_width=width, desktop_height=height)
    dimension = (width, height)[axis]
    tolerance = dimension / (2 * GRID) + 0.5
    # Centred, and every target checked to be on-screen: a clamped move lands
    # wherever the edge is, which is not a quantisation error and would make this
    # sweep measure clamping instead.
    origin = (width // 2, height // 2)
    checked = lossy = raised = 0

    for pixels in list(range(-40, 41)) + [77, 123, 456, -456]:
        delta = [0, 0]
        delta[axis] = pixels
        target = (origin[0] + delta[0], origin[1] + delta[1])
        if not (0 <= target[0] < width and 0 <= target[1] < height):
            continue
        stream = (ir.move_to(*target),)
        vanishes = norm_from_pixels(pixels, dimension) == 0 and pixels != 0
        if vanishes:
            with pytest.raises(MoveRelError, match="finer than the"):
                codec.action_from_operations(
                    stream, geometry=geometry, cursor=origin
                )
            raised += 1
            continue
        if pixels == 0:
            continue  # a zero-extent move has no representation; covered elsewhere
        checked += 1
        action = codec.action_from_operations(stream, geometry=geometry, cursor=origin)
        recompiled = codec.compile(codec.format(action), geometry, origin)
        landed = tuple(recompiled[-1].args)
        error = landed[axis] - target[axis]
        assert abs(error) <= tolerance, (
            f"{pixels} px on axis {axis} of {dimension} landed {error} px off, "
            f"outside the {tolerance:.2f} px grid tolerance"
        )
        if error:
            lossy += 1

    assert checked, "the sweep checked nothing"
    # Where the ceiling falls decides which screens a relative label is exact on:
    #
    #   dimension <= 1000  the grid is finer than pixels, so every delta gets its
    #                      own thousandth and the round trip is exact. No loss.
    #   dimension >  1000  thousandths are coarser than pixels; some deltas share
    #                      a grid value and come back on the wrong pixel.
    #   dimension >= 2000  a one-pixel delta normalises to zero and must raise.
    #                      At exactly 2000 it is ``round(0.5)``, which Python
    #                      rounds to even and therefore to zero, so the boundary
    #                      is inclusive.
    #
    # 1080p is already above the first threshold on both axes, so the ceiling
    # applies to every screen this program trains on.
    if dimension > GRID:
        assert lossy, f"axis {axis} of {dimension} lost nothing; grid changed?"
    else:
        assert not lossy, (
            f"axis {axis} of {dimension} is finer than the {GRID}ths grid, so "
            "every delta must round-trip exactly"
        )
    if dimension >= 2 * GRID:
        assert raised, f"axis {axis} of {dimension} should have sub-grid deltas"
    else:
        assert not raised, (
            f"nothing can vanish on axis {axis} of {dimension}: one pixel is at "
            f"least one {GRID}th of it"
        )


def test_move_rel_sub_grid_guard_is_per_axis():
    """The sub-grid guard fires per axis, not only when both axes vanish."""
    from .move_rel.codec import MoveRelError

    codec = _codec("move_rel")
    geometry = DisplayGeometry(desktop_width=4000, desktop_height=2000)
    for delta in ((1, 100), (100, 1), (1, 1)):
        with pytest.raises(MoveRelError, match="finer than the"):
            codec.action_from_operations(
                (ir.move_to(100 + delta[0], 100 + delta[1]),),
                geometry=geometry,
                cursor=(100, 100),
            )


def test_importing_grammars_imports_neither_a_codec_nor_desktop():
    """``available()`` reads metadata and imports nothing — including ``desktop``.

    That is what lets ``_explain_desktop`` report a wrong or missing install as
    one, after listing all seven names. Re-exporting the control channel eagerly
    from ``_support`` (which needs ``desktop``) turns ``import grammars`` itself
    into a bare ``No module named 'desktop.geometry'``, so the re-export is lazy
    and this asserts it in a fresh interpreter.
    """
    root = Path(grammars.__file__).parent.parent
    probe = (
        "import sys, grammars\n"
        "names = grammars.available()\n"
        "assert 'desktop' not in sys.modules, sorted(sys.modules)\n"
        "print(len(names))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        env={"PYTHONPATH": str(root), "PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(len(NAMES))


def test_entry_points_alone_discover_every_grammar():
    """Discovery must not depend on the source tree sitting next to the caller."""
    assert set(grammars._from_entry_points()) == set(NAMES)
    original = grammars._from_directories
    cached = dict(grammars._CACHE)
    try:
        grammars._from_directories = dict
        grammars._CACHE.clear()
        assert set(grammars.available()) == set(NAMES)
        for name in NAMES:
            assert grammars.load(name).name == name
    finally:
        grammars._from_directories = original
        grammars._CACHE.clear()
        grammars._CACHE.update(cached)


@pytest.mark.parametrize(
    "dropped",
    [
        "native_rel_v1",
        "native_rel_think",
        # `compact_absolute`'s former id. Both halves of it were wrong: it is not
        # the native tool-call grammar, and "control" meant control ARM while the
        # grammar has no control tokens. It must not resolve to `native_absolute`,
        # whose name it contains.
        "native_absolute_control",
    ],
)
def test_a_dropped_grammar_fails_loudly(dropped):
    """A retired id must not resolve to a neighbour that happens to be close."""
    assert dropped not in grammars.available()
    for call in (grammars.load, grammars.describe):
        with pytest.raises(KeyError, match=dropped):
            call(dropped)


def test_the_installed_desktop_is_ours():
    """The dependency is resolved by path and has no index presence at all.

    The substitution plain ``pip`` makes is still available under this name:
    PyPI's ``desktop`` 0.4.2 owns the same distribution and import name for a
    different package, and ``pip`` does not read ``[tool.uv.sources]``. So does a
    stale wheel or a leftover ``desktop_env.egg-info``. Asserting the version and
    the submodules is what tells ours apart from any of them.
    """
    import desktop

    assert desktop.__version__ == "0.1.0"
    for member in ("codec_protocol", "geometry", "ir"):
        importlib.import_module(f"desktop.{member}")


def test_the_import_name_no_longer_collides_with_osworlds():
    """Importing ours must not put anything under ``desktop_env`` in ``sys.modules``.

    ``evals/osworld.py`` imports OSWorld's ``desktop_env.controllers.setup`` and
    ``desktop_env.evaluators`` to score the benchmark. While this workspace owned
    that import name, that import resolved to our already-imported module and no
    ``$OSWORLD_ROOT`` could fix it: ``sys.path`` cannot override an entry already
    in ``sys.modules``.
    """
    import desktop  # noqa: F401
    import desktop.geometry  # noqa: F401
    import desktop.ir  # noqa: F401

    ours = pathlib.Path(sys.modules["desktop"].__file__).resolve().parent
    assert ours.name == "desktop"
    stolen = sys.modules.get("desktop_env")
    if stolen is not None:  # OSWorld may legitimately be imported by something else
        assert ours not in pathlib.Path(stolen.__file__).resolve().parents


def test_a_wrong_desktop_is_explained_not_just_reported():
    """The guard in ``load()``, without needing the wrong package installed.

    A bare ``No module named 'desktop.geometry'`` sends the reader looking
    for a missing file when the real fault is a wrong or missing install — and
    ``available()`` lists all seven beforehand regardless, because it reads
    metadata and imports nothing.
    """
    explained = grammars._explain_desktop(
        ImportError("No module named 'desktop.geometry'", name="desktop.geometry")
    )
    message = str(explained)
    assert "desktop.geometry" in message
    assert "[tool.uv.sources]" in message
    assert "uv pip install -e ../desktop" in message
    # A leftover desktop_env egg-info produces exactly this error.
    assert "xlang-ai/desktop_env" in message

    # And it must not dress up an unrelated ImportError as a packaging problem.
    original = grammars._from_entry_points
    cached = dict(grammars._CACHE)
    try:
        grammars._CACHE.clear()
        grammars._from_entry_points = lambda: {"broken": "grammars._absent:CODEC"}
        with pytest.raises(ImportError) as caught:
            grammars.load("broken")
        assert "xlang-ai" not in str(caught.value)
    finally:
        grammars._from_entry_points = original
        grammars._CACHE.clear()
        grammars._CACHE.update(cached)


def test_no_handler_table_comes_back():
    """No grammar may reintroduce a ``handlers.py`` dispatch table.

    Each grammar exported one, describing "the dispatch table it contributes to
    desktop's engine". No such engine existed, and the ``Handler`` those tables
    were annotated with runs in the opposite direction. Lowering an ``Operation``
    belongs in desktop, over a closed kind vocabulary.
    """
    root = Path(grammars.__file__).parent
    assert not list(root.glob("*/handlers.py"))
    assert not hasattr(grammars, "handlers")
    assert not hasattr(grammars, "handler_report")
    assert not hasattr(_support, "core_handlers")
