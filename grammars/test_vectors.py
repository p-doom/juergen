"""The conformance gate: every vector in every ``grammars/*/vectors/*.json``.

This module is the reason the vectors are worth having. They existed for a while
with nothing in the repository that executed them — 1,456 lines of pinned
behaviour, checked only by whatever throwaway script the last person wrote — and
a vector nobody runs is a comment. Each case below is one parametrised test, so a
regression names the grammar, the section and the case.

Run it with ``pytest grammars/`` (``pip install -e '.[dev]'`` for pytest).

Four invariants that are NOT expressible as vectors are asserted here as code,
each because it was found broken:

* ``isinstance(codec, Codec)`` for all seven. The protocol is
  ``@runtime_checkable``, so that is a gate a caller may legitimately write, and
  it returned False for every grammar while ``Codec`` demanded a ``handlers``
  table no codec had and a ``stop_sequences`` method no codec had.
* The matched pair's shared prose is byte-identical. It had drifted by a
  line-wrap, which is a different token sequence in the two arms of an A/B whose
  whole premise is byte-equal prose.
* Every canonical Operation kind can be grouped. ``drag``, ``click`` and
  ``ascii_type`` used to fall through to "unknown Operation kind", which made any
  recorded trajectory containing them unliftable in all seven grammars at once.
* The normalized grammar's quantisation ceiling. The vectors deliberately use
  deltas that round-trip exactly, so the lossy region — most of it — was pinned
  nowhere, and that region is precisely where a relative label is silently wrong.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

import grammars
from desktop_env import ir
from desktop_env.codec_protocol import Codec
from desktop_env.geometry import DisplayGeometry

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
    """Vector geometry -> ``DisplayGeometry``, using desktop-env's own field names.

    The vectors used to say ``width``/``height`` at the top level and
    ``desktop_width``/``desktop_height`` in the one per-case override — residue of
    the stub these were written against, and the exact drift that made the first
    real integration run interesting. One spelling now, and it is the package's.
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


def _codec(name: str):
    return grammars.load(name)


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


# --------------------------------------------------------------------------
# the three directions: parse · format · compile
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# the lift: operations -> action -> text -> operations
# --------------------------------------------------------------------------


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

    A lossy lift is allowed; a lift that loses something DIFFERENT on the second
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
    """An expressiveness ceiling must RAISE, never flatten silently."""
    geometry, cursor = _context(payload, case)
    with pytest.raises(ValueError, match=_message(case)):
        _codec(name).action_from_operations(
            _operations(case["operations"]),
            geometry=geometry,
            cursor=cursor,
            terminate=case.get("terminate"),
        )


# --------------------------------------------------------------------------
# the label-construction helpers each grammar owns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "payload", "case"), _cases("from_target"))
def test_from_target(name, payload, case):
    codec = _codec(name)
    geometry = _geometry(payload["geometry"])
    cursor = tuple(case["cursor"])
    target = tuple(case["target"])
    # The signatures differ on purpose and that difference is the measurement:
    # the relative arm needs a fresh cursor read, the absolute arm does not.
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
    """One intent, two encodings, ONE operation sequence."""
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


# --------------------------------------------------------------------------
# invariants that are not vector-shaped
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_codec_satisfies_the_protocol(name):
    """The seam, actually checked.

    ``Codec`` is ``@runtime_checkable``; a caller writing this gate got a false
    negative on every grammar while the protocol required a ``handlers`` table
    that described a dispatch engine desktop-env does not have.
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
    assert _support.spec_digest(first) == codec.digest
    for production in _support.productions(codec):
        assert production.syntax in first


@pytest.mark.parametrize("name", NAMES)
def test_report_never_raises(name):
    """A prompt digest is data. The predecessor raised on drift; nothing may."""
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


MATCHED_ARMS = ("compact_raw", "native_absolute_control")


def test_matched_arms_share_their_prose_byte_for_byte():
    """Everything except the two mouse-triple productions must be identical.

    Both arms take this text from ``_support.MATCHED_ARM_*``, so this test is a
    guard against someone reintroducing a local copy — which is how the two came
    to differ by a line-wrap the first time.
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
    """A converter does not get to choose what a recording contains.

    ``drag``, ``click`` and ``ascii_type`` are all kinds desktop-env's own
    executor handles — it SYNTHESISES ``click`` — and all three used to fall
    through to "unknown Operation kind", which made any stream containing one
    unliftable in every grammar simultaneously.
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
        # The whole reason ir.drag is its own kind: the press and the release
        # survive, and the stroke stays INSIDE them rather than being degraded
        # into a stationary click. Asserted on the parsed primitives, not on
        # substring positions -- the approach move to the drag's start point also
        # spells `move(`, and it correctly precedes the press.
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


# --------------------------------------------------------------------------
# the normalized grammar's quantisation ceiling
# --------------------------------------------------------------------------

SCREEN_SIZES = [(1920, 1080), (1280, 720), (2560, 1440), (4000, 2000), (3840, 2160)]


@pytest.mark.parametrize(("width", "height"), SCREEN_SIZES)
@pytest.mark.parametrize("axis", [0, 1])
def test_move_rel_quantisation_ceiling(width, height, axis):
    """Pin the region the vectors deliberately avoid.

    ``move_rel`` encodes a delta as thousandths of the screen, so most pixel
    deltas do NOT round-trip — and the vectors were chosen from the ones that do,
    which left the lossy region, the one where a relative label is quietly wrong,
    pinned nowhere at all.

    Two things are asserted for every delta on every axis:

    * a delta whose normalized value is zero while its pixel value is not RAISES,
      per axis and not only when both axes vanish. Silently zeroing one axis was a
      real defect: at 4000 wide a ``(1, 100)`` px move became ``[0, 50]``.
    * otherwise the recompiled pixel lies inside the grid's own tolerance,
      ``dimension / 2000 + 0.5`` — half a grid step from rounding the encode plus
      half a pixel from rounding the decode. Anything wider than that is a bug in
      the conversion, not the grid.
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
    # Where the ceiling exists at all is itself worth pinning, because it is not
    # obvious and it decides which screens a relative label is exact on:
    #
    #   dimension <= 1000  the grid is FINER than pixels, so every delta gets its
    #                      own thousandth and the round trip is exact. No loss.
    #   dimension >  1000  thousandths are coarser than pixels; some deltas share
    #                      a grid value and come back on the wrong pixel.
    #   dimension >= 2000  a one-pixel delta normalises to zero and MUST raise.
    #                      At exactly 2000 it is ``round(0.5)``, which Python
    #                      rounds to even and therefore to zero, so the boundary
    #                      is inclusive -- worth pinning rather than reasoning
    #                      about at the next screen-size change.
    #
    # 1080p is already above the first threshold on both axes, so the "documented
    # ceiling" applies to every screen this program actually trains on.
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
    """The regression this file exists to prevent, stated once, minimally."""
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


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


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


@pytest.mark.parametrize("dropped", ["native_rel_v1", "native_rel_think"])
def test_a_dropped_grammar_fails_loudly(dropped):
    """A retired id must not resolve to a neighbour that happens to be close."""
    assert dropped not in grammars.available()
    for call in (grammars.load, grammars.describe):
        with pytest.raises(KeyError, match=dropped):
            call(dropped)


def test_the_installed_desktop_env_is_ours():
    """The dependency is resolved by path, and the name is contested on PyPI.

    ``desktop-env`` on PyPI is xlang-ai/desktop_env (OSWorld) — same import name,
    different package — so an installer that does not read
    ``[tool.uv.sources]`` (plain ``pip``) silently installs the wrong one. This
    asserts the right one is present, so the suite cannot pass against a
    ``desktop_env`` that merely happens to expose what these tests touch.
    """
    import desktop_env

    assert desktop_env.__version__ == "0.1.0"
    for member in ("codec_protocol", "geometry", "ir"):
        importlib.import_module(f"desktop_env.{member}")


def test_a_wrong_desktop_env_is_explained_not_just_reported():
    """The guard in ``load()``, without needing the wrong package installed.

    A bare ``No module named 'desktop_env.geometry'`` sends the reader looking
    for a missing file when the real fault is a wrong package — and
    ``available()`` lists all seven beforehand regardless, because it reads
    metadata and imports nothing.
    """
    explained = grammars._explain_desktop_env(
        ImportError("No module named 'desktop_env.geometry'", name="desktop_env.geometry")
    )
    message = str(explained)
    assert "xlang-ai/desktop_env" in message
    assert "desktop_env.geometry" in message
    assert "uv pip install -e ../desktop-env" in message

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
    """The deleted fiction, pinned deleted.

    Each grammar exported a ``handlers.py`` describing "the dispatch table it
    contributes to desktop-env's engine". No such engine existed, and the
    ``Handler`` those tables were annotated with runs in the opposite direction.
    Lowering an ``Operation`` belongs in desktop-env, over a kind vocabulary that
    is closed by physics.
    """
    root = Path(grammars.__file__).parent
    assert not list(root.glob("*/handlers.py"))
    assert not hasattr(grammars, "handlers")
    assert not hasattr(grammars, "handler_report")
    assert not hasattr(_support, "core_handlers")
