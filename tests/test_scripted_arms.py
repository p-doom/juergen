"""Scripted arms as `Intent` + per-step rendering.

The renderer table is exact, not a substring heuristic: `compact_raw` and
`compact_absolute` share a prefix while meaning opposite things, `native_absolute`
and `compact_absolute` both contain "absolute", and `move_rel` contains neither
"compact" nor "absolute" while being relative.

One renderer emits `compact_absolute`'s bare-line absolute form, where
`0 0 0 ; +LMB -LMB` is a click at the top-left corner while the same bytes in
`compact_raw` mean "don't move, click here". That pair is why the table has to be
exact: the two surfaces are byte-identical, so picking the wrong one of those two
clicks somewhere else on the screen instead of failing.

Rendering happens per step, because the relative arms resolve a click against a
cursor read that must be fresh.
"""

from __future__ import annotations

import json

import pytest

from agent.agent import load_codec
from evals.signoflife.cells import ARMS
from evals.signoflife.guest import (
    DOCK_CHROME_COORDINATE,
    SCRIPT_RENDERERS,
    Intent,
    render_step,
    script_plan,
)
from evals.signoflife.suite import load_suite
from juergen_doubles import FakeSession, make_task_data

GEOMETRY_JSON = "SOLV2_GEOMETRY=" + json.dumps(
    {"window_id": "0x1", "x": 100, "y": 200, "width": 800, "height": 600, "window_line": "x"}
)


def _session(cursor=(0, 0)) -> FakeSession:
    return FakeSession(cursor=cursor, argv_responses={"python3": GEOMETRY_JSON})


def _task(kind: str, **kwargs):
    expected = {
        "terminal_command": {"command": "ls", "listing_marker": "m.txt"},
        "terminal_exact_text": {"text": "hello there"},
        "open_chrome": {"active_window_class_any": ["chrome"]},
        "focus_terminal_and_type": {"command": "printf x > /tmp/p", "file": "/tmp/p", "content": "x"},
    }[kind]
    return make_task_data(kind=kind, name=f"cell_{kind}", expected=expected, **kwargs)


def _geo():
    from desktop.geometry import DisplayGeometry

    return DisplayGeometry(desktop_width=1920, desktop_height=1080)


def test_the_renderer_table_is_exact_and_covers_every_arm_with_a_model_leg() -> None:
    """Every grammar `cells.ARMS` names must have a scripted renderer.

    Asserted against `ARMS` rather than a literal set: a model arm added without
    its oracle and negative pair produces an uncalibrated number, and a control
    arm without a renderer cannot run at all.
    """
    assert set(SCRIPT_RENDERERS) == {
        "native_absolute",
        "compact_absolute",
        "deltatype_v2",
        "compact_raw",
        "ordered_events_v3",
    }
    assert {arm.codec for arm in ARMS.values()} <= set(SCRIPT_RENDERERS)


def test_a_prefix_grammar_name_gets_its_own_renderer_not_the_others() -> None:
    """A substring heuristic collides on both halves of `compact_absolute`'s name.

    On "compact" it takes `compact_raw`'s relative renderer, which emits the same
    bytes for a different position and so clicks silently in the wrong place; on
    "absolute" it takes `native_absolute`'s tool-call renderer, which at least
    fails loudly.
    """
    assert SCRIPT_RENDERERS["compact_raw"] is not SCRIPT_RENDERERS["compact_absolute"]
    assert SCRIPT_RENDERERS["native_absolute"] is not SCRIPT_RENDERERS["compact_absolute"]
    assert SCRIPT_RENDERERS["deltatype_v2"] is SCRIPT_RENDERERS["compact_raw"], (
        "the two raw-relative grammars share one renderer, which is correct"
    )


@pytest.mark.parametrize("missing", ["move_rel", "diffabs"])
def test_a_grammar_with_no_scripted_arm_is_a_loud_lookup_error(missing: str) -> None:
    """A missing renderer must never be a silently substituted grammar."""
    with pytest.raises(LookupError, match="no scripted control arm"):
        render_step(_session(), _task("terminal_command"), codec=load_codec(missing), intent=Intent("submit"))


def test_an_unnamed_codec_object_is_refused() -> None:
    with pytest.raises(LookupError):
        render_step(_session(), _task("terminal_command"), codec=object(), intent=Intent("submit"))


def test_a_plan_is_intents_not_text() -> None:
    for cell in load_suite().tasks:
        task = _task(cell.kind)
        for negative in (False, True):
            plan = script_plan(task, negative=negative)
            assert plan and all(isinstance(i, Intent) for i in plan)
            assert all(i.kind in {"click", "type", "submit"} for i in plan)


def test_the_gold_plans_match_the_cells() -> None:
    assert [i.kind for i in script_plan(_task("terminal_command"), negative=False)] == [
        "type",
        "submit",
    ]
    assert [i.kind for i in script_plan(_task("open_chrome"), negative=False)] == ["click"]
    assert [i.kind for i in script_plan(_task("focus_terminal_and_type"), negative=False)] == [
        "click",
        "type",
        "submit",
    ]


def test_the_gold_plan_types_the_expected_command_and_the_negative_types_pwd() -> None:
    gold = script_plan(_task("terminal_command"), negative=False)
    bad = script_plan(_task("terminal_command"), negative=True)
    assert gold[0].text == "ls" and bad[0].text == "pwd"


def test_the_compound_negative_omits_the_focus_click() -> None:
    """Typing without focusing first: a real action that cannot succeed."""
    bad = script_plan(_task("focus_terminal_and_type"), negative=True)
    assert [i.kind for i in bad] == ["type", "submit"]


def test_the_chrome_negative_clicks_screen_centre_not_the_dock() -> None:
    bad = script_plan(_task("open_chrome"), negative=True)
    assert bad[0].target == (960, 540)
    gold = script_plan(_task("open_chrome"), negative=False)
    assert gold[0].target is None, "the gold target is resolved from the guest at render time"


def test_an_unknown_kind_is_refused_by_the_planner() -> None:
    with pytest.raises(ValueError):
        script_plan(make_task_data(kind="not_a_cell"), negative=False)


def test_native_absolute_renders_tool_calls_with_absolute_pixels() -> None:
    codec = load_codec("native_absolute")
    task = _task("open_chrome")
    text = render_step(_session(), task, codec=codec, intent=Intent("click", target=(300, 400)))
    payload = json.loads(text.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
    assert payload["name"] == "computer_use"
    assert payload["arguments"] == {"action": "left_click", "coordinate": [300, 400]}
    codec.parse(text)  # must round-trip through the real parser


def test_native_absolute_renders_type_and_submit() -> None:
    codec = load_codec("native_absolute")
    task = _task("terminal_command")
    typed = render_step(_session(), task, codec=codec, intent=Intent("type", text='say "hi"'))
    assert json.loads(typed.split("<tool_call>\n")[1].split("\n</tool_call>")[0])["arguments"] == {
        "action": "type",
        "text": 'say "hi"',
    }
    submit = render_step(_session(), task, codec=codec, intent=Intent("submit"))
    assert json.loads(submit.split("<tool_call>\n")[1].split("\n</tool_call>")[0])["arguments"] == {
        "action": "key",
        "keys": ["ENTER"],
    }
    codec.parse(typed)
    codec.parse(submit)


def test_the_native_absolute_renderer_emits_bare_line_ABSOLUTE_coordinates() -> None:
    """`compact_absolute` reads no cursor at all."""
    codec = load_codec("compact_absolute")
    task = _task("open_chrome")
    for cursor in [(0, 0), (900, 900), (1919, 1079)]:
        text = render_step(
            _session(cursor), task, codec=codec, intent=Intent("click", target=(300, 400))
        )
        assert text == "300 400 0 ; +LMB -LMB", (
            "the absolute arm needs only element geometry, so nothing can go stale"
        )
    codec.parse(text)


def test_the_relative_renderer_resolves_the_click_against_the_live_cursor() -> None:
    codec = load_codec("deltatype_v2")
    task = _task("open_chrome")
    assert (
        render_step(_session((100, 100)), task, codec=codec, intent=Intent("click", target=(300, 400)))
        == "200 300 0 ; +LMB -LMB"
    )
    assert (
        render_step(_session((250, 450)), task, codec=codec, intent=Intent("click", target=(300, 400)))
        == "50 -50 0 ; +LMB -LMB"
    )


def test_the_same_bytes_mean_different_actions_in_the_paired_arms() -> None:
    """`0 0 0 ; +LMB -LMB` is a click at the top-left corner in the absolute arm and
    "don't move, click here" in `compact_raw`, so a control arm must be rendered per
    grammar and never copied between them.
    """
    bytes_ = "0 0 0 ; +LMB -LMB"
    geometry = _geo()
    cursor = (640, 480)
    absolute = load_codec("compact_absolute").compile(bytes_, geometry, cursor)
    relative = load_codec("compact_raw").compile(bytes_, geometry, cursor)
    absolute_moves = [tuple(op.args[:2]) for op in absolute if op.kind in ("move_to", "glide_to")]
    relative_moves = [tuple(op.args[:2]) for op in relative if op.kind in ("move_to", "glide_to")]
    assert absolute_moves == [(0, 0)], absolute_moves
    assert relative_moves in ([], [cursor]), relative_moves
    assert absolute_moves != relative_moves or not relative_moves


def test_a_stale_cursor_read_is_exactly_the_drift_in_the_relative_arm() -> None:
    codec = load_codec("deltatype_v2")
    task = _task("open_chrome")
    fresh = render_step(_session((100, 100)), task, codec=codec, intent=Intent("click", target=(300, 400)))
    stale = render_step(_session((110, 90)), task, codec=codec, intent=Intent("click", target=(300, 400)))
    assert fresh == "200 300 0 ; +LMB -LMB"
    assert stale == "190 310 0 ; +LMB -LMB"
    absolute = load_codec("compact_absolute")
    a = render_step(_session((100, 100)), task, codec=absolute, intent=Intent("click", target=(300, 400)))
    b = render_step(_session((110, 90)), task, codec=absolute, intent=Intent("click", target=(300, 400)))
    assert a == b, "the absolute arm is immune to the same drift"


@pytest.mark.parametrize("codec_name", ["deltatype_v2", "compact_raw", "compact_absolute"])
def test_the_bare_line_renderers_json_escape_the_typed_text(codec_name: str) -> None:
    codec = load_codec(codec_name)
    task = _task("terminal_exact_text")
    text = render_step(_session(), task, codec=codec, intent=Intent("type", text='a "quoted" ; semi'))
    assert text == '0 0 0 ; type("a \\"quoted\\" ; semi")'
    parsed = codec.parse(text)
    from agent.agent import _action_record

    from evals.indicators import typed_texts

    assert typed_texts(_action_record(parsed)) == ['a "quoted" ; semi'], (
        "the separator inside the payload must survive the round trip"
    )


@pytest.mark.parametrize("codec_name", ["deltatype_v2", "compact_raw", "compact_absolute"])
def test_the_bare_line_renderers_submit_with_a_real_key_transition(codec_name: str) -> None:
    text = render_step(_session(), _task("terminal_command"), codec=load_codec(codec_name), intent=Intent("submit"))
    assert text == "0 0 0 ; +Return -Return"
    assert "\\n" not in text, "never a literal escape — that is indicator A's defect"


def test_an_unknown_intent_kind_is_refused_by_every_renderer() -> None:
    for name in SCRIPT_RENDERERS:
        with pytest.raises(ValueError):
            render_step(_session(), _task("terminal_command"), codec=load_codec(name), intent=Intent("teleport"))


def test_the_chrome_click_target_comes_from_the_dock_constant() -> None:
    codec = load_codec("compact_absolute")
    text = render_step(_session(), _task("open_chrome"), codec=codec, intent=Intent("click"))
    x, y = DOCK_CHROME_COORDINATE
    assert text == f"{x} {y} 0 ; +LMB -LMB"


def test_a_terminal_click_target_is_recomputed_from_guest_geometry() -> None:
    """Not cached on `self`: one `Preparer` instance serves every concurrent rollout."""
    codec = load_codec("compact_absolute")
    session = _session()
    text = render_step(session, _task("focus_terminal_and_type"), codec=codec, intent=Intent("click"))
    assert text == f"{100 + 800 // 2} {200 + 100} 0 ; +LMB -LMB"
    assert any("wmctrl" in " ".join(argv) for argv in session.argv_log), (
        "the coordinate is read from the guest, not from cached setup evidence"
    )


@pytest.mark.parametrize("codec_name", sorted(SCRIPT_RENDERERS))
@pytest.mark.parametrize("negative", [False, True])
def test_every_scripted_step_parses_and_compiles_through_the_real_codec(
    codec_name: str, negative: bool
) -> None:
    """The controls use the same `parse` and `compile` the model arm does, which is
    what makes 4/4 and 0/4 comparable to it."""
    codec = load_codec(codec_name)
    geometry = _geo()
    for cell in load_suite().tasks:
        task = _task(cell.kind)
        session = _session((640, 480))
        for intent in script_plan(task, negative=negative):
            text = render_step(session, task, codec=codec, intent=intent)
            action = codec.parse(text)
            operations = list(codec.compile(text, geometry, session.cursor))
            assert action is not None
            if intent.kind != "submit" or operations:
                pass
            assert all(hasattr(op, "kind") for op in operations), (codec_name, text)


@pytest.mark.parametrize("codec_name", sorted(SCRIPT_RENDERERS))
def test_a_scripted_click_compiles_to_a_press_and_a_release(codec_name: str) -> None:
    codec = load_codec(codec_name)
    task = _task("open_chrome")
    session = _session((640, 480))
    text = render_step(session, task, codec=codec, intent=Intent("click", target=(300, 400)))
    kinds = [op.kind for op in codec.compile(text, _geo(), session.cursor)]
    assert kinds.count("mouse_down") == 1 and kinds.count("mouse_up") == 1


@pytest.mark.parametrize("codec_name", sorted(SCRIPT_RENDERERS))
def test_a_scripted_click_lands_on_the_target_in_absolute_pixels(codec_name: str) -> None:
    """Whatever the encoding, `compile` must resolve to the same screen pixel."""
    codec = load_codec(codec_name)
    task = _task("open_chrome")
    session = _session((640, 480))
    text = render_step(session, task, codec=codec, intent=Intent("click", target=(300, 400)))
    moves = [
        tuple(op.args[:2])
        for op in codec.compile(text, _geo(), session.cursor)
        if op.kind in ("move_to", "glide_to")
    ]
    assert moves == [(300, 400)], (codec_name, text, moves)
