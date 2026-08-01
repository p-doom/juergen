"""Tests for the ordered_events_v2/v3 eval path: parser + VM dispatch.

The parser is the inverse of ``action_format.OrderedFormatter`` /
``OrderedTypingFormatter``; the round-trip property against the real renderer
is gated in data_pipeline/tests/test_action_format.py (which can import the
formatter). Here we pin the grammar itself and the dispatch ordering, both of
which are what makes the format worth having: ``move -> click -> move`` in one
turn must reach the VM as three ordered operations.
"""

from __future__ import annotations

import ast
import json
import sys
import types
import unittest
from pathlib import Path


def _install_import_stubs() -> None:
    """Let these helper tests run without syncing the heavy eval venv."""
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        image = types.ModuleType("PIL.Image")
        image.Image = object
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image
    if "requests" not in sys.modules:
        requests = types.ModuleType("requests")
        requests.RequestException = Exception
        requests.Session = lambda: None
        requests.get = lambda *args, **kwargs: None
        requests.post = lambda *args, **kwargs: None
        sys.modules["requests"] = requests


sys.path.insert(0, str(Path(__file__).resolve().parent))
_install_import_stubs()

import freeroll  # noqa: E402
import osworld_vm_client  # noqa: E402
from action_parser import (  # noqa: E402
    parse_ordered_action,
    parse_ordered_action_tolerant,
)
from osworld_system_prompts import (  # noqa: E402
    ACTION_FORMAT_AGGREGATE,
    ACTION_FORMAT_ORDERED,
    SYSTEM_PROMPTS,
    action_format_for_prompt,
)


def _kinds(line: str) -> list[str]:
    return [p.kind for p in parse_ordered_action(line).primitives]


class OrderedParserTests(unittest.TestCase):
    def test_no_op_yields_no_primitives(self) -> None:
        action = parse_ordered_action("NO_OP")
        self.assertTrue(action.no_op)
        self.assertEqual(action.primitives, ())

    def test_move_click_move_preserves_order(self) -> None:
        # The whole reason the format exists: the aggregate grammar cannot
        # express motion on both sides of a click.
        self.assertEqual(
            _kinds("move(4,-1); down(LMB); move(2,0); up(LMB)"),
            ["move", "down", "move", "up"],
        )

    def test_move_and_scroll_carry_signed_axes(self) -> None:
        (move,) = parse_ordered_action("move(-100,250)").primitives
        self.assertEqual((move.dx, move.dy), (-100, 250))
        (scroll,) = parse_ordered_action("scroll(-3,7)").primitives
        self.assertEqual((scroll.dx, scroll.dy), (-3, 7))

    def test_mouse_buttons_resolve_to_x11_codes(self) -> None:
        prims = parse_ordered_action(
            "down(LMB); down(MMB); down(RMB); down(KeyA)"
        ).primitives
        self.assertEqual([p.mouse_button for p in prims], [1, 2, 3, None])

    def test_key_events_projection_drops_motion_and_typing(self) -> None:
        action = parse_ordered_action(
            'move(5,5); down(ShiftLeft); type("hi"); up(ShiftLeft)'
        )
        self.assertEqual(
            [(e.kind, e.what) for e in action.key_events],
            [("press", "ShiftLeft"), ("release", "ShiftLeft")],
        )

    def test_left_click_detection(self) -> None:
        action = parse_ordered_action("move(1,1); down(LMB); up(LMB)")
        self.assertTrue(action.has_left_click_press)
        self.assertTrue(action.has_left_click_release)
        self.assertFalse(parse_ordered_action("down(RMB)").has_left_click_press)

    # ---------------------------------------------------------- type()
    def test_type_payload_is_unescaped(self) -> None:
        (prim,) = parse_ordered_action(r'type("say \"hi\"")').primitives
        self.assertEqual(prim.text, 'say "hi"')
        (prim,) = parse_ordered_action(r'type("back\\slash")').primitives
        self.assertEqual(prim.text, "back\\slash")

    def test_type_payload_may_contain_the_primitive_separator(self) -> None:
        # A line is NEVER safely split on "; " -- it must be scanned.
        prims = parse_ordered_action('move(2,3); type("a; b"); up(LMB)').primitives
        self.assertEqual([p.kind for p in prims], ["move", "type", "up"])
        self.assertEqual(prims[1].text, "a; b")

    def test_type_payload_may_contain_escaped_quote_before_separator(self) -> None:
        (prim,) = parse_ordered_action(r'type("a\"; b")').primitives
        self.assertEqual(prim.text, 'a"; b')

    def test_empty_type_payload_is_rejected(self) -> None:
        # The grammar requires chars = 1*char.
        with self.assertRaises(ValueError):
            parse_ordered_action('type("")')

    # ---------------------------------------------------------- rejection
    def test_aggregate_format_is_rejected(self) -> None:
        # An aggregate-trained checkpoint pointed at the ordered parser must
        # surface as a parse error, not be silently reinterpreted.
        with self.assertRaises(ValueError):
            parse_ordered_action("100 -50 0 ; +LMB -LMB")

    def test_malformed_lines_are_rejected(self) -> None:
        for bad in (
            "",
            "move(0)",
            "move(1,2) down(LMB)",   # missing separator
            "move(1,2);",            # dangling separator
            "move(1,2); ; up(LMB)",  # empty primitive
            "frobnicate(1)",
            "TERMINATE",             # intercepted upstream, never parsed here
            "move(1.5,2)",           # ints only
        ):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                parse_ordered_action(bad)

    def test_non_string_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            parse_ordered_action(None)

    # ---------------------------------------------------------- tolerance
    def test_whitespace_inside_primitives_is_tolerated(self) -> None:
        self.assertEqual(
            parse_ordered_action("move( 4 , -1 )").primitives,
            parse_ordered_action("move(4,-1)").primitives,
        )

    def test_trailing_chatter_after_newline_is_ignored(self) -> None:
        self.assertEqual(_kinds("move(4,-1)\n<eos>"), ["move"])

    def test_tolerant_parser_finds_action_marker_after_reasoning(self) -> None:
        text = "Reasoning: the icon is left.\nAction: move(-20,0); down(LMB); up(LMB)"
        self.assertEqual(
            [p.kind for p in parse_ordered_action_tolerant(text).primitives],
            ["move", "down", "up"],
        )

    def test_tolerant_parser_falls_back_to_last_nonblank_line(self) -> None:
        text = "I should click the button.\n\nmove(3,4); down(LMB); up(LMB)"
        self.assertEqual(
            [p.kind for p in parse_ordered_action_tolerant(text).primitives],
            ["move", "down", "up"],
        )

    def test_tolerant_parser_still_rejects_action_buried_mid_sentence(self) -> None:
        with self.assertRaises(ValueError):
            parse_ordered_action_tolerant("I will move(3,4) now and see.")

    def test_render_round_trips_through_the_parser(self) -> None:
        for line in (
            "NO_OP",
            "move(4,-1); down(LMB); up(LMB)",
            "scroll(0,-60)",
            'move(-100,250); type("ls -la"); down(Return); up(Return)',
            r'type("quote \" and back\\slash")',
            'type("semi; colon")',
        ):
            with self.subTest(line=line):
                self.assertEqual(parse_ordered_action(line).render(), line)


class _FakeClient(osworld_vm_client.OSWorldClient):
    """OSWorldClient with the HTTP boundary replaced by a command recorder.

    Deliberately skips OSWorldClient.__init__ (no session, no base_url):
    dispatch_ordered only touches cursor_position / screen_size / execute and
    the model_resolution scaling helper.
    """

    def __init__(self, *, screen=(1920, 1080), cursor=(100, 100),
                 model_resolution=None) -> None:
        self.model_resolution = model_resolution
        self._screen = screen
        self._cursor = cursor
        self.commands: list[str] = []

    def cursor_position(self):
        return self._cursor

    def screen_size(self):
        return self._screen

    def execute(self, command: str) -> None:
        self.commands.append(command)


class OrderedDispatchTests(unittest.TestCase):
    def _dispatch(self, line: str, **kwargs) -> tuple[_FakeClient, object]:
        client = _FakeClient(**kwargs)
        result = client.dispatch_ordered(parse_ordered_action(line))
        return client, result

    def test_no_op_touches_nothing(self) -> None:
        client, result = self._dispatch("NO_OP")
        self.assertEqual(client.commands, [])
        self.assertEqual(result.delta, (0, 0))
        self.assertEqual(result.events_dispatched, [])

    def test_primitives_dispatch_in_emitted_order(self) -> None:
        client, _ = self._dispatch("move(10,20); down(LMB); move(5,0); up(LMB)")
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(110, 120)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.moveTo(115, 120)",
            "pyautogui.mouseUp(button='left')",
        ])

    def test_moves_accumulate_from_the_running_cursor(self) -> None:
        # Second move is relative to where the first one landed, not to the
        # cursor position read at the start of the step.
        client, result = self._dispatch("move(10,10); move(10,10)")
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(110, 110)",
            "pyautogui.moveTo(120, 120)",
        ])
        self.assertEqual(result.intended_target, (120, 120))
        self.assertEqual(result.delta, (20, 20))

    def test_moves_clip_to_screen_bounds(self) -> None:
        client, result = self._dispatch(
            "move(-9999,-9999)", screen=(1920, 1080), cursor=(100, 100))
        self.assertEqual(client.commands, ["pyautogui.moveTo(0, 0)"])
        self.assertEqual(result.intended_target, (0, 0))
        # Reported delta is what was APPLIED, not what was emitted.
        self.assertEqual(result.delta, (-100, -100))

    def test_zero_delta_move_emits_nothing(self) -> None:
        # move(0,0) is never rendered by the formatter, but a model may emit it.
        client, _ = self._dispatch("move(0,0)")
        self.assertEqual(client.commands, [])

    def test_scroll_axes_map_to_scroll_and_hscroll(self) -> None:
        client, result = self._dispatch("scroll(3,-5)")
        self.assertEqual(client.commands, [
            "pyautogui.scroll(-5)",
            "pyautogui.hscroll(3)",
        ])
        self.assertEqual(result.scroll, -5)

    def test_scroll_skips_zero_axes(self) -> None:
        client, _ = self._dispatch("scroll(0,-5)")
        self.assertEqual(client.commands, ["pyautogui.scroll(-5)"])

    def test_key_chord_presses_and_releases_in_emitted_order(self) -> None:
        client, _ = self._dispatch(
            "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)")
        self.assertEqual(client.commands, [
            "pyautogui.keyDown('ctrlleft')",
            "pyautogui.keyDown('c')",
            "pyautogui.keyUp('c')",
            "pyautogui.keyUp('ctrlleft')",
        ])

    def test_type_dispatches_a_single_write(self) -> None:
        client, _ = self._dispatch('type("ls -la")')
        self.assertEqual(client.commands, ["pyautogui.write('ls -la', interval=0)"])

    def test_type_payload_is_embedded_as_a_python_literal(self) -> None:
        # Quotes/backslashes must survive as data, not break the python -c body.
        client, _ = self._dispatch(r'type("say \"hi\"; then go")')
        self.assertEqual(len(client.commands), 1)
        cmd = client.commands[0]
        # The command is python source run as `python -c` in the VM, so the
        # payload must be a valid literal that evaluates back to the text.
        literal = cmd[len("pyautogui.write("):-len(", interval=0)")]
        self.assertEqual(ast.literal_eval(literal), 'say "hi"; then go')

    def test_model_resolution_scales_move_deltas_only(self) -> None:
        # Model sees 1280x720, VM runs 1920x1080 -> 1.5x on both axes.
        client, result = self._dispatch(
            "move(10,20); scroll(0,-5)",
            screen=(1920, 1080), cursor=(0, 0), model_resolution=(1280, 720),
        )
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(15, 30)",
            "pyautogui.scroll(-5)",  # scroll units are not pixels: unscaled
        ])
        self.assertEqual(result.delta, (15, 30))

    def test_typing_after_a_click_reaches_the_vm_in_order(self) -> None:
        # The v3 shape stage 04 actually emits for "click a field and type".
        client, _ = self._dispatch(
            'move(50,60); down(LMB); up(LMB); type("hello"); down(Return); up(Return)')
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(150, 160)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.mouseUp(button='left')",
            "pyautogui.write('hello', interval=0)",
            "pyautogui.keyDown('enter')",
            "pyautogui.keyUp('enter')",
        ])


class ActionFormatRoutingTests(unittest.TestCase):
    def test_ordered_prompts_select_the_ordered_format(self) -> None:
        for prompt_id in (
            "yll_ordered_v1",
            "yll_ordered_v1_no_goal",
            "cua_ordered_v1",
            "cua_ordered_typing_v1",
        ):
            with self.subTest(prompt_id=prompt_id):
                self.assertIn(prompt_id, SYSTEM_PROMPTS)
                self.assertEqual(
                    action_format_for_prompt(prompt_id), ACTION_FORMAT_ORDERED)

    def test_aggregate_prompts_are_unaffected(self) -> None:
        for prompt_id in ("training_v1", "yll_v1", "cua_v1", "cot_directions_v1"):
            with self.subTest(prompt_id=prompt_id):
                self.assertEqual(
                    action_format_for_prompt(prompt_id), ACTION_FORMAT_AGGREGATE)

    def test_unknown_prompt_defaults_to_aggregate(self) -> None:
        self.assertEqual(
            action_format_for_prompt("not_a_prompt"), ACTION_FORMAT_AGGREGATE)

    def test_every_ordered_prompt_is_registered(self) -> None:
        # Guards the pairing the prompts' own docstrings promise: an "ordered"
        # prompt whose reply contract is the mini-program must not silently
        # fall through to the aggregate parser.
        for prompt_id in SYSTEM_PROMPTS:
            if "ordered" in prompt_id:
                with self.subTest(prompt_id=prompt_id):
                    self.assertEqual(
                        action_format_for_prompt(prompt_id),
                        ACTION_FORMAT_ORDERED,
                    )


class ParseAndDispatchRoutingTests(unittest.TestCase):
    """freeroll._parse_and_dispatch: the right parser and the right VM path."""

    def test_ordered_format_routes_to_ordered_dispatch(self) -> None:
        client = _FakeClient()
        parsed, result = freeroll._parse_and_dispatch(
            client, "move(10,20); down(LMB); up(LMB)", ACTION_FORMAT_ORDERED)
        self.assertEqual([p["kind"] for p in parsed["primitives"]],
                         ["move", "down", "up"])
        self.assertFalse(parsed["no_op"])
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(110, 120)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.mouseUp(button='left')",
        ])
        self.assertEqual(result.events_dispatched, client.commands)

    def test_ordered_parsed_record_is_json_safe(self) -> None:
        client = _FakeClient()
        parsed, _ = freeroll._parse_and_dispatch(
            client, 'move(1,2); type("hi"); down(Return); up(Return)',
            ACTION_FORMAT_ORDERED)
        json.dumps(parsed)  # must not raise -- it lands in trajectory.jsonl
        self.assertEqual(parsed["primitives"][1], {"kind": "type", "text": "hi"})

    def test_ordered_clicks_are_visible_to_stop_on_click(self) -> None:
        client = _FakeClient()
        parsed, _ = freeroll._parse_and_dispatch(
            client, "move(5,5); down(LMB); up(LMB)", ACTION_FORMAT_ORDERED)
        self.assertTrue(freeroll._is_left_click(parsed))
        parsed, _ = freeroll._parse_and_dispatch(
            client, "move(5,5); down(RMB); up(RMB)", ACTION_FORMAT_ORDERED)
        self.assertFalse(freeroll._is_left_click(parsed))

    def test_ordered_no_op_dispatches_nothing(self) -> None:
        client = _FakeClient()
        parsed, _ = freeroll._parse_and_dispatch(
            client, "NO_OP", ACTION_FORMAT_ORDERED)
        self.assertTrue(parsed["no_op"])
        self.assertEqual(client.commands, [])

    def test_ordered_format_rejects_aggregate_replies(self) -> None:
        # A format violation must raise (counted as a parse error upstream)
        # and leave the VM untouched.
        client = _FakeClient()
        with self.assertRaises(ValueError):
            freeroll._parse_and_dispatch(
                client, "100 -50 0 ; +LMB -LMB", ACTION_FORMAT_ORDERED)
        self.assertEqual(client.commands, [])

    def test_aggregate_format_is_unchanged(self) -> None:
        # Regression guard for every existing recipe: same parser, same
        # dispatch, same parsed record shape as before this change.
        client = _FakeClient()
        parsed, _ = freeroll._parse_and_dispatch(
            client, "10 20 0 ; +LMB -LMB", ACTION_FORMAT_AGGREGATE)
        self.assertEqual(parsed["dx"], 10)
        self.assertEqual(parsed["dy"], 20)
        self.assertEqual(parsed["scroll"], 0)
        self.assertEqual([e["what"] for e in parsed["events"]], ["LMB", "LMB"])
        self.assertNotIn("primitives", parsed)
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(110, 120)",
            "pyautogui.mouseDown(button='left')",
            "pyautogui.mouseUp(button='left')",
        ])

    def test_aggregate_format_still_accepts_tool_calls(self) -> None:
        # The aggregate branch keeps its historical two-shape tolerance.
        client = _FakeClient()
        parsed, _ = freeroll._parse_and_dispatch(
            client,
            '<tool_call>{"name": "computer_use", '
            '"arguments": {"action": "key", "keys": ["Escape"]}}</tool_call>',
            ACTION_FORMAT_AGGREGATE)
        self.assertEqual(parsed["computer_use"]["action"], "key")

    def test_ordered_format_does_not_accept_tool_calls(self) -> None:
        # Strict by design: an ordered-trained checkpoint emitting JSON is a
        # real failure, not something to reinterpret.
        client = _FakeClient()
        with self.assertRaises(ValueError):
            freeroll._parse_and_dispatch(
                client,
                '<tool_call>{"name": "computer_use", '
                '"arguments": {"action": "key", "keys": ["Escape"]}}</tool_call>',
                ACTION_FORMAT_ORDERED)


if __name__ == "__main__":
    unittest.main()
