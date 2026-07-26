from __future__ import annotations

import ast
import re
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
    OrderedPrimitive,
    parse_computer_use_action,
    parse_computer_use_action_tolerant,
    parse_ordered_action,
    parse_ordered_action_tolerant,
)


def _tc(arguments_json: str) -> str:
    """One <tool_call> block for a computer_use arguments object."""
    return (
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": ' + arguments_json + "}\n"
        "</tool_call>"
    )


class FreerollHelperTests(unittest.TestCase):
    def test_parse_instructions_splits_nonempty_noncomment_lines(self) -> None:
        self.assertEqual(
            freeroll._parse_instructions("first\n\n# skip\nsecond\n"),
            ["first", "second"],
        )

    def test_parse_instructions_preserves_legacy_empty_behavior(self) -> None:
        self.assertEqual(freeroll._parse_instructions(None), [None])
        self.assertEqual(freeroll._parse_instructions("\n# only comments"), [None])

    def test_rdev_key_mapping_covers_typing_punctuation_and_digits(self) -> None:
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("KeyA"), "a")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Digit1"), "1")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Comma"), ",")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Quote"), "'")


class OrderedActionParserTests(unittest.TestCase):
    def test_parses_click_program(self) -> None:
        a = parse_ordered_action("move(12,-4); down(LMB); up(LMB)")
        self.assertFalse(a.no_op)
        self.assertEqual(
            a.primitives,
            (
                OrderedPrimitive(kind="move", dx=12, dy=-4),
                OrderedPrimitive(kind="down", name="LMB", mouse_button=1),
                OrderedPrimitive(kind="up", name="LMB", mouse_button=1),
            ),
        )
        self.assertTrue(a.has_left_click_press)

    def test_keyboard_names_have_no_mouse_button(self) -> None:
        a = parse_ordered_action("down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)")
        self.assertEqual([p.name for p in a.primitives],
                         ["ControlLeft", "KeyC", "KeyC", "ControlLeft"])
        self.assertTrue(all(p.mouse_button is None for p in a.primitives))
        self.assertFalse(a.has_left_click_press)

    def test_scroll_keeps_both_axes_and_signs(self) -> None:
        a = parse_ordered_action("scroll(0,-3)")
        self.assertEqual(a.primitives, (OrderedPrimitive(kind="scroll", dx=0, dy=-3),))

    def test_no_op(self) -> None:
        a = parse_ordered_action("NO_OP\n")
        self.assertTrue(a.no_op)
        self.assertEqual(a.primitives, ())

    def test_type_payload_may_contain_separator_and_parens(self) -> None:
        a = parse_ordered_action('type("a; b(c), d"); down(Return); up(Return)')
        self.assertEqual(
            [p.kind for p in a.primitives], ["type", "down", "up"]
        )
        self.assertEqual(a.primitives[0].text, "a; b(c), d")

    def test_type_unescapes_quote_and_backslash(self) -> None:
        a = parse_ordered_action(r'type("say \"hi\" to C:\\tmp")')
        self.assertEqual(a.primitives[0].text, 'say "hi" to C:\\tmp')

    def test_round_trip_through_grammar_escaping(self) -> None:
        # Re-render the parsed program and get byte-identical text back.
        line = r'move(3,4); type("a; \"b\" \\ c"); down(Return); up(Return)'
        a = parse_ordered_action(line)

        def render(p: OrderedPrimitive) -> str:
            if p.kind in ("move", "scroll"):
                return f"{p.kind}({p.dx},{p.dy})"
            if p.kind == "type":
                esc = p.text.replace("\\", "\\\\").replace('"', '\\"')
                return f'type("{esc}")'
            return f"{p.kind}({p.name})"

        self.assertEqual("; ".join(render(p) for p in a.primitives), line)

    def test_rejects_empty_and_blank(self) -> None:
        for bad in ("", "   ", "\n\n"):
            with self.assertRaises(ValueError):
                parse_ordered_action(bad)

    def test_rejects_malformed_loudly(self) -> None:
        bad_inputs = [
            "move(1)",                       # arity
            "move(1,2,3)",                   # arity
            "move(1.5,2)",                   # non-int
            "down()",                        # empty name
            "down(LM B)",                    # whitespace in name
            "hover(1,2)",                    # unknown primitive
            "move(1,2) extra",               # trailing junk
            "move(1,2);; down(LMB)",         # empty primitive
            'type(unquoted)',                # missing quotes
            'type("")',                      # empty payload (grammar: 1*char)
            'type("a\\qb")',                 # invalid escape
            'type("dangling\\',              # escape at end of line
            "TERMINATE",                     # not an action primitive
            "hello world",
        ]
        for bad in bad_inputs:
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                parse_ordered_action(bad)

    def test_rejects_truncated_trailing_primitive(self) -> None:
        # A max_tokens cut mid-primitive must reject the WHOLE line — never
        # dispatch the complete prefix (down without up -> key-repeat flood).
        for bad in ("move(3,4); down(LM", 'move(3,4); type("hel', "down(LMB); up("):
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                parse_ordered_action(bad)

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(TypeError):
            parse_ordered_action(None)  # type: ignore[arg-type]

    def test_tolerant_takes_last_nonblank_line(self) -> None:
        a = parse_ordered_action_tolerant(
            "I will click the button.\n\nmove(5,6); down(LMB); up(LMB)"
        )
        self.assertEqual(a.primitives[0], OrderedPrimitive(kind="move", dx=5, dy=6))

    def test_tolerant_still_rejects_prose_only(self) -> None:
        with self.assertRaises(ValueError):
            parse_ordered_action_tolerant("no action here")

    def test_v2_lines_parse_with_same_parser(self) -> None:
        a = parse_ordered_action("move(4,-1); down(LMB); move(2,0); up(LMB)")
        self.assertEqual([p.kind for p in a.primitives], ["move", "down", "move", "up"])


class ThinkStripTests(unittest.TestCase):
    def test_leading_think_block_is_stripped(self) -> None:
        self.assertEqual(
            freeroll._strip_think("<think>plan the click</think>\nmove(1,2)"),
            "move(1,2)",
        )

    def test_strips_through_first_close_only(self) -> None:
        self.assertEqual(
            freeroll._strip_think("<think>a</think>\nNO_OP\n</think>"),
            "NO_OP\n</think>",
        )

    def test_dangling_close_without_opener_is_stripped(self) -> None:
        # Legacy checkpoints: the template injected <think>, so the reply
        # starts mid-thought and only carries the closer.
        self.assertEqual(
            freeroll._strip_think("finishing the thought</think>\nNO_OP"),
            "NO_OP",
        )

    def test_no_think_markers_passes_through(self) -> None:
        self.assertEqual(freeroll._strip_think("move(1,2); down(LMB)"),
                         "move(1,2); down(LMB)")

    def test_unterminated_think_yields_no_action_content(self) -> None:
        self.assertEqual(freeroll._strip_think("<think>never closed"), "")


class TerminateDetectionTests(unittest.TestCase):
    def test_bare_terminate(self) -> None:
        self.assertTrue(freeroll._is_terminate("TERMINATE"))
        self.assertTrue(freeroll._is_terminate(" TERMINATE\nignored"))

    def test_terminate_with_preceding_thought(self) -> None:
        self.assertTrue(freeroll._is_terminate(
            "<think>The reply is in the sent thread — done.</think>\nTERMINATE"
        ))

    def test_terminate_as_last_line_after_action(self) -> None:
        self.assertTrue(freeroll._is_terminate("down(Return); up(Return)\nTERMINATE"))

    def test_action_alone_is_not_terminate(self) -> None:
        self.assertFalse(freeroll._is_terminate("NO_OP"))
        self.assertFalse(freeroll._is_terminate("move(1,2)"))
        self.assertFalse(freeroll._is_terminate("<think>keep going</think>\nmove(1,2)"))


class TruncationGuardTests(unittest.TestCase):
    def test_length_finish_reason_dispatches_nothing(self) -> None:
        clean, err = freeroll._dispatch_plan("move(3,4); down(LM", "length")
        self.assertIsNone(clean)
        self.assertIsNotNone(err)
        self.assertIn("truncated", err)
        self.assertIn("nothing dispatched", err)

    def test_stop_finish_reason_strips_think_and_passes(self) -> None:
        clean, err = freeroll._dispatch_plan("<think>go</think>\nmove(1,2)", "stop")
        self.assertIsNone(err)
        self.assertEqual(clean, "move(1,2)")

    def test_missing_finish_reason_passes(self) -> None:
        clean, err = freeroll._dispatch_plan("NO_OP", None)
        self.assertIsNone(err)
        self.assertEqual(clean, "NO_OP")


class InstructionPlumbingTests(unittest.TestCase):
    def test_goal_conditioned_first_turn_is_goal_prefixed(self) -> None:
        self.assertEqual(
            freeroll._instruction_text("open firefox", goal_conditioned=True),
            "GOAL: open firefox",
        )

    def test_goal_block_has_no_so_far_at_episode_start(self) -> None:
        text = freeroll._instruction_text("open firefox", goal_conditioned=True)
        self.assertNotIn("So far:", text)

    def test_other_prompts_keep_verbatim_instruction(self) -> None:
        self.assertEqual(
            freeroll._instruction_text("open firefox", goal_conditioned=False),
            "open firefox",
        )

    def test_none_instruction_passes_through(self) -> None:
        self.assertIsNone(freeroll._instruction_text(None, goal_conditioned=True))

    def test_persist_reanchors_same_goal_text_every_step(self) -> None:
        instr = freeroll._instruction_text("open firefox", goal_conditioned=True)
        self.assertEqual(
            freeroll._instruction_for_step(instr, 1, persist_instruction=True),
            "GOAL: open firefox",
        )
        self.assertEqual(
            freeroll._instruction_for_step(instr, 7, persist_instruction=True),
            "GOAL: open firefox",
        )

    def test_no_persist_reverts_to_step_one_only(self) -> None:
        instr = freeroll._instruction_text("open firefox", goal_conditioned=True)
        self.assertEqual(
            freeroll._instruction_for_step(instr, 1, persist_instruction=False),
            "GOAL: open firefox",
        )
        self.assertIsNone(
            freeroll._instruction_for_step(instr, 2, persist_instruction=False)
        )

    def test_cua_v3_thinking_prompt_is_registered(self) -> None:
        self.assertIn("cua_v3_thinking", freeroll.SYSTEM_PROMPTS)
        self.assertIn("GOAL:", freeroll.SYSTEM_PROMPTS["cua_v3_thinking"])


class ActionFormatSelectionTests(unittest.TestCase):
    def test_cua_v3_thinking_defaults_to_ordered_v3(self) -> None:
        self.assertEqual(
            freeroll._resolve_action_format(None, "cua_v3_thinking"),
            "ordered_events_v3",
        )

    def test_legacy_prompts_default_to_canonical(self) -> None:
        for prompt_id in ("training_v1", "yll_v1", "cua_v1_thinking"):
            self.assertEqual(
                freeroll._resolve_action_format(None, prompt_id), "canonical"
            )

    def test_explicit_flag_wins(self) -> None:
        self.assertEqual(
            freeroll._resolve_action_format("ordered_events_v2", "training_v1"),
            "ordered_events_v2",
        )

    def test_unknown_explicit_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            freeroll._resolve_action_format("bogus", "training_v1")


class OrderedDispatchHelperTests(unittest.TestCase):
    def test_scale_ordered_moves_scales_move_only(self) -> None:
        a = parse_ordered_action('move(100,-50); scroll(0,-3); type("hi")')
        scaled = freeroll._scale_ordered_moves(a, 2.0, 1.5)
        self.assertEqual(scaled.primitives[0],
                         OrderedPrimitive(kind="move", dx=200, dy=-75))
        self.assertEqual(scaled.primitives[1],
                         OrderedPrimitive(kind="scroll", dx=0, dy=-3))
        self.assertEqual(scaled.primitives[2].text, "hi")

    def test_is_left_click_sees_ordered_down_lmb(self) -> None:
        parsed = {
            "no_op": False,
            "primitives": [
                {"kind": "move", "dx": 1, "dy": 2, "name": None, "text": None},
                {"kind": "down", "dx": None, "dy": None, "name": "LMB", "text": None},
                {"kind": "up", "dx": None, "dy": None, "name": "LMB", "text": None},
            ],
        }
        self.assertTrue(freeroll._is_left_click(parsed))
        parsed_no_click = {
            "no_op": False,
            "primitives": [
                {"kind": "type", "dx": None, "dy": None, "name": None, "text": "x"},
            ],
        }
        self.assertFalse(freeroll._is_left_click(parsed_no_click))

    def test_type_write_command_round_trips_payload(self) -> None:
        payload = 'say "hi"; C:\\path (weird), done'
        cmd = osworld_vm_client._type_write_command(payload)
        m = re.fullmatch(r"pyautogui\.write\((.*), interval=0\)", cmd)
        self.assertIsNotNone(m)
        self.assertEqual(ast.literal_eval(m.group(1)), payload)


class ComputerUseRelParserTests(unittest.TestCase):
    """computer_use_rel_v1 (cua_v4_thinking contract) parsing."""

    def test_single_call_round_trip(self) -> None:
        a = parse_computer_use_action(
            _tc('{"action": "mouse_move_rel", "delta": [30, -12]}')
        )
        self.assertFalse(a.no_op)
        self.assertEqual(
            a.primitives, (OrderedPrimitive(kind="move", dx=30, dy=-12),)
        )

    def test_multi_call_preserves_order(self) -> None:
        text = "\n".join([
            _tc('{"action": "mouse_move_rel", "delta": [30, -12]}'),
            _tc('{"action": "left_click"}'),
            _tc('{"action": "type", "text": "hi"}'),
            _tc('{"action": "key", "keys": ["ctrl", "s"]}'),
        ])
        a = parse_computer_use_action(text)
        self.assertEqual(
            [p.kind for p in a.primitives], ["move", "click", "type", "key_combo"]
        )
        self.assertEqual(a.primitives[1], OrderedPrimitive(kind="click", name="left", count=1))
        self.assertEqual(a.primitives[2].text, "hi")
        self.assertEqual(a.primitives[3].keys, ("ctrl", "s"))
        self.assertTrue(a.has_left_click_press)

    def test_leading_think_block_is_stripped(self) -> None:
        a = parse_computer_use_action(
            "<think>aim for the icon center</think>\n"
            + _tc('{"action": "left_click"}')
        )
        self.assertEqual(a.primitives[0].kind, "click")

    def test_terminate_call_carries_status(self) -> None:
        for status in ("success", "failure"):
            a = parse_computer_use_action(
                _tc('{"action": "terminate", "status": "%s"}' % status)
            )
            self.assertEqual(
                a.primitives, (OrderedPrimitive(kind="terminate", status=status),)
            )

    def test_whitespace_variants_inside_and_between_blocks(self) -> None:
        text = (
            '<tool_call>   {"name": "computer_use", "arguments":'
            ' {"action": "wait", "time": 1.5}}   </tool_call>'
            "\n\n   \n"
            "<tool_call>{\"name\": \"computer_use\", \"arguments\":"
            " {\"action\": \"scroll\", \"pixels\": -3}}</tool_call>\n"
        )
        a = parse_computer_use_action(text)
        self.assertEqual([p.kind for p in a.primitives], ["wait", "scroll"])
        self.assertEqual(a.primitives[1], OrderedPrimitive(kind="scroll", dx=0, dy=-3))

    def test_hscroll_maps_to_scroll_dx(self) -> None:
        a = parse_computer_use_action(_tc('{"action": "hscroll", "pixels": 40.6}'))
        self.assertEqual(a.primitives, (OrderedPrimitive(kind="scroll", dx=41, dy=0),))

    def test_click_family_buttons_and_counts(self) -> None:
        cases = {
            "left_click": ("left", 1),
            "right_click": ("right", 1),
            "middle_click": ("middle", 1),
            "double_click": ("left", 2),
            "triple_click": ("left", 3),
        }
        for action, (button, count) in cases.items():
            a = parse_computer_use_action(_tc('{"action": "%s"}' % action))
            self.assertEqual(
                a.primitives,
                (OrderedPrimitive(kind="click", name=button, count=count),),
            )

    def test_button_and_key_hold_actions(self) -> None:
        text = "\n".join([
            _tc('{"action": "button_down", "button": "left"}'),
            _tc('{"action": "key_down", "key": "shift"}'),
            _tc('{"action": "key_up", "key": "shift"}'),
            _tc('{"action": "button_up", "button": "left"}'),
        ])
        a = parse_computer_use_action(text)
        self.assertEqual(
            [(p.kind, p.name) for p in a.primitives],
            [("button_down", "left"), ("key_down", "shift"),
             ("key_up", "shift"), ("button_up", "left")],
        )
        self.assertTrue(a.has_left_click_press)

    def test_rejects_zero_blocks(self) -> None:
        for bad in ("", "TERMINATE", "just prose", "<think>only thought</think>"):
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                parse_computer_use_action(bad)

    def test_rejects_junk_outside_blocks_naming_fragment(self) -> None:
        text = "Sure, clicking now!\n" + _tc('{"action": "left_click"}')
        with self.assertRaises(ValueError) as ctx:
            parse_computer_use_action(text)
        self.assertIn("Sure, clicking now!", str(ctx.exception))
        trailing = _tc('{"action": "left_click"}') + "\nDone."
        with self.assertRaises(ValueError) as ctx:
            parse_computer_use_action(trailing)
        self.assertIn("Done.", str(ctx.exception))

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_computer_use_action('<tool_call>{"action": </tool_call>')

    def test_rejects_wrong_tool_name(self) -> None:
        with self.assertRaises(ValueError):
            parse_computer_use_action(
                '<tool_call>{"name": "browser_use", "arguments":'
                ' {"action": "left_click"}}</tool_call>'
            )

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            parse_computer_use_action(_tc('{"action": "hover"}'))

    def test_rejects_missing_and_extra_args(self) -> None:
        bad_args = [
            '{"action": "key"}',                                  # missing keys
            '{"action": "mouse_move_rel"}',                       # missing delta
            '{"action": "terminate"}',                            # missing status
            '{"action": "left_click", "delta": [1, 2]}',          # unknown arg
            '{"action": "type", "text": "x", "keys": ["a"]}',     # unknown arg
            '{"action": "wait"}',                                 # missing time
        ]
        for args in bad_args:
            with self.assertRaises(ValueError, msg=f"should reject {args!r}"):
                parse_computer_use_action(_tc(args))

    def test_rejects_wrong_types(self) -> None:
        bad_args = [
            '{"action": "mouse_move_rel", "delta": ["30", "-12"]}',
            '{"action": "mouse_move_rel", "delta": [1, 2, 3]}',
            '{"action": "mouse_move_rel", "delta": 30}',
            '{"action": "key", "keys": "ctrl"}',
            '{"action": "key", "keys": ["ctrl", 5]}',
            '{"action": "type", "text": 5}',
            '{"action": "button_down", "button": "back"}',
            '{"action": "key_down", "key": ""}',
            '{"action": "scroll", "pixels": "3"}',
            '{"action": "scroll", "pixels": true}',
            '{"action": "wait", "time": "soon"}',
            '{"action": "terminate", "status": "done"}',
        ]
        for args in bad_args:
            with self.assertRaises(ValueError, msg=f"should reject {args!r}"):
                parse_computer_use_action(_tc(args))

    def test_rejects_empty_keys_array(self) -> None:
        with self.assertRaises(ValueError):
            parse_computer_use_action(_tc('{"action": "key", "keys": []}'))

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(TypeError):
            parse_computer_use_action(None)  # type: ignore[arg-type]

    def test_any_bad_block_rejects_whole_reply(self) -> None:
        # Never partially dispatch: a valid first call followed by a
        # truncated/bad block must reject everything.
        text = _tc('{"action": "button_down", "button": "left"}') + \
            '\n<tool_call>{"name": "computer_use", "arguments": {"action": "butt'
        with self.assertRaises(ValueError):
            parse_computer_use_action(text)
        with self.assertRaises(ValueError):
            parse_computer_use_action_tolerant(text + "on_up</tool_call>")

    def test_tolerant_ignores_surrounding_prose(self) -> None:
        a = parse_computer_use_action_tolerant(
            "I'll save the file now.\n"
            + _tc('{"action": "key", "keys": ["ctrl", "s"]}')
            + "\nThat should do it."
        )
        self.assertEqual(a.primitives[0].keys, ("ctrl", "s"))

    def test_tolerant_still_rejects_no_blocks_and_bad_blocks(self) -> None:
        with self.assertRaises(ValueError):
            parse_computer_use_action_tolerant("no tool calls here")
        with self.assertRaises(ValueError):
            parse_computer_use_action_tolerant(
                "prose\n" + _tc('{"action": "hover"}')
            )


class ComputerUseRelFreerollTests(unittest.TestCase):
    def test_cua_v4_thinking_defaults_to_native_format(self) -> None:
        self.assertEqual(
            freeroll._resolve_action_format(None, "cua_v4_thinking"),
            "computer_use_rel_v1",
        )

    def test_explicit_native_flag_accepted(self) -> None:
        self.assertEqual(
            freeroll._resolve_action_format("computer_use_rel_v1", "training_v1"),
            "computer_use_rel_v1",
        )

    def test_cua_v4_thinking_prompt_is_registered(self) -> None:
        self.assertIn("cua_v4_thinking", freeroll.SYSTEM_PROMPTS)
        prompt = freeroll.SYSTEM_PROMPTS["cua_v4_thinking"]
        self.assertIn("GOAL:", prompt)
        self.assertIn("mouse_move_rel", prompt)
        self.assertIn("<tool_call>", prompt)

    def test_cua_v4_thinking_is_goal_conditioned(self) -> None:
        self.assertIn("cua_v4_thinking", freeroll._GOAL_CONDITIONED_PROMPT_IDS)

    def test_terminate_tool_call_detected_anywhere_in_reply(self) -> None:
        text = (
            "<think>The file is saved — goal complete.</think>\n"
            + _tc('{"action": "left_click"}')
            + "\n"
            + _tc('{"action": "terminate", "status": "success"}')
        )
        self.assertEqual(
            freeroll._computer_use_rel_terminate_status(text), "success"
        )
        self.assertEqual(
            freeroll._computer_use_rel_terminate_status(
                _tc('{"action": "terminate", "status": "failure"}')
            ),
            "failure",
        )

    def test_non_terminate_reply_returns_none(self) -> None:
        self.assertIsNone(
            freeroll._computer_use_rel_terminate_status(
                _tc('{"action": "left_click"}')
            )
        )

    def test_terminate_line_is_not_native_terminate(self) -> None:
        # The TERMINATE-line convention belongs to the other formats; for
        # computer_use_rel_v1 only a parsed terminate tool call stops.
        self.assertIsNone(freeroll._computer_use_rel_terminate_status("TERMINATE"))
        self.assertIsNone(
            freeroll._computer_use_rel_terminate_status(
                "<think>done</think>\nTERMINATE"
            )
        )

    def test_scaling_touches_mouse_move_rel_only(self) -> None:
        a = parse_computer_use_action("\n".join([
            _tc('{"action": "mouse_move_rel", "delta": [100, -50]}'),
            _tc('{"action": "scroll", "pixels": -3}'),
            _tc('{"action": "type", "text": "hi"}'),
        ]))
        scaled = freeroll._scale_ordered_moves(a, 2.0, 1.5)
        self.assertEqual(scaled.primitives[0],
                         OrderedPrimitive(kind="move", dx=200, dy=-75))
        self.assertEqual(scaled.primitives[1],
                         OrderedPrimitive(kind="scroll", dx=0, dy=-3))
        self.assertEqual(scaled.primitives[2].text, "hi")

    def test_is_left_click_sees_native_click_and_button_down(self) -> None:
        for prim in (
            {"kind": "click", "name": "left", "count": 1},
            {"kind": "click", "name": "left", "count": 3},
            {"kind": "button_down", "name": "left"},
        ):
            self.assertTrue(freeroll._is_left_click(
                {"no_op": False, "primitives": [prim]}
            ))
        for prim in (
            {"kind": "click", "name": "right", "count": 1},
            {"kind": "button_up", "name": "left"},
            {"kind": "key_combo", "keys": ["ctrl", "s"]},
        ):
            self.assertFalse(freeroll._is_left_click(
                {"no_op": False, "primitives": [prim]}
            ))


class _FakeClient(osworld_vm_client.OSWorldClient):
    """Records execute() calls; simulates cursor tracking via moveTo."""

    def __init__(self, *, pos=(100, 100), screen=(1920, 1080)) -> None:
        super().__init__("http://fake")
        self._pos = pos
        self._screen = screen
        self.commands: list[str] = []

    def cursor_position(self):  # type: ignore[override]
        return self._pos

    def screen_size(self):  # type: ignore[override]
        return self._screen

    def execute(self, command: str) -> None:  # type: ignore[override]
        self.commands.append(command)
        if m := re.match(r"pyautogui\.moveTo\((-?\d+), (-?\d+)\)", command):
            self._pos = (int(m.group(1)), int(m.group(2)))


class ComputerUseRelDispatchTests(unittest.TestCase):
    def test_rich_reply_dispatch_plan(self) -> None:
        client = _FakeClient()
        action = parse_computer_use_action("\n".join([
            _tc('{"action": "mouse_move_rel", "delta": [30, -12]}'),
            _tc('{"action": "left_click"}'),
            _tc('{"action": "type", "text": "hi"}'),
            _tc('{"action": "key", "keys": ["ctrl", "s"]}'),
        ]))
        sr = client.dispatch_ordered_action(action)
        self.assertEqual(client.commands, [
            "pyautogui.moveTo(130, 88)",
            "pyautogui.click(clicks=1, interval=0.05, button='left')",
            "pyautogui.write('hi', interval=0)",
            "pyautogui.keyDown('ctrl')",
            "pyautogui.keyDown('s')",
            "pyautogui.keyUp('s')",
            "pyautogui.keyUp('ctrl')",
        ])
        # Clicks land at the tracked cursor; type/key have no cursor effect.
        self.assertEqual(sr.intended_target, (130, 88))
        self.assertEqual(sr.delta, (30, -12))

    def test_resolution_scaled_delta_reaches_dispatch(self) -> None:
        client = _FakeClient()
        action = parse_computer_use_action(
            _tc('{"action": "mouse_move_rel", "delta": [100, -50]}')
        )
        scaled = freeroll._scale_ordered_moves(action, 2.0, 1.5)
        client.dispatch_ordered_action(scaled)
        self.assertEqual(client.commands, ["pyautogui.moveTo(300, 25)"])

    def test_command_key_maps_to_winleft_on_linux_vm(self) -> None:
        client = _FakeClient()
        client.dispatch_ordered_action(parse_computer_use_action(
            _tc('{"action": "key", "keys": ["command", "l"]}')
        ))
        self.assertEqual(client.commands, [
            "pyautogui.keyDown('winleft')",
            "pyautogui.keyDown('l')",
            "pyautogui.keyUp('l')",
            "pyautogui.keyUp('winleft')",
        ])
        client.commands.clear()
        client.dispatch_ordered_action(parse_computer_use_action(
            _tc('{"action": "key_down", "key": "command"}')
            + _tc('{"action": "key_up", "key": "command"}')
        ))
        self.assertEqual(client.commands, [
            "pyautogui.keyDown('winleft')",
            "pyautogui.keyUp('winleft')",
        ])

    def test_wait_dispatches_nothing(self) -> None:
        client = _FakeClient()
        sr = client.dispatch_ordered_action(parse_computer_use_action(
            _tc('{"action": "wait", "time": 3}')
        ))
        self.assertEqual(client.commands, [])
        self.assertEqual(sr.events_dispatched, [])
        self.assertEqual(sr.cursor_after, (100, 100))

    def test_terminate_never_dispatched(self) -> None:
        client = _FakeClient()
        with self.assertRaises(ValueError):
            client.dispatch_ordered_action(parse_computer_use_action(
                _tc('{"action": "terminate", "status": "success"}')
            ))

    def test_button_hold_and_scrolls(self) -> None:
        client = _FakeClient()
        client.dispatch_ordered_action(parse_computer_use_action("\n".join([
            _tc('{"action": "button_down", "button": "middle"}'),
            _tc('{"action": "button_up", "button": "middle"}'),
            _tc('{"action": "scroll", "pixels": -3}'),
            _tc('{"action": "hscroll", "pixels": 7}'),
            _tc('{"action": "triple_click"}'),
        ])))
        self.assertEqual(client.commands, [
            "pyautogui.mouseDown(button='middle')",
            "pyautogui.mouseUp(button='middle')",
            "pyautogui.scroll(-3)",
            "pyautogui.hscroll(7)",
            "pyautogui.click(clicks=3, interval=0.05, button='left')",
        ])

    def test_move_is_clipped_to_screen(self) -> None:
        client = _FakeClient(pos=(10, 10))
        client.dispatch_ordered_action(parse_computer_use_action(
            _tc('{"action": "mouse_move_rel", "delta": [-100, -100]}')
        ))
        self.assertEqual(client.commands, ["pyautogui.moveTo(0, 0)"])


class KeyTableConsistencyTests(unittest.TestCase):
    """Every key name the formatter's RDEV_TO_COMPUTER_USE_KEY table can emit
    must, after eval's _CUA_V4_KEY_TO_PYAUTOGUI remap, be a key pyautogui
    actually presses on the X11 VM. The pipeline is not importable from the
    eval venv, so the table is ast-loaded straight from the source file.

    _PYAUTOGUI_KEYS is vendored from pyautogui.KEYBOARD_KEYS (stable for
    years); pyautogui itself only exists inside the VM. Names in
    _X11_SILENT_KEYS are in pyautogui's list but map to None on the X11
    backend (keyDown silently no-ops), so they must be remapped, not passed
    through."""

    _FORMATTER_FILE = (
        Path(__file__).resolve().parents[1]
        / "data_pipeline/realigned_pipeline/lib/action_format.py"
    )

    _PYAUTOGUI_KEYS = frozenset(
        list("\t\n\r !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
        + [
            "accept", "add", "alt", "altleft", "altright", "apps", "backspace",
            "browserback", "browserfavorites", "browserforward", "browserhome",
            "browserrefresh", "browsersearch", "browserstop", "capslock",
            "clear", "convert", "ctrl", "ctrlleft", "ctrlright", "decimal",
            "del", "delete", "divide", "down", "end", "enter", "esc", "escape",
            "execute", "final", "fn", "hanguel", "hangul", "hanja", "help",
            "home", "insert", "junja", "kana", "kanji", "launchapp1",
            "launchapp2", "launchmail", "launchmediaselect", "left",
            "modechange", "multiply", "nexttrack", "nonconvert", "numlock",
            "pagedown", "pageup", "pause", "pgdn", "pgup", "playpause",
            "prevtrack", "print", "printscreen", "prntscrn", "prtsc", "prtscr",
            "return", "right", "scrolllock", "select", "separator", "shift",
            "shiftleft", "shiftright", "sleep", "space", "stop", "subtract",
            "tab", "up", "volumedown", "volumemute", "volumeup", "win",
            "winleft", "winright", "yen", "command", "option", "optionleft",
            "optionright",
        ]
        + [f"f{i}" for i in range(1, 25)]
        + [f"num{i}" for i in range(10)]
    )

    # In pyautogui's key list but None on the X11 backend: passing them
    # through would silently no-op on the Ubuntu VM.
    _X11_SILENT_KEYS = frozenset({"command", "option", "optionleft", "optionright"})

    def _formatter_table(self) -> dict:
        source = self._FORMATTER_FILE.read_text()
        tree = ast.parse(source)
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if "RDEV_TO_COMPUTER_USE_KEY" in targets:
                # The table is built from dict-comprehension spreads over
                # string literals (not literal_eval-able); the expression is
                # self-contained, so evaluate it with empty globals.
                expr = ast.get_source_segment(source, node.value)
                assert expr is not None
                table = eval(  # noqa: S307
                    expr, {"__builtins__": {"range": range, "str": str}}, {}
                )
                assert isinstance(table, dict)
                return table
        raise AssertionError(
            f"RDEV_TO_COMPUTER_USE_KEY not found in {self._FORMATTER_FILE}"
        )

    def test_formatter_key_names_dispatchable_on_vm(self) -> None:
        table = self._formatter_table()
        self.assertTrue(table, "formatter key table is empty")
        bad: list[tuple[str, str, str]] = []
        for rdev, cu_name in sorted(table.items()):
            target = osworld_vm_client._CUA_V4_KEY_TO_PYAUTOGUI.get(cu_name, cu_name)
            if target not in self._PYAUTOGUI_KEYS or target in self._X11_SILENT_KEYS:
                bad.append((rdev, cu_name, target))
        self.assertEqual(
            bad, [],
            "formatter emits key names eval cannot dispatch on the X11 VM "
            "(add remaps to _CUA_V4_KEY_TO_PYAUTOGUI): " + repr(bad[:20]),
        )


if __name__ == "__main__":
    unittest.main()
