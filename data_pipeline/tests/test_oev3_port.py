"""Contract tests for the ordered_events_v3 port.

Pins the OrderedV3Regression corpus from the shortgoal branch's
eval/test_shortgoal_grammar.py against the ported eval/action_parser.py, and
round-trips ``cuagym_pipeline.oev3_render`` output through
``parse_ordered_action`` (semantic equality + render/parse/render byte
identity).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from cuagym_pipeline import oev3_render
from cuagym_pipeline.oev3_render import (
    NO_OP,
    TERMINATE,
    join_primitives,
    render_down,
    render_move,
    render_scroll,
    render_type,
    render_up,
)

from action_parser import (  # noqa: E402
    OrderedAction,
    OrderedPrimitive,
    parse_ordered_action,
)


class OrderedV3RegressionTests(unittest.TestCase):
    """The ported v3 parser must behave exactly as on the shortgoal branch."""

    def test_v3_corpus_parses_unchanged(self) -> None:
        action = parse_ordered_action("move(4,-1); down(LMB); up(LMB); scroll(0,-3)")
        self.assertEqual(
            action.primitives,
            (
                OrderedPrimitive(kind="move", dx=4, dy=-1),
                OrderedPrimitive(kind="down", name="LMB", mouse_button=1),
                OrderedPrimitive(kind="up", name="LMB", mouse_button=1),
                OrderedPrimitive(kind="scroll", dx=0, dy=-3),
            ),
        )
        self.assertFalse(action.no_op)
        self.assertTrue(action.has_left_click_press)
        self.assertTrue(parse_ordered_action("NO_OP").no_op)

    def test_v3_type_escapes_unchanged(self) -> None:
        action = parse_ordered_action(r'type("say \"hi\" to C:\\tmp"); down(Return)')
        self.assertEqual(action.primitives[0].text, 'say "hi" to C:\\tmp')
        self.assertEqual(action.primitives[1],
                         OrderedPrimitive(kind="down", name="Return"))

    def test_v3_zero_and_hscroll_stay_legal(self) -> None:
        self.assertEqual(
            parse_ordered_action("move(0,0); scroll(0,0); scroll(-2,0)").primitives,
            (
                OrderedPrimitive(kind="move", dx=0, dy=0),
                OrderedPrimitive(kind="scroll", dx=0, dy=0),
                OrderedPrimitive(kind="scroll", dx=-2, dy=0),
            ),
        )
        self.assertEqual(
            parse_ordered_action("move(4000,-9000)").primitives,
            (OrderedPrimitive(kind="move", dx=4000, dy=-9000),),
        )

    def test_new_fields_default_to_none_on_legacy_primitives(self) -> None:
        for prim in parse_ordered_action('move(1,2); down(LMB); type("x")').primitives:
            self.assertIsNone(prim.x)
            self.assertIsNone(prim.y)
        self.assertEqual(
            OrderedPrimitive(kind="move", dx=1, dy=2),
            OrderedPrimitive(kind="move", dx=1, dy=2, x=None, y=None),
        )

    def test_v3_parser_never_accepts_v4_primitives(self) -> None:
        for bad in ("move_to(1,2)", "scroll(3)"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ordered_action(bad)

    def test_v3_rejections_unchanged(self) -> None:
        for bad in ("TERMINATE", "", "move(1)", "move(1,2) extra", 'type("")',
                    "move(1,2);; down(LMB)", "hello world"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ordered_action(bad)

    def test_ordered_action_dataclass_shape_is_additive(self) -> None:
        action = OrderedAction(primitives=(), no_op=True)
        self.assertTrue(action.no_op)
        self.assertFalse(action.has_left_click_press)


def _rerender(prim: OrderedPrimitive) -> str:
    if prim.kind == "move":
        return render_move(prim.dx, prim.dy)
    if prim.kind == "scroll":
        return render_scroll(prim.dx, prim.dy)
    if prim.kind == "down":
        return render_down(prim.name)
    if prim.kind == "up":
        return render_up(prim.name)
    if prim.kind == "type":
        return render_type(prim.text)
    raise AssertionError(f"unexpected kind after v3 parse: {prim.kind!r}")


def _rerender_line(action: OrderedAction) -> str:
    if action.no_op:
        return NO_OP
    return join_primitives([_rerender(p) for p in action.primitives])


_MOUSE = {"LMB": 1, "MMB": 2, "RMB": 3}
_KEYS = ("ControlLeft", "ControlRight", "ShiftLeft", "AltLeft", "MetaLeft",
         "Return", "Tab", "Escape", "BackSpace", "KeyA", "Digit7", "F5",
         "ArrowDown", "'", "\\")

_ROUND_TRIP_CORPUS: list[list[OrderedPrimitive]] = (
    [
        [OrderedPrimitive(kind="move", dx=dx, dy=dy)]
        for dx, dy in ((1, 1), (-1, -1), (0, 5), (-7, 0), (123, -456),
                       (99999, -99999), (-2147483648, 2147483647), (0, 0))
    ]
    + [
        [OrderedPrimitive(kind="scroll", dx=dx, dy=dy)]
        for dx, dy in ((0, -3), (0, 3), (-2, 0), (2, 0), (11, -11),
                       (-100000, 100000), (0, 0))
    ]
    + [
        [
            OrderedPrimitive(kind="down", name=name, mouse_button=_MOUSE.get(name)),
            OrderedPrimitive(kind="up", name=name, mouse_button=_MOUSE.get(name)),
        ]
        for name in tuple(_MOUSE) + _KEYS
    ]
    + [
        [OrderedPrimitive(kind="type", text=text)]
        for text in (
            "hello",
            "a; b",
            "move(1,2); down(LMB)",
            'she said "no"',
            "C:\\Users\\me\\file.txt",
            "\\",
            '"',
            '\\"',
            '\\\\""\\',
            "f(x, y) = (x; y)",
            "NO_OP",
            "TERMINATE",
            "type(\"nested\")",
            "commas, everywhere, always,",
            "; ",
            ";",
            " leading and trailing kept ",
            "`~!@#$%^&*()-_=+[]{}|;:'<>,./?",
        )
    ]
    + [
        [
            OrderedPrimitive(kind="move", dx=4, dy=-1),
            OrderedPrimitive(kind="down", name="LMB", mouse_button=1),
            OrderedPrimitive(kind="up", name="LMB", mouse_button=1),
            OrderedPrimitive(kind="scroll", dx=0, dy=-3),
        ],
        [
            OrderedPrimitive(kind="down", name="ControlLeft"),
            OrderedPrimitive(kind="down", name="KeyC"),
            OrderedPrimitive(kind="up", name="KeyC"),
            OrderedPrimitive(kind="up", name="ControlLeft"),
        ],
        [
            OrderedPrimitive(kind="type", text='cd "my dir"; ls -la'),
            OrderedPrimitive(kind="down", name="Return"),
            OrderedPrimitive(kind="up", name="Return"),
        ],
        [
            OrderedPrimitive(kind="move", dx=-300, dy=42),
            OrderedPrimitive(kind="type", text="echo \\\"quoted\\\""),
            OrderedPrimitive(kind="scroll", dx=0, dy=7),
            OrderedPrimitive(kind="type", text="second; burst, (with) parens"),
        ],
        [
            OrderedPrimitive(kind="down", name="RMB", mouse_button=3),
            OrderedPrimitive(kind="move", dx=15, dy=15),
            OrderedPrimitive(kind="up", name="RMB", mouse_button=3),
            OrderedPrimitive(kind="down", name="MMB", mouse_button=2),
            OrderedPrimitive(kind="up", name="MMB", mouse_button=2),
        ],
        [],
    ]
)


class Oev3RenderRoundTripTests(unittest.TestCase):
    def test_corpus_is_large_enough(self) -> None:
        self.assertGreaterEqual(len(_ROUND_TRIP_CORPUS), 50)

    def test_render_parse_round_trip(self) -> None:
        for prims in _ROUND_TRIP_CORPUS:
            with self.subTest(prims=prims):
                line = _rerender_line(
                    OrderedAction(primitives=tuple(prims), no_op=not prims)
                )
                action = parse_ordered_action(line)
                self.assertEqual(action.no_op, not prims)
                self.assertEqual(action.primitives, tuple(prims))
                self.assertEqual(_rerender_line(action), line)

    def test_constants_match_parser_contract(self) -> None:
        self.assertEqual(NO_OP, "NO_OP")
        self.assertEqual(TERMINATE, "TERMINATE")
        self.assertTrue(parse_ordered_action(NO_OP).no_op)
        with self.assertRaises(ValueError):
            parse_ordered_action(TERMINATE)

    def test_join_primitives_empty_is_no_op(self) -> None:
        self.assertEqual(join_primitives([]), NO_OP)
        self.assertEqual(join_primitives(["move(1,2)"]), "move(1,2)")
        self.assertEqual(
            join_primitives(["down(LMB)", "up(LMB)"]), "down(LMB); up(LMB)"
        )

    def test_render_type_escapes_exactly_backslash_and_quote(self) -> None:
        self.assertEqual(render_type('a\\b"c'), r'type("a\\b\"c")')
        self.assertEqual(render_type("; plain ; stays"), 'type("; plain ; stays")')

    def test_render_rejections(self) -> None:
        with self.assertRaises(ValueError):
            render_type("")
        for bad_text in ("line\nbreak", "carriage\rreturn", "tab\there"):
            with self.subTest(bad_text=bad_text), self.assertRaises(ValueError):
                render_type(bad_text)
        for bad_name in ("", "two words", "paren(", "paren)", "a,b", "a;b"):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises(ValueError):
                    render_down(bad_name)
                with self.assertRaises(ValueError):
                    render_up(bad_name)

    def test_grammar_constant_ported_verbatim(self) -> None:
        self.assertIn('"NO_OP" / primitive *("; " primitive)',
                      oev3_render.ORDERED_EVENTS_V3_GRAMMAR)
        self.assertIn(r'escape     = "\" ("\" / DQUOTE)',
                      oev3_render.ORDERED_EVENTS_V3_GRAMMAR)


if __name__ == "__main__":
    unittest.main()
