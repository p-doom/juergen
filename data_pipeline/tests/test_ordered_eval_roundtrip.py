"""Contract gate across the training/eval boundary.

``lib/action_format`` RENDERS assistant-turn labels (training data); the eval
side's ``eval/action_parser.parse_ordered_action`` PARSES model replies in that
same format and dispatches them to the VM. Nothing forces the two to agree —
they live in different packages with different dependency sets — so a change to
either that breaks the pairing shows up as a wall of eval parse errors and a
rollout where the VM never moves.

This module pins the inverse property from both directions:
  * ``parse(ActionPrimitive.render())`` reconstructs the primitive, and
  * every label the real formatter emits over synthetic event streams parses,
    and re-renders byte-identically.

The eval parser is stdlib-only, so importing it here costs nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import ClassVar

from realigned_pipeline.lib.action_format import (
    ActionPrimitive,
    get_formatter,
)
from realigned_pipeline.lib.events import RawEvent, Window

_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from action_parser import parse_ordered_action  # noqa: E402

# Same convention as test_action_format.py: 15 fps master ticks, one window
# covering [0, 30) == the first 2 s, default 10 Hz motor grid.
MASTER_FPS = 15.0


def _move(seq: int, t_s: float, dx: float, dy: float) -> RawEvent:
    return RawEvent(seq, t_s, "move", dx=dx, dy=dy)


def _scroll(seq: int, t_s: float, dx: float, dy: float) -> RawEvent:
    return RawEvent(seq, t_s, "scroll", dx=dx, dy=dy, scroll=dy if dy != 0 else dx)


def _key(seq: int, t_s: float, kind: str, name: str) -> RawEvent:
    return RawEvent(seq, t_s, kind, name=name)


class PrimitiveRoundTripTest(unittest.TestCase):
    """parse(render(primitive)) == primitive, for every primitive kind."""

    CASES: ClassVar[list[list[ActionPrimitive]]] = [
        [ActionPrimitive("move", dx=4, dy=-1)],
        [ActionPrimitive("move", dx=-100, dy=250)],
        [ActionPrimitive("scroll", dx=0, dy=-60)],
        [ActionPrimitive("scroll", dx=-3, dy=7)],
        [
            ActionPrimitive("move", dx=4, dy=-1),
            ActionPrimitive("down", input_name="LMB"),
            ActionPrimitive("move", dx=2, dy=0),
            ActionPrimitive("up", input_name="LMB"),
        ],
        [
            ActionPrimitive("down", input_name="ShiftLeft"),
            ActionPrimitive("down", input_name="KeyH"),
            ActionPrimitive("up", input_name="KeyH"),
            ActionPrimitive("up", input_name="ShiftLeft"),
        ],
        # v3 type() payloads, including every escape the grammar defines and
        # the primitive separator appearing INSIDE a quoted string.
        [ActionPrimitive("type", text="hello world")],
        [ActionPrimitive("type", text='quote " and back\\slash')],
        [ActionPrimitive("type", text="semi; colon")],
        [ActionPrimitive("type", text='a"; b')],
        [ActionPrimitive("type", text="~!@#$%^&*()-_=+[]{}|;:'<>,./?")],
        [
            ActionPrimitive("move", dx=-100, dy=250),
            ActionPrimitive("type", text="ls -la"),
            ActionPrimitive("down", input_name="Return"),
            ActionPrimitive("up", input_name="Return"),
        ],
    ]

    def test_parse_inverts_render(self) -> None:
        for prims in self.CASES:
            line = "; ".join(p.render() for p in prims)
            with self.subTest(line=line):
                parsed = parse_ordered_action(line)
                self.assertEqual(len(parsed.primitives), len(prims))
                for got, want in zip(parsed.primitives, prims, strict=True):
                    self.assertEqual(got.kind, want.kind)
                    self.assertEqual(got.dx, want.dx)
                    self.assertEqual(got.dy, want.dy)
                    self.assertEqual(got.input_name, want.input_name)
                    self.assertEqual(got.text, want.text)

    def test_reparse_renders_the_same_wire_line(self) -> None:
        for prims in self.CASES:
            line = "; ".join(p.render() for p in prims)
            with self.subTest(line=line):
                self.assertEqual(parse_ordered_action(line).render(), line)


class FormatterOutputIsParseableTest(unittest.TestCase):
    """Every label the real formatter emits must parse on the eval side."""

    STREAMS: ClassVar[list[list[RawEvent]]] = [
        # Motion split by a click -- the shape the aggregate format cannot hold.
        [
            _move(0, 0.01, 1.0, 0.0),
            _move(1, 0.02, 3.0, -1.0),
            _key(2, 0.03, "press", "LMB"),
            _move(3, 0.04, 2.0, 0.0),
            _key(4, 0.05, "release", "LMB"),
        ],
        # Scroll on both axes.
        [_scroll(0, 0.01, 0.0, -5.0), _scroll(1, 0.2, 3.0, 0.0)],
        # Modifier chord.
        [
            _key(0, 0.01, "press", "ControlLeft"),
            _key(1, 0.02, "press", "KeyC"),
            _key(2, 0.03, "release", "KeyC"),
            _key(3, 0.04, "release", "ControlLeft"),
        ],
        # A typing run -- v3 collapses this into type("..."), v2 does not.
        [
            _key(0, 0.01, "press", "KeyH"),
            _key(1, 0.02, "release", "KeyH"),
            _key(2, 0.03, "press", "KeyI"),
            _key(3, 0.04, "release", "KeyI"),
        ],
        # Shifted capital inside a typing run.
        [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "press", "KeyH"),
            _key(2, 0.03, "release", "KeyH"),
            _key(3, 0.04, "release", "ShiftLeft"),
            _key(4, 0.05, "press", "KeyI"),
            _key(5, 0.06, "release", "KeyI"),
        ],
        # Drag: press, move, release.
        [
            _key(0, 0.01, "press", "LMB"),
            _move(1, 0.05, 40.0, 10.0),
            _move(2, 0.15, 40.0, 10.0),
            _key(3, 0.25, "release", "LMB"),
        ],
        # Empty window -> NO_OP.
        [],
    ]

    def _labels(self, formatter_name: str, events: list[RawEvent]) -> list[str]:
        return get_formatter(formatter_name, continuous_action_hz=10.0).format_segment(
            events, [Window(master_idx=0, start=0, end=30)], [], master_fps=MASTER_FPS
        ).labels

    def test_every_formatter_label_parses_and_reparses(self) -> None:
        for formatter_name in ("ordered_events_v2", "ordered_events_v3"):
            for i, events in enumerate(self.STREAMS):
                for label in self._labels(formatter_name, events):
                    with self.subTest(fmt=formatter_name, stream=i, label=label):
                        parsed = parse_ordered_action(label)
                        self.assertEqual(parsed.render(), label)
                        self.assertEqual(parsed.no_op, label == "NO_OP")

    def test_v3_emits_a_type_primitive_the_parser_recovers(self) -> None:
        # Guards the v2/v3 difference itself: the typing run must arrive as one
        # type() payload, not ~8 down/up primitives.
        (label,) = self._labels("ordered_events_v3", self.STREAMS[3])
        parsed = parse_ordered_action(label)
        self.assertEqual([p.kind for p in parsed.primitives], ["type"])
        self.assertEqual(parsed.primitives[0].text, "hi")

    def test_v2_emits_no_type_primitive(self) -> None:
        # The shared parser serves both formats; v2 simply never uses type().
        for i, events in enumerate(self.STREAMS):
            for label in self._labels("ordered_events_v2", events):
                with self.subTest(stream=i, label=label):
                    kinds = {p.kind for p in parse_ordered_action(label).primitives}
                    self.assertNotIn("type", kinds)

    def test_terminate_line_is_not_a_primitive_line(self) -> None:
        # Stage 04 OVERWRITES the final action with TERMINATE, so it always
        # arrives alone and freeroll._is_terminate intercepts it before the
        # parser sees it. Parsing it must fail loudly rather than half-succeed.
        for formatter_name in ("ordered_events_v2", "ordered_events_v3"):
            terminate = get_formatter(formatter_name).terminate_line()
            with self.subTest(fmt=formatter_name):
                self.assertEqual(terminate, "TERMINATE")
                with self.assertRaises(ValueError):
                    parse_ordered_action(terminate)


if __name__ == "__main__":
    unittest.main()
