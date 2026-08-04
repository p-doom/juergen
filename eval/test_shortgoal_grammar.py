from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import action_parser  # noqa: E402
import shortgoal_grammar as sg  # noqa: E402
from action_parser import (  # noqa: E402
    OrderedAction,
    OrderedPrimitive,
    parse_ordered_action,
    parse_ordered_v4_action,
    parse_ordered_v4_action_tolerant,
)

_KEY_NAMES = (
    "LMB", "RMB", "MMB", "ControlLeft", "ShiftLeft", "Return", "Tab", "Escape",
    "KeyA", "Digit7", "MetaLeft", "BackSpace",
)
_TYPE_CHARS = tuple(
    "abcXYZ019 " + "\\\"';:,.()[]{}<>/?|`~!@#$%^&*-_=+"
)
_FUZZ_PROGRAMS = 5000
_SCREENS = ((1920, 1080), (1152, 720), (1237, 683))


def _prims(*prims: OrderedPrimitive) -> tuple[OrderedPrimitive, ...]:
    return prims


def _key(kind: str, name: str) -> OrderedPrimitive:
    return OrderedPrimitive(
        kind=kind, name=name,
        mouse_button=action_parser._MOUSE_BUTTON_CODES.get(name),
    )


def _random_primitive(rng: random.Random, arm: str) -> OrderedPrimitive:
    kind = rng.choice(
        ["mouse", "mouse", "scroll", "down", "up", "type"]
    )
    if kind == "mouse" and arm == sg.ARM_REL:
        dx, dy = 0, 0
        while dx == 0 and dy == 0:
            dx = rng.randint(-sg.GRID, sg.GRID)
            dy = rng.randint(-sg.GRID, sg.GRID)
        return OrderedPrimitive(kind="move", dx=dx, dy=dy)
    if kind == "mouse":
        return OrderedPrimitive(
            kind="move_to", x=rng.randint(0, sg.GRID), y=rng.randint(0, sg.GRID),
        )
    if kind == "scroll":
        notches = 0
        while notches == 0:
            notches = rng.randint(-15, 15)
        return OrderedPrimitive(kind="scroll", dx=0, dy=notches)
    if kind in ("down", "up"):
        return _key(kind, rng.choice(_KEY_NAMES))
    return OrderedPrimitive(
        kind="type",
        text="".join(rng.choice(_TYPE_CHARS) for _ in range(rng.randint(1, 14))),
    )


class GrammarContractTests(unittest.TestCase):
    def test_arm_names_and_registry(self) -> None:
        self.assertEqual(sg.ARMS, ("ordered_events_v4_rel", "ordered_events_v4_abs"))
        self.assertEqual(set(sg.ARMS), set(action_parser.ORDERED_V4_ARMS))
        self.assertEqual(sg.ARM_REL, action_parser.ORDERED_V4_ARM_REL)
        self.assertEqual(sg.ARM_ABS, action_parser.ORDERED_V4_ARM_ABS)
        self.assertEqual(sg.GRID, action_parser.ORDERED_V4_SCALE)
        self.assertEqual(sorted(sg.PROMPT_IDS), sorted(sg.ARMS))
        self.assertEqual(sg.PROMPT_IDS[sg.ARM_REL], "shortgoal_oev4_rel_v1")
        self.assertEqual(sg.PROMPT_IDS[sg.ARM_ABS], "shortgoal_oev4_abs_v1")

    def test_whole_line_reply_and_context_constants(self) -> None:
        self.assertEqual(sg.TERMINATE_LINE, "TERMINATE")
        self.assertEqual(sg.NO_OP_LINE, "NO_OP")
        self.assertEqual(sg.IMAGE_PLACEHOLDER, "<Image collapsed>")
        self.assertEqual((sg.K_IMAGES, sg.KEEP_IMAGES), (6, 3))
        self.assertLess(sg.KEEP_IMAGES, sg.K_IMAGES)

    def test_grammar_constant_documents_both_arms(self) -> None:
        text = sg.ORDERED_EVENTS_V4_GRAMMAR
        for needle in (
            '"NO_OP"', "TERMINATE", 'move("', 'move_to("', 'scroll("',
            'down("', 'up("', 'type("', sg.ARM_REL, sg.ARM_ABS,
        ):
            self.assertIn(needle, text)
        self.assertNotIn("hscroll(", text)


class HandCorpusTests(unittest.TestCase):
    def _round_trip(self, line: str, arm: str, expected: tuple[OrderedPrimitive, ...]) -> None:
        action = parse_ordered_v4_action(line, arm=arm)
        self.assertFalse(action.no_op)
        self.assertEqual(action.primitives, expected)
        self.assertEqual(sg.render_line(expected, arm), line)
        self.assertEqual(sg.render_line(action.primitives, arm), line)

    def test_rel_corpus(self) -> None:
        cases = [
            ("move(4,-1)", _prims(OrderedPrimitive(kind="move", dx=4, dy=-1))),
            ("move(-1000,1000)", _prims(OrderedPrimitive(kind="move", dx=-1000, dy=1000))),
            ("move(0,-7)", _prims(OrderedPrimitive(kind="move", dx=0, dy=-7))),
            (
                "move(120,-33); down(LMB); up(LMB)",
                _prims(
                    OrderedPrimitive(kind="move", dx=120, dy=-33),
                    _key("down", "LMB"), _key("up", "LMB"),
                ),
            ),
            (
                "down(LMB); move(300,0); up(LMB)",
                _prims(
                    _key("down", "LMB"),
                    OrderedPrimitive(kind="move", dx=300, dy=0),
                    _key("up", "LMB"),
                ),
            ),
        ]
        for line, expected in cases:
            with self.subTest(line=line):
                self._round_trip(line, sg.ARM_REL, expected)

    def test_abs_corpus(self) -> None:
        cases = [
            ("move_to(0,0)", _prims(OrderedPrimitive(kind="move_to", x=0, y=0))),
            ("move_to(1000,1000)", _prims(OrderedPrimitive(kind="move_to", x=1000, y=1000))),
            (
                "move_to(512,377); down(LMB); up(LMB); down(LMB); up(LMB)",
                _prims(
                    OrderedPrimitive(kind="move_to", x=512, y=377),
                    _key("down", "LMB"), _key("up", "LMB"),
                    _key("down", "LMB"), _key("up", "LMB"),
                ),
            ),
            (
                "move_to(10,20); down(RMB); up(RMB)",
                _prims(
                    OrderedPrimitive(kind="move_to", x=10, y=20),
                    _key("down", "RMB"), _key("up", "RMB"),
                ),
            ),
        ]
        for line, expected in cases:
            with self.subTest(line=line):
                self._round_trip(line, sg.ARM_ABS, expected)

    def test_shared_corpus_is_identical_in_both_arms(self) -> None:
        cases = [
            ("scroll(3)", _prims(OrderedPrimitive(kind="scroll", dx=0, dy=3))),
            ("scroll(-12)", _prims(OrderedPrimitive(kind="scroll", dx=0, dy=-12))),
            (
                "down(ControlLeft); down(KeyS); up(KeyS); up(ControlLeft)",
                _prims(
                    _key("down", "ControlLeft"), _key("down", "KeyS"),
                    _key("up", "KeyS"), _key("up", "ControlLeft"),
                ),
            ),
            (
                'type("touch /tmp/a.txt"); down(Return); up(Return)',
                _prims(
                    OrderedPrimitive(kind="type", text="touch /tmp/a.txt"),
                    _key("down", "Return"), _key("up", "Return"),
                ),
            ),
        ]
        for arm in sg.ARMS:
            for line, expected in cases:
                with self.subTest(arm=arm, line=line):
                    self._round_trip(line, arm, expected)

    def test_type_escape_stress(self) -> None:
        cases = [
            (r'type("say \"hi\"")', 'say "hi"'),
            (r'type("C:\\tmp\\x")', "C:\\tmp\\x"),
            (r'type("a; b(c), d")', "a; b(c), d"),
            (r'type("echo \\\"q\\\" > f")', 'echo \\"q\\" > f'),
            (r'type("trailing\\")', "trailing\\"),
            (r'type("\"")', '"'),
            (r'type("\\")', "\\"),
            (r'type(" ")', " "),
            (r'type("NO_OP")', "NO_OP"),
            (r'type("TERMINATE")', "TERMINATE"),
            (r'type("move(1,2); up(LMB)")', "move(1,2); up(LMB)"),
        ]
        for arm in sg.ARMS:
            for line, text in cases:
                with self.subTest(arm=arm, line=line):
                    self._round_trip(
                        line, arm, _prims(OrderedPrimitive(kind="type", text=text))
                    )

    def test_no_op_line_is_a_whole_line_reply(self) -> None:
        for arm in sg.ARMS:
            self.assertEqual(sg.render_line((), arm), "NO_OP")
            self.assertEqual(sg.render_line([], arm), "NO_OP")
            action = parse_ordered_v4_action("NO_OP\n", arm=arm)
            self.assertTrue(action.no_op)
            self.assertEqual(action.primitives, ())

    def test_parser_tolerates_interior_whitespace_but_renders_canonically(self) -> None:
        action = parse_ordered_v4_action("move( 4 , -1 ); down(LMB)", arm=sg.ARM_REL)
        self.assertEqual(sg.render_line(action.primitives, sg.ARM_REL),
                         "move(4,-1); down(LMB)")
        action = parse_ordered_v4_action("move_to( 4 , 1 ); scroll( -2 )", arm=sg.ARM_ABS)
        self.assertEqual(sg.render_line(action.primitives, sg.ARM_ABS),
                         "move_to(4,1); scroll(-2)")


class FuzzRoundTripTests(unittest.TestCase):
    def test_render_parse_render_is_byte_identical(self) -> None:
        for arm in sg.ARMS:
            kinds: set[str] = set()
            for i in range(_FUZZ_PROGRAMS):
                rng = random.Random(f"shortgoal_grammar_fuzz:{arm}:{i}")
                prims = tuple(
                    _random_primitive(rng, arm) for _ in range(rng.randint(1, 8))
                )
                line = sg.render_line(prims, arm)
                action = parse_ordered_v4_action(line, arm=arm)
                self.assertEqual(action.primitives, prims, msg=f"{arm}:{i} {line!r}")
                self.assertEqual(
                    sg.render_line(action.primitives, arm), line, msg=f"{arm}:{i}"
                )
                self.assertFalse(action.no_op)
                kinds.update(p.kind for p in prims)
            expected_mouse = "move" if arm == sg.ARM_REL else "move_to"
            self.assertEqual(
                kinds, {expected_mouse, "scroll", "down", "up", "type"}, msg=arm
            )

    def test_fuzz_is_seed_stable(self) -> None:
        for arm in sg.ARMS:
            rng_a = random.Random(f"shortgoal_grammar_fuzz:{arm}:0")
            rng_b = random.Random(f"shortgoal_grammar_fuzz:{arm}:0")
            self.assertEqual(
                [_random_primitive(rng_a, arm) for _ in range(16)],
                [_random_primitive(rng_b, arm) for _ in range(16)],
            )


class NormalizationTests(unittest.TestCase):
    def test_norm_delta_matches_freeroll_denorm_semantics(self) -> None:
        self.assertEqual(sg.norm_delta(960, 1920), 500)
        self.assertEqual(sg.norm_delta(-540, 1080), -500)
        self.assertEqual(sg.norm_delta(1920, 1920), 1000)
        self.assertEqual(sg.norm_delta(0, 1920), 0)
        action = parse_ordered_v4_action("move(500,500)", arm=sg.ARM_REL)
        denormed = sg.denorm_v4(action, (1920, 1080))
        self.assertEqual(denormed.primitives,
                         _prims(OrderedPrimitive(kind="move", dx=960, dy=540)))

    def test_norm_point_and_grid_bounds(self) -> None:
        self.assertEqual(sg.norm_point(0, 1920), 0)
        self.assertEqual(sg.norm_point(960, 1920), 500)
        self.assertEqual(sg.norm_point(1919, 1920), 999)
        self.assertEqual(sg.norm_point(719, 720), 999)

    def test_delta_round_trip_within_one_grid_unit(self) -> None:
        for sw, sh in _SCREENS:
            for axis, screen in (("x", sw), ("y", sh)):
                unit = max(1.0, screen / sg.GRID)
                for d_px in range(-screen, screen + 1, max(1, screen // 97)):
                    grid = sg.norm_delta(d_px, screen)
                    if grid == 0:
                        continue
                    line = (f"move({grid},0)" if axis == "x" else f"move(0,{grid})")
                    got = sg.denorm_v4(
                        parse_ordered_v4_action(line, arm=sg.ARM_REL), (sw, sh)
                    ).primitives[0]
                    back = got.dx if axis == "x" else got.dy
                    self.assertLessEqual(
                        abs(back - d_px), unit,
                        msg=f"{screen}px axis {axis}: {d_px} -> {grid} -> {back}",
                    )

    def test_point_round_trip_within_one_grid_unit(self) -> None:
        for sw, sh in _SCREENS:
            for axis, screen in (("x", sw), ("y", sh)):
                unit = max(1.0, screen / sg.GRID)
                for p_px in range(0, screen, max(1, screen // 89)):
                    grid = sg.norm_point(p_px, screen)
                    line = (f"move_to({grid},0)" if axis == "x" else f"move_to(0,{grid})")
                    got = sg.denorm_v4(
                        parse_ordered_v4_action(line, arm=sg.ARM_ABS), (sw, sh)
                    ).primitives[0]
                    back = got.x if axis == "x" else got.y
                    self.assertTrue(0 <= back < screen)
                    self.assertLessEqual(
                        abs(back - p_px), unit,
                        msg=f"{screen}px axis {axis}: {p_px} -> {grid} -> {back}",
                    )

    def test_snapped_pixels_are_exactly_representable(self) -> None:
        for sw, sh in _SCREENS:
            for screen in (sw, sh):
                for p_px in range(0, screen, max(1, screen // 83)):
                    snapped = sg.snap_point_px(p_px, screen)
                    self.assertTrue(0 <= snapped < screen)
                    self.assertLessEqual(abs(snapped - p_px), max(1.0, screen / sg.GRID))
                    self.assertEqual(sg.snap_point_px(snapped, screen), snapped)
                    self.assertEqual(
                        sg._grid_to_px(sg.norm_point(snapped, screen), screen), snapped
                    )

    def test_snapped_click_targets_survive_the_full_abs_pipeline(self) -> None:
        for sw, sh in _SCREENS:
            tx, ty = sg.snap_point_px(int(sw * 0.37), sw), sg.snap_point_px(int(sh * 0.61), sh)
            line = sg.render_line(
                (
                    OrderedPrimitive(
                        kind="move_to", x=sg.norm_point(tx, sw), y=sg.norm_point(ty, sh),
                    ),
                    _key("down", "LMB"), _key("up", "LMB"),
                ),
                sg.ARM_ABS,
            )
            denormed = sg.denorm_v4(parse_ordered_v4_action(line, arm=sg.ARM_ABS), (sw, sh))
            self.assertEqual((denormed.primitives[0].x, denormed.primitives[0].y), (tx, ty))

    def test_denorm_leaves_scroll_keys_and_typing_alone(self) -> None:
        for arm in sg.ARMS:
            action = parse_ordered_v4_action(
                'scroll(-3); down(ControlLeft); up(ControlLeft); type("hi")', arm=arm
            )
            denormed = sg.denorm_v4(action, (1920, 1080))
            self.assertEqual(denormed.primitives, action.primitives)

    def test_denorm_preserves_no_op(self) -> None:
        action = parse_ordered_v4_action("NO_OP", arm=sg.ARM_REL)
        denormed = sg.denorm_v4(action, (1920, 1080))
        self.assertTrue(denormed.no_op)
        self.assertEqual(denormed.primitives, ())

    def test_norm_helpers_reject_bad_inputs(self) -> None:
        for bad in (0, -1, 1.5, "1920"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sg.norm_delta(10, bad)  # type: ignore[arg-type]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sg.norm_point(10, bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sg.norm_delta(1921, 1920)
        with self.assertRaises(ValueError):
            sg.norm_delta(-1921, 1920)
        with self.assertRaises(ValueError):
            sg.norm_delta(1.0, 1920)  # type: ignore[arg-type]
        for bad_point in (-1, 1920, 5000):
            with self.subTest(bad_point=bad_point), self.assertRaises(ValueError):
                sg.norm_point(bad_point, 1920)
        with self.assertRaises(ValueError):
            sg.snap_point_px(-1, 1920)

    def test_denorm_rejects_bad_arguments(self) -> None:
        with self.assertRaises(TypeError):
            sg.denorm_v4("move(1,1)", (1920, 1080))  # type: ignore[arg-type]
        action = parse_ordered_v4_action("move(1,1)", arm=sg.ARM_REL)
        for bad in ((0, 1080), (1920, -1), (1920.0, 1080)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sg.denorm_v4(action, bad)  # type: ignore[arg-type]


class RenderRejectionTests(unittest.TestCase):
    def test_unknown_arm_is_rejected_everywhere(self) -> None:
        prim = OrderedPrimitive(kind="scroll", dx=0, dy=1)
        for bad_arm in ("ordered_events_v3", "rel", "", None):
            with self.subTest(bad_arm=bad_arm):
                with self.assertRaises(ValueError):
                    sg.render_line((prim,), bad_arm)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    sg.render_primitive(prim, bad_arm)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    parse_ordered_v4_action("scroll(1)", arm=bad_arm)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    parse_ordered_v4_action_tolerant("scroll(1)", arm=bad_arm)  # type: ignore[arg-type]

    def test_cross_arm_render_is_rejected_both_directions(self) -> None:
        move = OrderedPrimitive(kind="move", dx=3, dy=4)
        move_to = OrderedPrimitive(kind="move_to", x=3, y=4)
        with self.assertRaises(ValueError):
            sg.render_line((move,), sg.ARM_ABS)
        with self.assertRaises(ValueError):
            sg.render_line((move_to,), sg.ARM_REL)
        self.assertEqual(sg.render_line((move,), sg.ARM_REL), "move(3,4)")
        self.assertEqual(sg.render_line((move_to,), sg.ARM_ABS), "move_to(3,4)")

    def test_render_rejects_out_of_range_and_degenerate_primitives(self) -> None:
        bad_rel = [
            OrderedPrimitive(kind="move", dx=0, dy=0),
            OrderedPrimitive(kind="move", dx=1001, dy=0),
            OrderedPrimitive(kind="move", dx=0, dy=-1001),
            OrderedPrimitive(kind="move", dx=None, dy=3),
            OrderedPrimitive(kind="move", dx=1.5, dy=3),
        ]
        for prim in bad_rel:
            with self.subTest(prim=prim), self.assertRaises(ValueError):
                sg.render_primitive(prim, sg.ARM_REL)
        bad_abs = [
            OrderedPrimitive(kind="move_to", x=-1, y=0),
            OrderedPrimitive(kind="move_to", x=1001, y=0),
            OrderedPrimitive(kind="move_to", x=0, y=1001),
            OrderedPrimitive(kind="move_to", x=None, y=0),
        ]
        for prim in bad_abs:
            with self.subTest(prim=prim), self.assertRaises(ValueError):
                sg.render_primitive(prim, sg.ARM_ABS)

    def test_render_rejects_bad_scroll_names_and_payloads(self) -> None:
        bad = [
            OrderedPrimitive(kind="scroll", dx=0, dy=0),
            OrderedPrimitive(kind="scroll", dx=2, dy=3),
            OrderedPrimitive(kind="scroll", dx=0, dy=None),
            OrderedPrimitive(kind="down", name=""),
            OrderedPrimitive(kind="down", name="LM B"),
            OrderedPrimitive(kind="up", name="a;b"),
            OrderedPrimitive(kind="up", name="a(b"),
            OrderedPrimitive(kind="down", name=None),
            OrderedPrimitive(kind="type", text=""),
            OrderedPrimitive(kind="type", text=None),
            OrderedPrimitive(kind="type", text="line\nbreak"),
            OrderedPrimitive(kind="type", text="tab\there"),
            OrderedPrimitive(kind="click", name="left", count=1),
            OrderedPrimitive(kind="wait"),
            OrderedPrimitive(kind="terminate", status="success"),
            OrderedPrimitive(kind="key_combo", keys=("ctrl", "s")),
        ]
        for arm in sg.ARMS:
            for prim in bad:
                with self.subTest(arm=arm, prim=prim), self.assertRaises(ValueError):
                    sg.render_primitive(prim, arm)


class ParseRejectionTests(unittest.TestCase):
    def test_rejects_empty_and_blank(self) -> None:
        for arm in sg.ARMS:
            for bad in ("", "   ", "\n\n", "\t"):
                with self.subTest(arm=arm, bad=bad), self.assertRaises(ValueError):
                    parse_ordered_v4_action(bad, arm=arm)

    def test_rejects_degenerate_and_out_of_range_mouse(self) -> None:
        for bad in ("move(0,0)", "move(1001,0)", "move(0,-1001)", "move(-2000,2000)"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ordered_v4_action(bad, arm=sg.ARM_REL)
        for bad in ("move_to(1001,0)", "move_to(0,1001)", "move_to(-1,0)",
                    "move_to(1,)", "move_to(1)", "move_to(1,2,3)"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ordered_v4_action(bad, arm=sg.ARM_ABS)

    def test_rejects_scroll_zero_and_v3_two_axis_scroll(self) -> None:
        for arm in sg.ARMS:
            for bad in ("scroll(0)", "scroll(0,-3)", "scroll()", "scroll(1,2)",
                        "scroll(+3)", "scroll(1.5)"):
                with self.subTest(arm=arm, bad=bad), self.assertRaises(ValueError):
                    parse_ordered_v4_action(bad, arm=arm)

    def test_rejects_cross_arm_mouse_primitive_both_directions(self) -> None:
        for bad in ("move(3,4)", "move_to(3,4); down(LMB); up(LMB)"):
            arm = sg.ARM_ABS if bad.startswith("move(") else sg.ARM_REL
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ordered_v4_action(bad, arm=arm)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action("move_to(3,4)", arm=sg.ARM_REL)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action("down(LMB); move(1,1)", arm=sg.ARM_ABS)

    def test_rejects_terminate_line_and_terminate_with_primitives(self) -> None:
        for arm in sg.ARMS:
            for bad in ("TERMINATE", "TERMINATE; down(LMB)", "down(LMB); TERMINATE",
                        "NO_OP; down(LMB)", "TERMINATE NO_OP"):
                with self.subTest(arm=arm, bad=bad), self.assertRaises(ValueError):
                    parse_ordered_v4_action(bad, arm=arm)

    def test_rejects_garbage_and_truncation(self) -> None:
        shared_bad = [
            "hello world",
            "scroll(3) extra",
            "scroll(3);; down(LMB)",
            "down()",
            "down(LM B)",
            "hover(1,2)",
            'type(unquoted)',
            'type("")',
            'type("a\\qb")',
            'type("dangling\\',
            'type("hel',
            "down(LM",
            "up(",
            "; down(LMB)",
            "down(LMB);",
        ]
        for arm in sg.ARMS:
            for bad in shared_bad:
                with self.subTest(arm=arm, bad=bad), self.assertRaises(ValueError):
                    parse_ordered_v4_action(bad, arm=arm)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action("move(3,4); down(LM", arm=sg.ARM_REL)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action("move_to(3,4); type(\"hel", arm=sg.ARM_ABS)

    def test_rejects_non_string(self) -> None:
        for arm in sg.ARMS:
            with self.subTest(arm=arm), self.assertRaises(TypeError):
                parse_ordered_v4_action(None, arm=arm)  # type: ignore[arg-type]
            with self.subTest(arm=arm), self.assertRaises(TypeError):
                parse_ordered_v4_action_tolerant(None, arm=arm)  # type: ignore[arg-type]


class TolerantParserTests(unittest.TestCase):
    def test_takes_last_nonblank_line(self) -> None:
        action = parse_ordered_v4_action_tolerant(
            "I will click the button.\n\nmove(5,6); down(LMB); up(LMB)", arm=sg.ARM_REL
        )
        self.assertEqual(action.primitives[0], OrderedPrimitive(kind="move", dx=5, dy=6))
        action = parse_ordered_v4_action_tolerant(
            "Plan: click Save.\nmove_to(5,6); down(LMB); up(LMB)\n", arm=sg.ARM_ABS
        )
        self.assertEqual(action.primitives[0], OrderedPrimitive(kind="move_to", x=5, y=6))

    def test_strict_path_is_unchanged_for_clean_lines(self) -> None:
        for arm, line in ((sg.ARM_REL, "move(1,2)"), (sg.ARM_ABS, "move_to(1,2)")):
            with self.subTest(arm=arm):
                self.assertEqual(
                    parse_ordered_v4_action_tolerant(line, arm=arm),
                    parse_ordered_v4_action(line, arm=arm),
                )

    def test_still_rejects_prose_only_and_cross_arm(self) -> None:
        for arm in sg.ARMS:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                parse_ordered_v4_action_tolerant("no action here", arm=arm)
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                parse_ordered_v4_action_tolerant("\n \n", arm=arm)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action_tolerant("thinking\nmove(1,2)", arm=sg.ARM_ABS)
        with self.assertRaises(ValueError):
            parse_ordered_v4_action_tolerant("thinking\nmove_to(1,2)", arm=sg.ARM_REL)


class OrderedV3RegressionTests(unittest.TestCase):
    """The v3 parser must behave exactly as before the v4 additions."""

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


if __name__ == "__main__":
    unittest.main()
