import sys
import unittest
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from action_parser import parse_ordered_action
from cuagym_pipeline.key_names import UnmappableKeyError, pyautogui_to_rdev
from cuagym_pipeline.translate import (
    DropStep,
    StepTranslation,
    px_to_norm,
    reconstruct_target_px,
    rewrite_assistant,
    translate_step,
)

SCREEN = (1920, 1080)


def parses(line: str) -> bool:
    parse_ordered_action(line)
    return True


class KeyNamesTests(unittest.TestCase):
    def test_common_maps(self):
        self.assertEqual(pyautogui_to_rdev("ctrl"), "ControlLeft")
        self.assertEqual(pyautogui_to_rdev("enter"), "Return")
        self.assertEqual(pyautogui_to_rdev("Esc"), "Escape")
        self.assertEqual(pyautogui_to_rdev("a"), "KeyA")
        self.assertEqual(pyautogui_to_rdev("Z"), "KeyZ")
        self.assertEqual(pyautogui_to_rdev("7"), "Num7")
        self.assertEqual(pyautogui_to_rdev("f5"), "F5")
        self.assertEqual(pyautogui_to_rdev("f12"), "F12")
        self.assertEqual(pyautogui_to_rdev(","), "Comma")
        self.assertEqual(pyautogui_to_rdev("win"), "MetaLeft")

    def test_passthrough_valid_name(self):
        self.assertEqual(pyautogui_to_rdev("printscreen"), "printscreen")

    def test_rejects_grammar_violations(self):
        with self.assertRaises(UnmappableKeyError):
            pyautogui_to_rdev("(")
        with self.assertRaises(UnmappableKeyError):
            pyautogui_to_rdev("")


class TranslateStepTests(unittest.TestCase):
    def test_dataset_verified_click(self):
        t = translate_step(
            {"action": "left_click", "coordinate": [389, 308]}, (1728, 972), SCREEN
        )
        self.assertEqual(t.line, "move(-511,-592); down(LMB); up(LMB)")
        self.assertTrue(parses(t.line))
        self.assertEqual(t.move_delta, (-511, -592))
        self.assertEqual(
            reconstruct_target_px((1728, 972), t.move_delta, SCREEN), (747, 333)
        )

    def test_click_at_cursor_no_move(self):
        t = translate_step(
            {"action": "left_click", "coordinate": [900, 900]}, (1728, 972), SCREEN
        )
        self.assertEqual(t.line, "down(LMB); up(LMB)")
        self.assertEqual(t.move_delta, (0, 0))

    def test_click_without_coordinate(self):
        t = translate_step({"action": "right_click"}, (10, 10), SCREEN)
        self.assertEqual(t.line, "down(RMB); up(RMB)")
        self.assertIsNone(t.move_delta)

    def test_double_and_triple_click(self):
        t2 = translate_step(
            {"action": "double_click", "coordinate": [500, 500]}, (0, 0), SCREEN
        )
        self.assertEqual(
            t2.line, "move(500,500); down(LMB); up(LMB); down(LMB); up(LMB)"
        )
        t3 = translate_step({"action": "triple_click"}, (0, 0), SCREEN)
        self.assertEqual(t3.line, "down(LMB); up(LMB); " * 2 + "down(LMB); up(LMB)")

    def test_drag(self):
        t = translate_step(
            {"action": "left_click_drag", "coordinate": [100, 200]}, (0, 0), SCREEN
        )
        self.assertEqual(t.line, "down(LMB); move(100,200); up(LMB)")
        self.assertTrue(parses(t.line))

    def test_key_combo_press_reverse_release(self):
        t = translate_step({"action": "key", "keys": ["ctrl", "c"]}, (0, 0), SCREEN)
        self.assertEqual(
            t.line, "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)"
        )
        self.assertTrue(parses(t.line))

    def test_key_single(self):
        t = translate_step({"action": "key", "keys": ["enter"]}, (0, 0), SCREEN)
        self.assertEqual(t.line, "down(Return); up(Return)")

    def test_type_plain(self):
        t = translate_step({"action": "type", "text": "hello; (x)"}, (0, 0), SCREEN)
        self.assertEqual(t.line, 'type("hello; (x)")')
        self.assertTrue(parses(t.line))

    def test_type_with_newline_and_tab(self):
        t = translate_step({"action": "type", "text": "a\nb\tc"}, (0, 0), SCREEN)
        self.assertEqual(
            t.line,
            'type("a"); down(Return); up(Return); type("b"); down(Tab); up(Tab); type("c")',
        )
        self.assertTrue(parses(t.line))

    def test_type_escapes(self):
        t = translate_step({"action": "type", "text": 'say "hi" \\ done'}, (0, 0), SCREEN)
        self.assertEqual(t.line, 'type("say \\"hi\\" \\\\ done")')
        self.assertTrue(parses(t.line))

    def test_scroll_signs(self):
        up = translate_step({"action": "scroll", "pixels": 500}, (0, 0), SCREEN)
        self.assertEqual(up.line, "scroll(0,500)")
        down = translate_step({"action": "scroll", "pixels": -500.0}, (0, 0), SCREEN)
        self.assertEqual(down.line, "scroll(0,-500)")
        h = translate_step({"action": "hscroll", "pixels": -30}, (0, 0), SCREEN)
        self.assertEqual(h.line, "scroll(-30,0)")

    def test_scroll_zero_is_noop(self):
        t = translate_step({"action": "scroll", "pixels": 0}, (0, 0), SCREEN)
        self.assertEqual(t.line, "NO_OP")

    def test_wait_and_screenshot_noop(self):
        self.assertEqual(
            translate_step({"action": "wait", "time": 2}, (0, 0), SCREEN).line, "NO_OP"
        )
        self.assertEqual(
            translate_step({"action": "screenshot"}, (0, 0), SCREEN).line, "NO_OP"
        )

    def test_terminate(self):
        t = translate_step(
            {"action": "terminate", "status": "success"}, (0, 0), SCREEN
        )
        self.assertEqual(t.line, "TERMINATE")

    def test_mouse_move(self):
        t = translate_step(
            {"action": "mouse_move", "coordinate": [900, 900]}, (960, 540), SCREEN
        )
        self.assertEqual(t.line, "move(400,400)")
        zero = translate_step(
            {"action": "mouse_move", "coordinate": [500, 500]}, (960, 540), SCREEN
        )
        self.assertEqual(zero.line, "NO_OP")

    def test_mouse_down_up(self):
        t = translate_step(
            {"action": "left_mouse_down", "coordinate": [10, 10]}, (0, 0), SCREEN
        )
        self.assertEqual(t.line, "move(10,10); down(LMB)")
        u = translate_step({"action": "left_mouse_up"}, (0, 0), SCREEN)
        self.assertEqual(u.line, "up(LMB)")

    def test_drop_actions(self):
        t = translate_step({"action": "call_user", "text": "?"}, (0, 0), SCREEN)
        self.assertEqual(t.dropped_reason, "call_user")

    def test_unknown_action_raises(self):
        with self.assertRaises(DropStep):
            translate_step({"action": "warp"}, (0, 0), SCREEN)

    def test_malformed_coordinate_raises(self):
        with self.assertRaises(DropStep):
            translate_step(
                {"action": "left_click", "coordinate": [1, 2, 3]}, (0, 0), SCREEN
            )


class InvertibilityTests(unittest.TestCase):
    def test_grid_sweep(self):
        for cx in range(0, 1920, 191):
            for cy in range(0, 1080, 107):
                for tx, ty in ((0, 0), (999, 999), (389, 308), (500, 421)):
                    t = translate_step(
                        {"action": "left_click", "coordinate": [tx, ty]},
                        (cx, cy),
                        SCREEN,
                    )
                    if t.move_delta is None:
                        continue
                    px = reconstruct_target_px((cx, cy), t.move_delta, SCREEN)
                    expected = (
                        round(tx / 1000 * 1920),
                        round(ty / 1000 * 1080),
                    )
                    self.assertEqual(px, expected)


class RewriteAssistantTests(unittest.TestCase):
    def test_rewrite(self):
        raw = (
            "I should click the cell.</think>\n\n"
            'Action: Click the header cell.\n<tool_call>\n{"name": "computer_use", '
            '"arguments": {"action": "left_click", "coordinate": [389, 308]}}\n</tool_call>'
        )
        out = rewrite_assistant(raw, "move(-511,-592); down(LMB); up(LMB)")
        self.assertEqual(
            out,
            "<think>I should click the cell.</think>\n\n"
            "Action: Click the header cell.\n"
            "move(-511,-592); down(LMB); up(LMB)",
        )

    def test_rewrite_requires_tool_call(self):
        with self.assertRaises(DropStep):
            rewrite_assistant("no call here</think>", "NO_OP")

    def test_rewrite_requires_think_close(self):
        with self.assertRaises(DropStep):
            rewrite_assistant("<tool_call>x</tool_call>", "NO_OP")


if __name__ == "__main__":
    unittest.main()


class ClampTests(unittest.TestCase):
    def test_out_of_grid_coordinate_clamps_to_screen(self):
        t = translate_step(
            {"action": "mouse_move", "coordinate": [1548, 1521]}, (1728, 972), SCREEN
        )
        px = reconstruct_target_px((1728, 972), t.move_delta, SCREEN)
        self.assertLessEqual(abs(px[0] - 1919), 1)
        self.assertLessEqual(abs(px[1] - 1079), 1)

    def test_negative_coordinate_clamps_to_zero(self):
        t = translate_step(
            {"action": "left_click", "coordinate": [-5, -9]}, (960, 540), SCREEN
        )
        self.assertEqual(reconstruct_target_px((960, 540), t.move_delta, SCREEN), (0, 0))
