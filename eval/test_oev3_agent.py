import sys
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from action_parser import parse_ordered_action
from oev3_agent import (
    Oev3Agent,
    compile_primitives,
    extract_action_line,
    rdev_to_pyautogui,
    strip_think,
)

SCREEN = (1920, 1080)


class StripAndExtractTests(unittest.TestCase):
    def test_strip_think(self):
        self.assertEqual(strip_think("abc</think>\n\nAction: x\nmove(1,2)"), "Action: x\nmove(1,2)")
        self.assertEqual(strip_think("<think>abc</think>\n\nmove(1,2)"), "move(1,2)")

    def test_extract_last_line(self):
        r = "<think>plan</think>\n\nAction: click it.\nmove(-511,-592); down(LMB); up(LMB)"
        self.assertEqual(extract_action_line(r), "move(-511,-592); down(LMB); up(LMB)")

    def test_extract_terminate(self):
        self.assertEqual(extract_action_line("done</think>\n\nTERMINATE"), "TERMINATE")


class CompileTests(unittest.TestCase):
    def _compile(self, line):
        return compile_primitives(parse_ordered_action(line).primitives, SCREEN)

    def test_move_click(self):
        code = self._compile("move(-511,-592); down(LMB); up(LMB)")
        self.assertIn("pyautogui.moveRel(-981, -639)", code)
        self.assertIn("pyautogui.mouseDown(button='left')", code)
        self.assertIn("pyautogui.mouseUp(button='left')", code)

    def test_key_combo(self):
        code = self._compile("down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)")
        self.assertIn("pyautogui.keyDown('ctrlleft')", code)
        self.assertIn("pyautogui.keyDown('c')", code)
        self.assertIn("pyautogui.keyUp('ctrlleft')", code)

    def test_type_escapes(self):
        code = self._compile('type("say \\"hi\\" \\\\ done")')
        self.assertIn("pyautogui.write('say \"hi\" \\\\ done', interval=0.012)", code)

    def test_scroll(self):
        code = self._compile("scroll(0,-500)")
        self.assertIn("pyautogui.scroll(-500)", code)
        self.assertNotIn("hscroll", code)
        code_h = self._compile("scroll(-30,0)")
        self.assertIn("pyautogui.hscroll(-30)", code_h)

    def test_rdev_names(self):
        self.assertEqual(rdev_to_pyautogui("Return"), "enter")
        self.assertEqual(rdev_to_pyautogui("KeyA"), "a")
        self.assertEqual(rdev_to_pyautogui("Num7"), "7")
        self.assertEqual(rdev_to_pyautogui("F5"), "f5")


class MessageAssemblyTests(unittest.TestCase):
    def _agent(self):
        a = Oev3Agent(model="m", history_n=4)
        a.reset()
        return a

    def test_first_turn_layout(self):
        a = self._agent()
        msgs = a._build_messages("do the thing", "IMGB64")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        blocks = [b["type"] for b in msgs[1]["content"]]
        self.assertEqual(blocks, ["image_url", "text"])
        self.assertIn("Previous actions:\nNone", msgs[1]["content"][1]["text"])

    def test_window_layout_and_step_list(self):
        a = self._agent()
        for i in range(6):
            a.screenshots.append(f"S{i}")
            a.stripped_responses.append(f"resp{i}")
            a.action_lines.append(f"move({i},0)")
        msgs = a._build_messages("goal", "CUR")
        roles = [m["role"] for m in msgs]
        self.assertEqual(
            roles,
            ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user"],
        )
        instr = msgs[1]["content"][1]["text"]
        self.assertIn("Step 1: move(0,0)", instr)
        self.assertIn("Step 2: move(1,0)", instr)
        self.assertNotIn("Step 3", instr)
        self.assertEqual(msgs[2]["content"][0]["text"], "resp2")
        self.assertEqual(len(msgs[-1]["content"]), 1)


if __name__ == "__main__":
    unittest.main()


class ElideTests(unittest.TestCase):
    def test_short_line_unchanged(self):
        from oev3_agent import elide_step_line
        self.assertEqual(elide_step_line("move(1,2); down(LMB); up(LMB)"), "move(1,2); down(LMB); up(LMB)")

    def test_long_line_elided(self):
        from oev3_agent import elide_step_line
        line = 'type("' + "x" * 5000 + '")'
        out = elide_step_line(line)
        self.assertLessEqual(len(out), 175)
        self.assertIn("chars]", out)
