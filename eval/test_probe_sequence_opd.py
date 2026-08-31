import unittest

from probe_sequence_opd import oev3_to_native


SCREEN = (1920, 1080)


def arguments(result):
    return [call["arguments"] for call in result.calls]


class SequenceOpdConversionTest(unittest.TestCase):
    def test_relative_move_click_becomes_absolute_native_click(self):
        result = oev3_to_native(
            "move(100,-200); down(LMB); up(LMB)", (960, 540), SCREEN
        )
        self.assertEqual(arguments(result), [
            {"action": "left_click", "coordinate": [600, 300]}
        ])
        self.assertEqual(result.cursor_after, (1152, 324))

    def test_click_drag_and_double_click(self):
        drag = oev3_to_native(
            "down(LMB); move(-250,100); up(LMB)", (960, 540), SCREEN
        )
        self.assertEqual(arguments(drag), [
            {"action": "left_click_drag", "coordinate": [250, 600]}
        ])
        double = oev3_to_native(
            "down(LMB); up(LMB); down(LMB); up(LMB)", (960, 540), SCREEN
        )
        self.assertEqual(arguments(double), [
            {"action": "double_click", "coordinate": [500, 500]}
        ])

    def test_non_mouse_actions_are_canonicalized_too(self):
        result = oev3_to_native(
            'down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft); type("hi"); scroll(0,-500)',
            (960, 540),
            SCREEN,
        )
        self.assertEqual(arguments(result), [
            {"action": "key", "keys": ["ctrl", "c"]},
            {"action": "type", "text": "hi"},
            {"action": "scroll", "pixels": -500},
        ])

    def test_clipping_happens_before_absolute_grid_conversion(self):
        result = oev3_to_native("move(-1000,-1000)", (10, 10), SCREEN)
        self.assertEqual(arguments(result), [
            {"action": "mouse_move", "coordinate": [0, 0]}
        ])
        self.assertEqual(result.cursor_after, (0, 0))

    def test_special_actions(self):
        self.assertEqual(arguments(oev3_to_native("NO_OP", (1, 2), SCREEN)), [
            {"action": "wait", "time": 1}
        ])
        self.assertEqual(arguments(oev3_to_native("TERMINATE", (1, 2), SCREEN)), [
            {"action": "terminate", "status": "success"}
        ])


if __name__ == "__main__":
    unittest.main()
