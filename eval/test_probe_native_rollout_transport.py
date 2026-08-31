import unittest

from probe_native_rollout_transport import (
    _normalized_importance,
    native_response_parts,
    native_to_oev3,
)


SCREEN = (1920, 1080)


def call(arguments):
    import json

    payload = json.dumps({"name": "computer_use", "arguments": arguments})
    return f"<tool_call>\n{payload}\n</tool_call>"


class NativeRolloutTransportTest(unittest.TestCase):
    def test_native_click_becomes_relative_oev3(self):
        action = call({"action": "left_click", "coordinate": [600, 300]})
        converted, cursor = native_to_oev3(action, (960, 540), SCREEN)
        self.assertEqual(converted, "move(100,-200); down(LMB); up(LMB)")
        self.assertEqual(cursor, (1152, 324))

    def test_non_mouse_actions_are_converted(self):
        key = call({"action": "key", "keys": ["ctrl", "c"]})
        self.assertEqual(
            native_to_oev3(key, (960, 540), SCREEN)[0],
            "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)",
        )
        typed = call({"action": "type", "text": "hello"})
        self.assertEqual(native_to_oev3(typed, (960, 540), SCREEN)[0], 'type("hello")')

    def test_response_is_split_after_reasoning(self):
        action = call({"action": "wait", "time": 1})
        thought, parsed = native_response_parts("reason\n</think>\n\n" + action)
        self.assertEqual(thought, "reason\n</think>\n\n")
        self.assertEqual(parsed, action)

    def test_importance_weights_are_normalized(self):
        weights = _normalized_importance([0.0, 1.0, 2.0])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[2], weights[1])


if __name__ == "__main__":
    unittest.main()
