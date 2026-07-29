"""Unit tests for the canonical -> Mihir IDM action-format converter.

Spot-checks that known canonical action strings convert to the exact Mihir
serialized form (compact JSON, frame/type/details, per-frame type order), that
mouse deltas normalize per-axis by the recording resolution, that clicks/keys/
scroll map correctly, that modifiers fold across turns, and that no input crashes
the converter.
"""

from __future__ import annotations

import json
import unittest

from realigned_pipeline.lib.mihir_action_format import (
    ConversionCounters,
    convert_conversation,
    convert_turn,
    normalize_key_name,
    format_key_with_modifiers,
)

W = 2000  # easy round numbers: /2000*1000 = /2, /1000*1000 = identity
H = 1000


def conv(text, held=None, w=W, h=H, terminal=("TERMINATE",)):
    held = set() if held is None else held
    c = ConversionCounters()
    out = convert_turn(text, held, video_w=w, video_h=h, counters=c, terminal_tokens=terminal)
    return out, c


class TestKeyNormalization(unittest.TestCase):
    def test_letters_digits_specials(self):
        self.assertEqual(normalize_key_name("KeyA"), "A")
        self.assertEqual(normalize_key_name("Digit7"), "7")
        self.assertEqual(normalize_key_name("Return"), "Return")
        self.assertEqual(normalize_key_name("MetaLeft"), "Cmd")
        self.assertEqual(normalize_key_name("ControlRight"), "Ctrl")
        self.assertEqual(normalize_key_name("F5"), "F5")
        self.assertEqual(normalize_key_name("KC_179"), "KC_179")

    def test_modifier_order(self):
        self.assertEqual(
            format_key_with_modifiers("A", {"Shift", "Cmd", "Ctrl", "Alt"}),
            "Cmd+Ctrl+Alt+Shift+A",
        )
        self.assertEqual(format_key_with_modifiers("C", {"Cmd"}), "Cmd+C")


class TestSentinels(unittest.TestCase):
    def test_noop_to_empty_array(self):
        out, c = conv("NO_OP")
        self.assertEqual(out, "[]")
        self.assertEqual(c.n_noop, 1)

    def test_terminate_passthrough(self):
        out, c = conv("TERMINATE")
        self.assertEqual(out, "TERMINATE")
        self.assertEqual(c.n_terminal, 1)

    def test_custom_terminal_token(self):
        out, _ = conv("<terminate>", terminal=("<terminate>",))
        self.assertEqual(out, "<terminate>")

    def test_appended_terminal_token(self):
        out, c = conv("100 0 0\nTERMINATE")
        arr, _, tail = out.partition("\n")
        self.assertEqual(tail, "TERMINATE")
        self.assertEqual(json.loads(arr), [{"frame": "F00", "type": "MouseMove", "details": "50,0"}])
        self.assertEqual(c.n_terminal, 1)


class TestMouseMove(unittest.TestCase):
    def test_normalization_per_axis(self):
        # dx=100 /2000*1000 = 50 ; dy=-250 /1000*1000 = -250
        out, c = conv("100 -250 0")
        self.assertEqual(out, '[{"frame":"F00","type":"MouseMove","details":"50,-250"}]')
        self.assertEqual(c.n_mousemove, 1)

    def test_compact_json_no_spaces(self):
        out, _ = conv("100 0 0")
        self.assertNotIn(", ", out)
        self.assertNotIn('": ', out)

    def test_rounding(self):
        # dx=1 /2000*1000 = 0.5 -> round-half-to-even = 0 ; still emitted (raw nonzero)
        out, _ = conv("1 0 0")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "MouseMove", "details": "0,0"}])


class TestScroll(unittest.TestCase):
    def test_sign_flip_and_norm(self):
        # our scroll=-50 (down) ; mihir = round(-(-50)/1000*1000) = 50 (positive=down)
        out, c = conv("0 0 -50")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "MouseScroll", "details": "50"}])
        self.assertEqual(c.n_mousescroll, 1)

    def test_positive_up(self):
        out, _ = conv("0 0 30")  # our +30 up -> mihir -30
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "MouseScroll", "details": "-30"}])


class TestClicksAndKeys(unittest.TestCase):
    def test_left_click(self):
        out, c = conv("37 0 0 ; +LMB -LMB")
        # dx=37/2000*1000=18.5 -> 18 (round half to even) ; MouseClick then MouseMove
        self.assertEqual(
            json.loads(out),
            [
                {"frame": "F00", "type": "MouseClick", "details": "Left"},
                {"frame": "F00", "type": "MouseMove", "details": "18,0"},
            ],
        )
        self.assertEqual(c.n_mouseclick, 1)
        self.assertEqual(c.n_mousemove, 1)

    def test_right_middle_click(self):
        out, _ = conv("0 0 0 ; +RMB -RMB")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "MouseClick", "details": "Right"}])
        out, _ = conv("0 0 0 ; +MMB -MMB")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "MouseClick", "details": "Middle"}])

    def test_plain_letter_key(self):
        out, c = conv("0 0 0 ; +KeyA -KeyA")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "KeyPress", "details": "A"}])
        self.assertEqual(c.n_keypress, 1)

    def test_modifier_fold_within_turn(self):
        out, _ = conv("0 0 0 ; +MetaLeft +KeyC -KeyC -MetaLeft")
        self.assertEqual(json.loads(out), [{"frame": "F00", "type": "KeyPress", "details": "Cmd+C"}])

    def test_bare_modifier_emits_empty(self):
        held = set()
        out, c = conv("0 0 0 ; +ShiftLeft", held=held)
        self.assertEqual(out, "[]")
        self.assertEqual(c.n_empty_array, 1)
        self.assertIn("Shift", held)  # state persists for next turn

    def test_type_order_key_before_click_before_move(self):
        out, _ = conv("10 0 0 ; +LMB -LMB +KeyA -KeyA")
        types = [o["type"] for o in json.loads(out)]
        self.assertEqual(types, ["KeyPress", "MouseClick", "MouseMove"])

    def test_dedup_within_turn(self):
        out, _ = conv("0 0 0 ; +KeyA -KeyA +KeyA -KeyA")
        self.assertEqual(len(json.loads(out)), 1)

    def test_unknown_key_dropped(self):
        out, c = conv("0 0 0 ; +KC_179 -KC_179")
        self.assertEqual(out, "[]")
        self.assertEqual(c.n_dropped_unknown_key, 1)

    def test_extra_mouse_button_dropped(self):
        out, c = conv("0 0 0 ; +M_Button4 -M_Button4")
        self.assertEqual(out, "[]")
        self.assertEqual(c.n_dropped_unknown_button, 1)


class TestCrossTurnState(unittest.TestCase):
    def test_shift_held_across_turns(self):
        # Turn 1 presses Shift (no emit); turn 2 types A -> Shift+A; turn 3 releases.
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "0 0 0 ; +ShiftLeft"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "0 0 0 ; +KeyA -KeyA"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "0 0 0 ; -ShiftLeft"}]},
        ]
        c = ConversionCounters()
        convert_conversation(msgs, video_w=W, video_h=H, counters=c)
        texts = [m["content"][0]["text"] for m in msgs if m["role"] == "assistant"]
        self.assertEqual(texts[0], "[]")
        self.assertEqual(json.loads(texts[1]), [{"frame": "F00", "type": "KeyPress", "details": "Shift+A"}])
        self.assertEqual(texts[2], "[]")


class TestRobustness(unittest.TestCase):
    def test_no_input_crashes(self):
        samples = [
            "NO_OP",
            "TERMINATE",
            "0 0 0",
            "1 -1 1",
            "999 -999 5 ; +LMB -LMB",
            "0 0 0 ; +ControlLeft +KeyC -KeyC -ControlLeft",
            "0 0 0 ; +Return -Return",
            "0 0 0 ; +Space -Space",
            "-40 -33 0 ; +LMB",
            "",
        ]
        c = ConversionCounters()
        held = set()
        for s in samples:
            out = convert_turn(s, held, video_w=W, video_h=H, counters=c)
            self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
