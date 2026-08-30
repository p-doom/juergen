from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import freeroll
import osworld_vm_client


class FreerollHelperTests(unittest.TestCase):
    def test_parse_instructions_splits_nonempty_noncomment_lines(self) -> None:
        self.assertEqual(
            freeroll._parse_instructions("first\n\n# skip\nsecond\n"),
            ["first", "second"],
        )

    def test_parse_instructions_preserves_legacy_empty_behavior(self) -> None:
        self.assertEqual(freeroll._parse_instructions(None), [None])
        self.assertEqual(freeroll._parse_instructions("\n# only comments"), [None])

    def test_terminate_is_detected_as_first_line_token(self) -> None:
        self.assertTrue(freeroll._is_terminate("TERMINATE"))
        self.assertTrue(freeroll._is_terminate(" TERMINATE\nignored"))
        self.assertFalse(freeroll._is_terminate("NO_OP"))

    def test_rdev_key_mapping_covers_typing_punctuation_and_digits(self) -> None:
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("KeyA"), "a")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Digit1"), "1")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Comma"), ",")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Quote"), "'")


if __name__ == "__main__":
    unittest.main()
