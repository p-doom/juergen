from __future__ import annotations

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
