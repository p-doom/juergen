from __future__ import annotations

import sys
import unittest
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from cuagym_pipeline.onpolicy_curriculum import (
    classify_verifier,
    is_near_duplicate,
    jaccard,
    match_gap_signatures,
    normalize_tokens,
    strip_comments_and_docstrings,
)

PERSIST_SOURCE = '''
import os
import json

def verify():
    path = os.path.join('/home/user', 'out.json')
    with open(path) as f:
        data = json.load(f)
    return 1.0 if data.get('done') else 0.0

print('REWARD:', verify())
'''

TRANSIENT_SOURCE = '''
import pyautogui

def verify():
    shot = pyautogui.screenshot()
    px = pyautogui.pixel(10, 10)
    return 1.0 if px == (255, 0, 0) else 0.0

print('REWARD:', verify())
'''

SAVE_FLUSH_SOURCE = '''
import os
import pyautogui

def verify():
    pyautogui.hotkey('ctrl', 's')
    return 1.0 if os.path.exists('/home/user/Desktop/report.odt') else 0.0

print('REWARD:', verify())
'''

SAVED_IMAGE_PIXEL_SOURCE = '''
from PIL import Image

def verify():
    img = Image.open('/home/user/Desktop/out.png')
    return 1.0 if img.getpixel((0, 0)) == (255, 0, 0) else 0.0

print('REWARD:', verify())
'''

DOCSTRING_RED_HERRING_SOURCE = '''
"""Reward script: take a screenshot with pyautogui and inspect pixels via xdotool."""
import os

def verify():
    return 1.0 if os.path.exists('/home/user/report.pdf') else 0.0

print('REWARD:', verify())
'''

UNKNOWN_SOURCE = '''
def verify():
    return 0.5 + 0.5

print('REWARD:', verify())
'''

MIXED_TRANSIENT_LEANING_SOURCE = '''
import os

def verify():
    out = os.popen('wmctrl -l').read()
    return 1.0 if 'Writer' in out else 0.0

print('REWARD:', verify())
'''


class ClassifierTest(unittest.TestCase):
    def test_persist(self):
        cls, hits = classify_verifier(PERSIST_SOURCE)
        self.assertEqual(cls, "persist_verified")
        self.assertIn("file_io", hits["persist"])

    def test_transient(self):
        cls, hits = classify_verifier(TRANSIENT_SOURCE)
        self.assertEqual(cls, "transient_screen")
        self.assertIn("screen_capture", hits["transient"])

    def test_docstring_does_not_flip_class(self):
        cls, hits = classify_verifier(DOCSTRING_RED_HERRING_SOURCE)
        self.assertEqual(cls, "persist_verified")
        self.assertEqual(hits["transient"], [])

    def test_unknown(self):
        cls, _ = classify_verifier(UNKNOWN_SOURCE)
        self.assertEqual(cls, "unknown")

    def test_mixed_leans_transient(self):
        cls, hits = classify_verifier(MIXED_TRANSIENT_LEANING_SOURCE)
        self.assertEqual(cls, "transient_screen")
        self.assertIn("window_inspection", hits["transient"])

    def test_save_flush_hotkey_is_persist(self):
        cls, hits = classify_verifier(SAVE_FLUSH_SOURCE)
        self.assertEqual(cls, "persist_verified")
        self.assertEqual(hits["transient"], [])

    def test_saved_image_pixel_check_is_persist(self):
        cls, hits = classify_verifier(SAVED_IMAGE_PIXEL_SOURCE)
        self.assertEqual(cls, "persist_verified")
        self.assertEqual(hits["transient"], [])

    def test_empty_source(self):
        cls, _ = classify_verifier("   \n")
        self.assertEqual(cls, "empty")

    def test_strip_removes_comments(self):
        stripped = strip_comments_and_docstrings("x = 1  # pyautogui screenshot\n")
        self.assertNotIn("pyautogui", stripped)


class JaccardTest(unittest.TestCase):
    def test_identical(self):
        t = normalize_tokens("Rename the file to report.pdf")
        self.assertEqual(jaccard(t, t), 1.0)

    def test_disjoint(self):
        a = normalize_tokens("alpha beta gamma")
        b = normalize_tokens("delta epsilon zeta")
        self.assertEqual(jaccard(a, b), 0.0)

    def test_empty(self):
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_near_duplicate_flagged(self):
        a = "Please change the font size of the title to 24 in the presentation slide"
        b = "Please change the font size of the title to 26 in the presentation slide"
        dup, j, sub = is_near_duplicate(a, b, 0.8, 5)
        self.assertTrue(dup)
        self.assertGreaterEqual(j, 0.8)
        self.assertFalse(sub)

    def test_substring_flagged(self):
        a = "Export the current sheet as a CSV file into the Documents folder"
        b = "In LibreOffice Calc, export the current sheet as a CSV file into the Documents folder, then close the app"
        dup, _, sub = is_near_duplicate(a, b, 0.8, 5)
        self.assertTrue(dup)
        self.assertTrue(sub)

    def test_unrelated_not_flagged(self):
        a = "Mute the video playback in VLC"
        b = "Create a pivot table summarizing quarterly sales by region"
        dup, _, _ = is_near_duplicate(a, b, 0.8, 5)
        self.assertFalse(dup)

    def test_short_substring_not_flagged(self):
        dup, _, _ = is_near_duplicate("open file", "open file manager and browse", 0.8, 5)
        self.assertFalse(dup)


class GapSignatureTest(unittest.TestCase):
    def test_calc_dialog_requires_calc(self):
        instr = "Apply data validation with a list entry to column B"
        self.assertIn("calc_dialog_list_entry", match_gap_signatures(instr, "libreoffice_calc"))
        self.assertNotIn("calc_dialog_list_entry", match_gap_signatures(instr, "vscode"))

    def test_menu_word_boundary(self):
        self.assertIn("app_menu_navigation", match_gap_signatures("Use the Format menu to bold", "vscode"))
        self.assertEqual(match_gap_signatures("Open menus.txt configuration", "vscode"), [])


if __name__ == "__main__":
    unittest.main()
