"""Annotation package: method discovery, prompt packs (+sha, snapshot), unit
chunking (submission-snapped cuts, tail buffer), view-local -> master
conversion of a method result, and the plans quality flags. No labeler calls.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.crowdcast.annotation.lib.prompts import PromptPack
from pipeline.crowdcast.annotation.lib.registry import discover_methods, load_method
from pipeline.crowdcast.annotation.lib.units import (
    AnnotationUnit,
    _is_submission,
    build_units,
    is_typing,
    plan_windows,
)
from pipeline.crowdcast.annotation.methods.describe_extract.annotator import (
    clean_goals,
    snap_goal_starts,
)
from pipeline.crowdcast.annotation.methods.plans.annotator import goal_start_frame, plan_flags
from pipeline.crowdcast.lib.action_format import _US_PRINTABLE, WindowKeyboard, get_formatter
from pipeline.crowdcast.lib.events import RawEvent, Window
from pipeline.crowdcast.lib.goals import validate_goal_row, view_span_to_master
from pipeline.crowdcast.lib.views import build_segment_view


def _kb(*names: str) -> WindowKeyboard:
    """One frame's keyboard content. No arguments == nothing typed."""
    return WindowKeyboard(names=names, texts=())


def _view(n_records: int = 150, fps: float = 1.0):
    return build_segment_view(
        {
            "segment_id": "s0",
            "recording_id": "r0",
            "segment_idx": 0,
            "master_fps": 15.0,
            "n_master_records": n_records,
            "shard_path": "/nowhere/frames/s0/images.array_record",
            "keylog_path": None,
            "alignment_status": "aligned",
            "kept_ranges": [[0, n_records]],
            "dropped": [],
        },
        fps=fps,
    )


class RegistryTest(unittest.TestCase):
    def test_discovery_and_kinds(self) -> None:
        methods = discover_methods()
        self.assertIn("describe_extract", methods)
        self.assertIn("plans", methods)
        self.assertEqual(load_method("describe_extract").input_kind, "frames")
        self.assertEqual(load_method("plans").input_kind, "goals")

    def test_unknown_method_refused(self) -> None:
        with self.assertRaises(KeyError):
            load_method("nope")

    def test_prompt_pack_sha_and_snapshot(self) -> None:
        m = load_method("describe_extract")
        self.assertEqual(len(m.prompts.sha), 16)
        # Placeholders resolve; JSON braces pass through untouched.
        rendered = m.prompts.render("describe_prose", n_frames=42, frame_period_s="2")
        self.assertIn("42 frames attached", rendered)
        self.assertNotIn("${n_frames}", rendered)
        extract = m.prompts.render("extract", description="D", n_frames=3, frame_period_s="2")
        self.assertIn('"goals": [', extract)
        with tempfile.TemporaryDirectory() as tmp:
            snap = m.prompts.snapshot_to(Path(tmp))
            self.assertEqual(PromptPack(snap).sha, m.prompts.sha)


class UnitTest(unittest.TestCase):
    def test_single_window_keeps_segment_id(self) -> None:
        view = _view()
        # max_frames_per_window set explicitly: no image decode needed.
        units = build_units(view, [_kb()] * len(view.frames),
                            context_limit=10_000_000, completion_reserve=32000,
                            safety_margin=28000, max_frames_per_window=1000)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].unit_id, "s0")
        self.assertEqual(units[0].tail_buffer, 0)

    def test_split_units_snap_to_submission_and_get_tail_buffer(self) -> None:
        view = _view()
        n = len(view.frames)  # 10 frames
        keyboard = [_kb("KeyA", "KeyA")] * n
        keyboard[6] = _kb("Return", "Return")  # submission: ideal cut after it
        units = build_units(view, keyboard,
                            context_limit=10_000_000, completion_reserve=32000,
                            safety_margin=28000, max_frames_per_window=8,
                            snap_slack=3, tail_buffer=2)
        self.assertEqual([u.unit_id for u in units], ["s0__w0", "s0__w1"])
        self.assertEqual(units[0].hi, 7)  # cut BEFORE frame 7 (after the Return)
        self.assertEqual(units[0].tail_buffer, 2)
        self.assertEqual(units[1].tail_buffer, 0)
        # Sent frames include the buffer; owned range excludes it.
        self.assertEqual(units[0].sent_view_indices, list(range(0, 9)))
        self.assertEqual(units[0].owned_hi_view_idx, 6)

    def test_plan_windows_mid_burst_avoided(self) -> None:
        # All frames mid typing-burst except an idle pair at 8/9: the cut moves
        # off the ideal boundary (10, cost 3: both sides typing) to 9 (cost 1:
        # neither side typing), within the ±3 slack.
        keyboard = [_kb("KeyA")] * 20
        keyboard[8] = keyboard[9] = _kb()
        wins = plan_windows(20, 12, keyboard=keyboard,
                            times=[float(i) for i in range(20)], slack=3)
        self.assertEqual(wins, [(0, 9), (9, 20)])


#: Every key/button name the crowd-cast keylogs spell. Measured 2026-08-20 over
#: the first 3,000 keylogs by segment id of ccast0618d_dataset_full_v3 stage
#: 01+02 -- 2,436 segments with events, 388,113 windows, at the planner's 0.5 fps
#: geometry. Re-measure there; do not extend this by hand. A name that slice does
#: not spell stays pinned: ``common.resolve_key_name`` synthesises ``KC_<code>``
#: for macOS codes it does not map, and widening from 346 segments to 2,436 is
#: what turned up ``Delete`` and ``MMB``. So the awkward ones are here on
#: purpose: ``KC_*``, ``ISO_Section``, the media keys.
_OBSERVED_KEY_NAMES = (
    "Alt", "AltGr", "BackQuote", "BackSlash", "Backspace", "BrightnessDown",
    "BrightnessUp", "CapsLock", "Comma", "ControlLeft", "ControlRight",
    "Delete", "Dot", "DownArrow", "End", "Equal", "Escape", "F12", "F2", "F3",
    "ForwardDelete", "Home", "ISO_Section", "IntlBackslash",
    "KC_143", "KC_160", "KC_176", "KC_179", "KC_325", "KC_330", "KC_333",
    "KC_334", "KC_76",
    *(f"Key{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "LMB", "LeftArrow", "LeftBracket", "MMB", "M_Other_3", "M_Other_4",
    "MetaLeft", "MetaRight", "Minus", "NextTrack",
    *(f"Num{d}" for d in "0123456789"),
    "PageDown", "PageUp", "PlayCd", "PlayPause", "PreviousTrack",
    "PrintScreen", "Quote", "RMB", "Return", "RightArrow", "RightBracket",
    "SemiColon", "ShiftLeft", "ShiftRight", "Slash", "Space", "Tab",
    "UpArrow", "VolumeDown", "VolumeMute", "VolumeUp",
)


def _one_key(action_format: str, name: str) -> tuple[str, WindowKeyboard]:
    """One window holding a single balanced press/release of ``name``."""
    events = [RawEvent(0, 0.01, "press", name=name),
              RawEvent(1, 0.02, "release", name=name)]
    result = get_formatter(action_format).format_segment(
        events, [Window(master_idx=0, start=0, end=30)], [], master_fps=15.0
    )
    return result.labels[0], result.keyboard[0]


class PrintableNameTest(unittest.TestCase):
    """``lib/action_format._US_PRINTABLE`` against the names keylogs really spell."""

    def test_every_printable_name_is_one_the_keylogs_spell(self) -> None:
        """A name that cannot occur folds nothing, so the map is silently short of
        that character. ``BracketLeft``/``BracketRight`` were spelled backwards of
        the keylogs' ``LeftBracket``/``RightBracket``, so ``[`` and ``]`` trained as
        a key press where a printable character belongs."""
        self.assertEqual(set(_US_PRINTABLE) - set(_OBSERVED_KEY_NAMES), set())

    def test_the_bracket_keys_reach_a_typing_burst(self) -> None:
        for name, char in (("LeftBracket", "["), ("RightBracket", "]")):
            self.assertEqual(_one_key("ordered_events_v3", name)[0], f'type("{char}")')


class WindowActivityTest(unittest.TestCase):
    """The planner's two predicates, read off the formatter's structured keyboard
    projection rather than off a rendered label."""

    def test_the_two_grammars_render_the_same_activity_differently(self) -> None:
        """Guards the parity test below from passing because both formatters
        happened to emit the same text."""
        self.assertEqual(_one_key("canonical", "KeyH")[0], "0 0 0 ; +KeyH -KeyH")
        self.assertEqual(_one_key("ordered_events_v3", "KeyH")[0], 'type("h")')

    def test_the_two_grammars_agree_on_every_name_a_keylog_spells(self) -> None:
        """One definition of a text key, so the same keystroke reads the same
        whichever grammar spelled it. This replaces a pinned inventory of the gap
        between two definitions; the gap is closed, so there is nothing to except."""
        for name in _OBSERVED_KEY_NAMES:
            _, deltatype = _one_key("canonical", name)
            _, ordered = _one_key("ordered_events_v3", name)
            self.assertEqual(is_typing(deltatype), is_typing(ordered), name)
            self.assertEqual(_is_submission(deltatype), _is_submission(ordered), name)

    def test_a_burst_is_the_text_keys_plus_the_ones_that_edit_or_commit_one(self) -> None:
        """Parity alone would hold with both grammars always reading False, so the
        verdict itself is pinned. The seven punctuation names are the ones the
        reconciliation moved: the substring marker list called them non-typing.

        Both directions of delete count, or the same correction is a burst or not by
        which way the demonstrator deleted — real keylogs hold windows that are
        nothing but forward-deletes retracting the text typed in the window before,
        and those may not be cut through. ``ForwardDelete`` is the macOS spelling of
        ``Delete``, so dropping either one reinstates the asymmetry per platform."""
        for name in ("KeyA", "Num1", "Space", "Minus", "Slash", "BackSlash", "Comma",
                     "Return", "Backspace", "Delete", "ForwardDelete",
                     "BackQuote", "Dot", "Equal", "Quote", "SemiColon",
                     "LeftBracket", "RightBracket"):
            self.assertTrue(is_typing(_one_key("canonical", name)[1]), name)
        for name in ("Escape", "Tab", "LMB", "ShiftLeft", "ControlLeft",
                     "UpArrow", "VolumeMute", "KC_160", "ISO_Section"):
            self.assertFalse(is_typing(_one_key("canonical", name)[1]), name)


class ViewLocalConversionTest(unittest.TestCase):
    def test_clean_goals_clamps_and_drops_buffer_starts(self) -> None:
        parsed = {"goals": [
            {"instruction": "do a", "start_frame": 2, "end_frame": 5},
            {"instruction": "in buffer", "start_frame": 8, "end_frame": 9},  # past own_hi
            {"instruction": "swapped", "start_frame": 6, "end_frame": 4},
            {"instruction": ""},  # empty: dropped
            {"instruction": "unbounded"},
        ]}
        goals = clean_goals(parsed, frame_lo=0, frame_hi=9, own_hi=7)
        self.assertEqual([g["instruction"] for g in goals], ["do a", "swapped", "unbounded"])
        self.assertEqual((goals[1]["start_frame"], goals[1]["end_frame"]), (4, 6))
        self.assertIsNone(goals[2]["start_frame"])

    def test_snap_goal_starts_walks_back_typing_burst(self) -> None:
        view = _view()
        keyboard = [_kb()] * len(view.frames)
        keyboard[3] = _kb("KeyH", "KeyH")
        keyboard[4] = _kb("KeyI", "KeyI")
        keyboard[5] = _kb("Return", "Return")
        unit = AnnotationUnit(unit_id="s0", view=view, window_index=0, n_windows=1,
                              lo=0, hi=len(view.frames), tail_buffer=0, keyboard=keyboard)
        goals = [{"instruction": "x", "start_frame": 4, "end_frame": 6}]
        snap_goal_starts(goals, unit)
        self.assertEqual(goals[0]["start_frame"], 3)  # pulled to burst start
        # A mouse goal is untouched.
        goals2 = [{"instruction": "y", "start_frame": 8, "end_frame": 9}]
        snap_goal_starts(goals2, unit)
        self.assertEqual(goals2[0]["start_frame"], 8)

    def test_view_span_to_master_roundtrip_row_validates(self) -> None:
        view = _view()
        start_m, end_m = view_span_to_master(view, 2, 6)
        row = {
            "goal_id": "s0_g00", "segment_id": "s0", "recording_id": "r0",
            "start_master_idx": start_m, "end_master_idx": end_m,
            "instruction": "do the thing", "method": "describe_extract",
            "model": "m", "prompt_pack_sha": "abc", "unit_id": "s0",
        }
        validate_goal_row(row)
        self.assertEqual((start_m, end_m), (30, 90))


class PlansHelpersTest(unittest.TestCase):
    def test_plan_flags(self) -> None:
        self.assertEqual(plan_flags("", "any"), ["empty"])
        self.assertIn("restates_instruction",
                      plan_flags("I need to open the settings page. I'll open the settings page.",
                                 "open the settings page"))
        good = ("The build failed on the missing import earlier, so I'll check the "
                "dependency list first.")
        self.assertEqual(plan_flags(good, "fix the build"), [])
        self.assertIn("not_first_person", plan_flags("Open the file and edit it.", "edit the file"))

    def test_goal_start_frame_matching(self) -> None:
        view = _view()  # frames at ticks 0,15,...,135
        self.assertEqual(goal_start_frame(view, 45).master_idx, 45)   # exact
        self.assertEqual(goal_start_frame(view, 50).master_idx, 60)   # nearest after
        self.assertEqual(goal_start_frame(view, 149).master_idx, 135)  # before (tail)


if __name__ == "__main__":
    unittest.main()
