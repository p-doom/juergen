"""Foreground-app track + the stage-04 application filter.

Covers the three properties the pipeline relies on:
  * the tick fold is forward-fill STATE, and LAST WINS at equal timestamps (the
    realign pause-collapse clamps a whole run of switches onto one instant, and
    the last of those is the app in focus when recording resumed),
  * an UNCAPTURED (privacy blackout) span does not break a same-app run, but a
    different app always does,
  * splitting a segment by app cuts on the seams and leaves every surviving
    turn's action untouched.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import msgpack

from pipeline.crowdcast.lib.app_context import (
    UNCAPTURED,
    UNKNOWN,
    AppTrack,
    build_app_track,
    iter_app_spans,
    load_app_track,
    normalize_app_id,
    resolve_app_selector,
)
from pipeline.crowdcast.lib.events import iter_context, load_events
from pipeline.crowdcast.lib.app_filter import (
    AppFilter,
    label_view_frames,
    app_stats,
    plan_app_spans,
    split_app_selectors,
)

MASTER_FPS = 15.0
SAFARI = "com.apple.Safari"
TERMINAL = "com.apple.Terminal"
FIREFOX = "org.mozilla.firefox"
GHOSTTY = "com.mitchellh.ghostty"


def _us(t_s: float) -> int:
    return round(t_s * 1_000_000)


def _write_keylog(entries: list) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "input_seg.msgpack"
    tmp.write_bytes(msgpack.packb(entries, use_bin_type=True))
    return tmp


def _frames(apps: list[str | None], *, seams: set[int] | None = None) -> list[dict]:
    """One frame per app label, one master tick apart; ``seams`` marks the frames
    whose action window straddles a switch (stage 03's app_window_switches)."""
    seams = seams or set()
    return [
        {
            "global_frame_idx": i,
            "master_record_index": i,
            "action": f"act{i}",
            "app": app,
            "app_window_switches": 1 if i in seams else 0,
        }
        for i, app in enumerate(apps)
    ]


class TestParse(unittest.TestCase):
    def test_iter_context_reads_what_iter_events_skips(self) -> None:
        path = _write_keylog([
            [_us(0.0), ["ContextChanged", [SAFARI]]],
            [_us(0.5), ["KeyPress", [59, "KeyA"]]],
            [_us(0.6), ["KeyRelease", [59, "KeyA"]]],
            [_us(1.0), ["ContextChanged", [FIREFOX]]],
        ])
        self.assertEqual(
            list(iter_context(path)), [(0.0, SAFARI), (1.0, FIREFOX)]
        )
        # ...and the action path still ignores them entirely.
        events, stats = load_events(path)
        self.assertEqual([e.kind for e in events], ["press", "release"])
        self.assertEqual(stats.n_keypress, 1)

    def test_timemap_applies(self) -> None:
        path = _write_keylog([[_us(100.0), ["ContextChanged", [SAFARI]]]])
        self.assertEqual(
            list(iter_context(path, lambda t: t - 29.0)), [(71.0, SAFARI)]
        )

    def test_missing_keylog_is_unknown_not_a_crash(self) -> None:
        track = load_app_track(Path("/nonexistent/x.msgpack"), n_ticks=10,
                               master_fps=MASTER_FPS)
        self.assertEqual(track.at(0), UNKNOWN)
        self.assertEqual(track.runs(), [])

    def test_normalization_and_selectors(self) -> None:
        self.assertEqual(normalize_app_id("firefox"), FIREFOX)
        self.assertEqual(normalize_app_id("com.google.antigravity-ide"),
                         "com.google.antigravity")
        self.assertEqual(normalize_app_id("brand.new.app"), "brand.new.app")
        self.assertEqual(resolve_app_selector("cursor"), "com.todesktop.230313mzl4w4u92")
        self.assertEqual(resolve_app_selector(FIREFOX), FIREFOX)
        self.assertEqual(resolve_app_selector("uncaptured"), UNCAPTURED)
        with self.assertRaises(ValueError):
            resolve_app_selector("  ")


class TestSelectorFlattening(unittest.TestCase):
    """labctl renders one --key=value per arg, so a comma list must work too."""

    def test_repeatable_and_comma_separated(self) -> None:
        self.assertEqual(split_app_selectors(None), [])
        self.assertEqual(split_app_selectors(["firefox"]), ["firefox"])
        self.assertEqual(split_app_selectors(["firefox,safari"]), ["firefox", "safari"])
        self.assertEqual(split_app_selectors(["firefox", "safari,arc"]),
                         ["firefox", "safari", "arc"])
        self.assertEqual(split_app_selectors(["firefox, ,safari"]), ["firefox", "safari"])
        self.assertEqual(
            [resolve_app_selector(a) for a in split_app_selectors(["firefox,cursor"])],
            [FIREFOX, "com.todesktop.230313mzl4w4u92"],
        )


class TestTrack(unittest.TestCase):
    def test_forward_fill_and_head_is_unknown(self) -> None:
        track = build_app_track([(1.0, SAFARI), (2.0, FIREFOX)],
                                n_ticks=45, master_fps=MASTER_FPS)
        self.assertEqual(track.at(0), UNKNOWN)      # before the first event
        self.assertEqual(track.at(15), SAFARI)
        self.assertEqual(track.at(29), SAFARI)
        self.assertEqual(track.at(30), FIREFOX)
        self.assertEqual(track.at(44), FIREFOX)

    def test_last_wins_on_a_collapsed_pause(self) -> None:
        """realign clamps every switch inside a collapsed pause to the splice point,
        so the run shares one timestamp -- the LAST is the app at resume."""
        track = build_app_track(
            [(0.0, SAFARI), (10.0, GHOSTTY), (10.0, UNCAPTURED), (10.0, FIREFOX)],
            n_ticks=300, master_fps=MASTER_FPS,
        )
        self.assertEqual(track.at(150), FIREFOX)
        self.assertEqual(track.switch_ticks, [0, 150])

    def test_refocus_of_same_app_is_not_a_switch(self) -> None:
        track = build_app_track([(0.0, SAFARI), (1.0, SAFARI), (2.0, FIREFOX)],
                                n_ticks=45, master_fps=MASTER_FPS)
        self.assertEqual(track.switch_ticks, [0, 30])
        self.assertEqual(track.summary()["n_app_switches"], 1)

    def test_counts_are_interval_exact(self) -> None:
        track = build_app_track([(0.0, SAFARI), (1.0, FIREFOX)],
                                n_ticks=30, master_fps=MASTER_FPS)
        self.assertEqual(dict(track.counts(0, 30)), {SAFARI: 15, FIREFOX: 15})
        self.assertEqual(dict(track.counts(10, 20)), {SAFARI: 5, FIREFOX: 5})
        self.assertEqual(dict(track.counts(5, 5)), {})

    def test_counts_head_span_before_first_event(self) -> None:
        track = build_app_track([(1.0, SAFARI)], n_ticks=30, master_fps=MASTER_FPS)
        self.assertEqual(dict(track.counts(0, 30)), {UNKNOWN: 15, SAFARI: 15})

    def test_switches_in_window(self) -> None:
        track = build_app_track([(0.0, SAFARI), (0.5, FIREFOX), (2.0, GHOSTTY)],
                                n_ticks=45, master_fps=MASTER_FPS)
        self.assertEqual(track.switches_in(0, 15), [7])   # the 0.5 s switch
        self.assertEqual(track.switches_in(15, 30), [])
        self.assertEqual(track.switches_in(30, 45), [])

    def test_runs_merge_across_uncaptured_but_not_across_apps(self) -> None:
        track = build_app_track(
            [(0.0, SAFARI), (1.0, UNCAPTURED), (2.0, SAFARI), (3.0, FIREFOX)],
            n_ticks=60, master_fps=MASTER_FPS,
        )
        runs = track.runs()
        self.assertEqual([(r.app, r.start, r.end) for r in runs],
                         [(SAFARI, 0, 45), (FIREFOX, 45, 60)])
        self.assertAlmostEqual(runs[0].duration_s(MASTER_FPS), 3.0)
        # ...and with merging off, the blackout splits the Safari run in two.
        self.assertEqual(len(track.runs(merge_across_unresolved=False)), 3)

    def test_summary_is_tick_weighted_and_separately_named(self) -> None:
        """Coverage over the whole axis, deliberately NOT the key filtering reads --
        a tick-dominant app can differ from the frame-dominant one after thinning."""
        track = build_app_track([(0.0, SAFARI), (1.0, UNCAPTURED), (2.0, FIREFOX)],
                                n_ticks=60, master_fps=MASTER_FPS)
        s = track.summary()
        self.assertEqual(s["app_by_ticks"], FIREFOX)   # 30 ticks vs Safari's 15
        self.assertAlmostEqual(s["app_frac_by_ticks"], 30 / 45, places=6)
        self.assertAlmostEqual(s["app_uncaptured_frac"], 15 / 60)
        self.assertEqual(s["apps"], [FIREFOX, SAFARI])
        self.assertNotIn("app", s)       # must not collide with frame_app_stats
        self.assertNotIn("app_frac", s)

    def test_empty_track_summary_has_no_app(self) -> None:
        s = AppTrack([], [], 30, MASTER_FPS).summary()
        self.assertIsNone(s["app_by_ticks"])
        self.assertEqual(s["app_frac_by_ticks"], 0.0)


class TestSpans(unittest.TestCase):
    def test_iter_app_spans_breaks_on_app_and_on_unresolved(self) -> None:
        frames = _frames([SAFARI, SAFARI, FIREFOX, UNCAPTURED, FIREFOX, None])
        self.assertEqual(
            list(iter_app_spans(frames)),
            [(SAFARI, 0, 2), (FIREFOX, 2, 3), (FIREFOX, 4, 5)],
        )

    def test_app_stats_ignores_unresolved_for_dominance(self) -> None:
        stats = app_stats(_frames([SAFARI, SAFARI, FIREFOX, UNCAPTURED]))
        self.assertEqual(stats["app"], SAFARI)
        self.assertAlmostEqual(stats["app_frac"], 2 / 3, places=6)
        self.assertEqual(stats["app_mix"][UNCAPTURED], 1)


class TestAppFilter(unittest.TestCase):
    def test_inactive_filter_is_a_no_op(self) -> None:
        self.assertFalse(AppFilter().active)
        self.assertTrue(AppFilter(unknown="drop").active)

    def test_gate_include_exclude_and_purity(self) -> None:
        frames = _frames([SAFARI] * 8 + [FIREFOX] * 2)
        keep = plan_app_spans(frames, AppFilter(include=frozenset({SAFARI})),
                              min_frames=1)
        self.assertEqual([(s.app, s.lo, s.hi) for s in keep], [(SAFARI, 0, 10)])
        # dominant app is Safari, so a Firefox-only filter drops the segment
        self.assertEqual(
            plan_app_spans(frames, AppFilter(include=frozenset({FIREFOX})), min_frames=1),
            [],
        )
        self.assertEqual(
            plan_app_spans(frames, AppFilter(exclude=frozenset({SAFARI})), min_frames=1),
            [],
        )
        # purity floor: Safari holds 0.8
        self.assertEqual(len(plan_app_spans(frames, AppFilter(min_frac=0.8), min_frames=1)), 1)
        self.assertEqual(plan_app_spans(frames, AppFilter(min_frac=0.9), min_frames=1), [])

    def test_unlabeled_never_matches_an_include_list(self) -> None:
        frames = _frames([None, None])
        self.assertEqual(
            plan_app_spans(frames, AppFilter(include=frozenset({SAFARI})), min_frames=1), []
        )
        # kept when nothing is required...
        self.assertEqual(len(plan_app_spans(frames, AppFilter(unknown="keep"), min_frames=1)), 1)
        # ...and dropped on request
        self.assertEqual(plan_app_spans(frames, AppFilter(unknown="drop"), min_frames=1), [])

    def test_uncaptured_only_segment_counts_as_unlabeled(self) -> None:
        frames = _frames([UNCAPTURED, UNCAPTURED])
        self.assertEqual(plan_app_spans(frames, AppFilter(unknown="drop"), min_frames=1), [])

    def test_split_one_conversation_per_run(self) -> None:
        frames = _frames([SAFARI, SAFARI, SAFARI, FIREFOX, FIREFOX, SAFARI])
        spans = plan_app_spans(frames, AppFilter(split=True, drop_seam_turns=False),
                               min_frames=1)
        self.assertEqual([(s.app, s.lo, s.hi) for s in spans],
                         [(SAFARI, 0, 3), (FIREFOX, 3, 5), (SAFARI, 5, 6)])

    def test_split_drops_the_seam_turn(self) -> None:
        # frame 2 is the last Safari turn and its window straddles the switch
        frames = _frames([SAFARI, SAFARI, SAFARI, FIREFOX, FIREFOX], seams={2})
        spans = plan_app_spans(frames, AppFilter(split=True), min_frames=1)
        self.assertEqual([(s.app, s.lo, s.hi, s.seam_trimmed) for s in spans],
                         [(SAFARI, 0, 2, True), (FIREFOX, 2 + 1, 5, False)])
        # the surviving turns keep their ORIGINAL action labels
        self.assertEqual([f["action"] for f in frames[spans[0].lo:spans[0].hi]],
                         ["act0", "act1"])

    def test_split_min_run_frames(self) -> None:
        frames = _frames([SAFARI, SAFARI, FIREFOX, SAFARI, SAFARI, SAFARI])
        spans = plan_app_spans(
            frames, AppFilter(split=True, min_run_frames=3, drop_seam_turns=False),
            min_frames=1,
        )
        self.assertEqual([(s.app, s.lo, s.hi) for s in spans], [(SAFARI, 3, 6)])

    def test_split_with_include_keeps_only_that_app(self) -> None:
        frames = _frames([SAFARI, SAFARI, FIREFOX, FIREFOX, GHOSTTY])
        spans = plan_app_spans(
            frames,
            AppFilter(split=True, include=frozenset({FIREFOX}), drop_seam_turns=False),
            min_frames=1,
        )
        self.assertEqual([(s.app, s.lo, s.hi) for s in spans], [(FIREFOX, 2, 4)])

    def test_split_never_emits_unresolved_runs(self) -> None:
        frames = _frames([UNCAPTURED, UNCAPTURED, SAFARI, SAFARI])
        spans = plan_app_spans(frames, AppFilter(split=True, drop_seam_turns=False),
                               min_frames=1)
        self.assertEqual([(s.app, s.lo, s.hi) for s in spans], [(SAFARI, 2, 4)])


class TestEndToEndFold(unittest.TestCase):
    """A keylog on disk -> per-frame labels -> spans, the stage-03/04 path."""

    def test_keylog_to_spans(self) -> None:
        path = _write_keylog([
            [_us(0.0), ["ContextChanged", [SAFARI]]],
            [_us(0.2), ["MouseMove", [3.0, 4.0]]],
            [_us(2.0), ["ContextChanged", ["firefox"]]],      # process-name spelling
            [_us(2.5), ["KeyPress", [59, "KeyA"]]],
            [_us(4.0), ["ContextChanged", [UNCAPTURED]]],
            [_us(5.0), ["ContextChanged", ["org.mozilla.firefox"]]],
        ])
        n_ticks = 90  # 6 s @15 fps
        track = load_app_track(path, n_ticks=n_ticks, master_fps=MASTER_FPS)
        # one frame per second, like a 1 fps sample
        ticks = [i * 15 for i in range(6)]
        frames = [
            {
                "global_frame_idx": i,
                "master_record_index": t,
                "action": f"act{i}",
                "app": track.at(t),
                "app_window_switches": len(
                    track.switches_in(t, ticks[i + 1] if i + 1 < len(ticks) else n_ticks)
                ),
            }
            for i, t in enumerate(ticks)
        ]
        self.assertEqual([f["app"] for f in frames],
                         [SAFARI, SAFARI, FIREFOX, FIREFOX, UNCAPTURED, FIREFOX])
        # the blackout frame breaks the span list, but the run is one Firefox run
        self.assertEqual(list(iter_app_spans(frames)),
                         [(SAFARI, 0, 2), (FIREFOX, 2, 4), (FIREFOX, 5, 6)])
        self.assertEqual([(r.app, r.start, r.end) for r in track.runs()],
                         [(SAFARI, 0, 30), (FIREFOX, 30, 90)])


if __name__ == "__main__":
    unittest.main()


class _StubViewFrame:
    """The three attributes ``label_view_frames`` reads off a ViewFrame."""

    def __init__(self, master_idx: int, win_start: int, win_end: int) -> None:
        self.master_idx = master_idx
        self.win_start = win_start
        self.win_end = win_end


class _StubView:
    def __init__(self, frames, keylog_path, master_fps=MASTER_FPS) -> None:
        self.frames = frames
        self.keylog_path = keylog_path
        self.master_fps = master_fps


class LabelViewFramesTest(unittest.TestCase):
    """The bridge from a stage-03 FILTER view to app labels.

    The sampler writes ``app``/``app_window_switches`` per frame itself; a filter
    view carries only master ticks, so the labels are derived here from the same
    keylog on the same clock. These pin that the two agree.
    """

    def test_each_frame_takes_the_app_in_focus_at_its_own_tick(self) -> None:
        # Safari from t=0, Terminal from t=1s (tick 15 at 15 fps).
        keylog = _write_keylog([
            [0, ["ContextChanged", [SAFARI]]],
            [1_000_000, ["ContextChanged", [TERMINAL]]],
        ])
        frames = [_StubViewFrame(0, 0, 15), _StubViewFrame(15, 15, 30)]
        labels = label_view_frames(_StubView(frames, str(keylog)))
        self.assertEqual([r["app"] for r in labels], [SAFARI, TERMINAL])

    def test_a_window_straddling_a_switch_is_marked_as_a_seam(self) -> None:
        """The seam count is what `plan_app_spans` trims on: that turn's action
        label mixes the app being left with the one arriving."""
        keylog = _write_keylog([
            [0, ["ContextChanged", [SAFARI]]],
            [500_000, ["ContextChanged", [TERMINAL]]],  # mid-window (tick 7.5)
        ])
        frames = [_StubViewFrame(0, 0, 15), _StubViewFrame(15, 15, 30)]
        labels = label_view_frames(_StubView(frames, str(keylog)))
        self.assertEqual(labels[0]["app_window_switches"], 1)
        self.assertEqual(labels[1]["app_window_switches"], 0)

    def test_no_keylog_yields_unlabelled_frames_rather_than_raising(self) -> None:
        """`accepts` already refuses to read "no app" as any particular app, so
        an unreadable view is a filtering question, not a crash."""
        frames = [_StubViewFrame(0, 0, 15)]
        labels = label_view_frames(_StubView(frames, None))
        self.assertEqual(labels, [{"app": None, "app_window_switches": 0}])

    def test_an_empty_view_yields_nothing(self) -> None:
        self.assertEqual(label_view_frames(_StubView([], "irrelevant")), [])

    def test_labels_feed_the_span_planner_the_same_shape_stage_03_writes(self) -> None:
        """End to end: derived labels drive `plan_app_spans` exactly as the
        sampler's own do, so a filter-view dataset and a sample dataset split
        identically."""
        keylog = _write_keylog([
            [0, ["ContextChanged", [SAFARI]]],
            [2_000_000, ["ContextChanged", [TERMINAL]]],
        ])
        frames = [_StubViewFrame(i * 15, i * 15, (i + 1) * 15) for i in range(4)]
        labels = label_view_frames(_StubView(frames, str(keylog)))
        spans = plan_app_spans(
            labels, AppFilter(split=True, drop_seam_turns=False), min_frames=1
        )
        self.assertEqual(
            [(s.app, s.lo, s.hi) for s in spans],
            [(SAFARI, 0, 2), (TERMINAL, 2, 4)],
        )
