"""Stage 04 --mode thinking goal-bounded windows: clip-stride window alignment,
leak-free GOAL/So-far memory selection (exact preceding-clip only), outcome-
frame TERMINATE placement (clean/verified/all), near-miss thought attachment,
no-goal exclusion, the TERMINATE-is-own-turn invariant, and the legacy goal-
free mode's golden regression. All on synthetic DayStreams + sidecar rows — no
filter artifact, no frame store, no labeler.
"""

from __future__ import annotations

import unittest

from realigned_pipeline.annotation.lib.days import DayFrame, DayStream, fmt_t
from realigned_pipeline.lib.action_format import get_formatter
from realigned_pipeline.lib.conversations import CLIP_STRIDE, require_window_alignment
from realigned_pipeline.stage_04_conversations import (
    TERMINATE_TOKEN,
    build_goal_day_rows,
    build_legacy_day_rows,
    goal_system_prompt_file,
    group_goal_runs,
    reindex_active_rows,
    resolve_terminal_token,
    select_memory,
)

STEP_S = 2.0


def _day(n: int = 90, chunk_splits: tuple[int, ...] = (), step_s: float = STEP_S) -> DayStream:
    """A synthetic one-user day: n frames at step_s spacing (0.5 fps shape),
    optionally split into chunks at the given day indices. Every frame gets a
    distinct action label so assistant targets are checkable."""
    frames = [
        DayFrame(day_idx=i, t_day_s=step_s * i, segment_id="s0", recording_id="r0",
                 master_idx=i * 30, image=f"ar://fake/images.array_record#{i}",
                 action=f"act_{i}")
        for i in range(n)
    ]
    bounds = [0, *chunk_splits, n]
    chunks = [frames[a:b] for a, b in zip(bounds, bounds[1:]) if a < b]
    return DayStream(day_tag="u0_20260101", user_id="u0", date="2026-01-01",
                     frames=frames, chunks=chunks, gap_cut_s=180.0, n_segments=1)


def _clip(key: str, lo: int, hi: int, gid=None, text=None, t0=None, t1=None,
          step_s: float = STEP_S) -> dict:
    return {
        "clip_key": key,
        "day_idx_range": [lo, hi],
        "t_range": [fmt_t(step_s * lo), fmt_t(step_s * hi)],
        "segments": ["s0"],
        "goal_id": gid,
        "goal_text": text,
        "goal_t_start": t0,
        "goal_t_end": t1,
        "goal_long_ref": "L1" if gid is not None else None,
    }


def _mem(key: str, lo: int, hi: int, memory: str) -> dict:
    return {"clip_key": key, "chunk_index": 0, "day_idx_range": [lo, hi],
            "t_range": [fmt_t(STEP_S * lo), fmt_t(STEP_S * hi)],
            "memory": memory, "log": ""}


def _brow(gid, *, completed=True, confidence="high", final_thought="It is done.",
          near_miss=None) -> dict:
    return {"goal_id": gid, "goal_text": f"goal {gid}", "completed": completed,
            "confidence": confidence, "evidence": "on screen",
            "final_thought": final_thought, "near_miss": near_miss}


def _thought(i: int, text: str) -> dict:
    """An anchored verified-thought row (goals.jsonl shape, joined by day_idx)."""
    return {"goal_id": f"u0_20260101_t{i:04d}", "instruction": text,
            "segment_id": "s0", "start_master_idx": i * 30, "t_day_s": i * STEP_S}


def _build(day, active, memory=(), boundaries=None, anchored=None, **over):
    # 30 is the smallest aligned window >1 clip; individual tests override it.
    cfg = dict(window_frames=30, terminate_mode="clean", terminate_max_lag_s=180.0,
               min_anchor_lead=0, system_prompt="SYS", fps=0.5,
               action_format="canonical", annotation_method="lumine_thinking_goals")
    cfg.update(over)
    return build_goal_day_rows(day, list(active), list(memory), boundaries or {},
                               anchored or {}, **cfg)


def _assistant_texts(row) -> list[str]:
    return [m["content"][0]["text"] for m in row["messages"] if m["role"] == "assistant"]


def _user_turns(row) -> list[list[dict]]:
    return [m["content"] for m in row["messages"] if m["role"] == "user"]


def _frame_idxs(row) -> list[int]:
    """Day indices of every image shown in the conversation."""
    return [int(block["image"].rsplit("#", 1)[1])
            for turn in _user_turns(row) for block in turn if block["type"] == "image"]


def _goal_block(row) -> str:
    return _user_turns(row)[0][0]["text"]


class WindowAlignmentTest(unittest.TestCase):
    def test_multiples_of_the_clip_stride_pass(self) -> None:
        self.assertEqual(CLIP_STRIDE, 15)
        for wf in (15, 30, 45, 60, 300):
            require_window_alignment(wf)  # no raise

    def test_non_multiples_error_and_name_the_stride(self) -> None:
        for bad in (24, 14, 16, 1):  # positive, not a multiple of 15
            with self.assertRaises(SystemExit) as ctx:
                require_window_alignment(bad)
            self.assertIn(str(CLIP_STRIDE), str(ctx.exception))

    def test_nonpositive_errors(self) -> None:
        for bad in (0, -15):
            with self.assertRaises(SystemExit):
                require_window_alignment(bad)


class GroupGoalRunsTest(unittest.TestCase):
    def test_contiguous_runs_and_handoff_ids(self) -> None:
        active = [
            _clip("c0", 0, 14, gid=1, text="A", t0=0.0, t1=100.0),
            _clip("c1", 15, 29, gid=1, text="A", t0=0.0, t1=100.0),
            _clip("c2", 30, 44),  # no-goal clip: separates runs
            _clip("c3", 45, 59, gid=1, text="A", t0=0.0, t1=100.0),  # recurrence
            _clip("c4", 60, 74, gid=2, text="B", t0=120.0, t1=140.0),
        ]
        runs = group_goal_runs(active)
        self.assertEqual([(r["goal_id"], r["start_idx"], r["end_idx"]) for r in runs],
                         [(1, 0, 29), (1, 45, 59), (2, 60, 74)])
        # handoff ids: run0 -> no-goal clip (None); run1 -> g2; run2 -> day end
        self.assertEqual([r["next_goal_id"] for r in runs], [None, 2, None])


class DecisionFpsProjectionTest(unittest.TestCase):
    def test_annotation_clip_projects_by_time_not_dense_index(self) -> None:
        annotation = _day(n=20, step_s=2.0)
        decision = _day(n=160, step_s=0.25)
        active = [_clip("c0", 0, 14, gid=1, text="G", t0=0.0, t1=30.0)]
        projected = reindex_active_rows(
            active, annotation, decision, annotation_fps=0.5
        )
        self.assertEqual(projected[0]["source_day_idx_range"], [0, 14])
        # Annotation clip [0s,30s) becomes 120 causal 4 Hz decisions.
        self.assertEqual(projected[0]["day_idx_range"], [0, 119])


class SelectMemoryTest(unittest.TestCase):
    def test_exact_preceding_clip_wins(self) -> None:
        rows = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1")]
        # window starting at 30 -> predecessor clip ends at 29 == 30-1 -> M1
        self.assertEqual(select_memory(rows, 30), ("M1", "ok"))
        # window starting at 15 -> predecessor ends at 14 -> M0
        self.assertEqual(select_memory(rows, 15), ("M0", "ok"))

    def test_overlapping_or_gappy_clip_is_never_attached(self) -> None:
        rows = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1")]
        # window starts at 20 (mid clip [15,29]): [15,29] overlaps (ends 29>=20)
        # and [0,14] ends at 14, NOT at 19 -> gappy -> omit (no leak of M1).
        self.assertEqual(select_memory(rows, 20), ("", "gap"))
        # a clip ending exactly ON the window start overlaps too -> nothing before
        self.assertEqual(select_memory([_mem("c0", 0, 20, "M0")], 20), ("", "none"))

    def test_empty_predecessor_memory(self) -> None:
        self.assertEqual(select_memory([_mem("c0", 0, 14, "")], 15), ("", "empty"))

    def test_no_earlier_memory(self) -> None:
        self.assertEqual(select_memory([_mem("c0", 0, 14, "M0")], 0), ("", "none"))


class SpanTilingTest(unittest.TestCase):
    def test_tiles_span_at_30_and_excludes_no_goal_frames(self) -> None:
        day = _day(n=100)
        active = [
            _clip("c0", 0, 14),
            _clip("c1", 15, 29, gid=1, text="Do the thing", t0=30.0, t1=1000.0),
            _clip("c2", 30, 44, gid=1, text="Do the thing", t0=30.0, t1=1000.0),
            _clip("c3", 45, 59, gid=1, text="Do the thing", t0=30.0, t1=1000.0),
            _clip("c4", 60, 74),
        ]
        rows, stats = _build(day, active)  # goal_t_end beyond the chunk: no outcome
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["n_frames"] for r in rows], [30, 15])  # 45 frames, 30-tiled
        shown = sorted(i for r in rows for i in _frame_idxs(r))
        self.assertEqual(shown, list(range(15, 60)))  # in-span only, nothing else
        self.assertEqual(stats["n_terminate_turns"], 0)
        self.assertEqual(stats["n_terminate_skipped_no_outcome_frame"], 1)
        self.assertEqual(stats["n_no_goal_frames_excluded"], 100 - 45)
        for r in rows:
            self.assertTrue(r["goal_conditioned"])
            self.assertEqual(r["goal_text"], "Do the thing")
            self.assertTrue(_goal_block(r).startswith("GOAL: Do the thing"))

    def test_tiles_span_at_60_lands_on_clip_edges(self) -> None:
        day = _day(n=200)
        # one span of 8 clips, 120 in-span frames [15..134]
        active = [_clip("c0", 0, 14)] + [
            _clip(f"c{j}", 15 * j, 15 * j + 14, gid=1, text="G", t0=30.0, t1=1e4)
            for j in range(1, 9)]
        rows, _ = _build(day, active, window_frames=60)
        # 120 frames / 60 -> two full windows, edges at clip boundaries 15,75,135
        self.assertEqual([_frame_idxs(r)[0] for r in rows], [15, 75])
        self.assertEqual([_frame_idxs(r)[-1] for r in rows], [74, 134])
        self.assertEqual([r["n_frames"] for r in rows], [60, 60])

    def test_windows_never_cross_chunk_boundaries(self) -> None:
        day = _day(n=60, chunk_splits=(30,))
        active = [
            _clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
            _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0),
        ]
        rows, _ = _build(day, active)
        self.assertEqual(len(rows), 2)
        self.assertEqual(_frame_idxs(rows[0]), list(range(15, 30)))
        self.assertEqual(_frame_idxs(rows[1]), list(range(30, 45)))
        self.assertEqual([r["chunk_index"] for r in rows], [0, 1])


class FreshDecisionRecordTest(unittest.TestCase):
    def test_goal_plus_four_images_has_one_assistant_target_and_no_history(self) -> None:
        day = _day(n=20)
        active = [_clip("c0", 0, 14, gid=1, text="Click Save", t0=0.0, t1=1e4)]
        rows, _ = _build(
            day,
            active,
            context_images=4,
            omit_goal_memory=True,
            action_format="computer_use_rel_step_v1",
        )
        self.assertEqual(len(rows), 15)
        self.assertEqual([m["role"] for m in rows[4]["messages"]],
                         ["system", "user", "assistant"])
        self.assertEqual(_frame_idxs(rows[4]), [1, 2, 3, 4])
        self.assertEqual(_goal_block(rows[4]), "GOAL: Click Save")
        self.assertEqual(_assistant_texts(rows[4]), ["act_4"])
        self.assertEqual(rows[4]["n_turns"], 1)
        self.assertEqual(rows[4]["context_images"], 4)

    def test_rel_step_default_prompt_is_the_exact_rel_step_prompt(self) -> None:
        self.assertEqual(
            goal_system_prompt_file("computer_use_rel_step_v1").name,
            "cua_rel_step_v1_thinking.txt",
        )


class LeakFreeMemoryTest(unittest.TestCase):
    def test_span_start_has_no_memory_then_exact_predecessor(self) -> None:
        day = _day(n=60)
        active = [_clip("c0", 0, 14, gid=1, text="G", t0=0.0, t1=1000.0),
                  _clip("c1", 15, 29, gid=1, text="G", t0=0.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=0.0, t1=1000.0)]
        memory = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1"),
                  _mem("c2", 30, 44, "M2")]
        rows, stats = _build(day, active, memory=memory)  # window_frames=30
        # window 0 [0..29] is the span start -> bare GOAL, no So-far
        self.assertEqual(_goal_block(rows[0]), "GOAL: G")
        self.assertFalse(rows[0]["has_memory"])
        # window 1 [30..44] -> predecessor clip [15,29] ends at 29 == 30-1 -> M1
        self.assertEqual(_goal_block(rows[1]), "GOAL: G\nSo far: M1")
        self.assertTrue(rows[1]["has_memory"])
        self.assertEqual(stats["n_windows_with_memory"], 1)
        self.assertEqual(stats["n_windows_memory_omitted_boundary"], 0)

    def test_chunk_start_window_withholds_memory(self) -> None:
        # a span crossing a chunk boundary: the second chunk's first window is a
        # piece start -> no memory even though an in-span clip ends just before.
        day = _day(n=60, chunk_splits=(30,))
        active = [_clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0)]
        memory = [_mem("c1", 15, 29, "M1"), _mem("c2", 30, 44, "M2")]
        rows, _ = _build(day, active, memory=memory)
        self.assertEqual(_goal_block(rows[0]), "GOAL: G")   # span start
        self.assertEqual(_goal_block(rows[1]), "GOAL: G")   # chunk start, not So-far
        self.assertFalse(any(r["has_memory"] for r in rows))

    def test_later_window_omits_gappy_memory_and_counts_it(self) -> None:
        # a well-tiled window whose exact predecessor clip is MISSING from the
        # memory sidecar (only a farther-back row exists) -> omit + count.
        day = _day(n=60)
        active = [_clip(f"c{j}", 15 * j, 15 * j + 14, gid=1, text="G", t0=0.0, t1=1e4)
                  for j in range(0, 3)]
        memory = [_mem("c0", 0, 14, "M0")]  # clip ending at 29 (win 2's pred) absent
        rows, stats = _build(day, active, memory=memory, window_frames=15)
        # win0 span start (no mem); win1 pred [0,14] present -> M0; win2 pred
        # would end at 29 but that clip is missing from the sidecar -> gappy,
        # omitted rather than attaching the farther-back [0,14].
        self.assertEqual([r["has_memory"] for r in rows], [False, True, False])
        self.assertEqual(_goal_block(rows[1]), "GOAL: G\nSo far: M0")
        self.assertEqual(_goal_block(rows[2]), "GOAL: G")
        self.assertEqual(stats["n_windows_memory_omitted_boundary"], 1)


class CleanTerminateTest(unittest.TestCase):
    def _active(self, next_gid=2):
        nxt = _clip("c1", 15, 29, gid=next_gid, text="Next goal",
                    t0=30.0, t1=58.0) if next_gid else _clip("c1", 15, 29)
        return [_clip("c0", 0, 14, gid=1, text="First goal", t0=0.0, t1=27.0), nxt]

    def test_goal_to_goal_handoff_terminates_on_outcome_frame(self) -> None:
        day = _day(n=40)
        rows, stats = _build(day, self._active())
        g1 = [r for r in rows if r["goal_id"] == 1]
        self.assertEqual(len(g1), 1)
        row = g1[0]
        # supervised frames 0..13 (t <= 27), outcome frame 14 (t=28) appended
        self.assertEqual(_frame_idxs(row), list(range(0, 15)))
        self.assertEqual(row["n_frames"], 15)
        self.assertEqual(row["terminate"], "clean")
        texts = _assistant_texts(row)
        self.assertEqual(texts[:-1], [f"act_{i}" for i in range(14)])
        self.assertEqual(texts[-1], TERMINATE_TOKEN)  # plain, no thought, own turn
        # the outcome frame's user turn is image-only (no GOAL block)
        self.assertEqual(_user_turns(row)[-1], [
            {"type": "image", "image": "ar://fake/images.array_record#14"}])
        self.assertEqual(stats["n_terminate_turns"], 1)
        self.assertEqual(stats["n_frames_dropped_post_goal"], 1)  # frame 14's action

    def test_no_goal_handoff_means_no_terminate(self) -> None:
        day = _day(n=40)
        rows, stats = _build(day, self._active(next_gid=None))
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertIsNone(row["terminate"])
        self.assertEqual(_frame_idxs(row), list(range(0, 15)))  # full span kept
        self.assertEqual(_assistant_texts(row)[-1], "act_14")  # normal last action
        self.assertEqual(stats["n_terminate_skipped_no_goal_handoff"], 1)

    def test_lag_exceeded_means_no_terminate(self) -> None:
        # sparse frames (200 s apart): outcome lands 190 s after goal_t_end
        day = _day(n=4, step_s=200.0)
        active = [_clip("c0", 0, 1, gid=1, text="G", t0=0.0, t1=210.0, step_s=200.0),
                  _clip("c1", 2, 3, gid=2, text="H", t0=220.0, t1=600.0, step_s=200.0)]
        rows, stats = _build(day, active)
        self.assertEqual(stats["n_terminate_skipped_lag_exceeded"], 1)
        self.assertNotIn("clean", [r["terminate"] for r in rows if r["goal_id"] == 1])

    def test_all_mode_terminates_without_handoff(self) -> None:
        day = _day(n=40)
        rows, stats = _build(day, self._active(next_gid=None), terminate_mode="all")
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(row["terminate"], "all")
        self.assertEqual(_assistant_texts(row)[-1], TERMINATE_TOKEN)
        self.assertEqual(stats["n_terminate_turns"], 1)


class VerifiedTerminateTest(unittest.TestCase):
    def _active(self):
        return [_clip("c0", 0, 14, gid=1, text="Send the reply", t0=0.0, t1=55.0),
                _clip("c1", 15, 29, gid=1, text="Send the reply", t0=0.0, t1=55.0),
                _clip("c2", 30, 44, gid=2, text="Next", t0=60.0, t1=88.0)]

    def test_completed_high_gets_final_thought_terminate(self) -> None:
        day = _day(n=60)
        boundaries = {1: _brow(1, final_thought="The reply is in the sent thread.")}
        rows, stats = _build(day, self._active(), boundaries=boundaries,
                             terminate_mode="verified", window_frames=60)
        row = [r for r in rows if r["goal_id"] == 1][0]
        # supervised 0..27 (t <= 55), outcome frame 28 (t=56)
        self.assertEqual(_frame_idxs(row), list(range(0, 29)))
        self.assertEqual(
            _assistant_texts(row)[-1],
            "<think>\nThe reply is in the sent thread.\n</think>\nTERMINATE")
        self.assertEqual(row["terminate"], "verified")
        self.assertEqual(stats["n_terminate_turns"], 1)

    def test_not_completed_span_gets_no_terminate(self) -> None:
        day = _day(n=60)
        # goal 2 never has a boundaries row here, so it always skips too
        for boundaries, skips in (
            ({1: _brow(1, completed=False, final_thought="")},
             {"not_completed_high": 1, "no_boundaries_row": 1}),
            ({1: _brow(1, confidence="low")},
             {"not_completed_high": 1, "no_boundaries_row": 1}),
            ({}, {"no_boundaries_row": 2}),
        ):
            rows, stats = _build(day, self._active(), boundaries=boundaries,
                                 terminate_mode="verified", window_frames=60)
            row = [r for r in rows if r["goal_id"] == 1][0]
            self.assertIsNone(row["terminate"])
            # untruncated: ends on the last in-span frame's normal action
            self.assertEqual(_frame_idxs(row), list(range(0, 30)))
            self.assertEqual(_assistant_texts(row)[-1], "act_29")
            self.assertEqual(stats["n_terminate_turns"], 0)
            for reason, n in skips.items():
                self.assertEqual(stats[f"n_terminate_skipped_{reason}"], n)

    def test_near_miss_thought_replaces_annotation_thought(self) -> None:
        day = _day(n=60)
        near_miss = {"clip_key": "c0", "day_idx_range": [0, 14],
                     "not_done_reason": "draft not sent",
                     "next_step_thought": "The draft still needs sending."}
        boundaries = {1: _brow(1, near_miss=near_miss)}
        anchored = {0: _thought(0, "I open the mail client.")}
        rows, stats = _build(day, self._active(), boundaries=boundaries,
                             anchored=anchored, terminate_mode="verified",
                             window_frames=60)
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(
            _assistant_texts(row)[0],
            "<think>\nThe draft still needs sending.\n</think>\nact_0")
        self.assertIn("near_miss:g0001", row["thought_ids"])
        self.assertEqual(stats["n_near_miss_attached"], 1)
        self.assertEqual(stats["n_thoughts_placed"], 1)  # replacement, not addition


class RecurringGoalRunsTest(unittest.TestCase):
    def test_only_the_run_containing_goal_t_end_terminates(self) -> None:
        day = _day(n=90)
        active = [
            _clip("c0", 0, 14, gid=1, text="G", t0=0.0, t1=86.0),
            _clip("c1", 15, 29, gid=2, text="H", t0=30.0, t1=58.0),  # interruption
            _clip("c2", 30, 44, gid=1, text="G", t0=0.0, t1=86.0),   # resumed run
        ]
        rows, stats = _build(day, active, terminate_mode="all")
        g1 = [r for r in rows if r["goal_id"] == 1]
        self.assertEqual([r["run_index"] for r in g1], [0, 2])
        first, resumed = g1
        # first run: no terminate, full clip
        self.assertIsNone(first["terminate"])
        self.assertEqual(_frame_idxs(first), list(range(0, 15)))
        # resumed run contains goal_t_end (t=86 -> frame 43): terminates there
        self.assertEqual(resumed["terminate"], "all")
        self.assertEqual(_frame_idxs(resumed), list(range(30, 45)))  # 30..43 + outcome 44
        self.assertEqual(_assistant_texts(resumed)[-1], TERMINATE_TOKEN)
        g1_terminates = [r["terminate"] for r in g1 if r["terminate"]]
        self.assertEqual(g1_terminates, ["all"])
        self.assertEqual(stats["n_terminate_turns"], 2)


class TerminateOwnTurnInvariantTest(unittest.TestCase):
    def test_terminate_never_glued_to_an_action(self) -> None:
        day = _day(n=90)
        active = [
            _clip("c0", 0, 14, gid=1, text="G", t0=0.0, t1=27.0),
            _clip("c1", 15, 29, gid=2, text="H", t0=30.0, t1=57.0),
            _clip("c2", 30, 44, gid=3, text="I", t0=60.0, t1=87.0),
        ]
        boundaries = {g: _brow(g) for g in (1, 2, 3)}
        for mode in ("clean", "verified", "all"):
            rows, _ = _build(day, active, boundaries=boundaries, terminate_mode=mode)
            for row in rows:
                for text in _assistant_texts(row):
                    if TERMINATE_TOKEN not in text:
                        continue
                    # the whole action line is TERMINATE, optionally preceded by
                    # exactly one think block — never an action + TERMINATE
                    self.assertTrue(
                        text == TERMINATE_TOKEN
                        or (text.startswith("<think>\n")
                            and text.endswith("\n</think>\n" + TERMINATE_TOKEN)),
                        f"glued TERMINATE in {mode}: {text!r}")
                    self.assertNotRegex(text, r"act_\d+")


class TerminateHookTest(unittest.TestCase):
    """Terminate turns come from the formatter's ``terminate_line()`` — the
    TERMINATE literal for the text formats, the native terminate tool_call block
    for computer_use_rel_v1 — never a hardcoded literal."""

    NATIVE_TERMINATE = ('<tool_call>\n{"name": "computer_use", "arguments": '
                        '{"action": "terminate", "status": "success"}}\n</tool_call>')

    def _active(self):
        return [_clip("c0", 0, 14, gid=1, text="First goal", t0=0.0, t1=27.0),
                _clip("c1", 15, 29, gid=2, text="Next goal", t0=30.0, t1=58.0)]

    def test_terminate_line_values(self) -> None:
        for name in ("canonical", "ordered_events_v2", "ordered_events_v3"):
            self.assertEqual(get_formatter(name).terminate_line(), TERMINATE_TOKEN)
        self.assertEqual(get_formatter("computer_use_rel_v1").terminate_line(),
                         self.NATIVE_TERMINATE)

    def test_native_clean_terminate_turn_is_the_tool_call_block(self) -> None:
        rows, stats = _build(_day(n=40), self._active(),
                             action_format="computer_use_rel_v1")
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(row["terminate"], "clean")
        self.assertEqual(_assistant_texts(row)[-1], self.NATIVE_TERMINATE)
        self.assertEqual(stats["n_terminate_turns"], 1)

    def test_native_verified_terminate_keeps_the_think_shape(self) -> None:
        boundaries = {1: _brow(1, final_thought="The reply is sent.")}
        rows, _ = _build(_day(n=40), self._active(), boundaries=boundaries,
                         terminate_mode="verified",
                         action_format="computer_use_rel_v1")
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(
            _assistant_texts(row)[-1],
            f"<think>\nThe reply is sent.\n</think>\n{self.NATIVE_TERMINATE}")

    def test_v3_terminate_turn_stays_the_literal(self) -> None:
        rows, _ = _build(_day(n=40), self._active(),
                         action_format="ordered_events_v3")
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(_assistant_texts(row)[-1], TERMINATE_TOKEN)

    def test_resolve_terminal_token(self) -> None:
        self.assertEqual(resolve_terminal_token(TERMINATE_TOKEN, "canonical"),
                         TERMINATE_TOKEN)
        self.assertEqual(resolve_terminal_token(TERMINATE_TOKEN, "ordered_events_v3"),
                         TERMINATE_TOKEN)
        self.assertEqual(
            resolve_terminal_token(TERMINATE_TOKEN, "computer_use_rel_v1"),
            self.NATIVE_TERMINATE)
        self.assertIsNone(resolve_terminal_token(None, "computer_use_rel_v1"))
        self.assertEqual(resolve_terminal_token("<TERM>", "computer_use_rel_v1"),
                         "<TERM>")

    def test_goal_system_prompt_defaults_per_format(self) -> None:
        self.assertEqual(goal_system_prompt_file("computer_use_rel_v1").name,
                         "cua_v4_thinking.txt")
        self.assertEqual(goal_system_prompt_file("ordered_events_v2").name,
                         "cua_oev2_thinking.txt")
        for name in ("canonical", "ordered_events_v3"):
            self.assertEqual(goal_system_prompt_file(name).name,
                             "cua_v3_thinking.txt")


class ThoughtsInGoalWindowsTest(unittest.TestCase):
    def test_thoughts_attach_and_windows_without_thoughts_are_kept(self) -> None:
        day = _day(n=60)
        active = [_clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0)]
        anchored = {20: _thought(20, "I switch to the terminal.")}
        rows, stats = _build(day, active, anchored=anchored, window_frames=15)
        self.assertEqual(len(rows), 2)  # thought-less window 2 is KEPT
        self.assertEqual(rows[0]["n_thoughts"], 1)
        self.assertEqual(_assistant_texts(rows[0])[5],  # frame 20, pos 5
                         "<think>\nI switch to the terminal.\n</think>\nact_20")
        self.assertEqual(rows[1]["n_thoughts"], 0)
        self.assertEqual(stats["n_thoughts_placed"], 1)

    def test_min_anchor_lead_demotes_early_mid_chunk_anchors(self) -> None:
        day = _day(n=60)
        active = [_clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0)]
        # window 2 starts mid-chunk at 30; anchor at 40 sits 10 frames in
        anchored = {40: _thought(40, "T")}
        rows, stats = _build(day, active, anchored=anchored, min_anchor_lead=12,
                             window_frames=15)
        self.assertEqual(_assistant_texts(rows[1])[10], "act_40")  # demoted
        self.assertEqual(stats["n_demoted"], 1)
        # a chunk-start window is exempt: anchor at pos 0 of window 1 keeps it
        anchored = {15: _thought(15, "T")}
        day2 = _day(n=60, chunk_splits=(15,))
        rows, stats = _build(day2, active, anchored=anchored, min_anchor_lead=12,
                             window_frames=15)
        self.assertEqual(_assistant_texts(rows[0])[0], "<think>\nT\n</think>\nact_15")
        self.assertEqual(stats["n_demoted"], 0)


class LegacyModeRegressionTest(unittest.TestCase):
    """Golden regression for the pre-goal builder: default-flag legacy
    invocations must keep producing exactly this output."""

    def test_golden_thinking_window(self) -> None:
        day = _day(n=14)
        anchored = {2: _thought(2, "I check the diff."),
                    8: _thought(8, "I rerun the tests.")}
        rows, stats = build_legacy_day_rows(
            day, anchored, window_frames=6, context_thoughts=8, min_anchor_lead=3,
            thinking_only=True, system_prompt="LEGACY SYS", terminal_token=None,
            fps=0.5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["n_demoted"], 1)
        self.assertEqual(stats["n_thoughts_placed"], 1)

        expected_messages = [{"role": "system",
                              "content": [{"type": "text", "text": "LEGACY SYS"}]}]
        for i in range(6):
            expected_messages.append({
                "role": "user",
                "content": [{"type": "image",
                             "image": f"ar://fake/images.array_record#{i}"}]})
            text = f"act_{i}"
            if i == 2:
                text = f"<think>\nI check the diff.\n</think>\n{text}"
            expected_messages.append({
                "role": "assistant", "content": [{"type": "text", "text": text}]})
        self.assertEqual(rows[0], {
            "conversation_id": "u0_20260101_c00_w000",
            "day_tag": "u0_20260101",
            "chunk_index": 0,
            "recording_id": "r0",
            "segment_ids": ["s0"],
            "t_start": "+00:00:00",
            "t_end": "+00:00:10",
            "target_fps": 0.5,
            "window_frames": 6,
            "goal_conditioned": False,
            "annotation_method": "lumine_thinking",
            "n_frames": 6,
            "n_turns": 6,
            "n_thoughts": 1,
            "n_context_thoughts": 0,
            "thought_ids": ["u0_20260101_t0002"],
            "n_non_noop": 6,
            "messages": expected_messages,
        })

    def test_legacy_terminal_token_still_glues_to_every_window(self) -> None:
        day = _day(n=6)
        anchored = {1: _thought(1, "T")}
        rows, _ = build_legacy_day_rows(
            day, anchored, window_frames=6, context_thoughts=8, min_anchor_lead=3,
            thinking_only=True, system_prompt=None, terminal_token="<TERM>",
            fps=0.5)
        self.assertEqual(_assistant_texts(rows[0])[-1], "act_5\n<TERM>")

    def test_legacy_context_block_lists_earlier_chunk_thoughts(self) -> None:
        day = _day(n=12)
        anchored = {1: _thought(1, "First thought."), 8: _thought(8, "Second.")}
        rows, _ = build_legacy_day_rows(
            day, anchored, window_frames=6, context_thoughts=8, min_anchor_lead=2,
            thinking_only=True, system_prompt=None, terminal_token=None, fps=0.5)
        self.assertEqual(len(rows), 2)
        block = _user_turns(rows[1])[0][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"],
                         "Your thoughts so far this session:\n[+00:00:02] First thought.")


if __name__ == "__main__":
    unittest.main()
