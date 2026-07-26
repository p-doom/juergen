"""Stage 04t goal-bounded windows: span tiling, outcome-frame TERMINATE
placement (clean/verified/all), near-miss thought attachment, leak-free
GOAL/So-far memory selection, no-goal exclusion, memory-update samples, the
TERMINATE-is-own-turn invariant, and the legacy goal-free mode's golden
regression. All on synthetic DayStreams + sidecar rows — no filter artifact,
no frame store, no labeler.
"""

from __future__ import annotations

import unittest

from realigned_pipeline.annotation.lib.days import DayFrame, DayStream, fmt_t
from realigned_pipeline.lib.action_format import get_formatter
from realigned_pipeline.stage_04_thinking_conversations import (
    MEMORY_UPDATE_PROMPT,
    TERMINATE_TOKEN,
    build_goal_day_rows,
    build_legacy_day_rows,
    goal_system_prompt_file,
    group_goal_runs,
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
    cfg = dict(window_frames=24, terminate_mode="clean", terminate_max_lag_s=180.0,
               min_anchor_lead=0, memory_update_samples=False, system_prompt="SYS",
               fps=0.5, action_format="canonical",
               annotation_method="lumine_thinking_goals")
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


class SelectMemoryTest(unittest.TestCase):
    def test_latest_strictly_earlier_row_wins(self) -> None:
        rows = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1")]
        self.assertEqual(select_memory(rows, 30), "M1")
        self.assertEqual(select_memory(rows, 15), "M0")

    def test_overlapping_row_is_rejected(self) -> None:
        rows = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1")]
        # window starts at 20: [15, 29] overlaps it (ends at 29 >= 20) -> M0
        self.assertEqual(select_memory(rows, 20), "M0")
        # inclusive range ending exactly ON the window start also overlaps
        self.assertEqual(select_memory([_mem("c0", 0, 20, "M0")], 20), "")

    def test_no_earlier_memory(self) -> None:
        self.assertEqual(select_memory([_mem("c0", 0, 14, "M0")], 0), "")


class SpanTilingTest(unittest.TestCase):
    def test_tiles_span_and_excludes_no_goal_frames(self) -> None:
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
        self.assertEqual([r["n_frames"] for r in rows], [24, 21])  # 45 frames <= 24 each
        shown = sorted(i for r in rows for i in _frame_idxs(r))
        self.assertEqual(shown, list(range(15, 60)))  # in-span only, nothing else
        self.assertEqual(stats["n_terminate_turns"], 0)
        self.assertEqual(stats["n_terminate_skipped_no_outcome_frame"], 1)
        self.assertEqual(stats["n_no_goal_frames_excluded"], 100 - 45)
        for r in rows:
            self.assertTrue(r["goal_conditioned"])
            self.assertEqual(r["goal_text"], "Do the thing")
            first_user = _user_turns(r)[0]
            self.assertEqual(first_user[0]["type"], "text")
            self.assertTrue(first_user[0]["text"].startswith("GOAL: Do the thing"))

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
                             terminate_mode="verified", window_frames=48)
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
                                 terminate_mode="verified", window_frames=48)
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
                             window_frames=48)
        row = [r for r in rows if r["goal_id"] == 1][0]
        self.assertEqual(
            _assistant_texts(row)[0],
            "<think>\nThe draft still needs sending.\n</think>\nact_0")
        self.assertIn("near_miss:g0001", row["thought_ids"])
        self.assertEqual(stats["n_near_miss_attached"], 1)
        self.assertEqual(stats["n_thoughts_placed"], 1)  # replacement, not addition


class GoalMemoryBlockTest(unittest.TestCase):
    def test_first_window_no_memory_then_leak_free_selection(self) -> None:
        day = _day(n=60)
        active = [_clip("c0", 0, 14, gid=1, text="G", t0=0.0, t1=1000.0),
                  _clip("c1", 15, 29, gid=1, text="G", t0=0.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=0.0, t1=1000.0)]
        memory = [_mem("c0", 0, 14, "M0"), _mem("c1", 15, 29, "M1"),
                  _mem("c2", 30, 44, "M2")]
        rows, _ = _build(day, active, memory=memory)
        # window 1 [0..23]: no strictly-earlier memory -> GOAL line only
        b0 = _user_turns(rows[0])[0][0]["text"]
        self.assertEqual(b0, "GOAL: G")
        self.assertFalse(rows[0]["has_memory"])
        # window 2 [24..44]: [15,29] overlaps its start (24) -> only M0 usable
        b1 = _user_turns(rows[1])[0][0]["text"]
        self.assertEqual(b1, "GOAL: G\nSo far: M0")
        self.assertTrue(rows[1]["has_memory"])


class MemoryUpdateSamplesTest(unittest.TestCase):
    def _setup(self):
        day = _day(n=60)
        active = [_clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0)]
        memory = [_mem("c1", 15, 29, "The draft  reads 'Muss ich'."),
                  _mem("c2", 30, 44, "M2")]
        return day, active, memory

    def test_appendix_shape_and_verbatim_memory(self) -> None:
        day, active, memory = self._setup()
        rows, stats = _build(day, active, memory=memory, memory_update_samples=True)
        # window 1 [15..38] fully contains clip [15,29] -> appendix
        row = rows[0]
        self.assertTrue(row["memory_update"])
        self.assertEqual(row["messages"][-2]["role"], "user")
        self.assertEqual(row["messages"][-2]["content"],
                         [{"type": "text", "text": MEMORY_UPDATE_PROMPT}])  # no image
        self.assertEqual(row["messages"][-1]["role"], "assistant")
        self.assertEqual(row["messages"][-1]["content"][0]["text"],
                         "The draft  reads 'Muss ich'.")  # verbatim, not normalized
        # window 2 [39..44] fully contains no clip -> no appendix
        self.assertFalse(rows[1]["memory_update"])
        self.assertEqual(stats["n_memory_update_samples"], 1)

    def test_terminate_window_skips_the_appendix(self) -> None:
        day, active, memory = self._setup()
        active[0]["goal_t_end"] = active[1]["goal_t_end"] = 86.0  # inside c2
        rows, stats = _build(day, active, memory=memory, memory_update_samples=True,
                             terminate_mode="all", window_frames=48)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["terminate"], "all")
        self.assertFalse(row["memory_update"])
        self.assertEqual(_assistant_texts(row)[-1], TERMINATE_TOKEN)
        self.assertEqual(stats["n_memory_update_samples"], 0)

    def test_flag_off_appends_nothing(self) -> None:
        day, active, memory = self._setup()
        rows, stats = _build(day, active, memory=memory)
        self.assertFalse(any(r["memory_update"] for r in rows))
        self.assertEqual(stats["n_memory_update_samples"], 0)


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
        # exactly one terminate for goal 1 (plus goal 2's own, 'all' mode)
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
                    # the whole action line is TERMINATE, optionally preceded
                    # by exactly one think block — never an action + TERMINATE
                    self.assertTrue(
                        text == TERMINATE_TOKEN
                        or (text.startswith("<think>\n")
                            and text.endswith("\n</think>\n" + TERMINATE_TOKEN)),
                        f"glued TERMINATE in {mode}: {text!r}")
                    self.assertNotRegex(text, r"act_\d+")


class TerminateHookTest(unittest.TestCase):
    """Terminate turns come from the formatter's ``terminate_line()`` — the
    TERMINATE literal for the text formats, the native terminate tool_call
    block for computer_use_rel_v1 — never a hardcoded literal."""

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
        # the TERMINATE sentinel resolves through the formatter; canonical/v2/v3
        # are byte-identical, computer_use renders its native block
        self.assertEqual(resolve_terminal_token(TERMINATE_TOKEN, "canonical"),
                         TERMINATE_TOKEN)
        self.assertEqual(resolve_terminal_token(TERMINATE_TOKEN, "ordered_events_v3"),
                         TERMINATE_TOKEN)
        self.assertEqual(
            resolve_terminal_token(TERMINATE_TOKEN, "computer_use_rel_v1"),
            self.NATIVE_TERMINATE)
        # anything else (incl. the default None) passes through untouched
        self.assertIsNone(resolve_terminal_token(None, "computer_use_rel_v1"))
        self.assertEqual(resolve_terminal_token("<TERM>", "computer_use_rel_v1"),
                         "<TERM>")

    def test_goal_system_prompt_defaults_per_format(self) -> None:
        self.assertEqual(goal_system_prompt_file("computer_use_rel_v1").name,
                         "cua_v4_thinking.txt")
        for name in ("canonical", "ordered_events_v2", "ordered_events_v3"):
            self.assertEqual(goal_system_prompt_file(name).name,
                             "cua_v3_thinking.txt")


class ThoughtsInGoalWindowsTest(unittest.TestCase):
    def test_thoughts_attach_and_windows_without_thoughts_are_kept(self) -> None:
        day = _day(n=60)
        active = [_clip("c1", 15, 29, gid=1, text="G", t0=30.0, t1=1000.0),
                  _clip("c2", 30, 44, gid=1, text="G", t0=30.0, t1=1000.0)]
        anchored = {20: _thought(20, "I switch to the terminal.")}
        rows, stats = _build(day, active, anchored=anchored)
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
        # window 2 starts mid-chunk at 39; anchor at 40 sits 1 frame in
        anchored = {40: _thought(40, "T")}
        rows, stats = _build(day, active, anchored=anchored, min_anchor_lead=12)
        self.assertEqual(_assistant_texts(rows[1])[1], "act_40")  # demoted
        self.assertEqual(stats["n_demoted"], 1)
        # a chunk-start window is exempt: anchor at pos 0 of window 1 keeps it
        anchored = {15: _thought(15, "T")}
        day2 = _day(n=60, chunk_splits=(15,))
        rows, stats = _build(day2, active, anchored=anchored, min_anchor_lead=12)
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
        # window 0 [0..5]: thought@2 kept (chunk-start exemption); window 1
        # [6..11]: thought@8 at pos 2 < 3 demoted -> thinking_only drops it;
        # window 2 [12..13]: no thoughts -> dropped.
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
