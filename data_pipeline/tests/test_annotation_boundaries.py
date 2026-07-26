"""lumine_goal_boundaries: span grouping, judge frame-index math, fail-closed
JSON parsing, sidecar row assembly, and the full run_unit loop against a
scripted fake labeler (injected through MethodContext — the method only ever
touches ``ctx.labeler.call_json_full``). No labeler calls, no frame store:
``_render`` is patched so no ar:// reads happen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from realigned_pipeline.annotation.lib.days import DayFrame, DayStream, fmt_t, frame_label
from realigned_pipeline.annotation.lib.registry import MethodContext, load_method
from realigned_pipeline.annotation.methods.lumine_goal_boundaries import annotator as gb
from realigned_pipeline.lib.common import read_jsonl, write_jsonl


def _day(n: int = 20, chunk_splits: tuple[int, ...] = (), step_s: float = 2.0) -> DayStream:
    """A synthetic one-user day: n frames at step_s spacing (0.5 fps shape),
    optionally split into chunks at the given day indices."""
    frames = [
        DayFrame(day_idx=i, t_day_s=step_s * i, segment_id="s0", recording_id="r0",
                 master_idx=i * 30, image=f"ar://fake/images.array_record#{i}", action="NO_OP")
        for i in range(n)
    ]
    bounds = [0, *chunk_splits, n]
    chunks = [frames[a:b] for a, b in zip(bounds, bounds[1:]) if a < b]
    return DayStream(day_tag="u0_20260101", user_id="u0", date="2026-01-01",
                     frames=frames, chunks=chunks, gap_cut_s=180.0, n_segments=1)


def _clip_row(key: str, lo: int, hi: int, gid=None, text=None, t0=None, t1=None,
              step_s: float = 2.0) -> dict:
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


class FakeLabeler:
    """Scripted stand-in for lib.labeler.Labeler: same call_json_full surface,
    responses selected by cache-file stem, every call recorded. Raises on an
    unscripted call — a resume run must make none."""

    def __init__(self, responses: dict[str, dict]):
        self.config = SimpleNamespace(model="fake-model")
        self.responses = responses
        self.calls: list[dict] = []

    def call_json_full(self, system, user_text, images=None, image_labels=None,
                       cache_path=None, no_cache=False, max_completion_tokens=None):
        stem = Path(cache_path).stem if cache_path else "call"
        self.calls.append({"stem": stem, "system": system, "user": user_text,
                           "n_images": len(images or []), "labels": list(image_labels or [])})
        if stem not in self.responses:
            raise AssertionError(f"unscripted labeler call: {stem}")
        parsed = self.responses[stem]
        res = SimpleNamespace(content=json.dumps(parsed), reasoning="",
                              finish_reason="stop", usage={"total_tokens": 100},
                              model="fake-model")
        return parsed, res


def _fake_render(ctx, frames):
    return (["data:image/jpeg;base64,eA=="] * len(frames),
            [frame_label(fr) for fr in frames])


class RegistryTest(unittest.TestCase):
    def test_method_loads_and_prompts_render(self) -> None:
        m = load_method("lumine_goal_boundaries")
        self.assertEqual(m.input_kind, "days")
        self.assertEqual(m.labeler_defaults, {"temperature": 0.2, "reasoning_effort": "low"})
        judge = m.prompts.render("judge", goal="G", log="L", n_end=3, n_after=2,
                                 t_end_label="+00:01:00")
        self.assertIn("GOAL: G", judge)
        self.assertIn("last 3 frame(s)", judge)
        self.assertNotIn("${", judge)
        self.assertIn('{"completed": true | false', judge)  # JSON braces intact
        nm = m.prompts.render("near_miss", goal="G", log="L")
        self.assertIn("not_done_reason", nm)
        self.assertNotIn("${", nm)


class SpanGroupingTest(unittest.TestCase):
    def test_groups_skip_nogoal_and_merge_recurrence(self) -> None:
        rows = [
            _clip_row("clip_0000", 0, 4),
            _clip_row("clip_0001", 5, 9, gid=1, text="goal one", t0=10.0, t1=30.0),
            _clip_row("clip_0002", 10, 14, gid=1, text="goal one", t0=10.0, t1=30.0),
            _clip_row("clip_0003", 15, 19, gid=2, text="goal two", t0=31.0, t1=39.0),
            _clip_row("clip_0005", 25, 29, gid=1, text="goal one", t0=10.0, t1=30.0),
        ]
        spans = gb.group_goal_spans(rows)
        self.assertEqual([s["goal_id"] for s in spans], [2, 1])  # ordered by boundary
        g1 = spans[1]
        self.assertEqual(len(g1["clips"]), 3)  # recurrence merged: ONE span per goal_id
        self.assertEqual(g1["clips"][-1]["clip_key"], "clip_0005")  # boundary = last clip
        self.assertEqual(g1["goal_text"], "goal one")

    def test_all_nogoal_yields_no_spans(self) -> None:
        self.assertEqual(gb.group_goal_spans([_clip_row("clip_0000", 0, 4)]), [])


class FrameIndexTest(unittest.TestCase):
    def test_end_and_after_windows(self) -> None:
        end, after = gb.judge_frame_indices(5, 14, 3, 2, chunk_end=19)
        self.assertEqual((end, after), ([12, 13, 14], [15, 16]))

    def test_short_span_clamps_to_span_start(self) -> None:
        end, _ = gb.judge_frame_indices(5, 6, 3, 2, chunk_end=19)
        self.assertEqual(end, [5, 6])

    def test_after_clamped_to_chunk_end(self) -> None:
        _, after = gb.judge_frame_indices(5, 18, 3, 5, chunk_end=19)
        self.assertEqual(after, [19])
        _, after = gb.judge_frame_indices(5, 19, 3, 2, chunk_end=19)
        self.assertEqual(after, [])  # span ends the chunk: never cross a gap

    def test_inverted_span_refused(self) -> None:
        with self.assertRaises(ValueError):
            gb.judge_frame_indices(10, 9, 3, 2, chunk_end=19)

    def test_chunk_end_idx(self) -> None:
        day = _day(20, chunk_splits=(12,))
        self.assertEqual(gb.chunk_end_idx(day, 3), 11)
        self.assertEqual(gb.chunk_end_idx(day, 12), 19)
        with self.assertRaises(KeyError):
            gb.chunk_end_idx(day, 99)


class AlignmentTest(unittest.TestCase):
    def test_matching_clip_passes(self) -> None:
        gb.check_clip_alignment(_clip_row("clip_0001", 5, 9), _day())

    def test_time_mismatch_raises(self) -> None:
        row = _clip_row("clip_0001", 5, 9)
        row["t_range"] = ["+00:00:10", "+00:09:59"]  # not what the stream says
        with self.assertRaisesRegex(ValueError, "misaligned"):
            gb.check_clip_alignment(row, _day())

    def test_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            gb.check_clip_alignment(_clip_row("clip_0009", 15, 25), _day(20))


class ParseTest(unittest.TestCase):
    def test_clean_completion_passes_through(self) -> None:
        j = gb.parse_judgment({"completed": True, "confidence": "high",
                               "evidence": "the  reply shows", "final_thought": " sent.  done "})
        self.assertEqual(j, {"completed": True, "confidence": "high",
                             "evidence": "the reply shows", "final_thought": "sent. done"})

    def test_string_bool_and_case_normalize(self) -> None:
        j = gb.parse_judgment({"completed": "True", "confidence": "HIGH",
                               "evidence": "e", "final_thought": "t"})
        self.assertTrue(j["completed"])
        self.assertEqual(j["confidence"], "high")

    def test_fail_closed(self) -> None:
        # Unknown confidence -> low; junk completed -> false.
        self.assertEqual(gb.parse_judgment({"completed": "maybe", "confidence": "medium"}),
                         {"completed": False, "confidence": "low",
                          "evidence": "", "final_thought": ""})
        # Completed without evidence (or thought) is demoted: never trusted.
        j = gb.parse_judgment({"completed": True, "confidence": "high",
                               "evidence": "", "final_thought": "t"})
        self.assertEqual((j["completed"], j["confidence"]), (True, "low"))
        # Not completed clears any stray final_thought.
        j = gb.parse_judgment({"completed": False, "confidence": "high",
                               "evidence": "e", "final_thought": "leak"})
        self.assertEqual(j["final_thought"], "")
        self.assertEqual(gb.parse_judgment(None)["completed"], False)

    def test_near_miss_requires_both_fields(self) -> None:
        self.assertIsNone(gb.parse_near_miss({"not_done_reason": "r"}))
        self.assertIsNone(gb.parse_near_miss({"next_step_thought": "t"}))
        self.assertIsNone(gb.parse_near_miss(None))
        self.assertEqual(gb.parse_near_miss({"not_done_reason": " r ",
                                             "next_step_thought": "t"}),
                         {"not_done_reason": "r", "next_step_thought": "t"})


class RunUnitTest(unittest.TestCase):
    """Full day loop with the fake client: judge + near-miss calls, sidecar
    rows, single-clip-span rule, resume from the per-span ledger."""

    ROWS = [
        _clip_row("clip_0000", 0, 4),
        _clip_row("clip_0001", 5, 9, gid=1, text="reply to TUM", t0=10.0, t1=30.0),
        _clip_row("clip_0002", 10, 14, gid=1, text="reply to TUM", t0=10.0, t1=30.0),
        _clip_row("clip_0003", 15, 19, gid=2, text="file the form", t0=31.0, t1=39.0),
    ]
    RESPONSES = {
        "span_000_g1_judge": {
            "completed": True, "confidence": "high",
            "evidence": "the reply appears in the thread above the empty input box",
            "final_thought": "the reply to TUM is sent — it shows in the thread, done.",
        },
        "span_000_g1_nearmiss": {
            "not_done_reason": "the draft still ends mid-sentence and has not been sent",
            "next_step_thought": "the draft needs its closing line — I'll finish it and hit send",
        },
        # Clean high completion, but the span has ONE clip: no near-miss call
        # (an unscripted span_001_g2_nearmiss would raise in FakeLabeler).
        "span_001_g2_judge": {"completed": "true", "confidence": "HIGH",
                              "evidence": "the form shows its submitted banner",
                              "final_thought": "the form is filed — the submitted banner is up."},
    }

    def _setup(self, tmp: Path) -> tuple[Path, Path]:
        goals_dir = tmp / "in_artifact"
        day_units_dir = tmp / "out" / "units" / "u0_20260101"
        day_units_dir.mkdir(parents=True)
        (goals_dir / "units" / "u0_20260101").mkdir(parents=True)
        (goals_dir / "memory").mkdir(parents=True)
        write_jsonl(goals_dir / "units" / "u0_20260101" / "goals_active.jsonl", self.ROWS)
        write_jsonl(goals_dir / "memory" / "u0_20260101.jsonl",
                    [{"clip_key": r["clip_key"], "log": f"log for {r['clip_key']}"}
                     for r in self.ROWS])
        return goals_dir, day_units_dir

    def _ctx(self, fake: FakeLabeler, tmp: Path, goals_dir: Path,
             day_units_dir: Path) -> MethodContext:
        return MethodContext(
            labeler=fake, prompts=load_method("lumine_goal_boundaries").prompts,
            cache_dir=tmp / "calls", vlm_frame_height=720, jpeg_quality=80,
            params={"goals_dir": str(goals_dir), "day_units_dir": str(day_units_dir),
                    "memory_path": tmp / "out" / "memory" / "u0_20260101.jsonl"})

    def test_run_unit_and_resume(self) -> None:
        day = _day(20)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            goals_dir, day_units_dir = self._setup(tmp)
            fake = FakeLabeler(dict(self.RESPONSES))
            with mock.patch.object(gb, "_render", _fake_render):
                result = gb.run_unit({"id": day.day_tag, "day": day, "row": {}},
                                     self._ctx(fake, tmp, goals_dir, day_units_dir))

            self.assertEqual(result["thoughts"], [])
            self.assertEqual((result["n_spans"], result["n_spans_resumed"]), (2, 0))
            self.assertEqual((result["n_completed"], result["n_completed_high"],
                              result["n_near_miss"]), (2, 2, 1))
            self.assertEqual(result["actual_tokens"], 300)  # 2 judges + 1 near-miss

            # Judge for g1 saw the span's last 3 frames + 2 marked after-frames.
            judge = next(c for c in fake.calls if c["stem"] == "span_000_g1_judge")
            self.assertEqual(judge["n_images"], 5)
            self.assertTrue(judge["labels"][0].startswith("frame 12 |"))
            self.assertTrue(judge["labels"][3].startswith("(AFTER the goal stretch) frame 15"))
            self.assertIn("GOAL: reply to TUM", judge["user"])
            self.assertIn("log for clip_0002", judge["user"])
            # Near-miss saw the WHOLE preceding clip (clip_0001: frames 5..9).
            nm = next(c for c in fake.calls if c["stem"] == "span_000_g1_nearmiss")
            self.assertEqual(nm["n_images"], 5)
            self.assertTrue(nm["labels"][0].startswith("frame 5 |"))
            self.assertIn("log for clip_0001", nm["user"])
            # g2's judge saw NO after-frames (its span ends the chunk).
            judge2 = next(c for c in fake.calls if c["stem"] == "span_001_g2_judge")
            self.assertEqual(judge2["n_images"], 3)

            # Sidecar rows.
            out = read_jsonl(tmp / "out" / "boundaries" / "u0_20260101.jsonl")
            self.assertEqual(len(out), 2)
            r1, r2 = out
            self.assertEqual((r1["goal_id"], r1["clip_key"], r1["n_clips_in_span"]),
                             (1, "clip_0002", 2))
            self.assertTrue(r1["completed"])
            self.assertEqual(r1["confidence"], "high")
            self.assertEqual(r1["near_miss"]["clip_key"], "clip_0001")
            self.assertEqual(r1["near_miss"]["day_idx_range"], [5, 9])
            self.assertIn("closing line", r1["near_miss"]["next_step_thought"])
            self.assertEqual((r1["model"], r1["goal_t_end"]), ("fake-model", 30.0))
            self.assertTrue(r1["ts"])
            # g2: verified high, but single-clip span -> no clean negative exists.
            self.assertEqual((r2["completed"], r2["confidence"], r2["near_miss"]),
                             (True, "high", None))
            ledger = json.loads((day_units_dir / "span_001_g2.json").read_text())
            self.assertEqual(ledger["near_miss_status"], "single_clip_span")

            # Resume: a second run makes ZERO labeler calls, same rows.
            strict = FakeLabeler({})  # any call would raise
            with mock.patch.object(gb, "_render", _fake_render):
                again = gb.run_unit({"id": day.day_tag, "day": day, "row": {}},
                                    self._ctx(strict, tmp, goals_dir, day_units_dir))
            self.assertEqual(again["n_spans_resumed"], 2)
            self.assertEqual(strict.calls, [])
            self.assertEqual(read_jsonl(tmp / "out" / "boundaries" / "u0_20260101.jsonl"), out)

    def test_max_spans_and_missing_sidecar(self) -> None:
        day = _day(20)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            goals_dir, day_units_dir = self._setup(tmp)
            ctx = self._ctx(FakeLabeler({"span_000_g1_judge":
                                         {"completed": False, "confidence": "low",
                                          "evidence": "", "final_thought": ""}}),
                            tmp, goals_dir, day_units_dir)
            ctx.params["max_spans_per_day"] = "1"
            with mock.patch.object(gb, "_render", _fake_render):
                result = gb.run_unit({"id": day.day_tag, "day": day, "row": {}}, ctx)
            self.assertEqual((result["n_spans"], result["n_completed"]), (1, 0))

            # A day with no goal sidecar judges nothing and still writes its file.
            other = _day(20)
            other.day_tag = "u0_20260102"
            du2 = tmp / "out" / "units" / "u0_20260102"
            du2.mkdir(parents=True)
            ctx2 = self._ctx(FakeLabeler({}), tmp, goals_dir, du2)
            with mock.patch.object(gb, "_render", _fake_render):
                result = gb.run_unit({"id": other.day_tag, "day": other, "row": {}}, ctx2)
            self.assertEqual(result["n_spans"], 0)
            self.assertEqual(read_jsonl(tmp / "out" / "boundaries" / "u0_20260102.jsonl"), [])

    def test_goals_dir_required(self) -> None:
        ctx = MethodContext(labeler=FakeLabeler({}), prompts=None, cache_dir=Path("/x"),
                            vlm_frame_height=720, jpeg_quality=80,
                            params={"day_units_dir": "/x/units/d"})
        with self.assertRaisesRegex(ValueError, "goals_dir"):
            gb.run_unit({"id": "d", "day": _day(4), "row": {}}, ctx)


if __name__ == "__main__":
    unittest.main()
