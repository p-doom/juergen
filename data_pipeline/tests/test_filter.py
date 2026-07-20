"""Stage-03 filter: duration-based idle knobs reproduce the legacy NO_OP
head/tail thinning at equivalent params, black masking is per master tick, and
the worker caches on identical params.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import msgpack

from realigned_pipeline.stage_03_filter import (
    REASON_BLACK,
    REASON_IDLE,
    REASON_KEPT,
    _activity_mask,
    _coarsen_activity,
    _compress_reasons,
    _idle_interiors,
    filter_segment,
)

MASTER_FPS = 15.0
STRIDE_05 = 30  # 0.5 fps on a 15 fps master (the legacy sampling rate)


def _legacy_noop_thinning(noop_bins: list[bool], head: int = 1, tail: int = 1) -> list[bool]:
    """The old sampler's Pass 2 (stage_03_sample_frames): within each maximal
    run of NO_OP bins keep the first ``head`` + last ``tail``, drop the middle."""
    keep = [True] * len(noop_bins)
    i = 0
    while i < len(noop_bins):
        if not noop_bins[i]:
            i += 1
            continue
        j = i
        while j < len(noop_bins) and noop_bins[j]:
            j += 1
        if (j - i) > head + tail:
            for k in range(i + head, j - tail):
                keep[k] = False
        i = j
    return keep


class IdleLegacyEquivalenceTest(unittest.TestCase):
    def test_matches_legacy_thinning_on_bin_aligned_runs(self) -> None:
        # 40 legacy bins of 2 s (0.5 fps) = 1200 master ticks. Activity placed
        # at bin STARTS so idle runs align with bin boundaries — there the
        # duration knobs (>4 s, keep 2 s ends) must reproduce head/tail=1
        # exactly: the same slots survive.
        active_bins = {0, 1, 2, 9, 10, 15, 25, 39}  # NO_OP runs of len 6, 4, 9, 13
        n_bins = 40
        n_ticks = n_bins * STRIDE_05
        active = [False] * n_ticks
        for b in active_bins:
            active[b * STRIDE_05] = True

        interiors = _idle_interiors(
            active, MASTER_FPS, min_duration_s=4.0, keep_head_s=2.0, keep_tail_s=2.0
        )
        masked = [False] * n_ticks
        for s, e in interiors:
            for t in range(s, e):
                masked[t] = True
        new_kept_slots = [b for b in range(n_bins) if not masked[b * STRIDE_05]]

        noop_bins = [b not in active_bins for b in range(n_bins)]
        legacy = _legacy_noop_thinning(noop_bins)
        legacy_kept_slots = [b for b in range(n_bins) if legacy[b]]

        self.assertEqual(new_kept_slots, legacy_kept_slots)

    def test_runs_at_threshold_are_kept_whole(self) -> None:
        # An inactive run of exactly 4 s (60 ticks) is NOT thinned (> only),
        # matching legacy: a 2-bin NO_OP run survives head/tail=1.
        active = [True] + [False] * 60 + [True] * 10
        self.assertEqual(
            _idle_interiors(active, MASTER_FPS, 4.0, 2.0, 2.0),
            [],
        )

    def test_long_run_keeps_head_and_tail_seconds(self) -> None:
        # Inactive [10, 160) on a 15 fps axis (10 s): interior drops
        # [10+30, 160-30) = [40, 130), keeping 2 s at each end.
        active = [False] * 170
        for t in list(range(10)) + list(range(160, 170)):
            active[t] = True
        self.assertEqual(
            _idle_interiors(active, MASTER_FPS, 4.0, 2.0, 2.0),
            [(40, 130)],
        )

    def test_fps_agnostic_semantics(self) -> None:
        # The same real-time pattern judged on a 4 fps master: identical spans
        # in seconds. Inactive [2.5s, 40s): interior [4.5s, 38s) = ticks [18,152).
        n = 4 * 42
        active = [False] * n
        for t in list(range(10)) + list(range(160, n)):  # active first 2.5 s + from 40 s
            active[t] = True
        self.assertEqual(
            _idle_interiors(active, 4.0, 4.0, 2.0, 2.0),
            [(10 + 8, 160 - 8)],
        )


class MaskMechanicsTest(unittest.TestCase):
    def test_compress_reasons(self) -> None:
        reasons = (
            [REASON_KEPT] * 3 + [REASON_BLACK] * 2 + [REASON_IDLE] * 2 + [REASON_KEPT] * 1
        )
        kept, dropped = _compress_reasons(reasons)
        self.assertEqual(kept, [[0, 3], [7, 8]])
        self.assertEqual(
            dropped,
            [
                {"start": 3, "end": 5, "reason": "black"},
                {"start": 5, "end": 7, "reason": "idle_interior"},
            ],
        )

    def test_coarsen_activity(self) -> None:
        active = [True, False, False, False, False, False, True, False]
        self.assertEqual(
            _coarsen_activity(active, 3),
            [True, True, True, False, False, False, True, True],
        )
        self.assertIs(_coarsen_activity(active, 1), active)

    def test_activity_mask_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            keylog = Path(tmp) / "k.msgpack"
            keylog.write_bytes(
                msgpack.packb(
                    [
                        [500_000, ["MouseMove", [0.0, 0.0]]],  # zero delta: NOT activity
                        [1_500_000, ["MouseMove", [0.3, 0.0]]],  # sub-pixel drift: activity
                        [2_500_000, ["MouseScroll", [0, 0]]],  # zero scroll: NOT activity
                        [3_500_000, ["KeyPress", [0, "KeyA"]]],
                        [4_500_000, ["KeyRelease", [0, "KeyA"]]],
                        [99_000_000, ["KeyPress", [0, "KeyB"]]],  # past axis: ignored
                    ]
                )
            )
            active = _activity_mask(keylog, 10, master_fps=1.0)
        self.assertEqual(active, [False, True, False, True, True, False, False, False, False, False])


class FilterSegmentTest(unittest.TestCase):
    def _make_task(self, root: Path, *, mean_lumas: list[float], events: list[list]) -> dict:
        seg = "seg0"
        frames_dir = root / "frames" / seg
        frames_dir.mkdir(parents=True)
        shard = frames_dir / "images.array_record"
        rows = [
            {
                "record_index": i,
                "image": f"ar://{shard}#{i}",
                "mean_luma": luma,
                "frac_dark": 0.0,
            }
            for i, luma in enumerate(mean_lumas)
        ]
        with (frames_dir / "frame_manifest.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        keylog = root / "keylog.msgpack"
        keylog.write_bytes(msgpack.packb(events))
        (root / "filter").mkdir(exist_ok=True)
        return {
            "manifest_row": {
                "segment_id": seg,
                "recording_id": "rec0",
                "segment_idx": 0,
                "keylog_path": str(keylog),
                "alignment_status": "aligned",
                "video_duration_s": float(len(mean_lumas)),
            },
            "master_row": {
                "status": "ok",
                "master_fps": 1.0,
                "shard_path": str(shard),
            },
            "filter_dir": str(root / "filter"),
            "frames_dir": str(root / "frames"),
            "master_fps": 1.0,
            "drop_black_frames": True,
            "black_luma_max": 6.0,
            "black_dark_frac_min": 0.999,
            "idle_min_duration_s": 3.0,
            "idle_keep_head_s": 1.0,
            "idle_keep_tail_s": 1.0,
            "qc_view_fps": None,
            "qc_dir": None,
            "force": False,
        }

    def test_black_and_idle_masking_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 12 ticks; ticks 4-5 black; activity only at ticks 0 and 11 ->
            # inactive run [1,11) (10 ticks > 3 s): interior [2,10) idle.
            lumas = [50.0] * 12
            lumas[4] = lumas[5] = 2.0
            events = [
                [200_000, ["MouseMove", [3.0, 0.0]]],
                [11_200_000, ["MouseMove", [0.0, 4.0]]],
            ]
            task = self._make_task(root, mean_lumas=lumas, events=events)
            res = filter_segment(task)
            self.assertEqual(res["status"], "ok", res.get("error"))
            doc = json.loads((root / "filter" / "seg0.json").read_text())
            self.assertEqual(doc["kept_ranges"], [[0, 2], [10, 12]])
            self.assertEqual(
                doc["dropped"],
                [
                    {"start": 2, "end": 4, "reason": "idle_interior"},
                    {"start": 4, "end": 6, "reason": "black"},  # black wins over idle
                    {"start": 6, "end": 10, "reason": "idle_interior"},
                ],
            )
            self.assertEqual(doc["n_black"], 2)
            self.assertEqual(doc["n_idle_interior"], 6)
            self.assertEqual(doc["n_kept"], 4)

    def test_cache_hits_on_same_params_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._make_task(
                root, mean_lumas=[50.0] * 5, events=[[100_000, ["KeyPress", [0, "KeyA"]]]]
            )
            self.assertEqual(filter_segment(task)["status"], "ok")
            self.assertEqual(filter_segment(task)["status"], "cached")
            changed = dict(task, black_luma_max=9.0)
            self.assertEqual(filter_segment(changed)["status"], "ok")
            forced = dict(task, force=True)
            self.assertEqual(filter_segment(forced)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
