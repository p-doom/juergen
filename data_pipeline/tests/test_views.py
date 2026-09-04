"""Shared selector (lib/views): stride validation, masked-slot skip (never
substitute a neighbor), window tiling with conservation, idle spans not being
dead zones, and the artifact-id join refusal.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.lib.events import RawEvent, apply_label_policy
from pipeline.lib.manifest import make_artifact_id
from pipeline.lib.views import (
    FilterArtifact,
    build_segment_view,
    resolve_stride,
)


def _seg(
    *,
    n_records: int = 150,
    kept_ranges: list[list[int]],
    dropped: list[dict] | None = None,
    master_fps: float = 15.0,
) -> dict:
    return {
        "segment_id": "s0",
        "recording_id": "r0",
        "segment_idx": 0,
        "master_fps": master_fps,
        "n_master_records": n_records,
        "shard_path": "/nowhere/frames/s0/images.array_record",
        "keylog_path": None,
        "alignment_status": "aligned",
        "kept_ranges": kept_ranges,
        "dropped": dropped or [],
    }


class StrideTest(unittest.TestCase):
    def test_integer_strides(self) -> None:
        self.assertEqual(resolve_stride(15.0, 1.0), 15)
        self.assertEqual(resolve_stride(15.0, 0.5), 30)
        self.assertEqual(resolve_stride(15.0, 15.0), 1)
        self.assertEqual(resolve_stride(4.0, 0.5), 8)

    def test_non_integer_strides_fail_loudly(self) -> None:
        for master, fps in ((15.0, 2.0), (4.0, 5.0), (15.0, 0.4), (4.0, 3.0)):
            with self.assertRaises(ValueError):
                resolve_stride(master, fps)
        with self.assertRaises(ValueError):
            resolve_stride(15.0, 0.0)


class SelectorTest(unittest.TestCase):
    def test_masked_slot_is_skipped_never_substituted(self) -> None:
        # Black span [60,75) masks slot 4 (tick 60): the view has NO frame for
        # it — not tick 59, not tick 75-as-slot-4.
        view = build_segment_view(
            _seg(
                kept_ranges=[[0, 60], [75, 150]],
                dropped=[{"start": 60, "end": 75, "reason": "black"}],
            ),
            fps=1.0,
        )
        self.assertEqual(
            [f.master_idx for f in view.frames], [0, 15, 30, 45, 75, 90, 105, 120, 135]
        )
        self.assertEqual([f.slot for f in view.frames], [0, 1, 2, 3, 5, 6, 7, 8, 9])
        self.assertEqual(view.n_masked_slots, 1)
        self.assertEqual(
            view.frames[4].image,
            "ar:///nowhere/frames/s0/images.array_record#75",
        )

    def test_windows_tile_without_overlap(self) -> None:
        view = build_segment_view(
            _seg(
                kept_ranges=[[0, 60], [75, 150]],
                dropped=[{"start": 60, "end": 75, "reason": "black"}],
            ),
            fps=1.0,
        )
        for a, b in zip(view.frames, view.frames[1:], strict=False):
            self.assertEqual(a.win_end, b.win_start)  # contiguous, no overlap
            self.assertLess(a.win_start, a.win_end)
        self.assertEqual(view.frames[0].win_start, 0)
        self.assertEqual(view.frames[-1].win_end, 150)
        # The window before the black zone spans OVER it (zone handled by the
        # label policy, not by window geometry).
        self.assertEqual((view.frames[3].win_start, view.frames[3].win_end), (45, 75))

    def test_idle_spans_are_not_dead_zones(self) -> None:
        # Idle interior [30,45) masks slot 2; it must NOT appear as a dead
        # zone — the previous frame's window passes over it.
        view = build_segment_view(
            _seg(
                kept_ranges=[[0, 30], [45, 150]],
                dropped=[{"start": 30, "end": 45, "reason": "idle_interior"}],
            ),
            fps=1.0,
        )
        self.assertEqual([z.reason for z in view.dead_zones], [])
        self.assertEqual((view.frames[1].win_start, view.frames[1].win_end), (15, 45))

    def test_pre_first_frame_zone_absorbs_earlier_drops(self) -> None:
        view = build_segment_view(
            _seg(
                kept_ranges=[[30, 150]],
                dropped=[{"start": 0, "end": 30, "reason": "black"}],
            ),
            fps=1.0,
        )
        self.assertEqual([f.master_idx for f in view.frames][:2], [30, 45])
        self.assertEqual(
            [(z.start, z.end, z.reason) for z in view.dead_zones],
            [(0, 30, "pre_first_frame")],
        )

    def test_empty_view(self) -> None:
        view = build_segment_view(_seg(kept_ranges=[]), fps=1.0)
        self.assertEqual(view.frames, [])
        self.assertEqual(view.n_masked_slots, view.n_slots)

    def test_conservation_over_view(self) -> None:
        # Every event is owned by exactly one window or discarded-with-reason.
        view = build_segment_view(
            _seg(
                kept_ranges=[[0, 60], [75, 150]],
                dropped=[{"start": 60, "end": 75, "reason": "black"}],
            ),
            fps=1.0,
        )
        events = [
            RawEvent(i, t, "move", dx=1.0)
            for i, t in enumerate([0.2, 1.7, 3.99, 4.0, 4.3, 5.0, 8.6, 9.99, 11.0])
        ]  # seconds; ticks 3,25,59,60,64,75,129,149,165 at 15 fps
        labeled, _ = apply_label_policy(
            events, view.windows(), view.dead_zones, master_fps=view.master_fps
        )
        n_owned = sum(1 for le in labeled if le.window is not None)
        n_discarded = sum(1 for le in labeled if le.discard_reason is not None)
        self.assertEqual(n_owned + n_discarded, len(events))
        # ticks 60 and 64 are in the black zone; tick 165 is past coverage.
        self.assertEqual(n_discarded, 3)


class ArtifactJoinTest(unittest.TestCase):
    def _make_master(self, root: Path) -> Path:
        master = root / "master"
        master.mkdir()
        (master / "manifest.json").write_text(json.dumps({"artifact_type": "m", "v": 1}))
        return master

    def _make_filter(self, root: Path, master: Path) -> Path:
        fdir = root / "filter_art"
        (fdir / "filter").mkdir(parents=True)
        (fdir / "filter_index.jsonl").write_text("")
        (fdir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "realigned_filter_mask",
                    "master_fps": 15.0,
                    "master_store_id": make_artifact_id(master),
                }
            )
        )
        return fdir

    def test_matching_ids_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = self._make_master(root)
            art = FilterArtifact(self._make_filter(root, master))
            self.assertEqual(art.master_dir, master.resolve())
            self.assertEqual(art.stride_for(0.5), 30)

    def test_rebuilt_master_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = self._make_master(root)
            fdir = self._make_filter(root, master)
            (master / "manifest.json").write_text(json.dumps({"artifact_type": "m", "v": 2}))
            with self.assertRaises(ValueError):
                FilterArtifact(fdir)

    def test_wrong_artifact_type_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = self._make_master(root)
            fdir = self._make_filter(root, master)
            manifest = json.loads((fdir / "manifest.json").read_text())
            manifest["artifact_type"] = "something_else"
            (fdir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                FilterArtifact(fdir)


if __name__ == "__main__":
    unittest.main()
