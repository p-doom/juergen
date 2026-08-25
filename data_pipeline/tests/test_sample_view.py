"""The sampler's output, lifted into the ``SegmentView`` stage 04 consumes.

Stage 04 is written against exactly one shape, so the two stage-03 readings must
produce the same one. These pin the places where the adapter has to make a
choice the filter path makes for it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.crowdcast.lib.samples import SampleArtifact, build_sample_view

MASTER_FPS = 15.0
TARGET_FPS = 1.0


def _write(records, **row_overrides):
    td = Path(tempfile.mkdtemp())
    sample = td / "sample"
    (sample / "clips" / "seg0" / "stage_01").mkdir(parents=True)
    fr = sample / "clips" / "seg0" / "stage_01" / "frame_records.jsonl"
    fr.write_text("".join(json.dumps(r) + "\n" for r in records))
    row = {
        "segment_id": "seg0", "recording_id": "rec0", "segment_idx": 0,
        "status": "ok", "target_fps": TARGET_FPS, "master_fps": MASTER_FPS,
        "keylog_path": str(td / "keylog.msgpack"), "alignment_status": "ok",
        "frame_records": str(fr),
    }
    row.update(row_overrides)
    (sample / "sample_index.jsonl").write_text(json.dumps(row) + "\n")
    (sample / "manifest.json").write_text(json.dumps({
        "artifact_type": "juergen_sampled_frames", "master_fps": MASTER_FPS,
        "target_fps": TARGET_FPS, "master_store_id": "fake-master",
    }))
    return sample, row, fr


def _records(ticks):
    return [
        {
            "segment_id": "seg0", "recording_id": "rec0", "segment_idx": 0,
            "local_bin_idx": i, "global_frame_idx": i,
            "local_time_s": t / MASTER_FPS, "source_frame_idx": t,
            "master_record_index": t, "image_path": f"ar://store#{i}",
            "action": "SAMPLER_LABEL_THAT_MUST_BE_IGNORED",
        }
        for i, t in enumerate(ticks)
    ]


class SampleViewTest(unittest.TestCase):
    def test_windows_tile_contiguously(self):
        """``events._Locator`` requires a tiling with no holes; a gap would make
        an event belong to no window at all."""
        sample, row, fr = _write(_records([0, 15, 30]), n_master_records=45)
        view = build_sample_view(row, fr)
        bounds = [(f.win_start, f.win_end) for f in view.frames]
        self.assertEqual(bounds, [(0, 15), (15, 30), (30, 45)])

    def test_the_last_window_runs_to_the_master_axis_end(self):
        """NOT to the last kept tick. The sampler drops black and thinned-NO_OP
        frames, so real video usually continues past the last kept frame — ending
        coverage there would push every action in that tail into the trailing
        no_coverage zone, silently deleting it from the final turn."""
        sample, row, fr = _write(_records([0, 15, 75]), n_master_records=200)
        view = build_sample_view(row, fr)
        self.assertEqual(view.frames[-1].win_end, 200)

    def test_a_pre_field_index_falls_back_conservatively(self):
        """Without `n_master_records` the adapter cannot know where the video
        ends, so it ends coverage at the last frame rather than inventing a
        span — discarding a tail action instead of mislabelling one."""
        sample, row, fr = _write(_records([0, 15, 75]))
        view = build_sample_view(row, fr)
        self.assertEqual(view.frames[-1].win_end, 76)

    def test_the_gap_before_the_first_frame_is_a_dead_zone(self):
        """Nothing was visible there, so its keystrokes must not fold into the
        first turn's label."""
        sample, row, fr = _write(_records([30, 45]), n_master_records=60)
        view = build_sample_view(row, fr)
        self.assertEqual([(z.start, z.end) for z in view.dead_zones], [(0, 30)])

    def test_no_dead_zone_when_the_first_frame_is_tick_zero(self):
        sample, row, fr = _write(_records([0, 15]), n_master_records=30)
        self.assertEqual(build_sample_view(row, fr).dead_zones, [])

    def test_non_monotonic_ticks_are_refused(self):
        """Overlapping label windows would double-count events; a sampler that
        emitted these is broken and must not produce a dataset."""
        sample, row, fr = _write(_records([0, 30, 15]), n_master_records=45)
        with self.assertRaises(ValueError):
            build_sample_view(row, fr)

    def test_a_missing_master_fps_is_refused(self):
        sample, row, fr = _write(_records([0, 15]), master_fps=0)
        with self.assertRaises(ValueError):
            build_sample_view(row, fr)

    def test_the_view_reports_the_sampler_fps_not_a_recomputed_one(self):
        sample, row, fr = _write(_records([0, 15, 30]), n_master_records=45)
        view = build_sample_view(row, fr)
        self.assertEqual(view.fps, TARGET_FPS)
        self.assertEqual(view.fps_mode, "sampled")
        self.assertEqual(view.stride, MASTER_FPS / TARGET_FPS)

    def test_the_artifact_exposes_the_filter_paths_surface(self):
        """Stage 04 holds either artifact through the same three calls."""
        sample, row, fr = _write(_records([0, 15]), n_master_records=30)
        art = SampleArtifact(sample)
        self.assertEqual([r["segment_id"] for r in art.usable_rows()], ["seg0"])
        self.assertEqual(art.segment_path("seg0"), fr)
        self.assertEqual(art.master_fps, MASTER_FPS)
        self.assertEqual(art.filter_id, art.sample_id)

    def test_a_non_ok_segment_is_not_usable(self):
        sample, row, fr = _write(_records([0, 15]), status="no_master_frames")
        self.assertEqual(SampleArtifact(sample).usable_rows(), [])

    def test_a_directory_without_a_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                SampleArtifact(Path(td))


if __name__ == "__main__":
    unittest.main()
