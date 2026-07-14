import json
import tempfile
import unittest
from pathlib import Path

import msgpack

from annotation_pipeline import stage_02_observation_view
from annotation_pipeline.common import normalize_keylog_events
from annotation_pipeline.stage_02_observation_view import build_observation_view


def _frame(frame_idx: int, time_s: float) -> dict:
    return {
        "recording_id": "rec",
        "segment_id": "seg",
        "segment_idx": 0,
        "local_frame_idx": frame_idx,
        "local_time_s": time_s,
        "global_time_s": time_s,
        "image_path": f"ar:///tmp/images.array_record#{frame_idx}",
    }


def _key_event(source_event_idx: int, time_s: float, kind: str) -> dict:
    return {
        "recording_id": "rec",
        "segment_id": "seg",
        "segment_idx": 0,
        "source_event_idx": source_event_idx,
        "timestamp_us": int(time_s * 1_000_000),
        "local_time_s": time_s,
        "global_time_s": time_s,
        "kind": kind,
        "key": "KeyA",
    }


class ObservationViewTest(unittest.TestCase):
    def test_raw_move_click_move_order_is_preserved_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keylog = Path(tmpdir) / "input.msgpack"
            keylog.write_bytes(
                msgpack.packb(
                    [
                        [100_000, ["MouseMove", [1.0, 0.0]]],
                        [200_000, ["MousePress", ["Left"]]],
                        [300_000, ["MouseMove", [2.0, 0.0]]],
                    ],
                    use_bin_type=True,
                )
            )

            events, _ = normalize_keylog_events(
                keylog,
                recording_id="rec",
                segment_id="seg",
                segment_idx=0,
                segment_offset_s=0.0,
            )

        self.assertEqual([event["kind"] for event in events], ["move", "press", "move"])
        self.assertEqual(events[1]["key"], "LMB")

    def test_preserves_held_key_state_across_observation_intervals(self) -> None:
        observations, _ = build_observation_view(
            frames=[_frame(0, 0.0), _frame(1, 1.0)],
            events=[_key_event(0, 0.5, "press"), _key_event(1, 1.5, "release")],
            segment_summaries=[{"segment_id": "seg", "duration_s": 2.0}],
            base_fps=1.0,
            observation_fps=1.0,
            idle_keep_head=10,
            idle_keep_tail=10,
        )

        self.assertEqual(observations[0]["action_bin"]["events"], [["+", "KeyA"]])
        self.assertEqual(observations[1]["action_bin"]["events"], [["-", "KeyA"]])
        self.assertNotIn("action", observations[0])

    def test_materializes_an_observation_view_from_base_modalities(self) -> None:
        self.assertTrue(hasattr(stage_02_observation_view, "materialize_observation_view"))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            base.mkdir()
            (base / "manifest.json").write_text(
                json.dumps({"stage": "base_modalities", "base_fps": 1.0})
            )
            (base / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in [_frame(0, 0.0), _frame(1, 1.0)]) + "\n"
            )
            (base / "events.jsonl").write_text("")
            (base / "segment_summaries.json").write_text(
                json.dumps([{"segment_id": "seg", "duration_s": 2.0}])
            )

            manifest = stage_02_observation_view.materialize_observation_view(
                base_dir=base,
                output_dir=root / "view",
                observation_fps=1.0,
                idle_keep_head=10,
                idle_keep_tail=10,
            )

            self.assertNotIn("view_name", manifest)
            self.assertEqual(manifest["observation_fps"], 1.0)
            self.assertTrue((root / "view" / "observations.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
