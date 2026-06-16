import json
import tempfile
import unittest
from pathlib import Path

from annotation_pipeline.build_canonical_sft import build_canonical_sft


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _trajectory(*, clip_id: str, recording_id: str, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "recording_id": recording_id,
        "instruction": f"complete task for {recording_id}",
        "n_frames": 1,
        "duration_s": 1.0,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "system prompt"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{clip_id}/frame0.jpg"},
                    {"type": "text", "text": f"complete task for {recording_id}"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "10 20 0"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "old final answer"}],
            },
        ],
    }


class BuildCanonicalSftTest(unittest.TestCase):
    def test_rewrites_stage03_outputs_into_canonical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            out_dir = root / "canonical"

            for clip_id, recording_id in (("clip_a", "rec_a"), ("clip_b", "rec_b")):
                (run_dir / clip_id / "frame0.jpg").parent.mkdir(parents=True, exist_ok=True)
                (run_dir / clip_id / "frame0.jpg").write_bytes(b"fake image")
                _write_jsonl(
                    run_dir / clip_id / "stage_03_assemble" / "trajectories.jsonl",
                    [
                        _trajectory(
                            clip_id=clip_id,
                            recording_id=recording_id,
                            sample_id=f"{recording_id}_traj0000",
                        )
                    ],
                )

            manifest = build_canonical_sft(
                run_dir=run_dir,
                output_dir=out_dir,
                split_group="recording_id",
                val_frac=0.5,
                seed=0,
                image_mode="copy",
            )

            self.assertEqual(manifest["artifact_type"], "juergen_canonical_sft")
            self.assertEqual(manifest["n_samples"], 2)
            self.assertTrue((out_dir / "manifest.json").is_file())

            records = [
                json.loads(line)
                for line in (out_dir / "chat.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)

            groups_by_split: dict[str, set[str]] = {}
            for record in records:
                groups_by_split.setdefault(record["recording_id"], set()).add(record["split"])
                messages = record["messages"]
                self.assertEqual(messages[0]["role"], "system")
                first_user = next(msg for msg in messages if msg["role"] == "user")
                self.assertEqual(first_user["content"][0]["type"], "text")
                self.assertTrue(first_user["content"][0]["text"].startswith("complete task"))
                image_block = next(block for block in first_user["content"] if block["type"] == "image")
                image_path = Path(image_block["image"])
                self.assertFalse(image_path.is_absolute())
                self.assertTrue((out_dir / image_path).is_file())
                self.assertEqual(messages[-1]["content"], [{"type": "text", "text": "TERMINATE"}])

            self.assertTrue(all(len(splits) == 1 for splits in groups_by_split.values()))
            self.assertEqual(
                {"train", "val"},
                {record["split"] for record in records},
            )


if __name__ == "__main__":
    unittest.main()
