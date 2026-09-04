from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.lib.manifest import make_artifact_id
from pipeline.stage_01_master_frames import run_merge


def _write_merge_inputs(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    clips = source / "clips_manifest.jsonl"
    clips.write_text("{}\n")
    (source / "manifest.json").write_text("{}\n")

    output = tmp_path / "master"
    output.mkdir()
    source_sha256 = hashlib.sha256(clips.read_bytes()).hexdigest()
    source_id = make_artifact_id(source)
    suffix = "_of_0002"
    for index in range(2):
        row = {
            "segment_id": f"recording_seg{index:04d}",
            "status": "ok",
            "num_records": index + 1,
            "total_jpeg_bytes": (index + 1) * 100,
        }
        (output / f"segment_index.shard{index:04d}{suffix}.jsonl").write_text(
            json.dumps(row) + "\n"
        )
        summary = {
            "ffmpeg_bin": "/usr/bin/ffmpeg",
            "jpeg_quality": 92,
            "master_fps": 4.0,
            "num_shards": 2,
            "source_clips_id": source_id,
            "source_clips_manifest": str(clips),
            "source_clips_sha256": source_sha256,
            "target_height": 720,
            "shard_index": index,
            "n_segments": 1,
            "n_records_total": row["num_records"],
            "total_jpeg_bytes": row["total_jpeg_bytes"],
            "status_counts": {"ok": 1},
        }
        (output / f"frames_master_summary.shard{index:04d}{suffix}.json").write_text(
            json.dumps(summary)
        )
    return output


def _merge(output: Path) -> None:
    run_merge(argparse.Namespace(output_dir=output, num_shards=2))


def test_merge_requires_every_shard_summary_and_invalidates_stale_marker(
    tmp_path: Path,
):
    output = _write_merge_inputs(tmp_path)
    marker = output / "manifest.json"
    marker.write_text("stale")
    (output / "frames_master_summary.shard0001_of_0002.json").unlink()

    with pytest.raises(RuntimeError, match="summary set is incomplete"):
        _merge(output)
    assert not marker.exists()


def test_merge_requires_identical_shard_parameters(tmp_path: Path):
    output = _write_merge_inputs(tmp_path)
    path = output / "frames_master_summary.shard0001_of_0002.json"
    summary = json.loads(path.read_text())
    summary["master_fps"] = 8.0
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="shard summary mismatch"):
        _merge(output)


def test_merge_publishes_one_closed_artifact(tmp_path: Path):
    output = _write_merge_inputs(tmp_path)
    _merge(output)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status_counts"] == {"ok": 2}
    assert manifest["n_segments"] == 2
    assert manifest["n_records_total"] == 3
    assert manifest["merged_shards"] == [0, 1]
