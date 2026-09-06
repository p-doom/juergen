from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pytest
from image_domain import encode_jpeg_q92, validate_jpeg_q92
from PIL import Image

from pipeline.lib import master_frames, source_clips
from pipeline.lib.manifest import file_sha256_short, make_artifact_id
from pipeline.stage_01_master_frames import pack_master_arrayrecord, run_merge


def test_q92_contract_rejects_a_low_quality_jpeg() -> None:
    image = Image.new("RGB", (64, 48), "navy")
    with validate_jpeg_q92(encode_jpeg_q92(image)) as decoded:
        assert decoded.size == image.size
    output = io.BytesIO()
    image.save(output, "JPEG", quality=20, subsampling=2)
    with pytest.raises(ValueError, match="canonical JPEG q92"):
        validate_jpeg_q92(output.getvalue())


def _write_merge_inputs(tmp_path: Path) -> Path:
    source_root = (tmp_path / "dataset").resolve()
    source = tmp_path / "source"
    source.mkdir()
    clips = source / "clips_manifest.jsonl"
    output = (tmp_path / "master").resolve()
    output.mkdir()
    clip_rows = []
    index_rows = []
    for index in range(2):
        recording_id = "recording"
        segment_id = f"{recording_id}_seg{index:04d}"
        user_dir = source_root / "uploads" / "v1" / "user"
        video = user_dir / "recordings" / f"recording_{segment_id}.mp4"
        keylog = user_dir / "keylogs" / f"input_{segment_id}.msgpack"
        video.parent.mkdir(parents=True, exist_ok=True)
        keylog.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{index}".encode())
        keylog.write_bytes(f"keylog-{index}".encode())
        clip_row = {
            "keylog_path": str(keylog),
            "keylog_sha256": hashlib.sha256(keylog.read_bytes()).hexdigest(),
            "recording_id": recording_id,
            "segment_id": segment_id,
            "segment_idx": index,
            "user_id": "user",
            "version": "v1",
            "video_duration_s": 0.25,
            "video_fps": 4.0,
            "video_frame_count": 1,
            "video_height": 720,
            "video_ok": True,
            "video_path": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "video_width": 1280,
        }
        clip_rows.append(clip_row)
        frame_dir = output / "frames" / segment_id
        frame_dir.mkdir(parents=True)
        buffer = io.BytesIO()
        Image.new("RGB", (1280, 720), (index * 20, 40, 80)).save(
            buffer, "JPEG", quality=92, subsampling=2
        )
        frame_path = frame_dir / "frame_000000.jpg"
        frame_path.write_bytes(buffer.getvalue())
        packed = pack_master_arrayrecord([frame_path], frame_dir, 4.0, 4.0, 1)
        outputs = {
            key: packed[key]
            for key in (
                "frame_manifest_sha256",
                "num_records",
                "shard_sha256",
                "total_jpeg_bytes",
            )
        }
        (frame_dir / "segment_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "inputs": {
                        "jpeg_quality": 92,
                        "master_fps": 4.0,
                        "target_height": 720,
                        "video_sha256": clip_row["video_sha256"],
                    },
                    "outputs": outputs,
                }
            )
        )
        index_rows.append(
            {
                "segment_id": segment_id,
                "recording_id": recording_id,
                "segment_idx": index,
                "master_fps": 4.0,
                "target_height": 720,
                "jpeg_quality": 92,
                "video_duration_s": 0.25,
                "video_fps": 4.0,
                "video_sha256": clip_row["video_sha256"],
                "status": "ok",
                "shard_path": packed["shard_path"],
                "frame_manifest": packed["manifest_path"],
                **outputs,
            }
        )
    clips.write_text("".join(json.dumps(row) + "\n" for row in clip_rows))
    clips_sha256 = hashlib.sha256(clips.read_bytes()).hexdigest()
    exclusions = source / "exclusions.jsonl"
    exclusions.write_text("")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "crowdcast_source_clips",
                "schema_version": 1,
                "clips_file": "clips_manifest.jsonl",
                "clips_sha256": clips_sha256,
                "exclusions_file": "exclusions.jsonl",
                "exclusions_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
                "source_root": str(source_root),
                "n_segments": 2,
                "n_recordings": 1,
                "n_exclusions": 0,
                "n_source_videos": 2,
                "n_source_keylogs": 2,
                "exclusion_counts": {},
            }
        )
    )
    source_id = make_artifact_id(source)
    suffix = "_of_0002"
    for index, row in enumerate(index_rows):
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
            "source_clips_sha256": clips_sha256,
            "target_height": 720,
            "shard_index": index,
            "n_segments": 1,
            "n_records_total": 1,
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
    assert manifest["n_records_total"] == 2
    assert manifest["merged_shards"] == [0, 1]


def test_master_consumer_does_not_replay_raw_or_decode_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = _write_merge_inputs(tmp_path)
    _merge(output)
    original_source_hash = source_clips.file_sha256_short
    original_master_hash = master_frames.file_sha256_short

    def reject_raw_hash(path: Path, *, n: int) -> str:
        if path.suffix in {".mp4", ".msgpack"}:
            raise AssertionError(f"raw payload was replayed: {path}")
        return original_source_hash(path, n=n)

    def reject_frame_hash(path: Path, *, n: int) -> str:
        if path.name in {"images.array_record", "frame_manifest.jsonl"}:
            raise AssertionError(f"frame payload was replayed: {path}")
        return original_master_hash(path, n=n)

    monkeypatch.setattr(source_clips, "file_sha256_short", reject_raw_hash)
    monkeypatch.setattr(master_frames, "file_sha256_short", reject_frame_hash)
    monkeypatch.setattr(
        master_frames,
        "validate_jpeg_q92",
        lambda _payload: (_ for _ in ()).throw(AssertionError("JPEG was decoded")),
    )
    manifest, rows = master_frames.resolve_master_artifact(output)
    assert manifest["n_segments"] == len(rows) == 2


def test_master_consumer_rejects_changed_direct_index(tmp_path: Path):
    output = _write_merge_inputs(tmp_path)
    _merge(output)
    with (output / "segment_index.jsonl").open("a") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="master index digest mismatch"):
        master_frames.resolve_master_artifact(output)


def test_merge_rejects_master_row_rebound_to_changed_source(tmp_path: Path):
    output = _write_merge_inputs(tmp_path)
    source = tmp_path / "source"
    clips = source / "clips_manifest.jsonl"
    rows = [json.loads(line) for line in clips.read_text().splitlines()]
    rows[0]["video_sha256"] = "f" * 64
    clips.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_manifest["clips_sha256"] = file_sha256_short(clips, n=64)
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    source_id = make_artifact_id(source)
    for path in output.glob("frames_master_summary.shard*.json"):
        summary = json.loads(path.read_text())
        summary["source_clips_sha256"] = source_manifest["clips_sha256"]
        summary["source_clips_id"] = source_id
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="source payload digest mismatch"):
        _merge(output)
