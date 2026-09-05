from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import msgpack
import pytest
import synthetic_clip as clip

from pipeline.lib import realign
from pipeline.lib.source_clips import resolve_source_clips

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = REPO_ROOT / "pipeline"


def _segment(
    root: Path,
    segment_id: str,
    segment_idx: int,
    events: list[tuple[float, str]],
    video_duration_s: float,
) -> dict:
    keylog = root / f"{segment_id}.msgpack"
    video = root / f"{segment_id}.mp4"
    keylog.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        "KeyPress": [0, "KeyA"],
        "KeyRelease": [0, "KeyA"],
        "MousePress": ["Left", 0.0, 0.0],
        "MouseRelease": ["Left", 0.0, 0.0],
        "MouseScroll": [0.0, 1.0, 0.0, 0.0],
        "MouseMove": [1.0, 0.0],
        "WindowTitle": ["title"],
    }
    keylog.write_bytes(
        msgpack.packb(
            [[round(timestamp * 1e6), [kind, payloads[kind]]] for timestamp, kind in events],
            use_bin_type=True,
        )
    )
    video.touch()
    return {
        "segment_id": segment_id,
        "segment_idx": segment_idx,
        "keylog_path": str(keylog),
        "video_path": str(video),
        "video_dur_s": video_duration_s,
    }


def test_no_splice_cannot_certify_a_ten_second_overhang(tmp_path: Path):
    segment = _segment(tmp_path, "rec_seg0000", 0, [(20.0, "WindowTitle")], 10.0)

    result = realign.realign_recording([segment])["rec_seg0000"]

    assert result["closed"] is False
    assert result["exclusion_reason"] == "no_closed_candidate"
    assert result["candidates"] == {
        "naive": {"status": "UNDER", "residual_s": -10.0, "corr_end_s": 20.0},
        "global": {"status": "UNDER", "residual_s": -10.0, "corr_end_s": 20.0},
    }


def test_mp4_without_movie_header_has_no_creation_time(tmp_path: Path):
    video = tmp_path / "empty.mp4"
    video.touch()

    assert realign.mp4_creation_time(video) is None


def test_internal_pause_preserves_benign_idle_classification(tmp_path: Path):
    segment = _segment(
        tmp_path,
        "rec_seg0000",
        0,
        [(0.0, "MouseMove"), (200.0, "MouseMove")],
        150.0,
    )

    result = realign.realign_recording([segment])["rec_seg0000"]

    assert result["model"] == "naive"
    assert result["status"] == "benign_idle"
    assert result["closed"] is True
    assert result["total_collapse_s"] == 80.0


def test_creation_clock_global_candidate_closes_boundary_overhang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    segments = [
        _segment(tmp_path, "rec_seg0000", 0, [(0.0, "MouseMove")], 1.0),
        _segment(tmp_path, "rec_seg0001", 1, [(100.0, "MouseMove")], 20.0),
    ]
    epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    creation_times = {
        Path(segments[0]["video_path"]): epoch,
        Path(segments[1]["video_path"]): epoch + datetime.timedelta(seconds=101),
    }
    monkeypatch.setattr(realign, "mp4_creation_time", creation_times.__getitem__)

    result = realign.realign_recording(segments)["rec_seg0001"]

    assert result["model"] == "global"
    assert result["status"] == "exact"
    assert result["leading_method"] == "overhang"
    assert result["splices"] == [{"kp": 19.0, "vp": 19.0, "collapse": 80.0, "leading": True}]


@pytest.mark.parametrize("second_offset", [None, 0, -1])
def test_multi_segment_recording_requires_increasing_creation_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_offset: int | None,
):
    segments = [
        _segment(tmp_path, "rec_seg0000", 0, [(0.0, "MouseMove")], 1.0),
        _segment(tmp_path, "rec_seg0001", 1, [(0.0, "MouseMove")], 1.0),
    ]
    epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    creation_times = iter(
        [
            epoch,
            None if second_offset is None else epoch + datetime.timedelta(seconds=second_offset),
        ]
    )
    monkeypatch.setattr(realign, "mp4_creation_time", lambda _path: next(creation_times))

    results = realign.realign_recording(segments)

    reason = "unreadable_creation_time" if second_offset is None else "nonincreasing_creation_time"
    assert {result["exclusion_reason"] for result in results.values()} == {reason}


def test_global_splice_must_have_a_segment_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    segments = [
        _segment(tmp_path, "rec_seg0000", 0, [(0.0, "MouseMove")], 1.0),
        _segment(tmp_path, "rec_seg0001", 1, [(0.0, "MouseMove")], 1.0),
    ]
    epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    creation_times = iter([epoch, epoch + datetime.timedelta(seconds=10)])
    monkeypatch.setattr(realign, "mp4_creation_time", lambda _path: next(creation_times))
    monkeypatch.setattr(
        realign,
        "compute_splices",
        lambda _timestamps: [{"kp": 20.0, "vp": 20.0, "collapse": 1.0}],
    )

    results = realign.realign_recording(segments)

    assert {result["exclusion_reason"] for result in results.values()} == {
        "unattributed_global_splice"
    }


@pytest.mark.parametrize("duration", [None, 0.0, float("nan")])
def test_realign_requires_a_finite_manifest_duration(tmp_path: Path, duration: float):
    segment = _segment(tmp_path, "rec_seg0000", 0, [(0.0, "MouseMove")], 1.0)
    segment["video_dur_s"] = duration

    with pytest.raises(ValueError, match="manifest video duration"):
        realign.realign_recording([segment])


def test_source_manifest_duration_is_exact_frame_coverage(tmp_path: Path):
    source = tmp_path / "source"
    video = source / "uploads/v1/user/recordings/recording_rec_seg0000.mp4"
    keylog = source / "uploads/v1/user/keylogs/input_rec_seg0000.msgpack"
    video.parent.mkdir(parents=True)
    keylog.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    keylog.write_bytes(b"keylog")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    row = {
        "segment_id": "rec_seg0000",
        "segment_idx": 0,
        "recording_id": "rec",
        "video_path": str(video),
        "keylog_path": str(keylog),
        "video_sha256": digest(video),
        "keylog_sha256": digest(keylog),
        "user_id": "user",
        "version": "v1",
        "video_ok": True,
        "video_fps": 10.0,
        "video_frame_count": 100,
        "video_duration_s": 10.1,
        "video_width": 1920,
        "video_height": 1080,
    }
    clips = tmp_path / "artifact" / "clips_manifest.jsonl"
    clips.parent.mkdir()
    clips.write_text(json.dumps(row) + "\n")
    exclusions = clips.parent / "exclusions.jsonl"
    exclusions.write_text("")
    manifest = {
        "artifact_type": "crowdcast_source_clips",
        "schema_version": 1,
        "clips_file": clips.name,
        "clips_sha256": digest(clips),
        "exclusions_file": exclusions.name,
        "exclusions_sha256": digest(exclusions),
        "source_root": str(source),
        "n_segments": 1,
        "n_recordings": 1,
        "n_exclusions": 0,
        "n_source_videos": 1,
        "n_source_keylogs": 1,
        "exclusion_counts": {},
    }
    (clips.parent / "manifest.json").write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="inexact Crowd-Cast source video duration"):
        resolve_source_clips(clips)


def test_stage_02_publishes_only_closed_segments_with_a_complete_ledger(
    tmp_path: Path,
):
    source = clip.build_uploads_tree(tmp_path / "source")
    recording_id = "excluded"
    user_dir = source["video_path"].parent.parent
    video = user_dir / "recordings" / f"recording_{recording_id}_seg0000.mp4"
    keylog = user_dir / "keylogs" / f"input_{recording_id}_seg0000.msgpack"
    clip._write_video(video, clip.frames()[:40])
    keylog.write_bytes(
        msgpack.packb([[20_000_000, ["ContextChanged", ["app"]]]], use_bin_type=True)
    )
    stage_00 = tmp_path / "stage_00" / "clips_manifest.jsonl"
    stage_02 = tmp_path / "stage_02"
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    for script, args in (
        (
            "stage_00_clip_manifest.py",
            [
                "--dataset-root",
                str(tmp_path / "source"),
                "--out",
                str(stage_00),
                "--workers",
                "1",
            ],
        ),
        (
            "stage_02_realign.py",
            [
                "--clips-manifest",
                str(stage_00),
                "--output-dir",
                str(stage_02),
                "--num-workers",
                "1",
            ],
        ),
    ):
        process = subprocess.run(
            [sys.executable, str(STAGES / script), *args],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr

    clips = [
        json.loads(line) for line in (stage_02 / "clips_manifest.jsonl").read_text().splitlines()
    ]
    alignment = [
        json.loads(line) for line in (stage_02 / "alignment.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((stage_02 / "manifest.json").read_text())
    assert [row["segment_id"] for row in clips] == [clip.SEGMENT_ID]
    assert {row["segment_id"] for row in alignment} == {
        clip.SEGMENT_ID,
        "excluded_seg0000",
    }
    excluded = next(row for row in alignment if row["disposition"] == "excluded")
    assert excluded["exclusion_reason"] == "no_closed_candidate"
    assert not (stage_02 / "corrected_keylogs" / "excluded_seg0000.msgpack").exists()
    assert manifest["schema_version"] == 2
    assert manifest["n_source_segments"] == 2
    assert manifest["n_accepted_segments"] == 1
    assert manifest["n_excluded_segments"] == 1
    assert manifest["status_counts"] == {"aligned": 1}
    assert manifest["exclusion_counts"] == {"no_closed_candidate": 1}
