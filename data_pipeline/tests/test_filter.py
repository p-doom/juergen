from __future__ import annotations

import json
import sys
from pathlib import Path

import msgpack
import pytest

import pipeline.stage_03_filter as stage_03
from pipeline.lib.manifest import file_sha256_short
from pipeline.stage_03_filter import (
    FILTER_PARAMS,
    REASON_BLACK,
    REASON_IDLE,
    REASON_KEPT,
    _compress_reasons,
    _idle_interiors,
    _rounded_activity_mask,
    filter_segment,
)


def _task(root: Path) -> dict:
    segment = "seg0"
    frames = root / "frames" / segment
    frames.mkdir(parents=True)
    shard = frames / "images.array_record"
    shard.write_bytes(b"fixture")
    rows = [
        {
            "record_index": index,
            "image": f"ar://{shard}#{index}",
            "mean_luma": 2.0 if index in (4, 5) else 50.0,
            "frac_dark": 0.0,
        }
        for index in range(12)
    ]
    frame_manifest = frames / "frame_manifest.jsonl"
    frame_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    keylog = root / "keylog.msgpack"
    keylog.write_bytes(
        msgpack.packb(
            [
                [200_000, ["MouseMove", [3.0, 0.0]]],
                [11_200_000, ["MouseMove", [0.0, 4.0]]],
            ]
        )
    )
    keylog_sha256 = file_sha256_short(keylog, n=64)
    (root / "filter").mkdir()
    return {
        "manifest_row": {
            "segment_id": segment,
            "recording_id": "rec0",
            "segment_idx": 0,
            "keylog_path": str(keylog),
            "keylog_sha256": keylog_sha256,
            "alignment_status": "aligned",
            "alignment_closed": True,
            "video_duration_s": 12.0,
        },
        "master_row": {
            "status": "ok",
            "master_fps": 1.0,
            "shard_path": str(shard),
            "shard_sha256": file_sha256_short(shard, n=64),
            "frame_manifest_sha256": file_sha256_short(frame_manifest, n=64),
        },
        "filter_dir": str(root / "filter"),
    }


def _replace_keylog(task: dict, entries: list) -> None:
    keylog = Path(task["manifest_row"]["keylog_path"])
    keylog.write_bytes(msgpack.packb(entries))
    task["manifest_row"]["keylog_sha256"] = file_sha256_short(keylog, n=64)


def test_canonical_idle_activity_is_deltatype_v2(tmp_path: Path):
    keylog = tmp_path / "keylog.msgpack"
    keylog.write_bytes(
        msgpack.packb(
            [
                [200_000, ["MouseMove", [0.4, 0.0]]],
                [2_200_000, ["MouseMove", [0.6, 0.0]]],
            ]
        )
    )
    assert _rounded_activity_mask(keylog, 4, master_fps=1.0, bin_ticks=2) == [
        False,
        False,
        True,
        True,
    ]


def test_idle_and_black_masking_use_the_single_fixed_policy(tmp_path: Path):
    result = filter_segment(_task(tmp_path))
    assert result["status"] == "ok"
    document = json.loads((tmp_path / "filter" / "seg0.json").read_text())
    assert document["params"] == FILTER_PARAMS
    assert document["kept_ranges"] == [[0, 4], [8, 12]]
    assert document["dropped"] == [
        {"start": 4, "end": 6, "reason": "black"},
        {"start": 6, "end": 8, "reason": "idle_interior"},
    ]


def test_filter_requires_complete_inputs(tmp_path: Path):
    task = _task(tmp_path)
    Path(task["manifest_row"]["keylog_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="keylog is missing"):
        filter_segment(task)
    task["master_row"]["status"] = "failed"
    with pytest.raises(ValueError, match="not complete"):
        filter_segment(task)

    task = _task(tmp_path / "unclosed")
    task["manifest_row"]["alignment_closed"] = False
    with pytest.raises(ValueError, match="no closed alignment"):
        filter_segment(task)


def test_filter_excludes_the_whole_segment_for_an_unexecutable_action(
    tmp_path: Path,
):
    task = _task(tmp_path)
    _replace_keylog(task, [[0, ["KeyPress", [0, "PlayPause"]]]])

    result = filter_segment(task)

    assert result["status"] == "excluded_invalid_keylog"
    assert result["exclusion_reason"] == "unexecutable_action"
    assert result["filter_path"] is None
    assert result["filter_sha256"] is None
    assert result["n_kept"] == 0
    assert not (tmp_path / "filter" / "seg0.json").exists()


def test_filter_requires_stage_00_to_have_excluded_an_empty_keylog(tmp_path: Path):
    task = _task(tmp_path)
    _replace_keylog(task, [])

    with pytest.raises(ValueError, match="keylog is empty"):
        filter_segment(task)


def test_filter_refuses_a_segment_with_no_kept_frames(tmp_path: Path):
    task = _task(tmp_path)
    frame_manifest = Path(task["master_row"]["shard_path"]).parent / "frame_manifest.jsonl"
    rows = [json.loads(line) for line in frame_manifest.read_text().splitlines()]
    for row in rows:
        row["mean_luma"] = 0.0
        row["frac_dark"] = 1.0
    frame_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    task["master_row"]["frame_manifest_sha256"] = file_sha256_short(frame_manifest, n=64)
    with pytest.raises(ValueError, match="retained no frames"):
        filter_segment(task)


def test_mask_helpers_preserve_half_open_intervals():
    assert _idle_interiors([True, *([False] * 10), True], 1.0, 4.0, 2.0, 2.0) == [(3, 9)]
    reasons = [REASON_KEPT] * 3 + [REASON_BLACK] * 2 + [REASON_IDLE] * 2
    assert _compress_reasons(reasons) == (
        [[0, 3]],
        [
            {"start": 3, "end": 5, "reason": "black"},
            {"start": 5, "end": 7, "reason": "idle_interior"},
        ],
    )


def test_stage_03_consumes_closed_subset_of_attested_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    master_dir = tmp_path / "master"
    master_dir.mkdir()
    (master_dir / "manifest.json").write_text("{}\n")
    stage_02 = tmp_path / "stage_02"
    stage_02.mkdir()
    clips = stage_02 / "clips_manifest.jsonl"
    clips.write_text(
        json.dumps(
            {
                "segment_id": "accepted",
                "recording_id": "recording",
                "segment_idx": 0,
                "alignment_closed": True,
                "alignment_status": "aligned",
            }
        )
        + "\n"
    )
    alignment = stage_02 / "alignment.jsonl"
    alignment_rows = [
        {
            "segment_id": "accepted",
            "recording_id": "recording",
            "segment_idx": 0,
            "disposition": "accepted",
            "closed": True,
            "status": "aligned",
            "exclusion_reason": None,
            "model": "naive",
            "leading_method": "n/a",
            "n_pauses": 0,
            "total_collapse_s": 0.0,
            "overhang_s": 0.0,
            "residual_s": 0.0,
            "corr_end_s": 1.0,
            "keylog_span_s": 1.0,
            "video_dur_s": 1.0,
            "corrected_keylog_path": None,
            "corrected_keylog_sha256": None,
            "splices": [],
        },
        {
            "segment_id": "excluded",
            "recording_id": "recording",
            "segment_idx": 1,
            "disposition": "excluded",
            "closed": False,
            "exclusion_reason": "no_closed_candidate",
            "candidates": {},
        },
    ]
    alignment.write_text("".join(json.dumps(row) + "\n" for row in alignment_rows))
    source_sha = "a" * 64
    source_id = "/source::0123456789abcdef"
    stage_02_manifest = {
        "artifact_type": "juergen_annotation_clip_manifest_realigned",
        "schema_version": 2,
        "clips_file": clips.name,
        "clips_sha256": file_sha256_short(clips, n=64),
        "alignment_file": alignment.name,
        "alignment_sha256": file_sha256_short(alignment, n=64),
        "source_clips_sha256": source_sha,
        "source_clips_id": source_id,
        "idle_timeout_s": 120.0,
        "closure_tol_s": 2.0,
        "n_source_segments": 2,
        "n_accepted_segments": 1,
        "n_excluded_segments": 1,
        "n_recordings": 1,
        "n_corrected": 0,
        "n_keylogs_repointed": 0,
        "status_counts": {"aligned": 1},
        "exclusion_counts": {"no_closed_candidate": 1},
        "source_clips_manifest": "/source/clips_manifest.jsonl",
    }
    (stage_02 / "manifest.json").write_text(json.dumps(stage_02_manifest) + "\n")
    (stage_02 / "realign_summary.json").write_text(
        json.dumps({key: stage_02_manifest[key] for key in stage_03._STAGE02_SUMMARY_FIELDS}) + "\n"
    )
    master = {
        "master_fps": 1.0,
        "source_clips_sha256": source_sha,
        "source_clips_id": source_id,
    }
    master_rows = [
        {
            "segment_id": "accepted",
            "recording_id": "recording",
            "segment_idx": 0,
            "status": "ok",
        },
        {
            "segment_id": "excluded",
            "recording_id": "recording",
            "segment_idx": 1,
            "status": "ok",
        },
    ]
    monkeypatch.setattr(stage_03, "resolve_master_artifact", lambda _path: (master, master_rows))
    consumed: list[str] = []

    def fake_filter(task: dict) -> dict:
        segment_id = task["manifest_row"]["segment_id"]
        consumed.append(segment_id)
        return {
            "segment_id": segment_id,
            "status": "ok",
            "n_records": 1,
            "n_kept": 1,
            "n_black": 0,
            "n_idle_interior": 0,
        }

    monkeypatch.setattr(stage_03, "filter_segment", fake_filter)
    output = tmp_path / "filter_output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_03_filter.py",
            "--frames_master_dir",
            str(master_dir),
            "--clips_manifest",
            str(clips),
            "--output_dir",
            str(output),
            "--num_workers",
            "1",
        ],
    )

    stage_03.main()

    assert consumed == ["accepted"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n_master_segments"] == 2
    assert manifest["n_input_segments"] == 1
    assert manifest["n_alignment_excluded_segments"] == 1

    alignment.write_text(json.dumps(alignment_rows[0]) + "\n")
    stage_02_manifest["alignment_sha256"] = file_sha256_short(alignment, n=64)
    (stage_02 / "manifest.json").write_text(json.dumps(stage_02_manifest) + "\n")
    with pytest.raises(ValueError, match="partition the Stage01 master"):
        stage_03.main()
