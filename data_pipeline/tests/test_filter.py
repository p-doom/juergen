from __future__ import annotations

import json
from pathlib import Path

import msgpack
import pytest

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
