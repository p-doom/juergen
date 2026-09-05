"""Validate the canonical Crowd-Cast master frame store."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from image_domain import validate_jpeg_q92
from pipeline.lib import config
from pipeline.lib.image_store import parse_arrayrecord_image_uri
from pipeline.lib.manifest import file_sha256_short
from pipeline.lib.source_clips import resolve_source_clips

_INDEX_FIELDS = {
    "frame_manifest",
    "frame_manifest_sha256",
    "jpeg_quality",
    "master_fps",
    "num_records",
    "recording_id",
    "segment_id",
    "segment_idx",
    "shard_path",
    "shard_sha256",
    "status",
    "target_height",
    "total_jpeg_bytes",
    "video_duration_s",
    "video_fps",
    "video_sha256",
}
_FRAME_FIELDS = {
    "frac_dark",
    "image",
    "jpeg_bytes",
    "mean_luma",
    "record_index",
    "sha256",
    "shard_path",
    "source_frame_idx",
    "source_time_s",
}
_MANIFEST_FIELDS = {
    "artifact_type",
    "ffmpeg_bin",
    "jpeg_quality",
    "master_fps",
    "n_records_total",
    "n_segments",
    "schema_version",
    "segment_index",
    "segment_index_sha256",
    "source_clips_id",
    "source_clips_manifest",
    "source_clips_sha256",
    "status_counts",
    "target_height",
    "total_jpeg_bytes",
}


def _read_exact_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _luma_metrics(jpeg: bytes) -> tuple[float, float]:
    with Image.open(io.BytesIO(jpeg)) as image:
        image.load()
        histogram = image.convert("L").histogram()
    total = sum(histogram) or 1
    return (
        round(sum(index * count for index, count in enumerate(histogram)) / total, 3),
        round(sum(histogram[: config.BLACK_DARK_CUTOFF]) / total, 5),
    )


def _validate_jpeg(
    payload: bytes, *, height: int, width: int, path: Path, index: int
) -> None:
    with validate_jpeg_q92(payload) as image:
        if image.height != height or image.width != width:
            raise ValueError(f"invalid q92 JPEG frame {index} in {path}")


def validate_master_segment(
    row: dict[str, Any],
    *,
    root: Path,
    source_row: dict[str, Any],
) -> None:
    if set(row) != _INDEX_FIELDS:
        raise ValueError(f"invalid master index row fields: {sorted(row)}")
    segment_id = source_row["segment_id"]
    if (
        row["segment_id"] != segment_id
        or row["recording_id"] != source_row["recording_id"]
        or row["segment_idx"] != source_row["segment_idx"]
        or row["video_sha256"] != source_row["video_sha256"]
        or row["video_fps"] != source_row["video_fps"]
        or row["video_duration_s"] != source_row["video_duration_s"]
        or row["status"] != "ok"
        or row["jpeg_quality"] != 92
        or isinstance(row["master_fps"], bool)
        or not isinstance(row["master_fps"], (int, float))
        or row["master_fps"] <= 0
        or isinstance(row["target_height"], bool)
        or not isinstance(row["target_height"], int)
        or row["target_height"] <= 0
    ):
        raise ValueError(f"master/source contract mismatch for {segment_id}")
    expected_dir = (root / "frames" / segment_id).resolve()
    shard = Path(row["shard_path"]).resolve()
    frame_manifest = Path(row["frame_manifest"]).resolve()
    if (
        shard != expected_dir / "images.array_record"
        or frame_manifest != expected_dir / "frame_manifest.jsonl"
        or not shard.is_file()
        or not frame_manifest.is_file()
        or file_sha256_short(shard, n=64) != row["shard_sha256"]
        or file_sha256_short(frame_manifest, n=64) != row["frame_manifest_sha256"]
    ):
        raise ValueError(f"invalid master payload identity for {segment_id}")
    rows = _read_exact_jsonl(frame_manifest)
    expected_width = (
        round(
            source_row["video_width"]
            * row["target_height"]
            / source_row["video_height"]
            / 2
        )
        * 2
    )
    counts = (row["num_records"], row["total_jpeg_bytes"])
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
        or row["num_records"] <= 0
        or row["total_jpeg_bytes"] <= 0
        or len(rows) != row["num_records"]
    ):
        raise ValueError(f"invalid master payload counts for {segment_id}")

    from array_record.python.array_record_module import ArrayRecordReader

    reader = ArrayRecordReader(str(shard))
    if not reader.ok():
        raise ValueError(f"invalid master ArrayRecord: {shard}")
    total_bytes = 0
    try:
        if reader.num_records() != len(rows):
            raise ValueError(f"master ArrayRecord count mismatch: {shard}")
        for index, frame in enumerate(rows):
            if set(frame) != _FRAME_FIELDS:
                raise ValueError(f"invalid frame row {index}: {frame_manifest}")
            payload = reader.read()
            uri_path, uri_index = parse_arrayrecord_image_uri(frame["image"])
            expected_time = round(index / row["master_fps"], 6)
            expected_source_index = min(
                round((index / row["master_fps"]) * source_row["video_fps"]),
                source_row["video_frame_count"] - 1,
            )
            if (
                frame["record_index"] != index
                or frame["shard_path"] != str(shard)
                or uri_path.resolve() != shard
                or uri_index != index
                or frame["source_time_s"] != expected_time
                or isinstance(frame["record_index"], bool)
                or not isinstance(frame["record_index"], int)
                or isinstance(frame["source_time_s"], bool)
                or not isinstance(frame["source_time_s"], (int, float))
                or isinstance(frame["source_frame_idx"], bool)
                or not isinstance(frame["source_frame_idx"], int)
                or frame["source_frame_idx"] != expected_source_index
                or frame["jpeg_bytes"] != len(payload)
                or isinstance(frame["jpeg_bytes"], bool)
                or not isinstance(frame["jpeg_bytes"], int)
                or any(
                    isinstance(frame[field], bool)
                    or not isinstance(frame[field], (int, float))
                    for field in ("mean_luma", "frac_dark")
                )
                or frame["sha256"] != hashlib.sha256(payload).hexdigest()
                or (frame["mean_luma"], frame["frac_dark"]) != _luma_metrics(payload)
            ):
                raise ValueError(
                    f"frame row/payload mismatch at {frame_manifest}:{index}"
                )
            _validate_jpeg(
                payload,
                height=row["target_height"],
                width=expected_width,
                path=shard,
                index=index,
            )
            total_bytes += len(payload)
        try:
            reader.read()
        except IndexError:
            pass
        else:
            raise ValueError(f"master ArrayRecord has uncounted frames: {shard}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot read master ArrayRecord: {shard}") from exc
    finally:
        with contextlib.suppress(Exception):
            reader.close()
    if total_bytes != row["total_jpeg_bytes"]:
        raise ValueError(f"master JPEG byte count mismatch: {shard}")

    marker_path = expected_dir / "segment_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "schema_version": 1,
        "inputs": {
            "jpeg_quality": row["jpeg_quality"],
            "master_fps": row["master_fps"],
            "target_height": row["target_height"],
            "video_sha256": row["video_sha256"],
        },
        "outputs": {
            "frame_manifest_sha256": row["frame_manifest_sha256"],
            "num_records": row["num_records"],
            "shard_sha256": row["shard_sha256"],
            "total_jpeg_bytes": row["total_jpeg_bytes"],
        },
    }
    if marker != expected_marker or {path.name for path in expected_dir.iterdir()} != {
        "frame_manifest.jsonl",
        "images.array_record",
        "segment_manifest.json",
    }:
        raise ValueError(f"invalid master segment closure: {expected_dir}")


def resolve_master_artifact(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = set(manifest)
    merged = {"merged_shards", "num_shards"}
    if fields not in (_MANIFEST_FIELDS, _MANIFEST_FIELDS | merged):
        raise ValueError(f"invalid master manifest fields: {manifest_path}")
    if (
        manifest["artifact_type"] != "juergen_annotation_frames_master"
        or manifest["schema_version"] != 1
        or manifest["segment_index"] != "segment_index.jsonl"
        or manifest["jpeg_quality"] != 92
    ):
        raise ValueError(f"invalid master manifest contract: {manifest_path}")
    source_rows, source = resolve_source_clips(Path(manifest["source_clips_manifest"]))
    if (
        manifest["source_clips_manifest"] != source["path"]
        or manifest["source_clips_sha256"] != source["sha256"]
        or manifest["source_clips_id"] != source["artifact_id"]
    ):
        raise ValueError(f"master source identity mismatch: {manifest_path}")
    index_path = root / "segment_index.jsonl"
    if file_sha256_short(index_path, n=64) != manifest["segment_index_sha256"]:
        raise ValueError(f"master index digest mismatch: {index_path}")
    rows = _read_exact_jsonl(index_path)
    source_by_segment = {row["segment_id"]: row for row in source_rows}
    if (
        not rows
        or [row.get("segment_id") for row in rows] != sorted(source_by_segment)
        or len(rows) != len(source_by_segment)
    ):
        raise ValueError("master index does not cover the exact Stage00 segment set")
    for row in rows:
        validate_master_segment(
            row,
            root=root,
            source_row=source_by_segment[row["segment_id"]],
        )
    if {path.name for path in (root / "frames").iterdir() if path.is_dir()} != set(
        source_by_segment
    ):
        raise ValueError("master frame directories do not match Stage00")
    expected_summary = {
        key: manifest[key]
        for key in _MANIFEST_FIELDS
        - {
            "artifact_type",
            "schema_version",
            "segment_index",
            "segment_index_sha256",
        }
    }
    if fields == _MANIFEST_FIELDS | merged:
        expected_summary.update({key: manifest[key] for key in merged})
    summary = json.loads((root / "frames_master_summary.json").read_text())
    if summary != expected_summary:
        raise ValueError(f"master summary mismatch: {root}")
    integer_totals = (
        manifest["n_segments"],
        manifest["n_records_total"],
        manifest["total_jpeg_bytes"],
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_totals
        )
        or manifest["n_segments"] != len(rows)
        or manifest["status_counts"] != {"ok": len(rows)}
        or manifest["n_records_total"] != sum(row["num_records"] for row in rows)
        or manifest["total_jpeg_bytes"] != sum(row["total_jpeg_bytes"] for row in rows)
        or any(
            row[field] != manifest[field]
            for row in rows
            for field in ("jpeg_quality", "master_fps", "target_height")
        )
    ):
        raise ValueError(f"master receipt mismatch: {root}")
    return manifest, rows
