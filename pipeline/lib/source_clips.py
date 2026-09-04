"""Validate the canonical Crowd-Cast source inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.lib.common import read_jsonl
from pipeline.lib.manifest import file_sha256_short, make_artifact_id

_SEGMENT_ID = re.compile(r"(?P<recording>.+)_seg(?P<index>[0-9]{4})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROW_FIELDS = {
    "keylog_path",
    "keylog_sha256",
    "recording_id",
    "segment_id",
    "segment_idx",
    "user_id",
    "version",
    "video_duration_s",
    "video_fps",
    "video_frame_count",
    "video_height",
    "video_ok",
    "video_path",
    "video_sha256",
    "video_width",
}


def resolve_source_clips(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve()
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifact_type",
        "clips_file",
        "clips_sha256",
        "n_recordings",
        "n_segments",
        "schema_version",
        "source_root",
    }:
        raise ValueError(f"invalid Crowd-Cast source manifest: {manifest_path}")
    if (
        manifest["artifact_type"] != "crowdcast_source_clips"
        or manifest["schema_version"] != 1
        or manifest["clips_file"] != "clips_manifest.jsonl"
        or path != (path.parent / manifest["clips_file"]).resolve()
        or not isinstance(manifest["source_root"], str)
        or not Path(manifest["source_root"]).is_absolute()
    ):
        raise ValueError(f"Crowd-Cast source contract mismatch: {manifest_path}")
    observed_sha = file_sha256_short(path, n=64)
    if manifest["clips_sha256"] != observed_sha:
        raise ValueError(f"Crowd-Cast source digest mismatch: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Crowd-Cast source inventory is empty: {path}")
    root = Path(manifest["source_root"])
    identities: set[tuple[str, int]] = set()
    ordered_identities: list[tuple[str, int]] = []
    for row in rows:
        if set(row) != _ROW_FIELDS:
            raise ValueError(f"invalid Crowd-Cast source row fields: {sorted(row)}")
        recording = row["recording_id"]
        index = row["segment_idx"]
        match = (
            _SEGMENT_ID.fullmatch(row["segment_id"])
            if isinstance(row["segment_id"], str)
            else None
        )
        if (
            not isinstance(recording, str)
            or not recording
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or match is None
            or match.group("recording") != recording
            or int(match.group("index")) != index
            or row["video_ok"] is not True
        ):
            raise ValueError(f"invalid Crowd-Cast source identity: {row!r}")
        user = row["user_id"]
        version = row["version"]
        video = row["video_path"]
        keylog = row["keylog_path"]
        segment_tag = f"seg{index:04d}"
        if (
            not isinstance(user, str)
            or not user
            or not isinstance(version, str)
            or not version
            or not isinstance(video, str)
            or not Path(video).is_absolute()
            or Path(video)
            != root
            / "uploads"
            / version
            / user
            / "recordings"
            / f"recording_{recording}_{segment_tag}.mp4"
            or not isinstance(keylog, str)
            or not Path(keylog).is_absolute()
            or Path(keylog)
            != root
            / "uploads"
            / version
            / user
            / "keylogs"
            / f"input_{recording}_{segment_tag}.msgpack"
            or not isinstance(row["video_sha256"], str)
            or _SHA256.fullmatch(row["video_sha256"]) is None
            or not isinstance(row["keylog_sha256"], str)
            or _SHA256.fullmatch(row["keylog_sha256"]) is None
        ):
            raise ValueError(f"invalid Crowd-Cast source path or digest: {row!r}")
        for field in ("video_frame_count", "video_width", "video_height"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid Crowd-Cast source {field}: {row!r}")
        for field in ("video_duration_s", "video_fps"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"invalid Crowd-Cast source {field}: {row!r}")
        identity = (recording, index)
        if identity in identities:
            raise ValueError(f"duplicate Crowd-Cast source segment: {identity}")
        identities.add(identity)
        ordered_identities.append(identity)
    if ordered_identities != sorted(ordered_identities):
        raise ValueError(f"Crowd-Cast source rows are not canonical: {path}")
    counts = (manifest["n_segments"], manifest["n_recordings"])
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in counts
    ) or counts != (len(rows), len({row["recording_id"] for row in rows})):
        raise ValueError(f"Crowd-Cast source counts mismatch: {manifest_path}")
    return rows, {
        "path": str(path),
        "sha256": observed_sha,
        "artifact_id": make_artifact_id(path.parent),
    }
