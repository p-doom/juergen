"""Validate the canonical Crowd-Cast source inventory."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from pipeline.lib.common import read_jsonl
from pipeline.lib.manifest import file_sha256_short, make_artifact_id

_SEGMENT_ID = re.compile(r"(?P<recording>.+)_seg(?P<index>[0-9]{4})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_EXCLUSION_REASONS = frozenset(
    {
        "empty_keylog",
        "invalid_event",
        "invalid_msgpack",
        "missing_keylog",
        "noncanonical_keylog_name",
        "noncanonical_video_name",
        "orphan_keylog",
        "undecodable_video",
    }
)
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
_EXCLUSION_FIELDS = {
    "keylog_path",
    "keylog_sha256",
    "reason",
    "video_path",
    "video_sha256",
}


def _resolve_source_clips_receipt(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    path = path.resolve()
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifact_type",
        "clips_file",
        "clips_sha256",
        "exclusion_counts",
        "exclusions_file",
        "exclusions_sha256",
        "n_exclusions",
        "n_recordings",
        "n_segments",
        "n_source_keylogs",
        "n_source_videos",
        "schema_version",
        "source_root",
    }:
        raise ValueError(f"invalid Crowd-Cast source manifest: {manifest_path}")
    if (
        manifest["artifact_type"] != "crowdcast_source_clips"
        or manifest["schema_version"] != 1
        or manifest["clips_file"] != "clips_manifest.jsonl"
        or manifest["exclusions_file"] != "exclusions.jsonl"
        or path != (path.parent / manifest["clips_file"]).resolve()
        or not isinstance(manifest["source_root"], str)
        or not Path(manifest["source_root"]).is_absolute()
    ):
        raise ValueError(f"Crowd-Cast source contract mismatch: {manifest_path}")
    observed_sha = file_sha256_short(path, n=64)
    if manifest["clips_sha256"] != observed_sha:
        raise ValueError(f"Crowd-Cast source digest mismatch: {path}")
    exclusions_path = path.parent / manifest["exclusions_file"]
    if file_sha256_short(exclusions_path, n=64) != manifest["exclusions_sha256"]:
        raise ValueError(f"Crowd-Cast exclusions digest mismatch: {exclusions_path}")
    exclusions = read_jsonl(exclusions_path)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Crowd-Cast source inventory is empty: {path}")
    root = Path(manifest["source_root"])
    identities: set[tuple[str, int]] = set()
    ordered_identities: list[tuple[str, int]] = []
    attested_videos: set[Path] = set()
    attested_keylogs: set[Path] = set()
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
        attested_videos.add(Path(video))
        attested_keylogs.add(Path(keylog))
        for field in ("video_frame_count", "video_width", "video_height"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid Crowd-Cast source {field}: {row!r}")
        for field in ("video_duration_s", "video_fps"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid Crowd-Cast source {field}: {row!r}")
        if row["video_duration_s"] != row["video_frame_count"] / row["video_fps"]:
            raise ValueError(f"inexact Crowd-Cast source video duration: {row!r}")
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
    exclusion_counts: dict[str, int] = {}
    if exclusions != sorted(
        exclusions,
        key=lambda row: (row["video_path"] or "", row["keylog_path"] or ""),
    ):
        raise ValueError(f"Crowd-Cast source exclusions are not canonical: {path}")
    for exclusion in exclusions:
        if (
            set(exclusion) != _EXCLUSION_FIELDS
            or exclusion["reason"] not in SOURCE_EXCLUSION_REASONS
        ):
            raise ValueError(f"invalid Crowd-Cast source exclusion: {exclusion!r}")
        video = exclusion["video_path"]
        keylog = exclusion["keylog_path"]
        video_sha = exclusion["video_sha256"]
        keylog_sha = exclusion["keylog_sha256"]
        reason = exclusion["reason"]
        if reason in {"missing_keylog", "noncanonical_video_name"}:
            expected_payloads = (True, False)
        elif reason in {"noncanonical_keylog_name", "orphan_keylog"}:
            expected_payloads = (False, True)
        else:
            expected_payloads = (True, True)
        if (
            (video is None) != (video_sha is None)
            or (keylog is None) != (keylog_sha is None)
            or (video is not None, keylog is not None) != expected_payloads
        ):
            raise ValueError(f"invalid Crowd-Cast excluded payload: {exclusion!r}")
        for payload, digest in ((video, video_sha), (keylog, keylog_sha)):
            if payload is not None and (
                not isinstance(payload, str)
                or not Path(payload).is_absolute()
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError(f"invalid Crowd-Cast excluded payload: {exclusion!r}")
        if video is not None and Path(video) in attested_videos:
            raise ValueError(f"duplicate Crowd-Cast video inventory path: {video}")
        if keylog is not None and Path(keylog) in attested_keylogs:
            raise ValueError(f"duplicate Crowd-Cast keylog inventory path: {keylog}")
        if video is not None:
            attested_videos.add(Path(video))
        if keylog is not None:
            attested_keylogs.add(Path(keylog))
        exclusion_counts[exclusion["reason"]] = (
            exclusion_counts.get(exclusion["reason"], 0) + 1
        )
    exclusion_receipt = (
        manifest["n_exclusions"],
        manifest["n_source_videos"],
        manifest["n_source_keylogs"],
        manifest["exclusion_counts"],
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in exclusion_receipt[:3]
        )
        or not isinstance(manifest["exclusion_counts"], dict)
        or any(
            not isinstance(reason, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in manifest["exclusion_counts"].items()
        )
        or exclusion_receipt
        != (
            len(exclusions),
            len(attested_videos),
            len(attested_keylogs),
            dict(sorted(exclusion_counts.items())),
        )
    ):
        raise ValueError(
            f"Crowd-Cast source exclusion counts mismatch: {manifest_path}"
        )
    return rows, exclusions, manifest, observed_sha


def resolve_source_clips_receipt(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, _, _, observed_sha = _resolve_source_clips_receipt(path)
    path = path.resolve()
    return rows, {
        "path": str(path),
        "sha256": observed_sha,
        "artifact_id": make_artifact_id(path.parent),
    }


def resolve_source_clips(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, exclusions, manifest, observed_sha = _resolve_source_clips_receipt(path)
    root = Path(manifest["source_root"])
    attested_videos: set[Path] = set()
    attested_keylogs: set[Path] = set()
    for row in rows:
        video = Path(row["video_path"])
        keylog = Path(row["keylog_path"])
        if (
            not video.is_file()
            or not keylog.is_file()
            or file_sha256_short(video, n=64) != row["video_sha256"]
            or file_sha256_short(keylog, n=64) != row["keylog_sha256"]
        ):
            raise ValueError(f"Crowd-Cast source payload digest mismatch: {row!r}")
        attested_videos.add(video)
        attested_keylogs.add(keylog)
    for exclusion in exclusions:
        for payload, digest in (
            (exclusion["video_path"], exclusion["video_sha256"]),
            (exclusion["keylog_path"], exclusion["keylog_sha256"]),
        ):
            if payload is not None:
                payload_path = Path(payload)
                if (
                    not payload_path.is_file()
                    or file_sha256_short(payload_path, n=64) != digest
                ):
                    raise ValueError(
                        f"invalid Crowd-Cast excluded payload: {exclusion!r}"
                    )
                if payload_path.suffix == ".mp4":
                    attested_videos.add(payload_path)
                else:
                    attested_keylogs.add(payload_path)
    observed_videos = set(root.glob("uploads/*/*/recordings/*.mp4"))
    observed_keylogs = set(root.glob("uploads/*/*/keylogs/*.msgpack"))
    if attested_videos != observed_videos or attested_keylogs != observed_keylogs:
        raise ValueError("Crowd-Cast source artifact does not close its raw inventory")
    path = path.resolve()
    return rows, {
        "path": str(path),
        "sha256": observed_sha,
        "artifact_id": make_artifact_id(path.parent),
    }
