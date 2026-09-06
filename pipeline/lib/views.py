"""The shared frame selector: pick frames @fps within a stage-03 filter's
survivors and derive per-frame label windows + dead zones.

Used identically by stage 03b (annotation @k fps) and stage 04 (training
@x fps). Both rates must divide the master fps exactly.
The master axis is integer ticks == master record
indices; nothing here touches timestamps.

Selector semantics:
  * the integer stride master_fps/fps puts slot j at tick ``j * stride``;
    a masked tick yields no frame.
  * Window of selected frame i = ``[tick_i, tick_of_next_selected_frame)``;
    the last window runs to the end of master coverage. Windows never overlap.
    Idle-dropped spans are not dead zones: they are empty by definition and
    windows pass over them.
  * Dead zones = black spans + the span before the first selected frame
    (missing coverage past the axis end is derived implicitly by the label
    policy in lib/events.py).

Join safety: a view is built from a filter artifact; the filter's manifest
records the ``master_store_id`` it was built against, and this module verifies
both ids against the artifacts on disk, so a rebuilt master or filter fails
loudly instead of silently joining stale coordinates.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.lib.common import EVENT_EXCLUSION_REASONS, read_jsonl
from pipeline.lib.events import DeadZone, Window
from pipeline.lib.image_store import make_arrayrecord_image_uri
from pipeline.lib.manifest import check_artifact_id, file_sha256_short, make_artifact_id
from pipeline.lib.master_frames import resolve_master_artifact

_FPS_EPS = 1e-9


def resolve_stride(master_fps: float, fps: float) -> int:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    ratio = master_fps / fps
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > _FPS_EPS:
        raise ValueError(
            f"fps {fps} does not divide master_fps {master_fps}: stride {ratio:.6g} "
            "is not a positive integer"
        )
    return stride


@dataclass(frozen=True)
class ViewFrame:
    """One selected frame: master coordinates + its label-ownership window."""

    view_idx: int  # position within this segment's view (never persisted)
    slot: int  # ideal-slot index j (master_idx == round(j * stride))
    master_idx: int
    t_s: float  # master_idx / master_fps, for humans
    win_start: int  # label window [win_start, win_end) in master ticks
    win_end: int
    image: str  # ar:// URI into the master store


@dataclass
class SegmentView:
    segment_id: str
    recording_id: str
    segment_idx: int
    master_fps: float
    fps: float
    stride: int
    n_records: int  # master axis length (ticks)
    n_slots: int
    n_masked_slots: int
    frames: list[ViewFrame]
    dead_zones: list[DeadZone]
    keylog_path: str
    alignment_status: str

    def windows(self) -> list[Window]:
        return [Window(f.win_start, f.win_end) for f in self.frames]


class _KeptRanges:
    """Membership test over the filter's kept ``[start, end)`` ranges."""

    def __init__(self, ranges: list[list[int]]):
        self.ranges = ranges
        self._starts = [r[0] for r in ranges]

    def __contains__(self, tick: int) -> bool:
        i = bisect_right(self._starts, tick) - 1
        return i >= 0 and self.ranges[i][0] <= tick < self.ranges[i][1]


def build_segment_view(filter_seg: dict[str, Any], *, fps: float) -> SegmentView:
    """Pure selector over one segment's filter file (a picklable dict)."""
    master_fps = float(filter_seg["master_fps"])
    stride = resolve_stride(master_fps, fps)
    n_records = int(filter_seg["n_master_records"])
    kept = _KeptRanges(filter_seg["kept_ranges"])
    # Every frame's image is an ar:// URI into this shard. Without it a view
    # frame has no observation, and `str(None)` would silently reach a training
    # conversation as the image reference "None".
    shard_path = filter_seg["shard_path"]

    # A masked tick means the slot yields no frame.
    selected: list[tuple[int, int]] = []  # (slot, tick)
    n_slots = 0
    for slot in range(int(n_records / stride) + 1):
        tick = int(slot * stride + 0.5)
        if tick >= n_records:
            break
        n_slots += 1
        if tick in kept:
            selected.append((slot, tick))

    frames: list[ViewFrame] = []
    for view_idx, (slot, tick) in enumerate(selected):
        next_tick = (
            selected[view_idx + 1][1] if view_idx + 1 < len(selected) else n_records
        )
        frames.append(
            ViewFrame(
                view_idx=view_idx,
                slot=slot,
                master_idx=tick,
                t_s=tick / master_fps,
                win_start=tick,
                win_end=next_tick,
                image=make_arrayrecord_image_uri(shard_path, tick),
            )
        )

    # Dead zones: black spans (clipped to the visible region — everything
    # before the first selected frame is one pre_first_frame zone regardless
    # of why those ticks are unusable). idle_interior drops are NOT zones.
    first_tick = frames[0].master_idx if frames else n_records
    dead_zones: list[DeadZone] = []
    if first_tick > 0:
        dead_zones.append(DeadZone(0, first_tick, "pre_first_frame"))
    for span in filter_seg["dropped"]:
        reason = span["reason"]
        if reason not in {"black", "idle_interior"}:
            raise ValueError(f"unexpected dropped-span reason: {reason!r}")
        if reason == "idle_interior":
            continue
        start = max(int(span["start"]), first_tick)
        end = min(int(span["end"]), n_records)
        if start < end:
            dead_zones.append(DeadZone(start, end, "black"))

    return SegmentView(
        segment_id=str(filter_seg["segment_id"]),
        recording_id=filter_seg["recording_id"],
        segment_idx=filter_seg["segment_idx"],
        master_fps=master_fps,
        fps=fps,
        stride=stride,
        n_records=n_records,
        n_slots=n_slots,
        n_masked_slots=n_slots - len(frames),
        frames=frames,
        dead_zones=dead_zones,
        keylog_path=filter_seg["keylog_path"],
        alignment_status=filter_seg["alignment_status"],
    )


class FilterArtifact:
    """A verified stage-03 filter output."""

    def __init__(self, filter_dir: Path):
        self.dir = Path(filter_dir).resolve()
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no manifest.json under {self.dir} (not a filter artifact?)"
            )
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("artifact_type") != "realigned_filter_mask":
            raise ValueError(
                f"{self.dir} is not a stage-03 filter artifact "
                f"(artifact_type={self.manifest.get('artifact_type')!r})"
            )
        self.filter_id = make_artifact_id(self.dir)
        self.master_fps = float(self.manifest["master_fps"])
        self.master_store_id = str(self.manifest["master_store_id"])
        self.master_dir = check_artifact_id(self.master_store_id, what="master store")
        self._master_manifest, master_rows = resolve_master_artifact(self.master_dir)
        self._master_index = {str(item["segment_id"]): item for item in master_rows}
        self.source_clips_dir = check_artifact_id(
            str(self.manifest["source_clips_id"]), what="realigned clips"
        )
        index_path = self.dir / "filter_index.jsonl"
        if file_sha256_short(index_path, n=64) != self.manifest.get(
            "filter_index_sha256"
        ):
            raise ValueError(f"filter index digest mismatch: {index_path}")
        self.index_rows = read_jsonl(index_path)
        if not self.index_rows:
            raise ValueError(f"filter artifact contains no segments: {self.dir}")
        self._index = {str(row["segment_id"]): row for row in self.index_rows}
        if len(self._index) != len(self.index_rows):
            raise ValueError(f"filter artifact contains duplicate segments: {self.dir}")

    def usable_rows(self) -> list[dict[str, Any]]:
        for row in self.index_rows:
            status = row.get("status")
            if status == "ok":
                if (
                    not isinstance(row.get("filter_path"), str)
                    or not isinstance(row.get("filter_sha256"), str)
                    or "exclusion_reason" in row
                ):
                    raise ValueError(
                        f"filter artifact contains incomplete segment: {row!r}"
                    )
                continue
            if (
                status != "excluded_invalid_keylog"
                or row.get("exclusion_reason") not in EVENT_EXCLUSION_REASONS
                or not isinstance(row.get("keylog_path"), str)
                or not isinstance(row.get("keylog_sha256"), str)
                or row.get("filter_path") is not None
                or row.get("filter_sha256") is not None
                or row.get("n_kept") != 0
                or row.get("n_black") != 0
                or row.get("n_idle_interior") != 0
            ):
                raise ValueError(
                    f"filter artifact contains incomplete segment: {row!r}"
                )
        status_counts = Counter(row["status"] for row in self.index_rows)
        exclusion_counts = Counter(
            row["exclusion_reason"]
            for row in self.index_rows
            if row["status"] == "excluded_invalid_keylog"
        )
        if (
            self.manifest["n_segments"] != len(self.index_rows)
            or self.manifest["n_accepted_segments"] != status_counts["ok"]
            or self.manifest["n_excluded_segments"]
            != status_counts["excluded_invalid_keylog"]
            or self.manifest["status_counts"] != dict(sorted(status_counts.items()))
            or self.manifest["exclusion_counts"]
            != dict(sorted(exclusion_counts.items()))
        ):
            raise ValueError("filter manifest/index summary mismatch")
        rows = [row for row in self.index_rows if row.get("status") == "ok"]
        return rows

    def stride_for(self, fps: float) -> int:
        return resolve_stride(self.master_fps, fps)

    def segment_path(self, segment_id: str) -> Path:
        return self.dir / "filter" / f"{segment_id}.json"

    def load_segment(self, segment_id: str) -> dict[str, Any]:
        master_row = self._master_index[segment_id]
        row = self._index[segment_id]
        if row["status"] != "ok":
            raise ValueError(f"filter segment is not usable: {segment_id}")
        path = self.segment_path(segment_id)
        if Path(row["filter_path"]).resolve() != path.resolve():
            raise ValueError(f"filter index path mismatch for {segment_id}")
        if file_sha256_short(path, n=64) != row.get("filter_sha256"):
            raise ValueError(f"filter digest mismatch: {path}")
        segment = json.loads(path.read_text())
        if (
            segment["segment_id"] != segment_id
            or segment["recording_id"] != master_row["recording_id"]
            or segment["segment_idx"] != master_row["segment_idx"]
            or segment["master_fps"] != master_row["master_fps"]
            or segment["n_master_records"] != master_row["num_records"]
            or Path(segment["shard_path"]).resolve()
            != Path(master_row["shard_path"]).resolve()
            or segment["shard_sha256"] != master_row["shard_sha256"]
            or segment["frame_manifest_sha256"] != master_row["frame_manifest_sha256"]
        ):
            raise ValueError(f"filter/master contract mismatch for {segment_id}")
        keylog = Path(segment["keylog_path"])
        if file_sha256_short(keylog, n=64) != segment.get("keylog_sha256"):
            raise ValueError(f"filter keylog digest mismatch: {keylog}")
        shard = Path(segment["shard_path"])
        if file_sha256_short(shard, n=64) != segment.get("shard_sha256"):
            raise ValueError(f"filter master shard digest mismatch: {shard}")
        frame_manifest = shard.parent / "frame_manifest.jsonl"
        if file_sha256_short(frame_manifest, n=64) != segment.get(
            "frame_manifest_sha256"
        ):
            raise ValueError(f"filter frame manifest digest mismatch: {frame_manifest}")
        return segment

    def segment_view(self, segment_id: str, fps: float) -> SegmentView:
        return build_segment_view(self.load_segment(segment_id), fps=fps)
