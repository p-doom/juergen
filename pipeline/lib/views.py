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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.lib.common import read_jsonl
from pipeline.lib.events import DeadZone, Window
from pipeline.lib.image_store import make_arrayrecord_image_uri
from pipeline.lib.manifest import check_artifact_id, make_artifact_id

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
    recording_id: str | None
    segment_idx: int | None
    master_fps: float
    fps: float
    stride: int
    n_records: int  # master axis length (ticks)
    n_slots: int
    n_masked_slots: int
    frames: list[ViewFrame]
    dead_zones: list[DeadZone]
    keylog_path: str | None
    alignment_status: str | None

    def windows(self) -> list[Window]:
        return [Window(f.master_idx, f.win_start, f.win_end) for f in self.frames]


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
    for span in filter_seg.get("dropped", []):
        if span["reason"] != "black":
            continue
        start = max(int(span["start"]), first_tick)
        end = min(int(span["end"]), n_records)
        if start < end:
            dead_zones.append(DeadZone(start, end, "black"))

    return SegmentView(
        segment_id=str(filter_seg["segment_id"]),
        recording_id=filter_seg.get("recording_id"),
        segment_idx=filter_seg.get("segment_idx"),
        master_fps=master_fps,
        fps=fps,
        stride=stride,
        n_records=n_records,
        n_slots=n_slots,
        n_masked_slots=n_slots - len(frames),
        frames=frames,
        dead_zones=dead_zones,
        keylog_path=filter_seg.get("keylog_path"),
        alignment_status=filter_seg.get("alignment_status"),
    )


class FilterArtifact:
    """A stage-03 filter output: manifest + per-segment filter files.

    Construction verifies (a) the filter's own manifest exists (yielding
    ``filter_id``) and (b) the ``master_store_id`` it recorded still matches
    the master store on disk."""

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
        self.index_rows = read_jsonl(self.dir / "filter_index.jsonl")

    def usable_rows(self) -> list[dict[str, Any]]:
        rows = [row for row in self.index_rows if row.get("status") == "ok"]
        if len(rows) != len(self.index_rows):
            raise ValueError(
                f"filter artifact contains incomplete segments: {self.dir}"
            )
        return rows

    def stride_for(self, fps: float) -> int:
        return resolve_stride(self.master_fps, fps)

    def segment_path(self, segment_id: str) -> Path:
        return self.dir / "filter" / f"{segment_id}.json"

    def load_segment(self, segment_id: str) -> dict[str, Any]:
        return json.loads(self.segment_path(segment_id).read_text())

    def segment_view(self, segment_id: str, fps: float) -> SegmentView:
        return build_segment_view(self.load_segment(segment_id), fps=fps)
