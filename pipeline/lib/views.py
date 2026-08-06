"""The shared frame selector: pick frames @fps within a stage-03 filter's
survivors and derive per-frame label windows + dead zones.

Used identically by stage 03b (annotation @k fps) and stage 04 (training
@x fps) — the two rates are independent, each bounded only by the master fps
(integer stride in ``exact`` mode; any rate <= master in ``nearest`` mode).
The master axis is integer ticks == master record
indices; nothing here touches timestamps.

Selector semantics (load-bearing):
  * two fps modes (``FPS_MODES``): ``exact`` (default) requires an integer
    stride master_fps/fps and puts slot j at tick ``j * stride``; ``nearest``
    (opt-in) allows any fps <= master_fps and puts slot j at the tick nearest
    its ideal time (spacing jitters by up to half a master tick). In BOTH
    modes a masked tick means the slot yields NO frame — skip, never
    substitute a neighbor.
  * Window of selected frame i = ``[tick_i, tick_of_next_selected_frame)``;
    the last window runs to the end of master coverage. Windows never overlap.
    Idle-dropped spans are NOT dead zones: they are empty by definition and
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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.lib.common import read_jsonl
from pipeline.lib.events import DeadZone, Window
from pipeline.lib.image_store import make_arrayrecord_image_uri
from pipeline.lib.manifest import check_artifact_id, make_artifact_id

_FPS_EPS = 1e-9

# Filter-index statuses that carry a usable per-segment filter file.
USABLE_FILTER_STATUSES = {"ok", "cached"}


# Frame-selection modes:
#   exact   — fps must divide master_fps (integer stride); slots land ON ticks,
#             perfectly even spacing. The loud default.
#   nearest — any fps <= master_fps; slot j sits at the master tick NEAREST its
#             ideal time j/fps, so spacing jitters by up to half a master tick
#             (e.g. 4 fps on a 15 fps master picks ticks 0,4,8,11,15,...).
#             Causally correct: a frame's label window starts AT its actual tick
#             ([tick_i, tick_{i+1})), so no action ever precedes its observation
#             frame. A nearest-pick would smear up to half a tick.
FPS_MODES = ("exact", "nearest")


def resolve_stride(master_fps: float, fps: float, mode: str = "exact") -> float:
    """Tick stride (master ticks per selected frame) for sampling at ``fps``.

    ``exact``: raises for fps > master or non-integer ratios (5 fps on a 4 fps
    master has no exact tick alignment and must fail loudly); returns an int.
    ``nearest``: any fps <= master_fps is valid; returns the float ratio."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if mode not in FPS_MODES:
        raise ValueError(f"fps mode must be one of {FPS_MODES}, got {mode!r}")
    ratio = master_fps / fps
    if mode == "nearest":
        if ratio < 1 - _FPS_EPS:
            raise ValueError(f"fps {fps} exceeds master_fps {master_fps} (can only sample DOWN)")
        return ratio
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > _FPS_EPS:
        raise ValueError(
            f"fps {fps} does not divide master_fps {master_fps}: stride {ratio:.6g} "
            "is not a positive integer (pick an fps with integer master_fps/fps, "
            "or opt into fps_mode='nearest' for jittered spacing)"
        )
    return stride


@dataclass(frozen=True)
class ViewFrame:
    """One selected frame: master coordinates + its label-ownership window."""

    view_idx: int  # position within this segment's view (NEVER persisted)
    slot: int  # ideal-slot index j (master_idx == round(j * stride))
    master_idx: int
    t_s: float  # master_idx / master_fps, for humans
    win_start: int  # label window [win_start, win_end) in master ticks
    win_end: int
    image: str | None  # ar:// URI into the master store


@dataclass
class SegmentView:
    segment_id: str
    recording_id: str | None
    segment_idx: int | None
    master_fps: float
    fps: float
    fps_mode: str
    stride: float  # master ticks per selected frame (int in exact mode)
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


def build_segment_view(
    filter_seg: dict[str, Any], *, fps: float, fps_mode: str = "exact"
) -> SegmentView:
    """Pure selector over one segment's filter file (a picklable dict)."""
    master_fps = float(filter_seg["master_fps"])
    stride = resolve_stride(master_fps, fps, fps_mode)
    n_records = int(filter_seg["n_master_records"])
    kept = _KeptRanges(filter_seg["kept_ranges"])
    shard_path = filter_seg.get("shard_path")

    # Slot j's tick: j*stride exactly, or the nearest tick to the ideal time
    # (half-up, deterministic). stride >= 1 keeps ticks strictly increasing.
    # Either way a masked tick means the slot yields NO frame — never a
    # neighbor.
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
        next_tick = selected[view_idx + 1][1] if view_idx + 1 < len(selected) else n_records
        frames.append(
            ViewFrame(
                view_idx=view_idx,
                slot=slot,
                master_idx=tick,
                t_s=tick / master_fps,
                win_start=tick,
                win_end=next_tick,
                image=make_arrayrecord_image_uri(shard_path, tick) if shard_path else None,
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
        fps_mode=fps_mode,
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
            raise FileNotFoundError(f"no manifest.json under {self.dir} (not a filter artifact?)")
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
        return [r for r in self.index_rows if r.get("status") in USABLE_FILTER_STATUSES]

    def stride_for(self, fps: float, fps_mode: str = "exact") -> float:
        return resolve_stride(self.master_fps, fps, fps_mode)

    def segment_path(self, segment_id: str) -> Path:
        return self.dir / "filter" / f"{segment_id}.json"

    def load_segment(self, segment_id: str) -> dict[str, Any]:
        return json.loads(self.segment_path(segment_id).read_text())

    def segment_view(self, segment_id: str, fps: float, fps_mode: str = "exact") -> SegmentView:
        return build_segment_view(self.load_segment(segment_id), fps=fps, fps_mode=fps_mode)

    def iter_views(
        self, fps: float, *, fps_mode: str = "exact", limit: int | None = None
    ) -> Iterator[SegmentView]:
        self.stride_for(fps, fps_mode)  # validate once up front
        rows = self.usable_rows()
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            yield self.segment_view(str(row["segment_id"]), fps, fps_mode)
