"""The sampler's output, read as a ``SegmentView``.

Stage 03 has two readings of the master frame axis and stage 04 accepts either:

  * ``stage_03_filter``        — a keep/drop mask at master resolution, with fps
    decided downstream by ``lib/views.build_segment_view``.
  * ``stage_03_sample_frames`` — samples to a target fps up front and writes one
    ``frame_records.jsonl`` per segment.

This module is the second one's adapter. Stage 04 is written against exactly one
shape, ``SegmentView``, so the sampler's records are lifted into that rather than
the stage growing a second code path: every knob downstream of the view —
formatter, coalesce, app filter, goal projection — then behaves identically
whichever stage 03 produced the frames, and cannot drift apart.

What the sampler decided is not re-decided here. It already chose the ticks (at
its own ``--target-fps``, with NO_OP thinning and black-frame drops applied), so
``--fps`` is not a knob on this path and the view's ``fps``/``stride`` are
reported as what the sampler recorded, not recomputed. What IS re-derived is the
LABEL: the action format is a stage-04 flag, so the sampler's own ``action``
string is deliberately ignored and rebuilt from the realigned keylog through the
selected grammar's codec.

Windows tile from each kept tick to the next, the last running to the end of
master coverage — the same contiguous tiling ``lib/events._Locator`` requires,
and the same one the filter path builds. The gap before the first kept frame is a
``pre_first_frame`` dead zone: nothing was visible there, so its keystrokes must
not be folded into the first turn's label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.crowdcast.lib.common import read_jsonl
from pipeline.crowdcast.lib.events import DeadZone
from pipeline.crowdcast.lib.manifest import make_artifact_id
from pipeline.crowdcast.lib.views import SegmentView, ViewFrame

__all__ = ["SampleArtifact", "build_sample_view"]

#: Sampler statuses whose segment carries a usable frame_records.jsonl.
USABLE_STATUSES = frozenset({"ok", "cached"})


class SampleArtifact:
    """A ``stage_03_sample_frames --output-dir``, with the same surface as
    ``views.FilterArtifact`` so stage 04 can hold either."""

    def __init__(self, sample_dir: Path):
        self.dir = Path(sample_dir).resolve()
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no manifest.json under {self.dir} (not a sample artifact?)"
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.sample_id = make_artifact_id(self.dir)
        self.master_fps = float(self.manifest["master_fps"])
        self.master_store_id = str(self.manifest.get("master_store_id") or "")
        self.target_fps = float(self.manifest.get("target_fps") or 0.0)
        self.index_rows = read_jsonl(self.dir / "sample_index.jsonl")

    # `filter_id` is what stage 04 records and what a goals artifact is checked
    # against. A sample artifact has no filter, and naming its own id here keeps
    # the join auditable rather than recording a null.
    @property
    def filter_id(self) -> str:
        return self.sample_id

    def usable_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self.index_rows
            if str(row.get("status")) in USABLE_STATUSES
        ]

    def segment_path(self, segment_id: str) -> Path:
        """The segment's frame records. The index row names the path it wrote;
        the canonical layout is the fallback for an index written before that."""
        for row in self.index_rows:
            if str(row.get("segment_id")) == segment_id and row.get("frame_records"):
                return Path(str(row["frame_records"]))
        return self.dir / "clips" / segment_id / "stage_01" / "frame_records.jsonl"


def build_sample_view(index_row: dict[str, Any], records_path: Path) -> SegmentView:
    """One sampler segment -> the ``SegmentView`` stage 04 consumes."""
    records = read_jsonl(records_path)
    if not records:
        raise ValueError(f"empty frame records at {records_path}")

    ticks = [int(r["master_record_index"]) for r in records]
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError(
            f"{index_row.get('segment_id')}: master_record_index is not strictly "
            "increasing, so the label windows would overlap"
        )

    master_fps = float(index_row.get("master_fps") or 0.0)
    if master_fps <= 0:
        raise ValueError(
            f"{index_row.get('segment_id')}: no master_fps on the sample index row"
        )
    # The master axis length, which the sampler records as `n_master_records` and
    # uses for its OWN last window. Not `ticks[-1] + 1`: the sampler drops black
    # and thinned-NO_OP frames, so real video usually continues past the last kept
    # frame, and ending coverage there would push every action in that tail into
    # the trailing no_coverage zone — silently deleting it from the final turn
    # instead of labelling it. Falls back to the last tick only for an index
    # written before the field existed.
    axis_end = max(int(index_row.get("n_master_records") or 0), ticks[-1] + 1)

    frames = [
        ViewFrame(
            view_idx=i,
            slot=int(r.get("local_bin_idx", i)),
            master_idx=ticks[i],
            t_s=ticks[i] / master_fps,
            win_start=ticks[i],
            win_end=ticks[i + 1] if i + 1 < len(ticks) else axis_end,
            image=str(r["image_path"]),
        )
        for i, r in enumerate(records)
    ]

    dead_zones: list[DeadZone] = []
    if ticks[0] > 0:
        dead_zones.append(DeadZone(0, ticks[0], "pre_first_frame"))

    target_fps = float(index_row.get("target_fps") or 0.0)
    return SegmentView(
        segment_id=str(index_row["segment_id"]),
        recording_id=index_row.get("recording_id"),
        segment_idx=index_row.get("segment_idx"),
        master_fps=master_fps,
        fps=target_fps,
        fps_mode="sampled",
        stride=(master_fps / target_fps) if target_fps > 0 else 0.0,
        n_records=axis_end,
        n_slots=len(frames),
        n_masked_slots=0,
        frames=frames,
        dead_zones=dead_zones,
        keylog_path=index_row.get("keylog_path"),
        alignment_status=index_row.get("alignment_status"),
    )
