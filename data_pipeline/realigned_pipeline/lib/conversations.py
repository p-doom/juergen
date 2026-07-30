"""Shared conversation-builder plumbing for stage 04 (both --mode action and
--mode thinking): the chat.jsonl content-block schema, the four-file artifact
writer, day-index selection, and the window↔clip-stride alignment guard.

Mode-specific windowing/conditioning lives in ``stage_04_conversations.py``;
everything here is format-agnostic and used identically by both modes so the
merge stays a single clean surface, not a fork.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.days import DEFAULT_TZ, build_day_index
from realigned_pipeline.lib.common import ensure_dir, write_json, write_jsonl
from realigned_pipeline.lib.views import FilterArtifact

# Annotation memory/goals sidecars are emitted at a FIXED stride of this many
# sampled frames per clip (day_idx granularity): goals_active.jsonl /
# memory/<day>.jsonl rows tile each chunk in day_idx_range steps of this size
# (a short final clip per chunk). Thinking-mode windows must be a positive
# multiple of it so every window boundary coincides with a clip boundary, which
# is what makes the leak-free "So far:" selection exact.
CLIP_STRIDE = 15


# ---------------------------------------------------------------------------
# chat.jsonl content blocks (identical shape in both modes and both old scripts)
# ---------------------------------------------------------------------------


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


# ---------------------------------------------------------------------------
# Window/clip alignment guard (thinking mode)
# ---------------------------------------------------------------------------


def require_window_alignment(window_frames: int, *, stride: int = CLIP_STRIDE) -> None:
    """Enforce that a thinking-mode window is a positive multiple of the
    annotation clip stride, so window edges land on clip edges (and only then is
    the ``day_idx_range END == win_start-1`` memory predecessor well defined)."""
    if not isinstance(window_frames, int) or window_frames <= 0:
        raise SystemExit(
            f"--window-frames must be a positive integer, got {window_frames!r}")
    if window_frames % stride != 0:
        raise SystemExit(
            f"--window-frames {window_frames} is not a multiple of the annotation "
            f"clip stride {stride}: windows would not tile onto clip boundaries, so "
            "the leak-free 'So far:' memory predecessor is undefined. Pass a positive "
            f"multiple of {stride} (e.g. {stride}, {2 * stride}, {4 * stride})."
        )


# ---------------------------------------------------------------------------
# Day-index selection (day grouping via mvhd + stage-00/02 clips manifest)
# ---------------------------------------------------------------------------


def check_day_selection_args(day_filter: list[str] | None,
                             day_exclude: list[str] | None) -> None:
    if day_filter and day_exclude:
        raise SystemExit("--day-filter and --day-exclude are mutually exclusive")


def load_or_build_day_index(
    art: FilterArtifact,
    clips_manifest: Path,
    *,
    day_index_cache: Path | None,
    tz: str = DEFAULT_TZ,
) -> list[dict[str, Any]]:
    """Day rows for the filter artifact, reusing ``day_index_cache`` when it was
    built for the same filter_id + tz (the mvhd probe is ~minutes). Writes the
    cache on a miss so repeat runs are cheap."""
    if day_index_cache is not None and day_index_cache.is_file():
        doc = json.loads(day_index_cache.read_text())
        if doc.get("filter_id") == art.filter_id and doc.get("tz") == tz:
            return doc["days"]
    day_rows, counters = build_day_index(art, clips_manifest, tz=tz)
    print(f"[conversations] day index: {counters}", flush=True)
    if day_index_cache is not None:
        write_json(day_index_cache, {
            "filter_id": art.filter_id, "tz": tz,
            "clips_manifest": str(clips_manifest), "counters": counters,
            "days": day_rows,
        })
    return day_rows


def select_day_rows(
    day_rows: list[dict[str, Any]],
    *,
    day_filter: list[str] | None = None,
    day_exclude: list[str] | None = None,
    restrict_to: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply the shared day include/exclude selection. ``restrict_to`` is a
    mode-specific pre-filter (e.g. thinking goal mode keeps only days that have
    a goals_active sidecar)."""
    wanted = {d["day_tag"] for d in day_rows}
    if restrict_to is not None:
        wanted &= restrict_to
    if day_filter:
        wanted &= set(day_filter)
    if day_exclude:
        wanted -= set(day_exclude)
    return [d for d in day_rows if d["day_tag"] in wanted]


# ---------------------------------------------------------------------------
# Artifact writer (conversations.jsonl / chat.jsonl / summary / manifest)
# ---------------------------------------------------------------------------

CONVERSATIONS_ARTIFACT_TYPE = "juergen_annotation_conversations"
CONVERSATIONS_SCHEMA_VERSION = 2


def write_conversation_artifact(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    master_store_id: str,
    filter_id: str,
    goals_id: str | None = None,
) -> Path:
    """Write the canonical stage-04 output: conversations.jsonl (one row per
    conversation), chat.jsonl (byte-identical drop-in source for stages 05/06),
    conversations_summary.json, and manifest.json (summary + join guards). Rows
    are sorted by conversation_id for a stable, diffable artifact."""
    records.sort(key=lambda r: str(r["conversation_id"]))
    out_dir = ensure_dir(output_dir)
    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": CONVERSATIONS_ARTIFACT_TYPE,
        "schema_version": CONVERSATIONS_SCHEMA_VERSION,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",  # split-agnostic drop-in source_path for stages 05/06
        "master_store_id": master_store_id,
        "filter_id": filter_id,
        "goals_id": goals_id,
        **summary,
    })
    return out_dir
