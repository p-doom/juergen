"""Semantic-event action projection for sequential goal-memory data.

The legacy Stage 04 recipe bins actions into fixed-fps observation windows.
This module instead derives stable decision packets from the full Stage 02
event stream, after applying the Stage 03 label/dead-zone policy at master
resolution.  It never creates a computer action: every emitted packet records
the exact raw-event ids it abstracts, and every non-emitted event receives an
explicit disposition.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from realigned_pipeline.lib.action_format import (
    ActionPrimitive,
    ComputerUseRelNormFormatter,
    _collapse_typing,
    render_tool_call,
)
from realigned_pipeline.lib.events import LabeledEvent, apply_label_policy, load_events
from realigned_pipeline.lib.views import build_segment_view

ACTION_SPEC = "computer_use_rel_norm_v1"
_BUTTONS = {"LMB", "RMB", "MMB"}
_SUBMIT_KEYS = {"Return", "Enter"}
_MODIFIERS = {
    "Control", "ControlLeft", "ControlRight", "Alt", "AltLeft", "AltRight",
    "Meta", "MetaLeft", "MetaRight",
}


def stable_id(prefix: str, value: Any, n: int = 20) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:n]}"


def render_calls(calls: Sequence[dict[str, Any]]) -> str:
    return "\n".join(render_tool_call(call) for call in calls)


def _owned(labeled: Sequence[LabeledEvent]) -> list[LabeledEvent]:
    return sorted(
        (row for row in labeled if row.window is not None),
        key=lambda row: (row.label_t, row.event.seq),
    )


def _event_family(row: LabeledEvent) -> str:
    event = row.event
    if event.kind == "move":
        return "move"
    if event.kind == "scroll":
        return "scroll"
    if event.name in _BUTTONS:
        return "button"
    return "key"


def _packetize(events: Sequence[LabeledEvent], gap_s: float) -> list[list[LabeledEvent]]:
    """Split on genuine gaps and action-family boundaries, preserving order.

    Mouse motion immediately followed by a button transition stays together so
    move+click and drag remain one decision. Printable typing remains together;
    Enter and completed clicks close the current decision.
    """
    packets: list[list[LabeledEvent]] = []
    current: list[LabeledEvent] = []
    close_before_next = False
    held_mods: set[str] = set()
    for row in events:
        family = _event_family(row)
        previous_family = _event_family(current[-1]) if current else None
        gap = row.label_t - current[-1].label_t if current else 0.0
        family_break = bool(current) and (
            (previous_family == "key" and family != "key")
            or (previous_family == "scroll" and family != "scroll")
            or (previous_family == "move" and family not in ("move", "button"))
            or (previous_family == "button" and family not in ("button", "move"))
        )
        if current and (close_before_next or gap >= gap_s or family_break):
            packets.append(current)
            current = []
            held_mods.clear()
        close_before_next = False
        current.append(row)
        event = row.event
        if family == "key" and event.name in _MODIFIERS:
            if event.kind == "press":
                held_mods.add(str(event.name))
            else:
                held_mods.discard(str(event.name))
        if (family == "button" and event.kind == "release") or (
            family == "key" and event.kind == "release"
            and (event.name in _SUBMIT_KEYS or (event.name in _MODIFIERS and not held_mods))
        ):
            close_before_next = True
    if current:
        packets.append(current)
    return packets


def _primitives(packet: Sequence[LabeledEvent], width: int, height: int) -> list[ActionPrimitive]:
    """Raw packet -> ordered primitives; motor bursts coalesce once."""
    out: list[ActionPrimitive] = []
    i = 0
    while i < len(packet):
        row = packet[i]
        event = row.event
        if event.kind in ("move", "scroll"):
            kind = event.kind
            dx = dy = 0.0
            t_s = row.label_t
            while i < len(packet) and packet[i].event.kind == kind:
                dx += packet[i].event.dx
                dy += packet[i].event.dy
                i += 1
            if kind == "move":
                # The public action contract is a screen-fraction scale, not
                # unbounded device units.  A malformed/corrupt burst can run
                # beyond one screen, so clamp each endpoint to the declared
                # range while retaining its direction.
                rdx = max(-1000, min(1000, round(dx / width * 1000)))
                rdy = max(-1000, min(1000, round(dy / height * 1000)))
            else:
                rdx, rdy = round(dx), round(dy)
            if rdx or rdy:
                out.append(ActionPrimitive(kind=kind, dx=rdx, dy=rdy, t_s=t_s, owner=0))
            continue
        out.append(ActionPrimitive(
            kind="down" if event.kind == "press" else "up",
            input_name=event.name, t_s=row.label_t, owner=0,
        ))
        i += 1
    return _collapse_typing(out, set(), max_gap_s=1.0)


def semantic_events_from_labeled(
    labeled: Sequence[LabeledEvent], *, segment_id: str, recording_id: str | None,
    frames: Sequence[Any], frame_size: tuple[int, int], day_offset_s: float = 0.0,
    decision_gap_s: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid capture dimensions {frame_size!r}")
    raw_packets = _packetize(_owned(labeled), decision_gap_s)

    # Multiple sub-decisions in one master tick have no intervening screenshot;
    # merge them so each semantic event has a truthful current observation.
    packets: list[list[LabeledEvent]] = []
    for packet in raw_packets:
        anchor = int(frames[int(packet[0].window)].master_idx)
        if packets and int(frames[int(packets[-1][0].window)].master_idx) == anchor:
            packets[-1].extend(packet)
        else:
            packets.append(list(packet))

    formatter = ComputerUseRelNormFormatter()
    semantic: list[dict[str, Any]] = []
    seq_owner: dict[int, str] = {}
    for packet in packets:
        prims = _primitives(packet, width, height)
        counts: Counter = Counter()
        calls = formatter._window_tool_calls(prims, counts)
        if not calls:
            continue
        anchor_frame = frames[int(packet[0].window)]
        seqs = [row.event.seq for row in packet]
        event_id = stable_id("sem", {
            "segment_id": segment_id,
            "anchor_master_idx": int(anchor_frame.master_idx),
            "raw_event_seqs": seqs,
        })
        for seq in seqs:
            seq_owner[seq] = event_id
        semantic.append({
            "semantic_event_id": event_id,
            "segment_id": segment_id,
            "recording_id": recording_id,
            "anchor_master_idx": int(anchor_frame.master_idx),
            "end_master_idx": max(int(frames[int(row.window)].master_idx) for row in packet),
            "image": str(anchor_frame.image),
            "t_segment_s": float(packet[0].label_t),
            "t_day_s": day_offset_s + float(packet[0].label_t),
            "raw_event_seqs": seqs,
            "raw_event_ids": [f"{segment_id}:r{seq}" for seq in seqs],
            "tool_calls": calls,
            "assistant_action": render_calls(calls),
            "action_spec": ACTION_SPEC,
            "capture_size": [width, height],
        })

    dispositions: list[dict[str, Any]] = []
    for row in labeled:
        event = row.event
        owner = seq_owner.get(event.seq)
        if owner:
            disposition = "emitted"
        elif row.window is None:
            disposition = row.discard_reason or "filtered"
        else:
            disposition = "normalized_dead_zone"
        dispositions.append({
            "raw_event_id": f"{segment_id}:r{event.seq}",
            "raw_event_seq": event.seq,
            "kind": event.kind,
            "t_s": event.t_s,
            "semantic_event_id": owner,
            "disposition": disposition,
            "clamped": row.clamped,
        })
    return semantic, dispositions


def build_segment_semantic_events(
    filter_segment: dict[str, Any], *, source_row: dict[str, Any],
    day_offset_s: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a segment from Stage 02 keylogs and the Stage 03 mask only."""
    segment_id = str(filter_segment["segment_id"])
    keylog = filter_segment.get("keylog_path")
    events, _stats = load_events(Path(keylog)) if keylog else ([], None)
    width, height = source_row.get("video_width"), source_row.get("video_height")
    if not (isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0):
        dispositions = [{
            "raw_event_id": f"{segment_id}:r{event.seq}", "raw_event_seq": event.seq,
            "kind": event.kind, "t_s": event.t_s, "semantic_event_id": None,
            "disposition": "missing_capture_dimensions", "clamped": None,
        } for event in events]
        return [], dispositions, {"capture_size": None, "n_raw_events": len(events)}
    view = build_segment_view(filter_segment, fps=float(filter_segment["master_fps"]))
    if not view.frames:
        dispositions = [{
            "raw_event_id": f"{segment_id}:r{event.seq}", "raw_event_seq": event.seq,
            "kind": event.kind, "t_s": event.t_s, "semantic_event_id": None,
            "disposition": "empty_filtered_view", "clamped": None,
        } for event in events]
        return [], dispositions, {"capture_size": [width, height], "n_raw_events": len(events)}
    labeled, counters = apply_label_policy(
        events, view.windows(), view.dead_zones, master_fps=view.master_fps,
    )
    semantic, dispositions = semantic_events_from_labeled(
        labeled, segment_id=segment_id, recording_id=view.recording_id,
        frames=view.frames, frame_size=(width, height), day_offset_s=day_offset_s,
    )
    return semantic, dispositions, {
        "capture_size": [width, height], "n_raw_events": len(events),
        "policy_counters": asdict(counters),
    }
