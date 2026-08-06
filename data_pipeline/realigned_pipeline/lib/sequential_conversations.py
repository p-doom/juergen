"""Stage-04 packer for the sequential goal-memory annotation artifact.

Records are a SIMULATION of the eval runtime's context manager over one
annotated day rather than a fixed tiling of it: chronological turns accumulate
until the screenshot-capacity trigger fires (``lib/sequential_packing`` mirrors
``eval/osworld_runtime``'s ``ScreenshotCheckpointController``), the record ends
with the CHECKPOINT CONTROL user turn plus the annotated seven-field checkpoint,
and the continuation record re-opens with the same GOAL, the byte-identical
checkpoint text, and the same boundary screenshot. The model therefore trains on
the compaction cadence it meets at inference.

Explicit vs proactive conditioning is a per-segment RENDERING choice (hindsight
goal relabeling) drawn from the goal levels that actually cover the segment, not
a property of the annotation: there is no scheduled provenance mix, and a short
(motor) goal is never rendered as ``GOAL:``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable

from realigned_pipeline.lib.conversations import image_block, text_block
from realigned_pipeline.lib.sequential_goal_memory_contract import (
    ACTION_SPEC,
    CHECKPOINT_CONTROL_REQUEST,
    PROACTIVE_GOAL_TEXT,
    RECIPE,
    RESUME_UPWEIGHT_TURNS,
    goal_conditioning,
)
from realigned_pipeline.lib.sequential_packing import (
    DEFAULT_MODE_WEIGHTS,
    PackingConfig,
    boundary_events,
    eligible_modes,
    packing_config_hash,
    sample_mode,
    segments_from_boundaries,
)

# Goal level rendered by each explicit mode. ``proactive`` renders
# PROACTIVE_GOAL_TEXT and carries no goal_id/provenance.
_MODE_LEVEL = {"explicit_mid": "mid", "explicit_long": "long"}


@dataclass(frozen=True)
class _Day:
    """One annotated day document plus the by-id indexes the packer reads."""

    doc: dict[str, Any]
    events: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    thoughts: dict[str, str]
    memory_ids: dict[str, str]
    checkpoints: dict[str, dict[str, Any]]

    @property
    def day_tag(self) -> str:
        return str(self.doc["day_tag"])


def _index_day(day: dict[str, Any], *, expected_hash: str) -> _Day:
    """Validate one day document and index it by semantic-event id.

    The packer addresses events positionally, so the artifact's own
    ``day_event_index`` must agree with the stream order it publishes.
    """
    events = list(day["semantic_events"])
    ids = [str(row["semantic_event_id"]) for row in events]
    for position, event in enumerate(events):
        if int(event["day_event_index"]) != position:
            raise ValueError(
                f"{day['day_tag']}: semantic event {ids[position]!r} carries "
                f"day_event_index {event['day_event_index']!r} at stream position {position}"
            )
    snapshots = list(day.get("memory_snapshots") or [])
    if {str(row["anchor_semantic_event_id"]) for row in snapshots} != set(ids):
        raise ValueError("Stage 04 requires one rolling-memory snapshot per semantic event")
    checkpoints: dict[str, dict[str, Any]] = {}
    for row in day.get("checkpoints") or []:
        if str(row.get("packing_config_hash") or "") != expected_hash:
            continue
        anchor = str(row["anchor_semantic_event_id"])
        if anchor in checkpoints:
            raise ValueError(
                f"{day['day_tag']}: two checkpoints at anchor {anchor!r} carry "
                f"packing_config_hash {expected_hash}"
            )
        checkpoints[anchor] = row
    return _Day(
        doc=day, events=events, nodes=list(day["goal_nodes"]),
        thoughts={str(row["anchor_semantic_event_id"]): str(row.get("thought") or "").strip()
                  for row in day["decisions"]},
        memory_ids={str(row["anchor_semantic_event_id"]): str(row["memory_snapshot_id"])
                    for row in snapshots},
        checkpoints=checkpoints,
    )


def _covers(node: dict[str, Any], span: tuple[int, int]) -> bool:
    return (int(node["start_event_index"]) <= span[0]
            and span[1] <= int(node["end_event_index"]))


def _covering_node(nodes: list[dict[str, Any]], level: str,
                   span: tuple[int, int]) -> dict[str, Any] | None:
    """The narrowest single node of ``level`` spanning the whole action span.

    A well-formed tree has non-overlapping siblings per level, so this is
    normally the only candidate; narrowest-then-goal_id keeps the choice
    deterministic if an artifact ever nests two.
    """
    covering = [node for node in nodes
                if str(node["level"]) == level and _covers(node, span)]
    if not covering:
        return None
    return min(covering, key=lambda node: (int(node["end_event_index"])
                                           - int(node["start_event_index"]),
                                           str(node["goal_id"])))


def _checkpoint(day: _Day, index: int, *, cfg: PackingConfig, expected_hash: str,
                day_final: bool) -> dict[str, Any]:
    """The 03c checkpoint projected at one anchor, or a remedy-bearing error."""
    event = day.events[index]
    anchor = str(event["semantic_event_id"])
    row = day.checkpoints.get(anchor)
    kind = "day-final" if day_final else "boundary"
    if row is None:
        present = sorted({str(other.get("packing_config_hash"))
                          for other in day.doc.get("checkpoints") or []
                          if str(other["anchor_semantic_event_id"]) == anchor})
        raise ValueError(
            f"{day.day_tag}: no checkpoint at {kind} anchor {anchor!r} (event index "
            f"{index}) with packing_config_hash {expected_hash}; hashes present at that "
            f"anchor: {present or ['none']}. Remedy: rerun annotation pass 03c with this "
            f"packing config (--checkpoint-capacity {cfg.capacity} "
            f"--checkpoint-fraction-low {cfg.fraction_low} --checkpoint-fraction-high "
            f"{cfg.fraction_high} --packing-seed {cfg.seed} --n-packings {cfg.n_packings})"
        )
    if int(row["anchor_event_index"]) != index:
        raise ValueError(
            f"{day.day_tag}: checkpoint at anchor {anchor!r} claims event index "
            f"{row['anchor_event_index']!r}, but that anchor is event {index}"
        )
    if bool(row.get("is_day_final")) is not day_final:
        raise ValueError(
            f"{day.day_tag}: checkpoint at anchor {anchor!r} has "
            f"is_day_final={row.get('is_day_final')!r} at a {kind} anchor"
        )
    if not str(row["text"]).strip():
        raise ValueError(f"{day.day_tag}: checkpoint at anchor {anchor!r} has no text")
    return row


def _messages(
    events: list[dict[str, Any]], thoughts: dict[str, str], *, goal_text: str,
    system_prompt: str, checkpoint_in: str | None,
    control: tuple[dict[str, Any], str] | None, parse_reply: Callable[..., Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record's turns: goal/checkpoint conditioning, then screen/action pairs.

    ``control`` is ``(boundary event, checkpoint text)`` for a segment that ends
    at a compaction — its screenshot is the boundary frame, which the NEXT
    record shows again as its opening screen (``reset_to_current``). A final
    segment passes None and simply ends after the last action.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [text_block(system_prompt)]}
    ]
    n_thoughts = 0
    resume_turns: list[int] = []
    for offset, event in enumerate(events):
        content: list[dict[str, Any]] = []
        if offset == 0:
            content.append(text_block(goal_conditioning(goal_text, checkpoint_in)))
        content.append(image_block(str(event["image"])))
        messages.append({"role": "user", "content": content})
        assistant = str(event["assistant_action"])
        thought = thoughts.get(str(event["semantic_event_id"]))
        if thought:
            assistant = f"<think>{thought}</think>\n{assistant}"
            n_thoughts += 1
        parse_reply(assistant, expected="action")
        if checkpoint_in and len(resume_turns) < RESUME_UPWEIGHT_TURNS:
            resume_turns.append(len(messages))
        messages.append({"role": "assistant", "content": [text_block(assistant)]})
    if control is not None:
        boundary, checkpoint_text = control
        messages.append({"role": "user", "content": [
            text_block(CHECKPOINT_CONTROL_REQUEST), image_block(str(boundary["image"])),
        ]})
        parse_reply(checkpoint_text, expected="checkpoint")
        messages.append({"role": "assistant", "content": [text_block(checkpoint_text)]})
    return messages, {"n_thoughts": n_thoughts, "resume_upweight_turns": resume_turns}


def _record(
    day: _Day, *, span: tuple[int, int], mode: str, node: dict[str, Any] | None,
    checkpoint_in: dict[str, Any] | None, checkpoint_out: dict[str, Any] | None,
    conversation_id: str, episode_id: str, packing_index: int, segment_index: int,
    system_prompt: str, parse_reply: Callable[..., Any], cfg: PackingConfig,
    mission_link_id: str | None = None,
) -> dict[str, Any]:
    """One training record for one action span."""
    start, end = span
    if mode == "proactive":
        if node is not None:
            raise ValueError("proactive records must not carry a goal node")
        goal_text = PROACTIVE_GOAL_TEXT
    else:
        if node is None or str(node["level"]) != _MODE_LEVEL[mode]:
            raise ValueError(f"{conversation_id}: mode {mode!r} without a {mode!r} goal node")
        if not _covers(node, span):
            raise ValueError(
                f"{conversation_id}: goal {node['goal_id']!r} spans events "
                f"[{node['start_event_index']}, {node['end_event_index']}] and does not "
                f"cover the record's action span [{start}, {end}]"
            )
        goal_text = str(node["text"])
    window = day.events[start:end + 1]
    control = None
    if checkpoint_out is not None:
        if int(checkpoint_out["anchor_event_index"]) != end + 1:
            raise ValueError(
                f"{conversation_id}: outgoing checkpoint anchors event "
                f"{checkpoint_out['anchor_event_index']} but the span ends at {end}"
            )
        # The boundary frame: this record's control turn and the next record's
        # opening screen (the runtime's reset_to_current).
        control = (day.events[end + 1], str(checkpoint_out["text"]))
    messages, meta = _messages(
        window, day.thoughts, goal_text=goal_text, system_prompt=system_prompt,
        checkpoint_in=str(checkpoint_in["text"]) if checkpoint_in is not None else None,
        control=control, parse_reply=parse_reply,
    )
    n_images = len(window) + (control is not None)
    if n_images > cfg.capacity:
        raise ValueError(
            f"{conversation_id}: {n_images} screenshots exceed the packed capacity "
            f"{cfg.capacity}"
        )
    event_ids = [str(row["semantic_event_id"]) for row in window]
    return {
        "conversation_id": conversation_id, "episode_id": episode_id,
        "packing_index": packing_index, "segment_index": segment_index,
        "day_tag": day.day_tag, "user_id": day.doc["user_id"], "date": day.doc["date"],
        "recipe": RECIPE, "action_format": ACTION_SPEC,
        "goal_conditioned": True, "mode": mode,
        "goal_id": node["goal_id"] if node is not None else None,
        "instruction": goal_text,
        "goal_provenance": node["provenance"] if node is not None else None,
        "checkpoint_in_id": (checkpoint_in or {}).get("checkpoint_id"),
        "checkpoint_out_id": (checkpoint_out or {}).get("checkpoint_id"),
        "start_event_index": start, "end_event_index": end,
        "semantic_event_ids": event_ids,
        "memory_snapshot_ids": [day.memory_ids[event_id] for event_id in event_ids],
        "cross_day": mission_link_id is not None, "mission_link_id": mission_link_id,
        "n_images": n_images, "n_turns": len(window),
        "n_thoughts": meta["n_thoughts"],
        "n_checkpoint_turns": int(control is not None),
        "resume_upweight_turns": meta["resume_upweight_turns"],
        "messages": messages,
    }


def _pack_day(
    day: _Day, *, cfg: PackingConfig, expected_hash: str, mode_weights: dict[str, float],
    system_prompt: str, parse_reply: Callable[..., Any],
) -> list[dict[str, Any]]:
    """Every base record of one day: one episode per packing index."""
    records: list[dict[str, Any]] = []
    source_ids = {str(row["semantic_event_id"]) for row in day.events}
    for packing_index in range(cfg.n_packings):
        boundaries = boundary_events(len(day.events), day_tag=day.day_tag, cfg=cfg,
                                     packing_index=packing_index)
        spans = segments_from_boundaries(len(day.events), boundaries)
        episode_id = f"{day.day_tag}_p{packing_index}"
        anchors = [
            _checkpoint(day, boundary, cfg=cfg, expected_hash=expected_hash, day_final=False)
            for boundary in boundaries
        ]
        episode: list[dict[str, Any]] = []
        for segment_index, span in enumerate(spans):
            eligible = eligible_modes(span, day.nodes)
            mode = sample_mode(eligible, mode_weights, seed=cfg.seed, day_tag=day.day_tag,
                               packing_index=packing_index, segment_index=segment_index)
            node = (None if mode == "proactive"
                    else _covering_node(day.nodes, _MODE_LEVEL[mode], span))
            episode.append(_record(
                day, span=span, mode=mode, node=node,
                checkpoint_in=anchors[segment_index - 1] if segment_index else None,
                checkpoint_out=(anchors[segment_index]
                                if segment_index < len(anchors) else None),
                conversation_id=f"{episode_id}_s{segment_index:03d}",
                episode_id=episode_id, packing_index=packing_index,
                segment_index=segment_index, system_prompt=system_prompt,
                parse_reply=parse_reply, cfg=cfg,
            ))
        covered = [event_id for row in episode for event_id in row["semantic_event_ids"]]
        missing = source_ids - set(covered)
        if missing:
            raise ValueError(
                f"Stage 04 omitted {len(missing)} semantic action packet(s) from {episode_id}")
        if len(covered) != len(source_ids):
            raise ValueError(
                f"{episode_id} covers {len(covered)} action slots for {len(source_ids)} "
                "semantic events; a packing must partition the day exactly once")
        for earlier, later in zip(episode, episode[1:]):
            handoff = earlier["messages"][-1]["content"][0]["text"]
            if (earlier["checkpoint_out_id"] != later["checkpoint_in_id"]
                    or later["messages"][1]["content"][0]["text"]
                    != goal_conditioning(later["instruction"], handoff)):
                raise ValueError(
                    f"checkpoint handoff {earlier['conversation_id']} -> "
                    f"{later['conversation_id']} is not byte-identical")
        records.extend(episode)
    return records


def _cross_day_records(
    indexed: dict[str, _Day], mission_links: list[dict[str, Any]], *,
    cfg: PackingConfig, expected_hash: str, system_prompt: str,
    parse_reply: Callable[..., Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """One resume-across-days record per mission link, on day B's first segment.

    The record is day B / packing 0 / segment 0 re-rendered with day A's
    day-final checkpoint as the incoming state: the only training shape that
    shows a mission continuing after the machine was shut down. These are
    augmentation, so they stay out of the per-packing partition invariant.
    """
    owner = {str(node["goal_id"]): day.day_tag
             for day in indexed.values() for node in day.nodes}
    stats: Counter[str] = Counter({"n_cross_day_links_unresolved": 0})
    seen: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for link in sorted(mission_links, key=lambda row: str(row["mission_link_id"])):
        source_tag = owner.get(str(link["from_goal_id"]))
        target_tag = owner.get(str(link["to_goal_id"]))
        if source_tag is None or target_tag is None:
            stats["n_cross_day_links_unresolved"] += 1
            continue
        source, target = indexed[source_tag], indexed[target_tag]
        if str(source.doc["user_id"]) != str(target.doc["user_id"]):
            raise ValueError(f"mission link {link['mission_link_id']!r} crosses users")
        if str(source.doc["date"]) >= str(target.doc["date"]):
            raise ValueError(f"mission link {link['mission_link_id']!r} is not causal")
        if not source.events or not target.events:
            stats["n_cross_day_links_unresolved"] += 1
            continue
        boundaries = boundary_events(len(target.events), day_tag=target.day_tag, cfg=cfg,
                                     packing_index=0)
        span = segments_from_boundaries(len(target.events), boundaries)[0]
        node = next((row for row in target.nodes
                     if str(row["goal_id"]) == str(link["to_goal_id"])), None)
        mode = "explicit_long"
        if node is None or str(node["level"]) != "long" or not _covers(node, span):
            mode, node = "proactive", None
        # The spec's `_xday` suffix; numbered when several missions resume into
        # the same day, so conversation_ids stay unique.
        suffix = "_xday" if not seen[target_tag] else f"_xday{seen[target_tag]}"
        seen[target_tag] += 1
        records.append(_record(
            target, span=span, mode=mode, node=node,
            checkpoint_in=_checkpoint(source, len(source.events) - 1, cfg=cfg,
                                      expected_hash=expected_hash, day_final=True),
            checkpoint_out=(_checkpoint(target, boundaries[0], cfg=cfg,
                                        expected_hash=expected_hash, day_final=False)
                            if boundaries else None),
            conversation_id=f"{target.day_tag}_p0_s000{suffix}",
            episode_id=f"{target.day_tag}_p0", packing_index=0, segment_index=0,
            system_prompt=system_prompt, parse_reply=parse_reply, cfg=cfg,
            mission_link_id=str(link["mission_link_id"]),
        ))
    return records, stats


def build_sequential_conversations(
    days: list[dict[str, Any]], *, system_prompt: str, parse_reply: Callable[..., Any],
    cfg: PackingConfig, mode_weights: dict[str, float] = DEFAULT_MODE_WEIGHTS,
    mission_links: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project the annotation artifact into capacity-packed training records.

    Deterministic in ``cfg`` and the artifact alone: every boundary, mode and id
    is a seeded function of stable ids, so re-running writes the same bytes.
    """
    expected_hash = packing_config_hash(cfg)
    ordered_days = sorted(days, key=lambda row: (str(row["user_id"]), str(row["date"])))
    indexed = {str(day["day_tag"]): _index_day(day, expected_hash=expected_hash)
               for day in ordered_days}
    if len(indexed) != len(ordered_days):
        raise ValueError("Stage 04 received two annotated days with the same day_tag")

    source_events: set[str] = set()
    emitted_dispositions: set[str] = set()
    total_dispositions = 0
    total_memory_snapshots = 0
    for day in indexed.values():
        source_events.update(str(row["semantic_event_id"]) for row in day.events)
        total_dispositions += len(day.doc["event_dispositions"])
        emitted_dispositions.update(
            str(row["semantic_event_id"])
            for row in day.doc["event_dispositions"]
            if row.get("disposition") == "emitted" and row.get("semantic_event_id")
        )
        total_memory_snapshots += len(day.memory_ids)

    base: list[dict[str, Any]] = []
    for day in indexed.values():
        base.extend(_pack_day(day, cfg=cfg, expected_hash=expected_hash,
                              mode_weights=mode_weights, system_prompt=system_prompt,
                              parse_reply=parse_reply))
    cross_day, cross_stats = _cross_day_records(
        indexed, list(mission_links or []), cfg=cfg, expected_hash=expected_hash,
        system_prompt=system_prompt, parse_reply=parse_reply)
    records = [*base, *cross_day]

    represented = {event_id for row in records for event_id in row["semantic_event_ids"]}
    missing = source_events - represented
    if missing:
        raise ValueError(f"Stage 04 omitted {len(missing)} semantic action packet(s)")
    unresolved = emitted_dispositions - represented
    if unresolved:
        raise ValueError(f"Stage 04 lost provenance for {len(unresolved)} retained raw event(s)")

    n_action_events = sum(row["n_turns"] for row in base)
    return records, {
        "recipe": RECIPE, "action_format": ACTION_SPEC,
        "packing_config": {**asdict(cfg), "packing_config_hash": expected_hash},
        "mode_weights": dict(mode_weights),
        "n_conversations": len(records),
        "n_episodes": len({row["episode_id"] for row in base}),
        "n_segments": len(base),
        "n_cross_day_records": len(cross_day),
        "mean_segment_events": (n_action_events / len(base)) if base else 0.0,
        "mode_counts": dict(Counter(str(row["mode"]) for row in base)),
        "cross_day_mode_counts": dict(Counter(str(row["mode"]) for row in cross_day)),
        "goal_provenance_counts": dict(Counter(
            str(row["goal_provenance"]) for row in base if row["goal_id"] is not None)),
        "n_images": sum(row["n_images"] for row in records),
        "n_semantic_events": len(source_events),
        "n_event_dispositions": total_dispositions,
        "n_memory_snapshots": total_memory_snapshots,
        "n_thoughts": sum(row["n_thoughts"] for row in records),
        "n_checkpoint_turns": sum(row["n_checkpoint_turns"] for row in records),
        "n_resume_records": sum(1 for row in records if row["resume_upweight_turns"]),
        "resume_upweight_turns": RESUME_UPWEIGHT_TURNS,
        **cross_stats,
    }
