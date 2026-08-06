"""Resumable hierarchical-goal, causal-thought, and checkpoint annotation.

Per day, cached pass by pass (``units/<day_tag>/``):

  01_prepare       semantic action packets from the Stage 02/03 event stream.
  02_goal_tree     one nested long/mid/short goal tree over those events.
  03_causal_replay chained rolling memory per event, plus SPARSE thoughts: a
                   deterministic decision-boundary pre-gate (``gate.py``) and,
                   under ``thought_gating="agreement"``, a predict-then-reveal
                   agreement gate — a thought is written only where a predictor
                   with the same context but no sight of the real action
                   disagrees with the human.
  04_checkpoints   pass 03c: text-only checkpoint projections of that memory at
                   exactly the anchors the Stage 04 packer cuts on (shared
                   arithmetic in ``lib/sequential_packing``) plus the day-final
                   anchor. Requires ``checkpoint_capacity``.
  05_publish       the day's deployment-independent STATE, validated.

Nothing here decides how a training record is rendered: goal phrasing,
explicit/proactive framing and segmentation are Stage 04's choices.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.registry import DatasetFinalizeContext, MethodContext
from realigned_pipeline.annotation.lib.units import frames_to_data_urls
from realigned_pipeline.annotation.methods.sequential_goal_memory.gate import (
    DECISION_GAP_S, is_decision_boundary,
)
from realigned_pipeline.lib.common import write_json, write_jsonl
from realigned_pipeline.lib.semantic_actions import (
    ACTION_SPEC, build_segment_semantic_events, render_calls, stable_id,
)
from realigned_pipeline.lib.sequential_goal_memory_contract import (
    CHECKPOINT_FIELDS, CHECKPOINT_MAX_WORDS, THOUGHT_MAX_WORDS, render_checkpoint,
)
from realigned_pipeline.lib.sequential_packing import (
    PackingConfig, actions_agree, boundary_events, packing_config_hash,
)

INPUT_KIND = "days"
LABELER_DEFAULTS = {"temperature": 0.2, "reasoning_effort": "low"}
REQUIRES_PILOT_REVIEW = True
PROMPT_VERSIONS = {
    "prepare": "semantic_events_v1",
    "goal_tree": "goal_tree_v4",
    "causal_replay": "causal_replay_v4",
    "checkpoint_projection": "checkpoint_projection_v1",
    "mission_link": "mission_link_v1",
    "publish": "stage04_projection_v4",
}
LEVELS = ("long", "mid", "short")
PROVENANCE = ("explicit", "proactive")
# "agreement": predict the action without seeing it, reveal a thought only where
# the prediction diverges from the human. "boundary": offer an optional thought
# at every decision boundary (no predictor, ~2x cheaper, much less sparse).
THOUGHT_GATINGS = ("agreement", "boundary")
# `Completed` folds the previous checkpoint's `Completed` into at most two
# sentences plus the new interval; three is therefore the hard ceiling.
COMPLETED_MAX_SENTENCES = 3
INITIAL_MEMORY = (
    "No prior trajectory memory. Establish the current visible state and the "
    "user's active grounded intent without assuming any action has succeeded."
)
_MOTOR_SHORT_GOAL = re.compile(
    r"^(?:click|focus|activate|position|move (?:the )?cursor|scroll|hover|"
    r"select|highlight)\b", re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cached(path: Path, input_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    return doc if doc.get("input_hash") == input_hash else None


def _usage(result: Any) -> int:
    usage = getattr(result, "usage", None)
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens") or
               ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)))


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _prepare(item: dict[str, Any], ctx: MethodContext, pass_dir: Path) -> dict[str, Any]:
    day = item["day"]
    art = ctx.params["filter_artifact"]
    clips = ctx.params["clips_by_segment"]
    day_t1 = (float(ctx.params["day_t1"])
              if ctx.params.get("day_t1") is not None else None)
    input_hash = _hash({
        "version": PROMPT_VERSIONS["prepare"], "filter_id": art.filter_id,
        "day_row": item["row"], "day_t1": day_t1,
    })
    path = pass_dir / "01_prepare.json"
    hit = _cached(path, input_hash)
    if hit is not None:
        return hit
    events: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    segment_stats: list[dict[str, Any]] = []
    for segment in item["row"]["segments"]:
        segment_id = str(segment["segment_id"])
        if day_t1 is not None and float(segment["t_day_s"]) > day_t1:
            continue
        sem, disp, stats = build_segment_semantic_events(
            art.load_segment(segment_id), source_row=clips[segment_id],
            day_offset_s=float(segment["t_day_s"]),
        )
        if day_t1 is not None:
            sem = [row for row in sem if float(row["t_day_s"]) <= day_t1]
            disp = [row for row in disp
                    if float(segment["t_day_s"]) + float(row["t_s"]) <= day_t1]
        for row in sem:
            row.update({"day_tag": day.day_tag, "user_id": day.user_id, "date": day.date})
        for row in disp:
            row.update({"day_tag": day.day_tag, "user_id": day.user_id,
                        "segment_id": segment_id})
        events.extend(sem)
        dispositions.extend(disp)
        segment_stats.append({"segment_id": segment_id, **stats})
    events.sort(key=lambda row: (float(row["t_day_s"]), str(row["segment_id"]),
                                 int(row["raw_event_seqs"][0])))
    for index, event in enumerate(events):
        event["day_event_index"] = index
    doc = {
        "pass": "prepare", "version": PROMPT_VERSIONS["prepare"],
        "input_hash": input_hash, "day_tag": day.day_tag, "user_id": day.user_id,
        "date": day.date, "semantic_events": events,
        "event_dispositions": dispositions, "segment_stats": segment_stats,
    }
    write_json(path, doc)
    return doc


def _sample_frames(day: Any, maximum: int) -> list[Any]:
    frames = list(day.frames)
    if len(frames) <= maximum:
        return frames
    positions = sorted({round(i * (len(frames) - 1) / (maximum - 1))
                        for i in range(maximum)})
    return [frames[i] for i in positions]


def _normalize_tree(parsed: dict[str, Any], events: list[dict[str, Any]],
                    user_id: str, date: str) -> list[dict[str, Any]]:
    event_pos = {str(event["semantic_event_id"]): i for i, event in enumerate(events)}
    candidates: dict[str, dict[str, Any]] = {}
    for offset, raw in enumerate(parsed.get("goals") or []):
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("node_key") or f"node_{offset}")
        level = str(raw.get("level") or "").lower()
        text = " ".join(str(raw.get("text") or "").split())
        start = str(raw.get("start_event_id") or "")
        end = str(raw.get("end_event_id") or "")
        if level not in LEVELS or not text or start not in event_pos or end not in event_pos:
            continue
        lo, hi = event_pos[start], event_pos[end]
        if hi < lo:
            lo, hi, start, end = hi, lo, end, start
        provenance = str(raw.get("provenance") or "").lower()
        candidates[key] = {
            "key": key, "parent_key": (str(raw["parent_key"])
                                          if raw.get("parent_key") is not None else None),
            "level": level, "text": text,
            "provenance": provenance if provenance in PROVENANCE else "explicit",
            "grounding": " ".join(str(raw.get("grounding") or "").split()),
            "start_semantic_event_id": start, "end_semantic_event_id": end,
            "start_event_index": lo, "end_event_index": hi,
        }
    valid: dict[str, dict[str, Any]] = {
        key: row for key, row in candidates.items()
        if row["level"] == "long" and row["parent_key"] is None
    }
    for level, parent_level in (("mid", "long"), ("short", "mid")):
        for key, row in candidates.items():
            parent = valid.get(row["parent_key"] or "")
            if (row["level"] == level and parent is not None
                    and parent["level"] == parent_level
                    and parent["start_event_index"] <= row["start_event_index"]
                    <= row["end_event_index"] <= parent["end_event_index"]):
                valid[key] = row
    if events and any(not any(row["level"] == level for row in valid.values())
                      for level in LEVELS):
        raise ValueError("goal labeler did not return a valid long/mid/short hierarchy")
    ids = {key: stable_id("goal", {
        "user_id": user_id, "date": date, "key": key, "level": row["level"],
        "text": row["text"], "start": row["start_semantic_event_id"],
        "end": row["end_semantic_event_id"],
    }) for key, row in valid.items()}
    nodes: list[dict[str, Any]] = []
    for key, row in valid.items():
        start = events[row["start_event_index"]]
        end = events[row["end_event_index"]]
        nodes.append({
            "goal_id": ids[key], "parent_id": ids.get(row["parent_key"]),
            "level": row["level"], "text": row["text"],
            "provenance": row["provenance"], "grounding": row["grounding"],
            "start_semantic_event_id": row["start_semantic_event_id"],
            "end_semantic_event_id": row["end_semantic_event_id"],
            "start_event_index": row["start_event_index"],
            "end_event_index": row["end_event_index"],
            "start_segment_id": start["segment_id"], "end_segment_id": end["segment_id"],
            "start_master_idx": start["anchor_master_idx"],
            "end_master_idx": end["end_master_idx"],
            "start_raw_event_id": start["raw_event_ids"][0],
            "end_raw_event_id": end["raw_event_ids"][-1],
            "user_id": user_id, "date": date,
        })
    validate_goal_tree(nodes, len(events))
    for index in range(len(events)):
        levels = {node["level"] for node in nodes
                  if node["start_event_index"] <= index <= node["end_event_index"]}
        if levels != set(LEVELS):
            raise ValueError(f"semantic event {index} lacks a complete active goal path")
    return sorted(nodes, key=lambda node: (
        node["start_event_index"], LEVELS.index(node["level"]), node["goal_id"]))


def validate_goal_tree(nodes: list[dict[str, Any]], n_events: int) -> None:
    if nodes and n_events <= 0:
        raise ValueError("goal nodes cannot resolve against an empty event stream")
    by_id = {str(node["goal_id"]): node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate goal node id")
    for node in nodes:
        level = str(node.get("level") or "")
        if level not in LEVELS:
            raise ValueError(f"invalid goal level: {level!r}")
        parent_id = node.get("parent_id")
        if (level == "long") != (parent_id is None):
            raise ValueError("only long goals may be roots")
        if parent_id is not None:
            parent = by_id.get(str(parent_id))
            expected = "long" if level == "mid" else "mid"
            if parent is None or parent.get("level") != expected:
                raise ValueError(f"{level} goal has invalid parent level")
        lo, hi = int(node["start_event_index"]), int(node["end_event_index"])
        if not (0 <= lo <= hi < max(1, n_events)):
            raise ValueError(f"goal boundary out of range: {node['goal_id']}")
        seen = {str(node["goal_id"])}
        current = node
        while current.get("parent_id"):
            parent_id = str(current["parent_id"])
            if parent_id in seen or parent_id not in by_id:
                raise ValueError("cyclic or unresolved goal parent")
            seen.add(parent_id)
            parent = by_id[parent_id]
            if not (parent["start_event_index"] <= lo <= hi <= parent["end_event_index"]):
                raise ValueError("goal child is outside parent boundaries")
            current = parent


def _motor_short_goals(nodes: list[dict[str, Any]]) -> list[str]:
    return [str(node["text"]) for node in nodes
            if node.get("level") == "short"
            and _MOTOR_SHORT_GOAL.search(str(node.get("text") or "").strip())]


def _goal_tree(item: dict[str, Any], ctx: MethodContext, pass_dir: Path,
               prepared: dict[str, Any], track) -> dict[str, Any]:
    events = prepared["semantic_events"]
    input_hash = _hash({"version": PROMPT_VERSIONS["goal_tree"],
                        "prompt_sha": ctx.prompts.sha,
                        "prepare_hash": prepared["input_hash"]})
    path = pass_dir / "02_goal_tree.json"
    hit = _cached(path, input_hash)
    if hit is not None:
        return hit
    if not events:
        doc = {"pass": "goal_tree", "version": PROMPT_VERSIONS["goal_tree"],
               "input_hash": input_hash, "goal_nodes": [], "actual_tokens": 0}
        write_json(path, doc)
        return doc
    frames = _sample_frames(item["day"], int(ctx.params.get("goal_tree_max_frames", 96)))
    images = frames_to_data_urls([str(frame.image) for frame in frames],
                                 ctx.vlm_frame_height, ctx.jpeg_quality)
    labels = [f"frame {frame.day_idx} | t_day_s={frame.t_day_s:.3f} | "
              f"segment={frame.segment_id} master={frame.master_idx}" for frame in frames]
    event_block = "\n".join(
        f"{event['day_event_index']}: {event['semantic_event_id']} t={event['t_day_s']:.3f} "
        f"segment={event['segment_id']} calls={json.dumps(event['tool_calls'], ensure_ascii=False)}"
        for event in events
    )
    user_prompt = ctx.prompts.render(
        "goal_tree", user_id=item["day"].user_id,
        date=item["day"].date, event_block=event_block)
    parsed, result = ctx.labeler.call_json_full(
        ctx.prompts.get("goal_tree_system"), user_prompt,
        images=images, image_labels=labels,
        cache_path=ctx.cache_dir / f"02_goal_tree_{input_hash[:16]}.txt",
        no_cache=ctx.no_cache,
    )
    used = _usage(result)
    track(used)
    try:
        nodes = _normalize_tree(parsed, events, item["day"].user_id, item["day"].date)
        bad = _motor_short_goals(nodes)
        if bad:
            raise ValueError(f"motor-only short goals: {bad}")
    except ValueError as first_error:
        repair_prompt = (
            user_prompt
            + "\n\nCORRECTION REQUIRED: the prior answer failed validation: "
            + str(first_error)
            + ". Return a complete replacement tree. Group every click, focus, cursor "
              "movement, scroll, hover, selection, and highlight under the stable "
              "user-visible outcome it serves; none may be a short-goal label."
        )
        parsed, result = ctx.labeler.call_json_full(
            ctx.prompts.get("goal_tree_system"), repair_prompt,
            images=images, image_labels=labels,
            cache_path=ctx.cache_dir / f"02_goal_tree_repair_{input_hash[:16]}.txt",
            no_cache=ctx.no_cache,
        )
        repair_used = _usage(result)
        used += repair_used
        track(repair_used)
        nodes = _normalize_tree(parsed, events, item["day"].user_id, item["day"].date)
        bad = _motor_short_goals(nodes)
        if bad:
            raise ValueError(f"goal-tree repair retained motor-only short goals: {bad}")
    doc = {"pass": "goal_tree", "version": PROMPT_VERSIONS["goal_tree"],
           "input_hash": input_hash, "goal_nodes": nodes, "actual_tokens": used}
    write_json(path, doc)
    return doc


def _active_path(nodes: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    return sorted(
        [node for node in nodes
         if node["start_event_index"] <= index <= node["end_event_index"]],
        key=lambda node: LEVELS.index(node["level"]),
    )


def _checkpoint_values(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    values = {field: " ".join(str(value.get(field) or "").split())
              for field in CHECKPOINT_FIELDS}
    return values if all(values.values()) else None


def _wrapped(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bare argument dicts as the ``computer_use`` tool calls ``actions_agree``
    compares (the annotated packets store arguments only)."""
    return [{"name": "computer_use",
             "arguments": call if isinstance(call, dict) else {}} for call in calls]


def _predicted_calls(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """The predictor's ordered calls as bare argument dicts.

    A malformed entry becomes an empty dict rather than being dropped: the
    agreement comparison must see the length the predictor actually claimed, and
    an unrecognized call can then only disagree — the safe direction for a gate
    that decides whether a thought is revealed."""
    raw = parsed.get("calls")
    calls: list[dict[str, Any]] = []
    for call in (raw if isinstance(raw, list) else []):
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(arguments, dict):
            calls.append(arguments)
        else:
            calls.append(call if isinstance(call, dict) else {})
    return calls


def _thought_problem(thought: str) -> str | None:
    """Why a revealed thought is unusable, or None when it is fine."""
    if not thought:
        return "the thought is empty at a divergent decision"
    words = len(thought.split())
    if words > THOUGHT_MAX_WORDS:
        return f"the thought is {words} words, over the {THOUGHT_MAX_WORDS}-word budget"
    return None


def _causal_reply(parsed: dict[str, Any], event: dict[str, Any],
                  visible: list[dict[str, Any]], *,
                  offer_thought: bool) -> tuple[str, str, list[str]]:
    references = [str(ref) for ref in (parsed.get("references") or [])]
    visible_ids = {str(row["semantic_event_id"]) for row in visible}
    if any(ref not in visible_ids for ref in references):
        raise ValueError(f"future/unknown causal reference at {event['semantic_event_id']}")
    memory_after = " ".join(str(parsed.get("memory_after") or "").split())
    if not memory_after:
        raise ValueError(f"rolling memory omitted at {event['semantic_event_id']}")
    # A memory-only variant never offered a thought field; anything the labeler
    # volunteers there is discarded rather than trained on.
    thought = " ".join(str(parsed.get("thought") or "").split()) if offer_thought else ""
    return thought, memory_after, references


def _causal_event(ctx: MethodContext, *, event: dict[str, Any], index: int,
                  visible: list[dict[str, Any]], path_nodes: list[dict[str, Any]],
                  memory_before: str, boundary: bool, gating: str,
                  input_hash: str, track) -> dict[str, Any]:
    """One event's causal annotation: at most one predict call, one memory call,
    and one corrective retry of a required-but-invalid revealed thought."""
    event_id = str(event["semantic_event_id"])
    prior_visible = visible[:-1]
    image_refs: list[str] = []
    labels: list[str] = []
    for visible_event in visible:
        if not image_refs or image_refs[-1] != visible_event["image"]:
            image_refs.append(str(visible_event["image"]))
            labels.append(f"semantic {visible_event['semantic_event_id']} | "
                          f"t_day_s={visible_event['t_day_s']:.3f}")
    images = frames_to_data_urls(image_refs, ctx.vlm_frame_height, ctx.jpeg_quality)
    goal_path = "\n".join(
        f"{node['level']}: {node['text']} [{node['provenance']}]" for node in path_nodes
    ) or "(no active grounded goal)"
    prior_actions = "\n".join(
        f"{row['semantic_event_id']}: {row['assistant_action']}" for row in prior_visible
    ) or "(none; this is the first causally visible action)"
    time_label = f"+{event['t_day_s']:.3f}s"
    used = 0
    predicted: list[dict[str, Any]] | None = None
    agreed: bool | None = None
    variant = "causal_event_motor"
    if boundary and gating == "agreement":
        parsed, result = ctx.labeler.call_json_full(
            ctx.prompts.get("predict_system"),
            ctx.prompts.render(
                "predict_action", event_id=event_id, time_label=time_label,
                goal_path=goal_path, memory_before=memory_before,
                prior_actions=prior_actions,
            ),
            images=images, image_labels=labels,
            cache_path=ctx.cache_dir / f"03predict_{event_id}_{input_hash[:16]}.txt",
            no_cache=ctx.no_cache,
        )
        spent = _usage(result)
        used += spent
        track(spent)
        predicted = _predicted_calls(parsed)
        agreed = actions_agree(_wrapped(predicted), _wrapped(event["tool_calls"]))
        variant = "causal_event_motor" if agreed else "causal_event_reveal"
    elif boundary:
        variant = "causal_event"
    user_prompt = ctx.prompts.render(
        variant, event_id=event_id, time_label=time_label, goal_path=goal_path,
        memory_before=memory_before, prior_actions=prior_actions,
        tool_calls=render_calls(event["tool_calls"]), max_words=THOUGHT_MAX_WORDS,
    )
    offer_thought = variant != "causal_event_motor"
    parsed, result = ctx.labeler.call_json_full(
        ctx.prompts.get("causal_system"), user_prompt,
        images=images, image_labels=labels,
        cache_path=ctx.cache_dir / f"03_{variant}_{event_id}_{input_hash[:16]}.txt",
        no_cache=ctx.no_cache,
    )
    spent = _usage(result)
    used += spent
    track(spent)
    thought, memory_after, references = _causal_reply(
        parsed, event, visible, offer_thought=offer_thought)
    problem = _thought_problem(thought) if variant == "causal_event_reveal" else None
    if problem is not None:
        parsed, result = ctx.labeler.call_json_full(
            ctx.prompts.get("causal_system"),
            user_prompt + "\n\nCORRECTION REQUIRED: the prior answer failed validation: "
            + problem + ". Reply again with the same memory_after contract and one "
            f"decisive thought of at most {THOUGHT_MAX_WORDS} words naming the visible or "
            "remembered evidence that forces this exact action.",
            images=images, image_labels=labels,
            cache_path=ctx.cache_dir / f"03repair_{event_id}_{input_hash[:16]}.txt",
            no_cache=ctx.no_cache,
        )
        spent = _usage(result)
        used += spent
        track(spent)
        thought, memory_after, references = _causal_reply(
            parsed, event, visible, offer_thought=True)
        problem = _thought_problem(thought)
        if problem is not None:
            raise ValueError(f"revealed thought at {event_id} still invalid: {problem}")
    return {
        "pass": "causal_event", "version": PROMPT_VERSIONS["causal_replay"],
        "input_hash": input_hash,
        "anchor_semantic_event_id": event_id,
        "anchor_event_index": index, "anchor_master_idx": event["anchor_master_idx"],
        "segment_id": event["segment_id"],
        "visible_event_ids": [row["semantic_event_id"] for row in visible],
        "visible_through_event_id": event_id,
        "active_goal_path": event["active_goal_path"],
        "prior_action_event_ids": [row["semantic_event_id"] for row in prior_visible],
        "memory_before": memory_before, "memory_after": memory_after,
        "thought": thought, "references": references,
        "is_decision_boundary": boundary, "thought_gating": gating,
        "prompt_variant": variant, "predicted_calls": predicted, "agreed": agreed,
        "actual_tokens": used,
    }


def _causal(item: dict[str, Any], ctx: MethodContext, pass_dir: Path,
            prepared: dict[str, Any], tree: dict[str, Any], track) -> dict[str, Any]:
    events = prepared["semantic_events"]
    nodes = tree["goal_nodes"]
    causal_dir = pass_dir / "causal"
    causal_dir.mkdir(parents=True, exist_ok=True)
    gating = str(ctx.params.get("thought_gating") or THOUGHT_GATINGS[0])
    if gating not in THOUGHT_GATINGS:
        raise ValueError(f"thought_gating must be one of {THOUGHT_GATINGS}, got {gating!r}")
    gap_s = float(ctx.params.get("decision_gap_s") or DECISION_GAP_S)
    context_events = max(1, int(ctx.params.get("causal_context_events", 6)))
    decisions: list[dict[str, Any]] = []
    memory_snapshots: list[dict[str, Any]] = []
    rolling_memory = INITIAL_MEMORY
    hashes: list[str] = []
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        path_nodes = _active_path(nodes, index)
        event["active_goal_path"] = [node["goal_id"] for node in path_nodes]
        boundary = is_decision_boundary(events, index, nodes, gap_s=gap_s)
        visible = events[max(0, index - context_events + 1):index + 1]
        payload = {
            "version": PROMPT_VERSIONS["causal_replay"], "prompt_sha": ctx.prompts.sha,
            "event": event["semantic_event_id"],
            "visible": [row["semantic_event_id"] for row in visible],
            "path": event["active_goal_path"],
            "memory_before": rolling_memory,
            "prior_actions": [row["semantic_event_id"] for row in visible[:-1]],
            "is_decision_boundary": boundary, "thought_gating": gating,
        }
        input_hash = _hash(payload)
        hashes.append(input_hash)
        path = causal_dir / f"{event['semantic_event_id']}.json"
        rec = _cached(path, input_hash)
        if rec is None:
            rec = _causal_event(
                ctx, event=event, index=index, visible=visible, path_nodes=path_nodes,
                memory_before=rolling_memory, boundary=boundary, gating=gating,
                input_hash=input_hash, track=track)
            write_json(path, rec)
        if rec.get("memory_before") != rolling_memory:
            raise ValueError(f"rolling memory chain diverged at {event['semantic_event_id']}")
        rolling_memory = str(rec["memory_after"])
        records.append(rec)
        if rec.get("thought"):
            decisions.append({
                "decision_id": stable_id("decision", {
                    "anchor": event["semantic_event_id"], "thought": rec["thought"]}),
                "day_tag": item["day"].day_tag, "user_id": item["day"].user_id,
                **{key: rec[key] for key in (
                    "anchor_semantic_event_id", "anchor_event_index", "anchor_master_idx",
                    "segment_id", "visible_through_event_id", "active_goal_path",
                    "memory_before", "memory_after", "thought", "references")},
                "gate": "divergence" if rec.get("agreed") is False else "offered",
            })
        memory_snapshots.append({
            "memory_snapshot_id": stable_id("memory", {
                "anchor": event["semantic_event_id"], "memory": rec["memory_after"]}),
            "day_tag": item["day"].day_tag, "user_id": item["day"].user_id,
            "date": item["day"].date,
            **{key: rec[key] for key in (
                "anchor_semantic_event_id", "anchor_event_index", "anchor_master_idx",
                "segment_id", "visible_event_ids", "visible_through_event_id",
                "active_goal_path", "prior_action_event_ids", "memory_before",
                "memory_after", "thought", "references", "is_decision_boundary",
                "thought_gating", "predicted_calls", "agreed", "input_hash")},
        })
    doc = {
        "pass": "causal_replay", "version": PROMPT_VERSIONS["causal_replay"],
        "input_hash": _hash({"version": PROMPT_VERSIONS["causal_replay"],
                             "event_hashes": hashes}),
        "thought_gating": gating, "decision_gap_s": gap_s,
        "decisions": decisions, "memory_snapshots": memory_snapshots,
        "n_events_replayed": len(memory_snapshots),
        "n_decision_boundaries": sum(1 for rec in records
                                     if rec.get("is_decision_boundary")),
        "n_predicted": sum(1 for rec in records if rec.get("predicted_calls") is not None),
        "n_divergent": sum(1 for rec in records if rec.get("agreed") is False),
    }
    write_json(pass_dir / "03_causal_replay.json", doc)
    return doc


def _param(params: dict[str, Any], key: str, default: Any, cast) -> Any:
    value = params.get(key)
    return default if value is None or str(value).strip() == "" else cast(value)


def packing_config(params: dict[str, Any]) -> PackingConfig:
    """The packing geometry pass 03c projects checkpoints for.

    ``checkpoint_capacity`` is REQUIRED: it must equal the runtime screenshot
    capacity Stage 04 packs to, and a silent default would silently produce
    checkpoints at anchors no training record ever ends on."""
    capacity = params.get("checkpoint_capacity")
    if capacity is None or str(capacity).strip() == "":
        raise ValueError(
            "sequential_goal_memory needs --checkpoint-capacity (the runtime screenshot "
            "capacity Stage 04 will pack to); there is no default"
        )
    return PackingConfig(
        capacity=int(capacity),
        fraction_low=_param(params, "checkpoint_fraction_low",
                            PackingConfig.fraction_low, float),
        fraction_high=_param(params, "checkpoint_fraction_high",
                             PackingConfig.fraction_high, float),
        seed=_param(params, "packing_seed", PackingConfig.seed, int),
        n_packings=_param(params, "n_packings", PackingConfig.n_packings, int),
    )


def checkpoint_anchors(n_events: int, *, day_tag: str, cfg: PackingConfig) -> list[int]:
    """Event indices needing a checkpoint: every packing's compaction boundary
    (a record that ends there must emit one, and the next record embeds it),
    plus the day's final event (the cross-day handoff)."""
    if n_events <= 0:
        return []
    anchors = {n_events - 1}
    for packing_index in range(cfg.n_packings):
        anchors.update(boundary_events(n_events, day_tag=day_tag, cfg=cfg,
                                       packing_index=packing_index))
    return sorted(anchors)


def _checkpoint_problem(values: Any, previous: dict[str, str] | None) -> str | None:
    """Why a projected checkpoint is unusable, or None when it is fine."""
    values = _checkpoint_values(values)
    if values is None:
        return f"the checkpoint omits one of the seven fields {list(CHECKPOINT_FIELDS)}"
    words = sum(len(values[field].split()) for field in CHECKPOINT_FIELDS)
    if words > CHECKPOINT_MAX_WORDS:
        return (f"the seven field bodies total {words} words, over the "
                f"{CHECKPOINT_MAX_WORDS}-word budget")
    if previous is not None:
        sentences = [part for part in _SENTENCE_SPLIT.split(values["Completed"])
                     if part.strip()]
        if len(sentences) > COMPLETED_MAX_SENTENCES:
            return (f"`Completed` has {len(sentences)} sentences: fold the previous "
                    f"checkpoint's `Completed` into at most {COMPLETED_MAX_SENTENCES} "
                    "sentences total instead of restating history")
    return None


def _project_checkpoint(ctx: MethodContext, *, event: dict[str, Any], index: int,
                        memory_after: str, path_nodes: list[dict[str, Any]],
                        previous: dict[str, str] | None, input_hash: str,
                        track) -> dict[str, Any]:
    """One anchor's text-only checkpoint projection, with one corrective retry."""
    event_id = str(event["semantic_event_id"])
    goal_path = "\n".join(
        f"{node['level']}: {node['text']}" for node in path_nodes
    ) or "(no active grounded goal)"
    previous_checkpoint = (
        "(none; this is the first compaction of this day)" if previous is None
        else "\n".join(f"## {field}\n{previous[field]}" for field in CHECKPOINT_FIELDS))
    user_prompt = ctx.prompts.render(
        "checkpoint_projection", event_id=event_id,
        time_label=f"+{event['t_day_s']:.3f}s", goal_path=goal_path,
        memory_after=memory_after, previous_checkpoint=previous_checkpoint,
        max_words=CHECKPOINT_MAX_WORDS, max_sentences=COMPLETED_MAX_SENTENCES,
    )
    parsed, result = ctx.labeler.call_json_full(
        ctx.prompts.get("checkpoint_system"), user_prompt,
        cache_path=ctx.cache_dir / f"04checkpoint_{event_id}_{input_hash[:16]}.txt",
        no_cache=ctx.no_cache,
    )
    used = _usage(result)
    track(used)
    values = _checkpoint_values(parsed.get("checkpoint"))
    problem = _checkpoint_problem(values, previous)
    if problem is not None:
        parsed, result = ctx.labeler.call_json_full(
            ctx.prompts.get("checkpoint_system"),
            user_prompt + "\n\nCORRECTION REQUIRED: the prior answer failed validation: "
            + problem + ". Return a complete replacement checkpoint that satisfies every "
            "rule above; drop detail rather than exceeding a budget.",
            cache_path=ctx.cache_dir / f"04repair_{event_id}_{input_hash[:16]}.txt",
            no_cache=ctx.no_cache,
        )
        spent = _usage(result)
        used += spent
        track(spent)
        values = _checkpoint_values(parsed.get("checkpoint"))
        problem = _checkpoint_problem(values, previous)
        if problem is not None:
            raise ValueError(f"checkpoint projection at {event_id} still invalid: {problem}")
    return {
        "pass": "checkpoint_projection", "version": PROMPT_VERSIONS["checkpoint_projection"],
        "input_hash": input_hash, "anchor_semantic_event_id": event_id,
        "anchor_event_index": index, "values": values, "actual_tokens": used,
    }


def _checkpoints(item: dict[str, Any], ctx: MethodContext, pass_dir: Path,
                 prepared: dict[str, Any], tree: dict[str, Any],
                 causal: dict[str, Any], track) -> dict[str, Any]:
    """Pass 03c: lazy text-only checkpoint projections of the annotated rolling
    memory, at exactly the anchors the packer will cut on.

    Projection convention (identical to the checkpoints this replaces): from
    ``memory_after`` of the anchor event, whose action is recorded as INTENDED,
    never completed — the runtime asks for the checkpoint before executing the
    action belonging to the boundary screenshot."""
    events = prepared["semantic_events"]
    nodes = tree["goal_nodes"]
    cfg = packing_config(ctx.params)
    cfg_hash = packing_config_hash(cfg)
    day_tag = str(item["day"].day_tag)
    anchors = checkpoint_anchors(len(events), day_tag=day_tag, cfg=cfg)
    snapshots = {int(row["anchor_event_index"]): row for row in causal["memory_snapshots"]}
    checkpoint_dir = pass_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    previous: dict[str, str] | None = None
    for anchor in anchors:
        event = events[anchor]
        snapshot = snapshots.get(anchor)
        if snapshot is None:
            raise ValueError(f"no rolling-memory snapshot to project at event {anchor}")
        path_nodes = _active_path(nodes, anchor)
        payload = {
            "version": PROMPT_VERSIONS["checkpoint_projection"],
            "prompt_sha": ctx.prompts.sha, "packing_config_hash": cfg_hash,
            "anchor": event["semantic_event_id"],
            "memory_after": snapshot["memory_after"],
            "path": [node["goal_id"] for node in path_nodes],
            "previous": previous,
        }
        input_hash = _hash(payload)
        hashes.append(input_hash)
        path = checkpoint_dir / f"{event['semantic_event_id']}.json"
        rec = _cached(path, input_hash)
        if rec is None:
            rec = _project_checkpoint(
                ctx, event=event, index=anchor, memory_after=str(snapshot["memory_after"]),
                path_nodes=path_nodes, previous=previous, input_hash=input_hash, track=track)
            write_json(path, rec)
        values = dict(rec["values"])
        text = render_checkpoint(values)
        rows.append({
            "checkpoint_id": stable_id("checkpoint", {
                "anchor": event["semantic_event_id"], "text": text}),
            "day_tag": day_tag, "user_id": item["day"].user_id,
            "anchor_semantic_event_id": event["semantic_event_id"],
            "anchor_event_index": anchor,
            "anchor_master_idx": event["anchor_master_idx"],
            "segment_id": event["segment_id"],
            "visible_through_event_id": event["semantic_event_id"],
            "active_goal_path": [node["goal_id"] for node in path_nodes],
            "values": values, "text": text,
            "packing_config_hash": cfg_hash,
            "is_day_final": anchor == len(events) - 1,
            "source_memory_snapshot_id": snapshot["memory_snapshot_id"],
        })
        previous = values
    doc = {
        "pass": "checkpoint_projection", "version": PROMPT_VERSIONS["checkpoint_projection"],
        "input_hash": _hash({"version": PROMPT_VERSIONS["checkpoint_projection"],
                             "packing_config_hash": cfg_hash, "anchor_hashes": hashes}),
        "packing_config": asdict(cfg), "packing_config_hash": cfg_hash,
        "anchor_event_indices": anchors, "checkpoints": rows,
    }
    write_json(pass_dir / "04_checkpoints.json", doc)
    return doc


def _compatibility_goals(nodes: list[dict[str, Any]],
                         events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if node["level"] != "short":
            continue
        covered = events[node["start_event_index"]:node["end_event_index"] + 1]
        by_segment: dict[str, list[dict[str, Any]]] = {}
        for event in covered:
            by_segment.setdefault(str(event["segment_id"]), []).append(event)
        for segment_id, segment_events in by_segment.items():
            rows.append({
                "goal_id": f"{node['goal_id']}:{segment_id}",
                "goal_node_id": node["goal_id"], "segment_id": segment_id,
                "recording_id": segment_events[0].get("recording_id"),
                "start_master_idx": min(int(row["anchor_master_idx"])
                                        for row in segment_events),
                "end_master_idx": max(int(row["end_master_idx"])
                                      for row in segment_events) + 1,
                "instruction": node["text"], "instruction_variants": [],
                "provenance": node["provenance"], "grounding": node["grounding"],
            })
    return rows


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    pass_dir = Path(ctx.params["day_units_dir"])
    pass_dir.mkdir(parents=True, exist_ok=True)
    report = ctx.params.get("report_tokens") or (lambda _n: None)
    spent = 0

    def track(n: int) -> None:
        nonlocal spent
        spent += int(n)
        report(int(n))

    prepared = _prepare(item, ctx, pass_dir)
    tree = _goal_tree(item, ctx, pass_dir, prepared, track)
    causal = _causal(item, ctx, pass_dir, prepared, tree, track)
    checkpoints = _checkpoints(item, ctx, pass_dir, prepared, tree, causal, track)
    publish_hash = _hash({
        "version": PROMPT_VERSIONS["publish"], "prepare": prepared["input_hash"],
        "tree": tree["input_hash"], "causal": causal["input_hash"],
        "checkpoints": checkpoints["input_hash"],
    })
    publish_path = pass_dir / "05_publish.json"
    published = _cached(publish_path, publish_hash)
    if published is None:
        validate_goal_tree(tree["goal_nodes"], len(prepared["semantic_events"]))
        events_by_id = {
            str(event["semantic_event_id"]): event
            for event in prepared["semantic_events"]
        }
        checkpoint_ids = {int(row["anchor_event_index"]): str(row["checkpoint_id"])
                          for row in checkpoints["checkpoints"]}
        memory_snapshots = []
        for snapshot in causal["memory_snapshots"]:
            event = events_by_id[str(snapshot["anchor_semantic_event_id"])]
            memory_snapshots.append({
                **snapshot,
                "t_day_s": event["t_day_s"], "image": event["image"],
                "upcoming_tool_calls": event["tool_calls"],
                "raw_event_ids": event["raw_event_ids"],
                "action_spec": event["action_spec"],
                "checkpoint_id": checkpoint_ids.get(int(snapshot["anchor_event_index"])),
            })
        published = {
            "pass": "publish", "version": PROMPT_VERSIONS["publish"],
            "input_hash": publish_hash, "day_tag": item["day"].day_tag,
            "user_id": item["day"].user_id, "date": item["day"].date,
            "goal_nodes": tree["goal_nodes"],
            "semantic_events": prepared["semantic_events"],
            "event_dispositions": prepared["event_dispositions"],
            "decisions": causal["decisions"],
            "checkpoints": checkpoints["checkpoints"],
            "memory_snapshots": memory_snapshots,
            "packing_config": checkpoints["packing_config"],
            "packing_config_hash": checkpoints["packing_config_hash"],
            "thought_gating": causal["thought_gating"],
            "decision_gap_s": causal["decision_gap_s"],
            "gate_stats": {key: causal[key] for key in (
                "n_decision_boundaries", "n_predicted", "n_divergent")},
            "compatibility_goals": _compatibility_goals(
                tree["goal_nodes"], prepared["semantic_events"]),
        }
        write_json(publish_path, published)
    return {
        "compatibility_goals": published["compatibility_goals"],
        "n_semantic_events": len(published["semantic_events"]),
        "n_goal_nodes": len(published["goal_nodes"]),
        "n_decisions": len(published["decisions"]),
        "n_checkpoints": len(published["checkpoints"]),
        "n_memory_snapshots": len(published["memory_snapshots"]),
        **published["gate_stats"],
        "actual_tokens": spent,
    }


def goal_rows_from_result(day_tag: str, result: dict[str, Any], *, method: Any,
                          model: str | None, fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for goal in result.get("compatibility_goals", []):
        rows.append({
            **goal, "anchor": goal["goal_node_id"], "method": method.name,
            "model": model or "env", "prompt_pack_sha": method.prompts.sha,
            "unit_id": day_tag, "annotation_fps": fps, "day_tag": day_tag,
        })
    return rows


def _validate_checkpoints(day: dict[str, Any], ids: list[str],
                          snapshots: list[dict[str, Any]]) -> None:
    """Pass-03c checkpoint rows: exactly the packer's anchors, in order, each a
    valid folded projection of the rolling memory snapshot it names."""
    if not isinstance(day.get("packing_config"), dict):
        raise ValueError(
            f"day {day.get('day_tag')!r} was published before the 03c checkpoint pass "
            "existed; delete its units/<day_tag>/05_publish.json and re-run annotation"
        )
    cfg = PackingConfig(**day["packing_config"])
    cfg_hash = packing_config_hash(cfg)
    if str(day.get("packing_config_hash") or "") != cfg_hash:
        raise ValueError("day packing_config_hash does not match its packing_config")
    rows = sorted(day["checkpoints"], key=lambda row: int(row["anchor_event_index"]))
    expected = checkpoint_anchors(len(ids), day_tag=str(day["day_tag"]), cfg=cfg)
    if [int(row["anchor_event_index"]) for row in rows] != expected:
        raise ValueError(
            f"checkpoint anchors {[int(row['anchor_event_index']) for row in rows]} are not "
            f"the packing's anchors {expected} (rerun pass 03c with this packing config)"
        )
    by_anchor = {int(row["anchor_event_index"]): row for row in snapshots}
    previous: dict[str, str] | None = None
    for row in rows:
        index = int(row["anchor_event_index"])
        if str(row["anchor_semantic_event_id"]) != ids[index]:
            raise ValueError("checkpoint anchor id does not match its event index")
        if str(row.get("packing_config_hash") or "") != cfg_hash:
            raise ValueError("checkpoint row was projected for a different packing config")
        if bool(row["is_day_final"]) != (index == len(ids) - 1):
            raise ValueError("checkpoint is_day_final does not match its anchor")
        values = _checkpoint_values(row.get("values"))
        problem = _checkpoint_problem(values, previous)
        if problem is not None:
            raise ValueError(f"published checkpoint at {ids[index]} is invalid: {problem}")
        if render_checkpoint(values) != row["text"]:
            raise ValueError("checkpoint text does not render from its own values")
        if str(row["source_memory_snapshot_id"]) != str(
                by_anchor[index]["memory_snapshot_id"]):
            raise ValueError("checkpoint does not name the memory snapshot at its anchor")
        if str(by_anchor[index].get("checkpoint_id") or "") != str(row["checkpoint_id"]):
            raise ValueError("memory snapshot at a checkpoint anchor does not link back")
        previous = values


def _validate_days(days: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    raw_seen: set[str] = set()
    for day in days:
        events = day["semantic_events"]
        validate_goal_tree(day["goal_nodes"], len(events))
        ids = [str(event["semantic_event_id"]) for event in events]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("semantic event ids are not globally unique")
        seen.update(ids)
        position = {event_id: index for index, event_id in enumerate(ids)}
        goal_ids = {str(node["goal_id"]) for node in day["goal_nodes"]}
        for node in day["goal_nodes"]:
            lo, hi = int(node["start_event_index"]), int(node["end_event_index"])
            if (str(node["start_semantic_event_id"]) != ids[lo]
                    or str(node["end_semantic_event_id"]) != ids[hi]):
                raise ValueError("goal event-id boundary does not match its event index")
            if (int(node["start_master_idx"]) != int(events[lo]["anchor_master_idx"])
                    or int(node["end_master_idx"]) != int(events[hi]["end_master_idx"])):
                raise ValueError("goal master-frame boundary does not resolve")
        for index, event in enumerate(events):
            path = {str(goal_id) for goal_id in event.get("active_goal_path", [])}
            if not path or not path <= goal_ids:
                raise ValueError(f"event {index} has an unresolved active goal path")
        snapshots = sorted(day.get("memory_snapshots") or [],
                           key=lambda row: int(row["anchor_event_index"]))
        if len(snapshots) != len(events):
            raise ValueError("rolling memory does not have exactly one snapshot per event")
        memory_before = INITIAL_MEMORY
        checkpoint_ids = {str(row["checkpoint_id"]) for row in day["checkpoints"]}
        for index, snapshot in enumerate(snapshots):
            anchor = ids[index]
            if (int(snapshot["anchor_event_index"]) != index
                    or str(snapshot["anchor_semantic_event_id"]) != anchor
                    or snapshot["visible_through_event_id"] != anchor):
                raise ValueError("rolling memory snapshot has an unresolved anchor")
            event = events[index]
            if (snapshot.get("image") != event.get("image")
                    or snapshot.get("upcoming_tool_calls") != event.get("tool_calls")
                    or snapshot.get("raw_event_ids") != event.get("raw_event_ids")):
                raise ValueError("rolling memory snapshot does not resolve to its source event")
            if snapshot.get("memory_before") != memory_before:
                raise ValueError("rolling memory snapshots do not form a causal chain")
            memory_after = str(snapshot.get("memory_after") or "").strip()
            if not memory_after:
                raise ValueError("rolling memory snapshot is empty")
            if any(position.get(str(ref), len(ids)) > index
                   for ref in snapshot.get("references", [])):
                raise ValueError("rolling memory snapshot references a future event")
            checkpoint_id = snapshot.get("checkpoint_id")
            if checkpoint_id is not None and str(checkpoint_id) not in checkpoint_ids:
                raise ValueError("rolling memory snapshot references an unknown checkpoint")
            memory_before = memory_after
        for row in [*day["decisions"], *day["checkpoints"]]:
            anchor = str(row["anchor_semantic_event_id"])
            if anchor not in position or row["visible_through_event_id"] != anchor:
                raise ValueError("causal row has unresolved/newer anchor")
            if any(position.get(str(ref), len(ids)) > position[anchor]
                   for ref in row.get("references", [])):
                raise ValueError("causal row references a future event")
        for row in day["decisions"]:
            if row.get("gate") not in ("divergence", "offered"):
                raise ValueError(f"decision row has an invalid gate: {row.get('gate')!r}")
        _validate_checkpoints(day, ids, snapshots)
        for row in day["event_dispositions"]:
            raw_id = str(row.get("raw_event_id") or "")
            if not raw_id or raw_id in raw_seen:
                raise ValueError("raw event ids are empty or duplicated")
            raw_seen.add(raw_id)
            owner = row.get("semantic_event_id")
            if row.get("disposition") == "emitted":
                if str(owner) not in position:
                    raise ValueError("emitted raw event has no semantic action packet")
            elif owner is not None:
                raise ValueError("non-emitted raw event unexpectedly owns an action packet")


def finalize_dataset(ctx: DatasetFinalizeContext) -> dict[str, Any]:
    days = [json.loads(path.read_text())
            for path in sorted(ctx.units_dir.glob("*/05_publish.json"))]
    _validate_days(days)
    users: dict[str, list[dict[str, Any]]] = {}
    for day in days:
        users.setdefault(str(day["user_id"]), []).append(day)
    users_dir = ctx.output_dir / "finalize" / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, Any]] = []
    for user_id, user_days in sorted(users.items()):
        user_days.sort(key=lambda row: str(row["date"]))
        goals = [node for day in user_days for node in day["goal_nodes"]
                 if node["level"] == "long"]
        input_hash = _hash({"version": PROMPT_VERSIONS["mission_link"],
                            "user_id": user_id, "goals": goals,
                            "prompt_sha": ctx.method.prompts.sha})
        path = users_dir / f"{_safe(user_id)}.json"
        hit = _cached(path, input_hash)
        if hit is None:
            parsed: dict[str, Any] = {"links": []}
            used = 0
            if len({str(goal["date"]) for goal in goals}) > 1:
                block = "\n".join(
                    f"{goal['date']} {goal['goal_id']}: {goal['text']} "
                    f"[{goal['provenance']}]" for goal in goals)
                parsed, result = ctx.labeler.call_json_full(
                    ctx.method.prompts.get("mission_system"),
                    ctx.method.prompts.render("mission_link", user_id=user_id,
                                              goal_block=block),
                    cache_path=(ctx.calls_dir / "finalize"
                                / f"{_safe(user_id)}_{input_hash[:16]}.txt"),
                    no_cache=ctx.no_cache,
                )
                used = _usage(result)
            by_id = {str(goal["goal_id"]): goal for goal in goals}
            user_links: list[dict[str, Any]] = []
            for raw in parsed.get("links") or []:
                if not isinstance(raw, dict):
                    continue
                source = str(raw.get("from_goal_id") or "")
                target = str(raw.get("to_goal_id") or "")
                if source not in by_id or target not in by_id:
                    continue
                if by_id[source]["user_id"] != user_id or by_id[target]["user_id"] != user_id:
                    raise ValueError("mission link crosses users")
                if str(by_id[source]["date"]) >= str(by_id[target]["date"]):
                    continue
                relation = str(raw.get("relation") or "continues")
                if relation not in ("continues", "resumes", "refines"):
                    relation = "continues"
                user_links.append({
                    "mission_link_id": stable_id("mission", {
                        "user": user_id, "from": source, "to": target,
                        "relation": relation}),
                    "user_id": user_id, "from_goal_id": source, "to_goal_id": target,
                    "relation": relation,
                    "evidence": " ".join(str(raw.get("evidence") or "").split()),
                })
            hit = {"pass": "mission_link", "version": PROMPT_VERSIONS["mission_link"],
                   "input_hash": input_hash, "user_id": user_id,
                   "mission_links": user_links, "actual_tokens": used}
            write_json(path, hit)
        links.extend(hit["mission_links"])

    node_index = {str(node["goal_id"]): node for day in days for node in day["goal_nodes"]}
    for link in links:
        source = node_index.get(str(link["from_goal_id"]))
        target = node_index.get(str(link["to_goal_id"]))
        if (source is None or target is None
                or source["user_id"] != target["user_id"]
                or source["user_id"] != link["user_id"]
                or source["date"] >= target["date"]):
            raise ValueError("cached mission link is unresolved, non-causal, or cross-user")

    events = [row for day in days for row in day["semantic_events"]]
    dispositions = [row for day in days for row in day["event_dispositions"]]
    nodes = [row for day in days for row in day["goal_nodes"]]
    decisions = [row for day in days for row in day["decisions"]]
    checkpoints = [row for day in days for row in day["checkpoints"]]
    memory_snapshots = [row for day in days for row in day["memory_snapshots"]]
    write_jsonl(ctx.output_dir / "days.jsonl", days)
    write_jsonl(ctx.output_dir / "goal_nodes.jsonl", nodes)
    write_jsonl(ctx.output_dir / "semantic_events.jsonl", events)
    write_jsonl(ctx.output_dir / "event_dispositions.jsonl", dispositions)
    write_jsonl(ctx.output_dir / "decision_thoughts.jsonl", decisions)
    write_jsonl(ctx.output_dir / "checkpoints.jsonl", checkpoints)
    write_jsonl(ctx.output_dir / "memory_snapshots.jsonl", memory_snapshots)
    write_jsonl(ctx.output_dir / "mission_links.jsonl", links)
    write_jsonl(ctx.output_dir / "goals_active.jsonl", [
        {"semantic_event_id": event["semantic_event_id"], "day_tag": event["day_tag"],
         "user_id": event["user_id"], "active_goal_path": event["active_goal_path"]}
        for event in events
    ])
    write_jsonl(ctx.output_dir / "memory.jsonl", memory_snapshots)
    # Stage 04 packs every day with ONE packing config; a mixed artifact would
    # ask for checkpoints at anchors that were never projected.
    packing_hashes = {str(day["packing_config_hash"]) for day in days}
    packing_configs = {json.dumps(day["packing_config"], sort_keys=True) for day in days}
    gating_params = {json.dumps({key: day[key] for key in
                                 ("thought_gating", "decision_gap_s")}, sort_keys=True)
                     for day in days}
    if len(packing_hashes) > 1 or len(packing_configs) > 1:
        raise ValueError(f"artifact mixes packing configs: {sorted(packing_configs)}")
    if len(gating_params) > 1:
        raise ValueError(f"artifact mixes thought gating params: {sorted(gating_params)}")
    gate_stats = {key: sum(int(day["gate_stats"][key]) for day in days)
                  for key in ("n_decision_boundaries", "n_predicted", "n_divergent")}
    write_jsonl(ctx.output_dir / "stage04_index.jsonl", [
        {"day_tag": day["day_tag"], "user_id": day["user_id"], "date": day["date"],
         "semantic_event_ids": [row["semantic_event_id"] for row in day["semantic_events"]],
         "goal_ids": [row["goal_id"] for row in day["goal_nodes"]],
         "decision_ids": [row["decision_id"] for row in day["decisions"]],
         "checkpoint_ids": [row["checkpoint_id"] for row in day["checkpoints"]],
         "memory_snapshot_ids": [row["memory_snapshot_id"]
                                 for row in day["memory_snapshots"]]}
        for day in days
    ])
    return {
        "method_schema_version": 2, "prompt_versions": PROMPT_VERSIONS,
        "labeler_provenance": {
            "model": ctx.model or "env", "prompt_pack_sha": ctx.method.prompts.sha,
        },
        "action_spec": ACTION_SPEC, "source_manifests": ctx.source_manifests,
        "packing_config": days[0]["packing_config"] if days else None,
        "packing_config_hash": next(iter(sorted(packing_hashes)), None),
        "gating_params": json.loads(next(iter(gating_params))) if days else None,
        "n_days": len(days), "n_users": len(users), "n_goal_nodes": len(nodes),
        "n_semantic_events": len(events), "n_decisions": len(decisions),
        "n_checkpoints": len(checkpoints),
        "n_memory_snapshots": len(memory_snapshots),
        "n_mission_links": len(links),
        **gate_stats,
        "checkpoints": "checkpoints.jsonl",
        "memory_snapshots": "memory_snapshots.jsonl",
        "stage04_index": "stage04_index.jsonl",
    }
