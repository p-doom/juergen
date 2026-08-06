"""Stage 04 --mode sequential_goal_memory: the capacity-driven packer.

Covers the six packer invariants (per-packing partition, byte-identical
checkpoint handoff, capacity ceiling, parser round-trip, determinism, explicit
goal coverage), goal-mode eligibility, cross-day resume records,
``resume_upweight_turns``, the missing-checkpoint diagnostic, and the sequential
CLI helpers. Everything runs on synthetic annotation days built here — the
checkpoint rows are hand-made in the shape annotation pass 03c publishes, so
this suite does not depend on the annotator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from realigned_pipeline.lib.action_format import render_tool_call
from realigned_pipeline.lib.sequential_conversations import build_sequential_conversations
from realigned_pipeline.lib.sequential_goal_memory_contract import (
    ACTION_SPEC,
    CHECKPOINT_CONTROL_REQUEST,
    CHECKPOINT_FIELDS,
    PROACTIVE_GOAL_TEXT,
    RECIPE,
    RESUME_UPWEIGHT_TURNS,
    goal_conditioning,
    render_checkpoint,
    system_prompt,
)
from realigned_pipeline.lib.sequential_packing import (
    DEFAULT_MODE_WEIGHTS,
    PackingConfig,
    boundary_events,
    eligible_modes,
    packing_config_hash,
    segments_from_boundaries,
)
from realigned_pipeline.stage_04_conversations import (
    sequential_mode_weights,
    sequential_packing_config,
)

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from action_parser import parse_sequential_reply  # noqa: E402

DAY_A = "u0_2026-01-01"
DAY_B = "u0_2026-01-02"
# fraction_low == fraction_high with an exact binary product: threshold 5, so
# anchors step by 4 and a 13-event day cuts at [4, 8] — checkable by hand.
FIXED = PackingConfig(capacity=10, fraction_low=0.5, fraction_high=0.5, seed=3)
JITTER = PackingConfig(capacity=12, seed=7)
SWEEP = [
    (PackingConfig(capacity=capacity, fraction_low=low, fraction_high=high, seed=seed),
     n_events)
    for capacity, low, high in ((3, 0.3, 0.3), (6, 0.5, 0.9), (16, 0.4, 1.0))
    for seed in (0, 5)
    for n_events in (1, 2, 7, 23, 60)
]
PROACTIVE_ONLY = {"proactive": 1.0}
EXPLICIT_ONLY = {"explicit_mid": 1.0, "explicit_long": 1.0}
# Overwhelming mid mass: a segment renders its mid unless no single mid covers
# it, which pins the fallback the eligibility rule is supposed to force.
MID_FIRST = {"explicit_mid": 1.0, "explicit_long": 1e-6}


# ---------------------------------------------------------------------------
# synthetic annotation artifact
# ---------------------------------------------------------------------------

def _tree(n_events: int, day_tag: str, *, mid_split: int | None = None,
          long_start: int = 0) -> list[dict]:
    """long over the day, one or two mids partitioning it, one motor short."""
    if n_events <= 0:
        return []
    nodes = [{"goal_id": f"{day_tag}_long", "parent_id": None, "level": "long",
              "text": "Prepare the quarterly revenue report", "provenance": "explicit",
              "start_event_index": long_start, "end_event_index": n_events - 1}]
    split = (n_events + 1) // 2 if mid_split is None else mid_split
    if split >= n_events:
        nodes.append({"goal_id": f"{day_tag}_mid0", "parent_id": f"{day_tag}_long",
                      "level": "mid", "text": "Open the revenue spreadsheet",
                      "provenance": "explicit", "start_event_index": 0,
                      "end_event_index": n_events - 1})
    else:
        nodes.append({"goal_id": f"{day_tag}_mid0", "parent_id": f"{day_tag}_long",
                      "level": "mid", "text": "Open the revenue spreadsheet",
                      "provenance": "explicit", "start_event_index": 0,
                      "end_event_index": split - 1})
        nodes.append({"goal_id": f"{day_tag}_mid1", "parent_id": f"{day_tag}_long",
                      "level": "mid", "text": "Chart the revenue column",
                      "provenance": "proactive", "start_event_index": split,
                      "end_event_index": n_events - 1})
    if n_events >= 3:
        nodes.append({"goal_id": f"{day_tag}_short", "parent_id": f"{day_tag}_mid0",
                      "level": "short", "text": "Click the file-name field",
                      "provenance": "explicit", "start_event_index": 1,
                      "end_event_index": 2})
    return nodes


def _checkpoint_rows(day_tag: str, events: list[dict], cfg: PackingConfig, *,
                     drop: tuple[int, ...] = (), config_hash: str | None = None,
                     day_final_at: int | None = -1) -> list[dict]:
    """Pass-03c rows: the union of boundary anchors over packings + the day final."""
    anchors = {
        boundary
        for packing_index in range(cfg.n_packings)
        for boundary in boundary_events(len(events), day_tag=day_tag, cfg=cfg,
                                        packing_index=packing_index)
    }
    final = len(events) - 1 if day_final_at == -1 else day_final_at
    if final is not None:
        anchors.add(final)
    rows = []
    for index in sorted(anchors - set(drop)):
        anchor = events[index]["semantic_event_id"]
        values = {field: f"{field} known at {anchor}." for field in CHECKPOINT_FIELDS}
        rows.append({
            "checkpoint_id": f"cp_{anchor}", "day_tag": day_tag,
            "anchor_semantic_event_id": anchor, "anchor_event_index": index,
            "values": values, "text": render_checkpoint(values),
            "packing_config_hash": config_hash or packing_config_hash(cfg),
            "is_day_final": index == final,
            "source_memory_snapshot_id": f"mem_{anchor}",
        })
    return rows


def _day(n_events: int, *, cfg: PackingConfig, day_tag: str = DAY_A,
         user_id: str = "u0", date: str = "2026-01-01",
         nodes: list[dict] | None = None, thought_at: tuple[int, ...] = (0, 3),
         **checkpoint_kwargs) -> dict:
    nodes = _tree(n_events, day_tag) if nodes is None else nodes
    events, dispositions, snapshots = [], [], []
    for index in range(n_events):
        anchor = f"{day_tag}_sem{index:03d}"
        call = {"action": "mouse_move_rel", "delta": [index % 900 + 1, 0]}
        events.append({
            "semantic_event_id": anchor, "day_event_index": index, "segment_id": "s0",
            "image": f"/frames/{day_tag}/{index:03d}.jpg", "t_day_s": float(index),
            "tool_calls": [call], "assistant_action": render_tool_call(call),
            "active_goal_path": [node["goal_id"] for node in nodes
                                 if node["start_event_index"] <= index
                                 <= node["end_event_index"]],
        })
        dispositions.append({"raw_event_id": f"{anchor}:r0", "semantic_event_id": anchor,
                             "disposition": "emitted"})
        dispositions.append({"raw_event_id": f"{anchor}:r1", "semantic_event_id": None,
                             "disposition": "filtered"})
        snapshots.append({"memory_snapshot_id": f"mem_{anchor}",
                          "anchor_semantic_event_id": anchor, "anchor_event_index": index,
                          "memory_after": f"State after {anchor}."})
    return {
        "day_tag": day_tag, "user_id": user_id, "date": date,
        "semantic_events": events, "event_dispositions": dispositions,
        "goal_nodes": nodes, "memory_snapshots": snapshots,
        "decisions": [{"anchor_semantic_event_id": events[index]["semantic_event_id"],
                       "thought": f"The panel at event {index} is clipped; scroll first."}
                      for index in thought_at if index < n_events],
        "checkpoints": _checkpoint_rows(day_tag, events, cfg, **checkpoint_kwargs),
    }


def _build(days: list[dict], *, cfg: PackingConfig, weights=DEFAULT_MODE_WEIGHTS,
           mission_links: list[dict] | None = None):
    return build_sequential_conversations(
        days, system_prompt=system_prompt(), parse_reply=parse_sequential_reply,
        cfg=cfg, mode_weights=weights, mission_links=mission_links)


def _texts(record: dict, role: str) -> list[str]:
    return [block["text"] for message in record["messages"] if message["role"] == role
            for block in message["content"] if block["type"] == "text"]


def _images(record: dict) -> list[str]:
    return [block["image"] for message in record["messages"]
            for block in message["content"] if block["type"] == "image"]


# ---------------------------------------------------------------------------
# invariant 1: per-packing partition   +   invariant 4: parser round-trip
# ---------------------------------------------------------------------------

def test_base_records_partition_every_packing_exactly_once() -> None:
    for cfg, n_events in SWEEP:
        days = [_day(n_events, cfg=cfg)]
        records, summary = _build(days, cfg=cfg)
        assert summary["n_semantic_events"] == n_events
        for packing_index in range(cfg.n_packings):
            episode = [row for row in records if row["packing_index"] == packing_index]
            covered = [event_id for row in episode for event_id in row["semantic_event_ids"]]
            assert covered == [row["semantic_event_id"]
                               for row in days[0]["semantic_events"]], (cfg, n_events)
            spans = segments_from_boundaries(
                n_events, boundary_events(n_events, day_tag=DAY_A, cfg=cfg,
                                          packing_index=packing_index))
            assert [(row["start_event_index"], row["end_event_index"])
                    for row in episode] == spans, (cfg, n_events)


def test_multiple_packings_are_independent_partitions_of_the_same_day() -> None:
    cfg = PackingConfig(capacity=12, seed=7, n_packings=3)
    records, summary = _build([_day(40, cfg=cfg)], cfg=cfg)
    assert summary["n_episodes"] == 3
    assert summary["n_segments"] == len(records)
    episodes = {row["episode_id"] for row in records}
    assert episodes == {f"{DAY_A}_p{index}" for index in range(3)}
    shapes = {tuple((row["start_event_index"], row["end_event_index"])
                    for row in records if row["packing_index"] == index)
              for index in range(3)}
    assert len(shapes) > 1  # different boundaries, same events


def test_every_assistant_turn_round_trips_through_the_evaluator_parser() -> None:
    cfg = PackingConfig(capacity=8, seed=2, n_packings=2)
    records, _ = _build([_day(29, cfg=cfg)], cfg=cfg)
    n_checkpoints = 0
    for record in records:
        for message in record["messages"]:
            if message["role"] != "assistant":
                continue
            text = message["content"][0]["text"]
            kind = "checkpoint" if text.startswith("<checkpoint>") else "action"
            n_checkpoints += kind == "checkpoint"
            assert parse_sequential_reply(text, expected=kind).kind == kind
    assert n_checkpoints == sum(row["n_checkpoint_turns"] for row in records) > 0


def test_a_missing_memory_snapshot_still_fails_stage_04() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    day["memory_snapshots"].pop()
    with pytest.raises(ValueError, match="one rolling-memory snapshot per semantic event"):
        _build([day], cfg=cfg)


def test_a_relabelled_day_event_index_is_rejected() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    day["semantic_events"][5]["day_event_index"] = 9
    with pytest.raises(ValueError, match="day_event_index 9 at stream position 5"):
        _build([day], cfg=cfg)


# ---------------------------------------------------------------------------
# invariant 2: byte-identical checkpoint handoff + the shared boundary frame
# ---------------------------------------------------------------------------

def test_checkpoint_handoff_is_byte_identical_and_shares_the_boundary_frame() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    records, _ = _build([day], cfg=cfg, weights=PROACTIVE_ONLY)
    assert [row["conversation_id"] for row in records] == [
        f"{DAY_A}_p0_s000", f"{DAY_A}_p0_s001", f"{DAY_A}_p0_s002"]
    for earlier, later in zip(records, records[1:]):
        handoff = earlier["messages"][-1]["content"][0]["text"]
        assert handoff.startswith("<checkpoint>")
        assert earlier["checkpoint_out_id"] == later["checkpoint_in_id"]
        # the continuation opens with the same GOAL and those exact bytes
        assert (later["messages"][1]["content"][0]["text"]
                == goal_conditioning(PROACTIVE_GOAL_TEXT, handoff))
        # ... and re-shows the boundary screenshot the control turn ended on
        assert _images(earlier)[-1] == _images(later)[0]
    assert records[-1]["messages"][-1]["content"][0]["text"].startswith("<tool_call>")
    assert [row["checkpoint_in_id"] for row in records] == [
        None, f"cp_{DAY_A}_sem004", f"cp_{DAY_A}_sem008"]
    assert [row["checkpoint_out_id"] for row in records] == [
        f"cp_{DAY_A}_sem004", f"cp_{DAY_A}_sem008", None]


def test_record_turn_sequence_mirrors_the_runtime_context() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg, thought_at=(1,))
    records, _ = _build([day], cfg=cfg, weights=PROACTIVE_ONLY)
    first = records[0]
    assert [message["role"] for message in first["messages"]] == [
        "system", *(["user", "assistant"] * 5)]
    assert first["messages"][0]["content"] == [
        {"type": "text", "text": system_prompt()}]
    assert first["messages"][1]["content"] == [
        {"type": "text", "text": goal_conditioning(PROACTIVE_GOAL_TEXT, None)},
        {"type": "image", "image": f"/frames/{DAY_A}/000.jpg"}]
    # image-only user turns for the rest of the span, then the control turn
    assert [message["content"] for message in first["messages"][3:8:2]] == [
        [{"type": "image", "image": f"/frames/{DAY_A}/{index:03d}.jpg"}]
        for index in (1, 2, 3)]
    assert first["messages"][9]["content"] == [
        {"type": "text", "text": CHECKPOINT_CONTROL_REQUEST},
        {"type": "image", "image": f"/frames/{DAY_A}/004.jpg"}]
    assert first["messages"][10]["content"][0]["text"] == day["checkpoints"][0]["text"]
    assert first["messages"][4]["content"][0]["text"].startswith(
        "<think>The panel at event 1 is clipped; scroll first.</think>\n<tool_call>")
    assert first["n_thoughts"] == 1 and first["n_checkpoint_turns"] == 1
    assert first["n_images"] == 5 and first["n_turns"] == 4
    assert first["semantic_event_ids"] == [f"{DAY_A}_sem{i:03d}" for i in range(4)]
    assert first["memory_snapshot_ids"] == [f"mem_{DAY_A}_sem{i:03d}" for i in range(4)]
    assert first["recipe"] == RECIPE and first["action_format"] == ACTION_SPEC
    assert (first["day_tag"], first["user_id"], first["date"]) == (DAY_A, "u0", "2026-01-01")


def test_the_final_segment_of_a_day_has_no_control_turn() -> None:
    cfg = FIXED
    records, summary = _build([_day(13, cfg=cfg)], cfg=cfg)
    assert [row["n_checkpoint_turns"] for row in records] == [1, 1, 0]
    assert summary["n_checkpoint_turns"] == 2
    assert CHECKPOINT_CONTROL_REQUEST not in _texts(records[-1], "user")


# ---------------------------------------------------------------------------
# invariant 3: capacity ceiling
# ---------------------------------------------------------------------------

def test_no_record_shows_more_screenshots_than_the_capacity() -> None:
    for cfg, n_events in SWEEP:
        records, _ = _build([_day(n_events, cfg=cfg)], cfg=cfg)
        for record in records:
            assert record["n_images"] == len(_images(record)), record["conversation_id"]
            assert record["n_images"] <= cfg.capacity, (cfg, n_events)
            assert record["n_images"] == record["n_turns"] + record["n_checkpoint_turns"]


# ---------------------------------------------------------------------------
# invariant 5: determinism
# ---------------------------------------------------------------------------

def test_the_same_config_reproduces_identical_records() -> None:
    cfg = PackingConfig(capacity=9, seed=13, n_packings=2)
    days = [_day(31, cfg=cfg)]
    first, first_summary = _build(days, cfg=cfg)
    second, second_summary = _build([json.loads(json.dumps(days[0]))], cfg=cfg)
    assert first == second
    assert first_summary == second_summary


def test_the_packing_seed_changes_the_packing() -> None:
    shapes = set()
    for seed in range(6):
        cfg = PackingConfig(capacity=12, seed=seed)
        records, _ = _build([_day(60, cfg=cfg)], cfg=cfg)
        shapes.add(tuple((row["start_event_index"], row["mode"]) for row in records))
    assert len(shapes) > 1


# ---------------------------------------------------------------------------
# invariant 6 + mode eligibility: the GOAL always covers the whole span
# ---------------------------------------------------------------------------

def test_the_rendered_goal_always_covers_the_whole_action_span() -> None:
    for cfg, n_events in SWEEP:
        day = _day(n_events, cfg=cfg)
        nodes = {node["goal_id"]: node for node in day["goal_nodes"]}
        records, _ = _build([day], cfg=cfg)
        for record in records:
            span = (record["start_event_index"], record["end_event_index"])
            assert record["mode"] in eligible_modes(span, day["goal_nodes"])
            if record["mode"] == "proactive":
                assert record["goal_id"] is None and record["goal_provenance"] is None
                assert record["instruction"] == PROACTIVE_GOAL_TEXT
                continue
            node = nodes[record["goal_id"]]
            assert node["level"] == record["mode"].removeprefix("explicit_")
            assert node["start_event_index"] <= span[0] <= span[1] <= node["end_event_index"]
            assert record["instruction"] == node["text"]
            assert record["goal_provenance"] == node["provenance"]


def test_a_span_crossing_two_mids_falls_back_to_the_long_goal() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)  # mids split at 7; spans (0,3) (4,7) (8,12)
    records, summary = _build([day], cfg=cfg, weights=MID_FIRST)
    assert [row["mode"] for row in records] == [
        "explicit_mid", "explicit_long", "explicit_mid"]
    assert [row["goal_id"] for row in records] == [
        f"{DAY_A}_mid0", f"{DAY_A}_long", f"{DAY_A}_mid1"]
    assert summary["mode_counts"] == {"explicit_mid": 2, "explicit_long": 1}
    assert summary["goal_provenance_counts"] == {"explicit": 2, "proactive": 1}


def test_a_short_goal_is_never_rendered_as_the_goal() -> None:
    cfg = PackingConfig(capacity=4, fraction_low=0.5, fraction_high=0.5, seed=1)
    day = _day(9, cfg=cfg)  # threshold 2 -> every span is one event, inside the short
    short = next(node for node in day["goal_nodes"] if node["level"] == "short")
    records, _ = _build([day], cfg=cfg, weights=EXPLICIT_ONLY)
    assert any(row["start_event_index"] == short["start_event_index"] for row in records)
    assert all(short["text"] not in _texts(row, "user")[0] for row in records)
    assert all(row["goal_id"] != short["goal_id"] for row in records)


def test_a_day_without_goal_coverage_is_packed_proactively() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg, nodes=[])
    records, summary = _build([day], cfg=cfg)
    assert summary["mode_counts"] == {"proactive": 3}
    assert summary["goal_provenance_counts"] == {}
    assert all(row["instruction"] == PROACTIVE_GOAL_TEXT for row in records)


# ---------------------------------------------------------------------------
# resume upweighting
# ---------------------------------------------------------------------------

def test_resume_upweight_turns_index_the_first_assistant_actions_after_a_resume() -> None:
    cfg = FIXED
    records, summary = _build([_day(13, cfg=cfg)], cfg=cfg)
    assert records[0]["resume_upweight_turns"] == []
    for record in records[1:]:
        turns = record["resume_upweight_turns"]
        assert turns == [2, 4, 6][:min(RESUME_UPWEIGHT_TURNS, record["n_turns"])]
        for index in turns:
            message = record["messages"][index]
            assert message["role"] == "assistant"
            assert not message["content"][0]["text"].startswith("<checkpoint>")
    assert summary["resume_upweight_turns"] == RESUME_UPWEIGHT_TURNS
    assert summary["n_resume_records"] == 2


def test_a_resumed_segment_shorter_than_the_upweight_window_is_truncated() -> None:
    # threshold 2 -> one action per segment, so only one turn can be upweighted.
    cfg = PackingConfig(capacity=3, fraction_low=0.5, fraction_high=0.5, seed=4)
    records, _ = _build([_day(6, cfg=cfg)], cfg=cfg)
    assert [row["n_turns"] for row in records] == [1, 1, 1, 1, 2]
    assert [row["resume_upweight_turns"] for row in records] == [[], [2], [2], [2], [2, 4]]


# ---------------------------------------------------------------------------
# missing / mismatched checkpoints
# ---------------------------------------------------------------------------

def test_a_missing_boundary_checkpoint_names_the_anchor_hash_and_remedy() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg, drop=(4,))
    with pytest.raises(ValueError) as excinfo:
        _build([day], cfg=cfg)
    message = str(excinfo.value)
    assert f"{DAY_A}_sem004" in message and "event index 4" in message
    assert packing_config_hash(cfg) in message
    assert "rerun annotation pass 03c" in message
    assert "--checkpoint-capacity 10" in message and "--packing-seed 3" in message
    assert "--checkpoint-fraction-low 0.5" in message
    assert "--n-packings 1" in message


def test_checkpoints_projected_for_another_capacity_do_not_count() -> None:
    cfg = FIXED
    stale = packing_config_hash(PackingConfig(capacity=20, seed=3))
    day = _day(13, cfg=cfg, config_hash=stale)
    with pytest.raises(ValueError) as excinfo:
        _build([day], cfg=cfg)
    message = str(excinfo.value)
    assert stale in message  # the diagnostic reports what the artifact does carry
    assert packing_config_hash(cfg) in message


def test_two_checkpoints_at_one_anchor_and_one_hash_are_ambiguous() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    day["checkpoints"].append(dict(day["checkpoints"][0], checkpoint_id="cp_other"))
    with pytest.raises(ValueError, match="two checkpoints at anchor"):
        _build([day], cfg=cfg)


def test_a_day_final_flag_at_a_boundary_anchor_is_rejected() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    day["checkpoints"][0]["is_day_final"] = True
    with pytest.raises(ValueError, match="is_day_final=True at a boundary anchor"):
        _build([day], cfg=cfg)


def test_a_checkpoint_disagreeing_with_its_anchor_index_is_rejected() -> None:
    cfg = FIXED
    day = _day(13, cfg=cfg)
    day["checkpoints"][0]["anchor_event_index"] = 5
    with pytest.raises(ValueError, match="claims event index 5"):
        _build([day], cfg=cfg)


# ---------------------------------------------------------------------------
# cross-day resume records
# ---------------------------------------------------------------------------

def _link(from_goal: str, to_goal: str, *, link_id: str = "mission_0") -> dict:
    return {"mission_link_id": link_id, "user_id": "u0", "from_goal_id": from_goal,
            "to_goal_id": to_goal, "relation": "continues",
            "evidence": "the same report is still open the next morning"}


def test_a_mission_link_emits_one_cross_day_resume_record() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg),
            _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02")]
    links = [_link(f"{DAY_A}_long", f"{DAY_B}_long")]
    records, summary = _build(days, cfg=cfg, mission_links=links)
    assert summary["n_cross_day_records"] == 1
    assert summary["n_segments"] == 6 and summary["n_conversations"] == 7
    assert summary["cross_day_mode_counts"] == {"explicit_long": 1}
    extra = next(row for row in records if row["cross_day"])
    base = next(row for row in records
                if row["conversation_id"] == f"{DAY_B}_p0_s000")
    assert extra["conversation_id"] == f"{DAY_B}_p0_s000_xday"
    assert extra["mission_link_id"] == "mission_0"
    assert extra["mode"] == "explicit_long" and extra["goal_id"] == f"{DAY_B}_long"
    assert extra["semantic_event_ids"] == base["semantic_event_ids"]
    assert extra["checkpoint_out_id"] == base["checkpoint_out_id"]
    # it resumes from day A's day-final checkpoint, not from a day-B boundary
    prior = next(row for row in days[0]["checkpoints"] if row["is_day_final"])
    assert extra["checkpoint_in_id"] == prior["checkpoint_id"]
    assert (extra["messages"][1]["content"][0]["text"]
            == goal_conditioning("Prepare the quarterly revenue report", prior["text"]))
    assert extra["resume_upweight_turns"] == [2, 4, 6]
    assert base["resume_upweight_turns"] == []
    for message in extra["messages"]:
        if message["role"] == "assistant":
            text = message["content"][0]["text"]
            parse_sequential_reply(
                text, expected="checkpoint" if text.startswith("<checkpoint>") else "action")


def test_cross_day_records_stay_out_of_the_partition_invariant() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg), _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02")]
    records, _ = _build(days, cfg=cfg,
                        mission_links=[_link(f"{DAY_A}_long", f"{DAY_B}_long")])
    base = [row for row in records if not row["cross_day"] and row["day_tag"] == DAY_B]
    covered = [event_id for row in base for event_id in row["semantic_event_ids"]]
    assert covered == [row["semantic_event_id"] for row in days[1]["semantic_events"]]


def test_a_link_whose_goal_does_not_cover_the_first_segment_goes_proactive() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg),
            _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02",
                 nodes=_tree(13, DAY_B, long_start=6))]
    records, summary = _build(days, cfg=cfg,
                              mission_links=[_link(f"{DAY_A}_long", f"{DAY_B}_long")])
    extra = next(row for row in records if row["cross_day"])
    assert summary["cross_day_mode_counts"] == {"proactive": 1}
    assert extra["mode"] == "proactive" and extra["goal_id"] is None
    assert extra["messages"][1]["content"][0]["text"].startswith(
        f"GOAL: {PROACTIVE_GOAL_TEXT}")


def test_a_link_into_an_unselected_day_is_skipped_not_fatal() -> None:
    cfg = FIXED
    records, summary = _build([_day(13, cfg=cfg)], cfg=cfg,
                              mission_links=[_link(f"{DAY_A}_long", f"{DAY_B}_long")])
    assert summary["n_cross_day_records"] == 0
    assert summary["n_cross_day_links_unresolved"] == 1
    assert all(not row["cross_day"] for row in records)


def test_a_non_causal_mission_link_is_fatal() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg), _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02")]
    with pytest.raises(ValueError, match="is not causal"):
        _build(days, cfg=cfg, mission_links=[_link(f"{DAY_B}_long", f"{DAY_A}_long")])


def test_two_missions_resuming_into_one_day_keep_unique_ids() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg),
            _day(13, cfg=cfg, day_tag="u0_2026-01-03", date="2026-01-03"),
            _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02")]
    links = [_link(f"{DAY_A}_long", "u0_2026-01-03_long", link_id="mission_a"),
             _link(f"{DAY_B}_long", "u0_2026-01-03_long", link_id="mission_b")]
    records, summary = _build(days, cfg=cfg, mission_links=links)
    assert summary["n_cross_day_records"] == 2
    extra = sorted(row["conversation_id"] for row in records if row["cross_day"])
    assert extra == ["u0_2026-01-03_p0_s000_xday", "u0_2026-01-03_p0_s000_xday1"]
    assert len({row["conversation_id"] for row in records}) == len(records)


def test_a_cross_day_link_needs_the_prior_day_final_checkpoint() -> None:
    cfg = FIXED
    days = [_day(13, cfg=cfg, day_final_at=None),
            _day(13, cfg=cfg, day_tag=DAY_B, date="2026-01-02")]
    with pytest.raises(ValueError, match="no checkpoint at day-final anchor"):
        _build(days, cfg=cfg, mission_links=[_link(f"{DAY_A}_long", f"{DAY_B}_long")])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_reports_the_packing_config_not_a_window_schedule() -> None:
    cfg = PackingConfig(capacity=10, fraction_low=0.5, fraction_high=0.5, seed=3,
                        n_packings=2)
    _records, summary = _build([_day(13, cfg=cfg)], cfg=cfg, weights=EXPLICIT_ONLY)
    assert summary["packing_config"] == {
        "capacity": 10, "fraction_low": 0.5, "fraction_high": 0.5, "seed": 3,
        "n_packings": 2, "packing_config_hash": packing_config_hash(cfg)}
    assert summary["mode_weights"] == EXPLICIT_ONLY
    assert summary["n_episodes"] == 2 and summary["n_segments"] == 6
    assert summary["mean_segment_events"] == pytest.approx(13 * 2 / 6)
    assert summary["recipe"] == RECIPE and summary["action_format"] == ACTION_SPEC
    assert summary["n_event_dispositions"] == 26 and summary["n_memory_snapshots"] == 13
    assert summary["n_thoughts"] == 4  # two thought anchors x two packings
    assert summary["n_cross_day_records"] == 0
    assert summary["n_cross_day_links_unresolved"] == 0
    assert summary["cross_day_mode_counts"] == {}
    for dead in ("window_decisions", "window_stride", "n_windows",
                 "desired_provenance_counts", "explicit_target_fraction"):
        assert dead not in summary


def test_the_obsolete_window_api_is_gone() -> None:
    from realigned_pipeline.lib import sequential_conversations

    for dead in ("WINDOW_DECISIONS", "WINDOW_STRIDE", "_window_starts", "_goal_for_event"):
        assert not hasattr(sequential_conversations, dead)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"capacity": 16, "fraction_low": 0.5, "fraction_high": 0.85,
                              "packing_seed": 0, "n_packings": 1, **overrides})


def test_capacity_is_required_for_the_sequential_recipe() -> None:
    with pytest.raises(SystemExit, match="requires --capacity"):
        sequential_packing_config(_args(capacity=None))
    assert sequential_packing_config(_args()) == PackingConfig(
        capacity=16, fraction_low=0.5, fraction_high=0.85, seed=0, n_packings=1)


@pytest.mark.parametrize("overrides", [
    {"capacity": 2}, {"fraction_low": 0.9, "fraction_high": 0.5}, {"n_packings": 0},
    {"fraction_high": 1.5},
])
def test_an_impossible_packing_geometry_exits_cleanly(overrides) -> None:
    with pytest.raises(SystemExit, match="invalid packing config"):
        sequential_packing_config(_args(**overrides))


def test_mode_weights_default_and_parse() -> None:
    assert sequential_mode_weights(None) == DEFAULT_MODE_WEIGHTS
    assert sequential_mode_weights('{"explicit_mid": 1, "proactive": 3}') == {
        "explicit_mid": 1.0, "proactive": 3.0}


@pytest.mark.parametrize(("raw", "match"), [
    ("{", "not valid JSON"),
    ("[]", "non-empty JSON object"),
    ("{}", "non-empty JSON object"),
    ('{"explicit_short": 1}', "unknown mode"),
    ('{"proactive": "1"}', "non-negative number"),
    ('{"proactive": true}', "non-negative number"),
    ('{"proactive": -1}', "non-negative number"),
    ('{"proactive": 0}', "positive total mass"),
])
def test_mode_weights_rejects_bad_input(raw, match) -> None:
    with pytest.raises(SystemExit, match=match):
        sequential_mode_weights(raw)
