from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from realigned_pipeline.annotation.lib.registry import (
    DatasetFinalizeContext,
    MethodContext,
    load_method,
)
from realigned_pipeline.annotation.methods.sequential_goal_memory import annotator
from realigned_pipeline.annotation.methods.sequential_goal_memory.annotator import (
    COMPLETED_MAX_SENTENCES,
    INITIAL_MEMORY,
    _causal,
    _checkpoint_problem,
    _checkpoints,
    _motor_short_goals,
    _validate_days,
    checkpoint_anchors,
    finalize_dataset,
    packing_config,
    run_unit,
    validate_goal_tree,
)
from realigned_pipeline.annotation.methods.sequential_goal_memory.gate import (
    is_decision_boundary,
)
from realigned_pipeline.annotation.stage_annotate import _require_pilot_review
from realigned_pipeline.lib.events import LabeledEvent, RawEvent
from realigned_pipeline.lib.semantic_actions import render_calls, semantic_events_from_labeled
from realigned_pipeline.lib.sequential_goal_memory_contract import (
    CHECKPOINT_FIELDS,
    CHECKPOINT_MAX_WORDS,
    THOUGHT_MAX_WORDS,
    render_checkpoint,
    system_prompt,
)
from realigned_pipeline.lib.sequential_conversations import build_sequential_conversations
from realigned_pipeline.lib.sequential_packing import (
    PackingConfig,
    boundary_events,
    packing_config_hash,
)

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from action_parser import parse_sequential_reply  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402


def _labeled(events: list[RawEvent], windows: list[int] | None = None):
    windows = windows or [0] * len(events)
    return [LabeledEvent(event, event.t_s, window) for event, window in zip(events, windows)]


def _frames(n: int = 4):
    return [SimpleNamespace(master_idx=i * 15, image=f"/frames/{i}.jpg") for i in range(n)]


def test_registry_exposes_standalone_method() -> None:
    method = load_method("sequential_goal_memory")
    assert method.input_kind == "days"
    assert method.finalize_dataset is not None
    assert method.requires_pilot_review


def test_full_run_requires_all_review_gates(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="blocked"):
        _require_pilot_review(None)
    review = {
        "reviewed_by": "human",
        **{key: True for key in (
            "goal_grounding", "causal_thoughts", "cross_day_links", "checkpoints",
            "action_provenance", "parser_validity",
        )},
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    _require_pilot_review(path)


@pytest.mark.parametrize(
    ("size", "delta"), [((1920, 1080), (960, -540)), ((1280, 720), (640, -360))]
)
def test_relative_movement_is_resolution_independent(size, delta) -> None:
    events = [RawEvent(0, 0.1, "move", dx=delta[0], dy=delta[1])]
    semantic, dispositions = semantic_events_from_labeled(
        _labeled(events), segment_id="s", recording_id="r", frames=_frames(1),
        frame_size=size,
    )
    assert semantic[0]["tool_calls"] == [
        {"action": "mouse_move_rel", "delta": [500, -500]}
    ]
    assert dispositions[0]["disposition"] == "emitted"


def test_motion_typing_and_drag_preserve_semantics_and_provenance() -> None:
    events = [
        RawEvent(0, 0.10, "move", dx=10, dy=5),
        RawEvent(1, 0.11, "move", dx=20, dy=-5),
        RawEvent(2, 1.20, "press", name="KeyH"),
        RawEvent(3, 1.21, "release", name="KeyH"),
        RawEvent(4, 1.22, "press", name="KeyI"),
        RawEvent(5, 1.23, "release", name="KeyI"),
        RawEvent(6, 2.30, "press", name="LMB"),
        RawEvent(7, 2.31, "move", dx=100, dy=0),
        RawEvent(8, 2.32, "release", name="LMB"),
    ]
    semantic, dispositions = semantic_events_from_labeled(
        _labeled(events, [0, 0, 1, 1, 1, 1, 2, 2, 2]),
        segment_id="s", recording_id="r", frames=_frames(3), frame_size=(1000, 500),
    )
    assert semantic[0]["tool_calls"] == [
        {"action": "mouse_move_rel", "delta": [30, 0]}
    ]
    assert semantic[1]["tool_calls"] == [{"action": "type", "text": "hi"}]
    assert [call["action"] for call in semantic[2]["tool_calls"]] == [
        "button_down", "mouse_move_rel", "button_up"
    ]
    assert len({raw_id for row in semantic for raw_id in row["raw_event_ids"]}) == len(events)
    assert all(row["disposition"] == "emitted" for row in dispositions)


def _day(n: int = 9) -> dict:
    events = []
    dispositions = []
    for index in range(n):
        event_id = f"sem_{index}"
        call = {"action": "mouse_move_rel", "delta": [index + 1, 0]}
        events.append({
            "semantic_event_id": event_id, "image": f"/frames/{index}.jpg",
            "assistant_action": (
                '<tool_call>\n{"name":"computer_use","arguments":'
                + str(call).replace("'", '"') + "}\n</tool_call>"
            ),
        })
        dispositions.append({
            "raw_event_id": f"raw_{index}", "semantic_event_id": event_id,
            "disposition": "emitted",
        })
    nodes = [
        {"goal_id": "long", "parent_id": None, "level": "long", "text": "Finish task",
         "provenance": "explicit", "start_event_index": 0, "end_event_index": n - 1},
        {"goal_id": "mid", "parent_id": "long", "level": "mid", "text": "Edit item",
         "provenance": "explicit", "start_event_index": 0, "end_event_index": n - 1},
        {"goal_id": "short", "parent_id": "mid", "level": "short", "text": "Open menu",
         "provenance": "proactive", "start_event_index": 0, "end_event_index": n - 1},
    ]
    cp_text = render_checkpoint({field: f"known {field.lower()}" for field in CHECKPOINT_FIELDS})
    memory_snapshots = []
    memory_before = (
        "No prior trajectory memory. Establish the current visible state and the "
        "user's active grounded intent without assuming any action has succeeded."
    )
    for index in range(n):
        memory_after = f"The task is active through semantic event {index}."
        memory_snapshots.append({
            "memory_snapshot_id": f"memory_{index}",
            "anchor_semantic_event_id": f"sem_{index}",
            "anchor_event_index": index,
            "memory_before": memory_before,
            "memory_after": memory_after,
        })
        memory_before = memory_after
    return {
        "day_tag": "u_2026-01-01", "user_id": "u", "date": "2026-01-01",
        "semantic_events": events, "event_dispositions": dispositions,
        "goal_nodes": nodes,
        "decisions": [{"anchor_semantic_event_id": "sem_0", "thought": "I should open the menu."}],
        "checkpoints": [{
            "anchor_semantic_event_id": "sem_4", "anchor_event_index": 4, "text": cp_text,
        }],
        "memory_snapshots": memory_snapshots,
    }


def test_goal_tree_validation_rejects_cycles_and_prompt_is_identical() -> None:
    nodes = _day()["goal_nodes"]
    validate_goal_tree(nodes, 9)
    broken = [dict(node) for node in nodes]
    broken[0]["parent_id"] = "short"
    with pytest.raises(ValueError, match="cyclic|only long goals may be roots"):
        validate_goal_tree(broken, 9)
    assert system_prompt() == SYSTEM_PROMPTS["sequential_goal_memory_v1"]


def test_motor_only_short_goals_require_repair() -> None:
    nodes = _day()["goal_nodes"]
    assert _motor_short_goals(nodes) == []
    broken = [dict(node) for node in nodes]
    broken[-1]["text"] = "Click the input field"
    assert _motor_short_goals(broken) == ["Click the input field"]


# ---------------------------------------------------------------------------
# decision-boundary pre-gate (annotation/methods/sequential_goal_memory/gate.py)
# ---------------------------------------------------------------------------

CLICK = [{"action": "left_click"}]
DAY_TAG = "u_2026-01-01"
# fraction_low == fraction_high and capacity * fraction is exact in binary, so
# the compaction threshold is pinned at 5 events and the anchors are hand-checkable.
CHECKPOINT_CFG = PackingConfig(capacity=10, fraction_low=0.5, fraction_high=0.5, seed=3)


def _gate_event(index: int, *, calls=None, t: float | None = None,
                segment: str = "s") -> dict:
    return {
        "semantic_event_id": f"sem_{index}", "segment_id": segment,
        "t_day_s": float(index) if t is None else t,
        "tool_calls": list(calls or CLICK),
    }


def _gate_tree(n: int, *, starts: tuple[int, ...] = (0,)) -> list[dict]:
    return [{"goal_id": f"g{start}", "level": "short",
             "start_event_index": start, "end_event_index": n - 1} for start in starts]


def test_gate_opens_at_the_first_event_and_closes_on_motor_continuation() -> None:
    events = [_gate_event(0), _gate_event(1)]
    tree = _gate_tree(2)
    assert is_decision_boundary(events, 0, tree)
    assert not is_decision_boundary(events, 1, tree)
    with pytest.raises(IndexError):
        is_decision_boundary(events, 2, tree)


def test_gate_opens_where_a_goal_node_starts() -> None:
    events = [_gate_event(i) for i in range(4)]
    assert is_decision_boundary(events, 2, _gate_tree(4, starts=(0, 2)))
    assert not is_decision_boundary(events, 2, _gate_tree(4, starts=(0, 3)))


def test_gate_opens_on_a_new_segment() -> None:
    events = [_gate_event(0), _gate_event(1, segment="s2")]
    assert is_decision_boundary(events, 1, _gate_tree(2))


def test_gate_opens_after_a_real_pause_only_beyond_the_threshold() -> None:
    events = [_gate_event(0, t=0.0), _gate_event(1, t=6.0)]
    assert is_decision_boundary(events, 1, _gate_tree(2))
    assert not is_decision_boundary(events, 1, _gate_tree(2), gap_s=6.0)
    assert is_decision_boundary(events, 1, _gate_tree(2), gap_s=1.0)


@pytest.mark.parametrize(
    ("previous_calls", "boundary"),
    [
        ([{"action": "key", "keys": ["enter"]}], True),
        ([{"action": "key", "keys": ["ctrl", "c"]}], True),
        ([{"action": "key", "keys": ["command"]}], True),
        ([{"action": "wait", "time": 1.0}], True),
        ([{"action": "key", "keys": ["a"]}], False),
        ([{"action": "key", "keys": ["shift", "a"]}], False),
        ([{"action": "type", "text": "hello"}], False),
    ],
)
def test_gate_opens_when_the_previous_action_handed_over_control(
        previous_calls, boundary) -> None:
    events = [_gate_event(0, calls=previous_calls), _gate_event(1)]
    assert is_decision_boundary(events, 1, _gate_tree(2)) is boundary


@pytest.mark.parametrize(
    ("before", "now", "boundary"),
    [(-120, 120, True), (120, -120, True), (120, 120, False), (120, 0, False)],
)
def test_gate_opens_when_a_scroll_reverses(before, now, boundary) -> None:
    events = [_gate_event(0, calls=[{"action": "scroll", "pixels": before}]),
              _gate_event(1, calls=[{"action": "scroll", "pixels": now}])]
    assert is_decision_boundary(events, 1, _gate_tree(2)) is boundary


# ---------------------------------------------------------------------------
# causal replay: predict-then-reveal agreement gate, motor memory-only variant
# ---------------------------------------------------------------------------

_CACHE_KINDS = (
    "02_goal_tree_", "03predict_", "03repair_", "03_causal_event_motor_",
    "03_causal_event_reveal_", "03_causal_event_", "04checkpoint_", "04repair_",
)


def _kind(cache_path: Path) -> str:
    for prefix in _CACHE_KINDS:
        if cache_path.name.startswith(prefix):
            return prefix.strip("_")
    raise AssertionError(f"unexpected labeler cache path {cache_path.name}")


def _values(text: str) -> dict[str, str]:
    return {field: f"{field}: {text}." for field in CHECKPOINT_FIELDS}


class _FakeLabeler:
    """Records (kind, event_id) per call and answers the prompt variant it was
    actually asked for. ``predictions`` overrides what the agreement-gate
    predictor guesses (default: exactly the real action, i.e. agreement);
    ``thoughts``/``checkpoints`` are per-event reply queues."""

    def __init__(self, *, predictions=None, thoughts=None, checkpoints=None,
                 event_ids=None, fail: bool = False):
        self.predictions = dict(predictions or {})
        self.thoughts = {key: list(value) for key, value in (thoughts or {}).items()}
        self.checkpoints = {key: list(value) for key, value in (checkpoints or {}).items()}
        self.event_ids = list(event_ids or [])
        self.fail = fail
        self.calls: list[tuple[str, str]] = []
        self.prompts: dict[str, str] = {}

    def call_json_full(self, _system, user, **kwargs):
        if self.fail:
            raise AssertionError("resume repeated a completed labeler call")
        kind = _kind(Path(kwargs["cache_path"]))
        result = SimpleNamespace(usage={"total_tokens": 11})
        if kind == "02_goal_tree":
            self.calls.append((kind, "day"))
            self.prompts[f"{kind}:day"] = user
            return ({"goals": [
                {"node_key": key, "parent_key": parent, "level": level, "text": text,
                 "provenance": "explicit", "grounding": "visible in the recording",
                 "start_event_id": self.event_ids[0],
                 "end_event_id": self.event_ids[-1]}
                for key, parent, level, text in (
                    ("L1", None, "long", "Finish the quarterly report"),
                    ("M1", "L1", "mid", "Update the revenue summary"),
                    ("S1", "M1", "short", "Correct the quarterly total"),
                )
            ]}, result)
        event_id = user.split("semantic event ", 1)[1].split()[0]
        self.calls.append((kind, event_id))
        self.prompts[f"{kind}:{event_id}"] = user
        if kind == "03predict":
            calls = self.predictions.get(event_id, CLICK)
            return ({"calls": [{"name": "computer_use", "arguments": call}
                               for call in calls]}, result)
        if kind in ("04checkpoint", "04repair"):
            queue = self.checkpoints.get(event_id) or []
            return ({"checkpoint": queue.pop(0) if queue
                     else _values(f"known through {event_id}")}, result)
        reply = {"memory_after": f"Causal memory after {event_id}.",
                 "references": [event_id]}
        if kind == "03_causal_event_motor":
            # A labeler that volunteers a thought where none was offered: the
            # method must drop it rather than train on it.
            return ({**reply, "thought": "Unsolicited motor commentary."}, result)
        queue = self.thoughts.get(event_id) or []
        thought = queue.pop(0) if queue else f"The visible state at {event_id} forces this."
        return ({**reply, "thought": thought}, result)


def _causal_day(tmp_path: Path, times=(0.0, 0.5, 10.0, 10.5)) -> list[dict]:
    """A day whose events 0 and 2 are decision boundaries (first event, real
    pause) and whose events 1 and 3 are motor continuations."""
    events = []
    for index, t_day_s in enumerate(times):
        image = tmp_path / f"{index}.jpg"
        Image.new("RGB", (32, 24), color=(index * 20, 0, 0)).save(image)
        events.append({
            "semantic_event_id": f"sem_{index}", "anchor_master_idx": index * 15,
            "end_master_idx": index * 15 + 5, "segment_id": "s", "image": str(image),
            "t_day_s": float(t_day_s), "tool_calls": list(CLICK),
            "raw_event_ids": [f"s:r{index}"], "action_spec": "computer_use_rel_norm_v1",
            "assistant_action": render_calls(CLICK),
        })
    return events


def _tree(n: int) -> list[dict]:
    return [
        {"goal_id": "long", "parent_id": None, "level": "long", "text": "Do the task",
         "provenance": "explicit", "start_event_index": 0, "end_event_index": n - 1},
        {"goal_id": "mid", "parent_id": "long", "level": "mid", "text": "Edit the item",
         "provenance": "explicit", "start_event_index": 0, "end_event_index": n - 1},
        {"goal_id": "short", "parent_id": "mid", "level": "short", "text": "Update the item",
         "provenance": "explicit", "start_event_index": 0, "end_event_index": n - 1},
    ]


def _context(labeler, tmp_path: Path, **params) -> MethodContext:
    return MethodContext(
        labeler=labeler, prompts=load_method("sequential_goal_memory").prompts,
        cache_dir=tmp_path / "calls", vlm_frame_height=24, jpeg_quality=80,
        params={"causal_context_events": 2, **params},
    )


_ITEM = {"day": SimpleNamespace(day_tag=DAY_TAG, user_id="u", date="2026-01-01")}


def _run_causal(labeler, tmp_path: Path, events, **params) -> dict:
    return _causal(_ITEM, _context(labeler, tmp_path, **params), tmp_path / "passes",
                   {"semantic_events": events}, {"goal_nodes": _tree(len(events))},
                   lambda _n: None)


def test_agreement_gate_reveals_a_thought_only_where_the_predictor_diverges(
        tmp_path: Path) -> None:
    events = _causal_day(tmp_path)
    # sem_0: the predictor guesses the real click -> agreement -> memory only.
    # sem_2: it guesses a different action -> divergence -> reveal.
    labeler = _FakeLabeler(predictions={"sem_2": [{"action": "right_click"}]})
    result = _run_causal(labeler, tmp_path, events)
    assert labeler.calls == [
        ("03predict", "sem_0"), ("03_causal_event_motor", "sem_0"),
        ("03_causal_event_motor", "sem_1"),
        ("03predict", "sem_2"), ("03_causal_event_reveal", "sem_2"),
        ("03_causal_event_motor", "sem_3"),
    ]
    assert (result["n_decision_boundaries"], result["n_predicted"],
            result["n_divergent"]) == (2, 2, 1)
    snapshots = result["memory_snapshots"]
    assert [row["is_decision_boundary"] for row in snapshots] == [True, False, True, False]
    assert [row["thought"] for row in snapshots] == [
        "", "", "The visible state at sem_2 forces this.", ""]
    assert [row["agreed"] for row in snapshots] == [True, None, False, None]
    assert snapshots[0]["predicted_calls"] == CLICK
    assert snapshots[1]["predicted_calls"] is None
    assert [row["thought_gating"] for row in snapshots] == ["agreement"] * 4
    assert [row["memory_before"] for row in snapshots[1:]] == [
        row["memory_after"] for row in snapshots[:-1]]
    assert snapshots[0]["memory_before"] == INITIAL_MEMORY
    decisions = result["decisions"]
    assert [(row["anchor_semantic_event_id"], row["gate"]) for row in decisions] == [
        ("sem_2", "divergence")]
    assert 0 < len(decisions[0]["thought"].split()) <= THOUGHT_MAX_WORDS
    # Resume: every event is cached, so a labeler that refuses to answer is fine.
    assert _run_causal(_FakeLabeler(fail=True), tmp_path, events) == result


def test_motor_events_are_never_offered_a_thought(tmp_path: Path) -> None:
    events = _causal_day(tmp_path)
    labeler = _FakeLabeler()
    result = _run_causal(labeler, tmp_path, events)
    assert [kind for kind, _ in labeler.calls].count("03_causal_event_reveal") == 0
    assert all(not row["thought"] for row in result["memory_snapshots"])
    assert result["decisions"] == []
    motor_prompt = labeler.prompts["03_causal_event_motor:sem_1"]
    assert '"thought"' not in motor_prompt
    assert '{"memory_after"' in motor_prompt


def test_boundary_gating_offers_thoughts_without_predicting(tmp_path: Path) -> None:
    events = _causal_day(tmp_path)
    labeler = _FakeLabeler(thoughts={"sem_2": ["I switch to the second document."]})
    result = _run_causal(labeler, tmp_path, events, thought_gating="boundary")
    assert labeler.calls == [
        ("03_causal_event", "sem_0"), ("03_causal_event_motor", "sem_1"),
        ("03_causal_event", "sem_2"), ("03_causal_event_motor", "sem_3"),
    ]
    assert (result["n_predicted"], result["n_divergent"]) == (0, 0)
    assert result["n_decision_boundaries"] == 2
    assert {row["gate"] for row in result["decisions"]} == {"offered"}
    assert [row["anchor_semantic_event_id"] for row in result["decisions"]] == [
        "sem_0", "sem_2"]
    assert all(row["predicted_calls"] is None and row["agreed"] is None
               for row in result["memory_snapshots"])


def test_unknown_thought_gating_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thought_gating"):
        _run_causal(_FakeLabeler(), tmp_path, _causal_day(tmp_path),
                    thought_gating="vibes")


def test_revealed_thought_over_budget_gets_one_corrective_retry(tmp_path: Path) -> None:
    events = _causal_day(tmp_path)
    long_thought = " ".join(["word"] * (THOUGHT_MAX_WORDS + 5))
    labeler = _FakeLabeler(
        predictions={"sem_0": [{"action": "double_click"}]},
        thoughts={"sem_0": [long_thought, "The dialog is already focused here."]},
    )
    result = _run_causal(labeler, tmp_path, events)
    assert ("03repair", "sem_0") in labeler.calls
    assert result["memory_snapshots"][0]["thought"] == "The dialog is already focused here."
    assert "CORRECTION REQUIRED" in labeler.prompts["03repair:sem_0"]


def test_revealed_thought_still_invalid_after_the_retry_is_fatal(tmp_path: Path) -> None:
    events = _causal_day(tmp_path)
    labeler = _FakeLabeler(predictions={"sem_0": [{"action": "double_click"}]},
                           thoughts={"sem_0": ["", ""]})
    with pytest.raises(ValueError, match="still invalid"):
        _run_causal(labeler, tmp_path, events)


# ---------------------------------------------------------------------------
# pass 03c: checkpoint projection
# ---------------------------------------------------------------------------

def _causal_doc(n: int) -> dict:
    memory_before = INITIAL_MEMORY
    snapshots = []
    for index in range(n):
        memory_after = f"The task is active through semantic event {index}."
        snapshots.append({
            "memory_snapshot_id": f"memory_{index}",
            "anchor_semantic_event_id": f"sem_{index}", "anchor_event_index": index,
            "memory_before": memory_before, "memory_after": memory_after,
        })
        memory_before = memory_after
    return {"memory_snapshots": snapshots, "thought_gating": "agreement",
            "n_decision_boundaries": 1, "n_predicted": 1, "n_divergent": 0}


def _run_checkpoints(labeler, tmp_path: Path, events, **params) -> dict:
    return _checkpoints(
        _ITEM, _context(labeler, tmp_path, checkpoint_capacity=CHECKPOINT_CFG.capacity,
                        checkpoint_fraction_low=CHECKPOINT_CFG.fraction_low,
                        checkpoint_fraction_high=CHECKPOINT_CFG.fraction_high,
                        packing_seed=CHECKPOINT_CFG.seed, **params),
        tmp_path / "passes", {"semantic_events": events}, {"goal_nodes": _tree(len(events))},
        _causal_doc(len(events)), lambda _n: None)


def test_checkpoint_capacity_has_no_silent_default() -> None:
    with pytest.raises(ValueError, match="--checkpoint-capacity"):
        packing_config({})
    assert packing_config({"checkpoint_capacity": "10", "checkpoint_fraction_low": "0.5",
                           "checkpoint_fraction_high": "0.5", "packing_seed": "3"}) \
        == CHECKPOINT_CFG


def test_checkpoint_anchors_are_the_packer_boundaries_plus_the_day_final() -> None:
    boundaries = boundary_events(9, day_tag=DAY_TAG, cfg=CHECKPOINT_CFG)
    assert boundaries == [4]
    assert checkpoint_anchors(9, day_tag=DAY_TAG, cfg=CHECKPOINT_CFG) == [4, 8]
    assert checkpoint_anchors(1, day_tag=DAY_TAG, cfg=CHECKPOINT_CFG) == [0]
    assert checkpoint_anchors(0, day_tag=DAY_TAG, cfg=CHECKPOINT_CFG) == []
    # Every packing's boundaries must be covered, not just packing 0's.
    multi = PackingConfig(capacity=12, seed=7, n_packings=4)
    anchors = checkpoint_anchors(60, day_tag=DAY_TAG, cfg=multi)
    for packing_index in range(multi.n_packings):
        assert set(boundary_events(60, day_tag=DAY_TAG, cfg=multi,
                                   packing_index=packing_index)) <= set(anchors)
    assert anchors == sorted(set(anchors)) and anchors[-1] == 59


def test_checkpoint_projection_folds_and_resumes(tmp_path: Path) -> None:
    events = _causal_day(tmp_path, times=tuple(float(i) for i in range(9)))
    labeler = _FakeLabeler()
    doc = _run_checkpoints(labeler, tmp_path, events)
    assert doc["anchor_event_indices"] == [4, 8]
    assert labeler.calls == [("04checkpoint", "sem_4"), ("04checkpoint", "sem_8")]
    assert doc["packing_config_hash"] == packing_config_hash(CHECKPOINT_CFG)
    rows = doc["checkpoints"]
    assert [row["anchor_event_index"] for row in rows] == [4, 8]
    assert [row["is_day_final"] for row in rows] == [False, True]
    assert [row["source_memory_snapshot_id"] for row in rows] == ["memory_4", "memory_8"]
    assert all(row["packing_config_hash"] == packing_config_hash(CHECKPOINT_CFG)
               for row in rows)
    assert all(row["text"] == render_checkpoint(row["values"]) for row in rows)
    assert all(row["active_goal_path"] == ["long", "mid", "short"] for row in rows)
    # Anchor 4 sees no previous checkpoint; anchor 8 must fold anchor 4's.
    assert "(none; this is the first compaction" in labeler.prompts["04checkpoint:sem_4"]
    assert rows[0]["values"]["Completed"] in labeler.prompts["04checkpoint:sem_8"]
    # Every anchor is cached: a resume must not re-call the labeler.
    assert _run_checkpoints(_FakeLabeler(fail=True), tmp_path, events) == doc


def test_checkpoint_budget_and_folding_are_validated() -> None:
    assert _checkpoint_problem(_values("known"), None) is None
    assert "seven fields" in _checkpoint_problem({"Completed": "x"}, None)
    fat = {**_values("known"), "Completed": " ".join(["word"] * (CHECKPOINT_MAX_WORDS + 1))}
    assert "word budget" in _checkpoint_problem(fat, None)
    history = {**_values("known"),
               "Completed": " ".join(f"Step {i} is done." for i in range(
                   COMPLETED_MAX_SENTENCES + 1))}
    assert _checkpoint_problem(history, None) is None  # no previous: nothing to fold
    assert "fold the previous" in _checkpoint_problem(history, _values("known"))


def test_checkpoint_projection_retries_once_then_fails(tmp_path: Path) -> None:
    events = _causal_day(tmp_path, times=tuple(float(i) for i in range(9)))
    fat = {**_values("known"),
           "Current state": " ".join(["word"] * (CHECKPOINT_MAX_WORDS + 1))}
    labeler = _FakeLabeler(checkpoints={"sem_4": [fat, _values("compact")]})
    doc = _run_checkpoints(labeler, tmp_path, events)
    assert labeler.calls[:2] == [("04checkpoint", "sem_4"), ("04repair", "sem_4")]
    assert doc["checkpoints"][0]["values"] == _values("compact")
    assert "CORRECTION REQUIRED" in labeler.prompts["04repair:sem_4"]

    hopeless = _FakeLabeler(checkpoints={"sem_4": [fat, fat]})
    with pytest.raises(ValueError, match="still invalid"):
        _run_checkpoints(hopeless, tmp_path / "again", events)


# ---------------------------------------------------------------------------
# publish / finalize validation of the 03c checkpoint shape
# ---------------------------------------------------------------------------

def _published_day(n: int = 9, *, cfg: PackingConfig = CHECKPOINT_CFG,
                   tag: str = DAY_TAG, prefix: str = "sem") -> dict:
    """One day exactly as pass 05 publishes it, with 03c checkpoints."""
    user_id, date = tag.split("_", 1)
    events, dispositions, snapshots = [], [], []
    memory_before = INITIAL_MEMORY
    for index in range(n):
        event_id = f"{prefix}_{index}"
        events.append({
            "semantic_event_id": event_id, "segment_id": "s", "day_tag": tag,
            "user_id": user_id, "date": date, "anchor_master_idx": index * 15,
            "end_master_idx": index * 15 + 5, "image": f"/frames/{index}.jpg",
            "t_day_s": float(index), "tool_calls": list(CLICK),
            "raw_event_ids": [f"{prefix}:r{index}"],
            "action_spec": "computer_use_rel_norm_v1",
            "assistant_action": render_calls(CLICK),
            "active_goal_path": [f"{prefix}_long", f"{prefix}_mid", f"{prefix}_short"],
        })
        dispositions.append({"raw_event_id": f"{prefix}:r{index}",
                             "semantic_event_id": event_id, "disposition": "emitted"})
        memory_after = f"The task is active through {event_id}."
        snapshots.append({
            "memory_snapshot_id": f"{prefix}_memory_{index}", "day_tag": tag,
            "user_id": user_id, "date": date, "anchor_semantic_event_id": event_id,
            "anchor_event_index": index, "visible_through_event_id": event_id,
            "image": f"/frames/{index}.jpg", "upcoming_tool_calls": list(CLICK),
            "raw_event_ids": [f"{prefix}:r{index}"], "memory_before": memory_before,
            "memory_after": memory_after, "references": [event_id], "thought": "",
            "is_decision_boundary": index == 0, "thought_gating": "agreement",
            "predicted_calls": None, "agreed": None, "checkpoint_id": None,
        })
        memory_before = memory_after
    nodes = []
    for level, parent, text in (("long", None, "Finish the report"),
                                ("mid", "long", "Edit the summary"),
                                ("short", "mid", "Update the total")):
        nodes.append({
            "goal_id": f"{prefix}_{level}",
            "parent_id": f"{prefix}_{parent}" if parent else None,
            "level": level, "text": text, "provenance": "explicit",
            "grounding": "visible in the recording",
            "start_event_index": 0, "end_event_index": n - 1,
            "start_semantic_event_id": f"{prefix}_0",
            "end_semantic_event_id": f"{prefix}_{n - 1}",
            "start_master_idx": 0, "end_master_idx": (n - 1) * 15 + 5,
            "user_id": user_id, "date": date,
        })
    checkpoints = []
    for anchor in checkpoint_anchors(n, day_tag=tag, cfg=cfg):
        values = _values(f"known through {prefix}_{anchor}")
        checkpoint_id = f"{prefix}_checkpoint_{anchor}"
        snapshots[anchor]["checkpoint_id"] = checkpoint_id
        checkpoints.append({
            "checkpoint_id": checkpoint_id, "day_tag": tag, "user_id": user_id,
            "anchor_semantic_event_id": f"{prefix}_{anchor}", "anchor_event_index": anchor,
            "anchor_master_idx": anchor * 15, "segment_id": "s",
            "visible_through_event_id": f"{prefix}_{anchor}",
            "active_goal_path": [f"{prefix}_long", f"{prefix}_mid", f"{prefix}_short"],
            "values": values, "text": render_checkpoint(values),
            "packing_config_hash": packing_config_hash(cfg),
            "is_day_final": anchor == n - 1,
            "source_memory_snapshot_id": f"{prefix}_memory_{anchor}",
        })
    return {
        "day_tag": tag, "user_id": user_id, "date": date,
        "semantic_events": events, "event_dispositions": dispositions,
        "goal_nodes": nodes, "memory_snapshots": snapshots, "checkpoints": checkpoints,
        "decisions": [{
            "decision_id": f"{prefix}_decision_0",
            "anchor_semantic_event_id": f"{prefix}_0", "anchor_event_index": 0,
            "visible_through_event_id": f"{prefix}_0", "references": [f"{prefix}_0"],
            "thought": "The report is already open here.", "gate": "divergence",
        }],
        "packing_config": {"capacity": cfg.capacity, "fraction_low": cfg.fraction_low,
                           "fraction_high": cfg.fraction_high, "seed": cfg.seed,
                           "n_packings": cfg.n_packings},
        "packing_config_hash": packing_config_hash(cfg),
        "thought_gating": "agreement", "decision_gap_s": 5.0,
        "gate_stats": {"n_decision_boundaries": 1, "n_predicted": 1, "n_divergent": 1},
    }


def test_published_day_validates_with_the_03c_checkpoint_shape() -> None:
    _validate_days([_published_day()])


def _drop_final_checkpoint(day: dict) -> None:
    row = day["checkpoints"].pop()
    day["memory_snapshots"][int(row["anchor_event_index"])]["checkpoint_id"] = None


def _move_checkpoint(day: dict) -> None:
    """A well-formed checkpoint at an anchor this packing never cuts on — what a
    day projected under a different capacity looks like."""
    row = day["checkpoints"][0]
    anchor = int(row["anchor_event_index"])
    day["memory_snapshots"][anchor]["checkpoint_id"] = None
    row.update(anchor_event_index=anchor - 1, anchor_semantic_event_id=f"sem_{anchor - 1}",
               visible_through_event_id=f"sem_{anchor - 1}",
               anchor_master_idx=(anchor - 1) * 15,
               source_memory_snapshot_id=f"memory_{anchor - 1}")
    day["memory_snapshots"][anchor - 1]["checkpoint_id"] = row["checkpoint_id"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_drop_final_checkpoint, "not the packing's anchors"),
        (_move_checkpoint, "not the packing's anchors"),
        (lambda day: day["checkpoints"][0].update(packing_config_hash="deadbeef"),
         "different packing config"),
        (lambda day: day["checkpoints"][-1].update(is_day_final=False),
         "is_day_final"),
        (lambda day: day["checkpoints"][0].update(text="<checkpoint>x</checkpoint>"),
         "does not render"),
        (lambda day: day["checkpoints"][0].update(
            source_memory_snapshot_id="sem_memory_0"), "name the memory snapshot"),
        (lambda day: day["memory_snapshots"][4].update(checkpoint_id="sem_checkpoint_8"),
         "link back"),
        (lambda day: day.update(packing_config_hash="deadbeef"),
         "does not match its packing_config"),
        (lambda day: day["checkpoints"][0]["values"].update(Completed=""),
         "invalid"),
        (lambda day: day["decisions"][0].pop("gate"), "invalid gate"),
    ],
)
def test_published_checkpoint_validation_rejects(mutate, match) -> None:
    day = _published_day()
    mutate(day)
    with pytest.raises(ValueError, match=match):
        _validate_days([day])


def test_finalize_carries_the_packing_config_gating_params_and_divergence_stats(
        tmp_path: Path) -> None:
    day = _published_day()
    unit_dir = tmp_path / "units" / day["day_tag"]
    unit_dir.mkdir(parents=True)
    (unit_dir / "05_publish.json").write_text(json.dumps(day))
    method = load_method("sequential_goal_memory")
    manifest = finalize_dataset(DatasetFinalizeContext(
        output_dir=tmp_path, units_dir=tmp_path / "units", calls_dir=tmp_path / "calls",
        method=method, labeler=_FakeLabeler(fail=True), model=None, no_cache=False,
        source_manifests={}))
    assert manifest["packing_config"] == day["packing_config"]
    assert manifest["packing_config_hash"] == packing_config_hash(CHECKPOINT_CFG)
    assert manifest["gating_params"] == {"thought_gating": "agreement",
                                        "decision_gap_s": 5.0}
    assert {key: manifest[key] for key in day["gate_stats"]} == day["gate_stats"]
    assert manifest["n_checkpoints"] == 2 == len(day["checkpoints"])
    assert manifest["checkpoints"] == "checkpoints.jsonl"
    rows = [json.loads(line) for line in
            (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    assert [row["is_day_final"] for row in rows] == [False, True]
    assert all(row["packing_config_hash"] == packing_config_hash(CHECKPOINT_CFG)
               and row["source_memory_snapshot_id"] for row in rows)


def test_finalize_refuses_an_artifact_that_mixes_packing_configs(tmp_path: Path) -> None:
    other = PackingConfig(capacity=20, fraction_low=0.5, fraction_high=0.5, seed=3)
    for day in (_published_day(),
                _published_day(tag="v_2026-01-02", prefix="two", cfg=other)):
        unit_dir = tmp_path / "units" / day["day_tag"]
        unit_dir.mkdir(parents=True)
        (unit_dir / "05_publish.json").write_text(json.dumps(day))
    with pytest.raises(ValueError, match="mixes packing configs"):
        finalize_dataset(DatasetFinalizeContext(
            output_dir=tmp_path, units_dir=tmp_path / "units",
            calls_dir=tmp_path / "calls", method=load_method("sequential_goal_memory"),
            labeler=_FakeLabeler(fail=True), model=None, no_cache=False,
            source_manifests={}))


# ---------------------------------------------------------------------------
# integration: the real published day docs feed the real Stage-04 packer
# ---------------------------------------------------------------------------

# capacity * fraction is exact in binary, so the compaction threshold is pinned
# at 3 events for every day and packing: boundaries [2, 4] over a 7-event day.
PACKED_CFG = PackingConfig(capacity=4, fraction_low=0.75, fraction_high=0.75, seed=5)
# events 0 and 3 are decision boundaries: the day's first event, and one after a
# 7-second pause. Everything else is motor continuation.
PACKED_TIMES = (0.0, 0.5, 1.0, 8.0, 8.5, 9.0, 9.5)


def _prepared_day(tmp_path: Path, *, tag: str, prefix: str,
                  times: tuple[float, ...]) -> tuple[dict, SimpleNamespace]:
    """A pass-01 document (and the day frames pass 02 samples) for a fake day.

    Only pass 01 is synthesized: it is the one pass that needs a real Stage-03
    filter artifact and frame store, and it makes no labeler call.
    """
    user_id, date = tag.split("_", 1)
    events, dispositions, frames = [], [], []
    for index, t_day_s in enumerate(times):
        image = tmp_path / f"{prefix}_{index}.jpg"
        Image.new("RGB", (32, 24), color=(index * 20, 40, 80)).save(image)
        events.append({
            "semantic_event_id": f"{prefix}_{index}", "day_event_index": index,
            "segment_id": "s", "recording_id": "r", "day_tag": tag, "user_id": user_id,
            "date": date, "anchor_master_idx": index * 15,
            "end_master_idx": index * 15 + 5, "image": str(image),
            "t_segment_s": float(t_day_s), "t_day_s": float(t_day_s),
            "raw_event_seqs": [index], "raw_event_ids": [f"{prefix}:r{index}"],
            "tool_calls": list(CLICK), "assistant_action": render_calls(CLICK),
            "action_spec": "computer_use_rel_norm_v1", "capture_size": [1000, 500],
        })
        dispositions.append({
            "raw_event_id": f"{prefix}:r{index}", "raw_event_seq": index, "kind": "press",
            "t_s": float(t_day_s), "semantic_event_id": f"{prefix}_{index}",
            "disposition": "emitted", "day_tag": tag, "user_id": user_id,
            "segment_id": "s",
        })
        frames.append(SimpleNamespace(image=str(image), day_idx=index,
                                      t_day_s=float(t_day_s), segment_id="s",
                                      master_idx=index * 15))
    prepared = {
        "pass": "prepare", "version": "semantic_events_v1",
        "input_hash": f"prepared_{prefix}", "day_tag": tag, "user_id": user_id,
        "date": date, "semantic_events": events, "event_dispositions": dispositions,
        "segment_stats": [{"segment_id": "s"}],
    }
    return prepared, SimpleNamespace(day_tag=tag, user_id=user_id, date=date, frames=frames)


def _publish_through_run_unit(tmp_path: Path, monkeypatch, *, tag: str, prefix: str,
                              cfg: PackingConfig = PACKED_CFG) -> tuple[dict, _FakeLabeler]:
    """Run passes 02..05 for real and return the published day document."""
    prepared, day = _prepared_day(tmp_path, tag=tag, prefix=prefix, times=PACKED_TIMES)
    monkeypatch.setattr(annotator, "_prepare", lambda item, ctx, pass_dir: prepared)
    labeler = _FakeLabeler(
        event_ids=[row["semantic_event_id"] for row in prepared["semantic_events"]],
        # the predictor diverges at the pause, so exactly one thought is revealed
        predictions={f"{prefix}_3": [{"action": "right_click"}]},
    )
    unit_dir = tmp_path / "units" / tag
    ctx = _context(
        labeler, tmp_path / prefix, day_units_dir=unit_dir, thought_gating="agreement",
        checkpoint_capacity=cfg.capacity, checkpoint_fraction_low=cfg.fraction_low,
        checkpoint_fraction_high=cfg.fraction_high, packing_seed=cfg.seed,
        n_packings=cfg.n_packings,
    )
    result = run_unit({"id": tag, "day": day, "row": {}}, ctx)
    assert result["n_divergent"] == 1 and result["n_decisions"] == 1
    return json.loads((unit_dir / "05_publish.json").read_text()), labeler


def test_published_days_pack_into_stage04_records(tmp_path: Path, monkeypatch) -> None:
    day_a, _ = _publish_through_run_unit(
        tmp_path, monkeypatch, tag="u_2026-01-01", prefix="one")
    day_b, _ = _publish_through_run_unit(
        tmp_path, monkeypatch, tag="u_2026-01-02", prefix="two")
    _validate_days([day_a, day_b])
    for day in (day_a, day_b):
        assert [row["anchor_event_index"] for row in day["checkpoints"]] == \
            checkpoint_anchors(7, day_tag=day["day_tag"], cfg=PACKED_CFG) == [2, 4, 6]
    long_a = next(node["goal_id"] for node in day_a["goal_nodes"]
                  if node["level"] == "long")
    long_b = next(node["goal_id"] for node in day_b["goal_nodes"]
                  if node["level"] == "long")
    link = {"mission_link_id": "mission_1", "user_id": "u", "from_goal_id": long_a,
            "to_goal_id": long_b, "relation": "continues", "evidence": "same report"}

    records, summary = build_sequential_conversations(
        [day_a, day_b], system_prompt=system_prompt(),
        parse_reply=parse_sequential_reply, cfg=PACKED_CFG, mission_links=[link])

    base = [row for row in records if not row["cross_day"]]
    assert len(base) == 6  # two days x spans [(0,1), (2,3), (4,6)]
    assert summary["packing_config"]["packing_config_hash"] == \
        packing_config_hash(PACKED_CFG) == day_a["packing_config_hash"]
    assert all(row["n_images"] <= PACKED_CFG.capacity for row in records)
    # Every semantic event of every day is trained on exactly once per packing.
    for day in (day_a, day_b):
        covered = [event_id for row in base if row["day_tag"] == day["day_tag"]
                   for event_id in row["semantic_event_ids"]]
        assert covered == [row["semantic_event_id"] for row in day["semantic_events"]]
    # The revealed thought survives into the record that owns its event.
    thinking = [row for row in base if row["n_thoughts"]]
    assert len(thinking) == 2 and all(row["n_thoughts"] == 1 for row in thinking)
    assert any("<think>The visible state at one_3 forces this.</think>" in block["text"]
               for row in thinking for message in row["messages"]
               for block in message["content"] if "text" in block)
    # Byte-identical handoff: record k's checkpoint turn is record k+1's opening state.
    episodes: dict[str, list[dict]] = {}
    for row in base:
        episodes.setdefault(str(row["episode_id"]), []).append(row)
    assert len(episodes) == 2
    for episode in episodes.values():
        for earlier, later in zip(episode, episode[1:]):
            handoff = earlier["messages"][-1]["content"][0]["text"]
            assert handoff.startswith("<checkpoint>") and handoff.endswith("</checkpoint>")
            assert earlier["checkpoint_out_id"] == later["checkpoint_in_id"]
            assert handoff in later["messages"][1]["content"][0]["text"]
        assert episode[-1]["checkpoint_out_id"] is None

    cross = [row for row in records if row["cross_day"]]
    assert len(cross) == 1 and summary["n_cross_day_records"] == 1
    assert cross[0]["mission_link_id"] == "mission_1"
    assert cross[0]["day_tag"] == day_b["day_tag"]
    day_final = next(row for row in day_a["checkpoints"] if row["is_day_final"])
    assert cross[0]["checkpoint_in_id"] == day_final["checkpoint_id"]
    assert day_final["text"] in cross[0]["messages"][1]["content"][0]["text"]


def test_packer_refuses_days_published_for_another_capacity(
        tmp_path: Path, monkeypatch) -> None:
    day, _ = _publish_through_run_unit(tmp_path, monkeypatch, tag="u_2026-01-01",
                                       prefix="one")
    other = PackingConfig(capacity=8, fraction_low=0.75, fraction_high=0.75, seed=5)
    with pytest.raises(ValueError, match="rerun annotation pass 03c"):
        build_sequential_conversations(
            [day], system_prompt=system_prompt(), parse_reply=parse_sequential_reply,
            cfg=other)
