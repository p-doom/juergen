"""Functional-QA tests: synthetic PUBLISHED artifacts written to tmp_path plus a
fake labeler. Nothing here touches a real dataset, a frame store or a network —
each test bends exactly one property of a known-good artifact and asserts the QA
script notices, then asserts the drafted review can never unblock a full run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from realigned_pipeline.annotation.qa_sequential_goal_memory import (
    FULL_RUN_GATES,
    PROBE_PASS_RATE_MIN,
    check_stage04,
    draft_review,
    load_parse_reply,
    run_qa,
    sample_boundary_checkpoints,
    would_pass_full_run_gate,
)
from realigned_pipeline.annotation.stage_annotate import _require_pilot_review
from realigned_pipeline.lib.sequential_goal_memory_contract import (
    CHECKPOINT_FIELDS,
    METHOD,
    goal_conditioning,
    render_checkpoint,
)

DAY = "u0_20260101"
PACKING_HASH = "cafe" * 16
INITIAL_MEMORY = "No prior trajectory memory. Establish the current visible state."


def _call(**arguments: Any) -> dict[str, Any]:
    return dict(arguments)


def _assistant_action(calls: list[dict[str, Any]]) -> str:
    return "\n".join(
        '<tool_call>\n' + json.dumps({"name": "computer_use", "arguments": call})
        + '\n</tool_call>' for call in calls
    )


def _checkpoint_text(completed: str = "Opened the editor.") -> str:
    values = {name: f"Known {name.lower()}." for name in CHECKPOINT_FIELDS}
    values["Completed"] = completed
    return render_checkpoint(values)


def _artifact(
    tmp_path: Path, *, n_events: int = 6, calls: dict[int, list[dict[str, Any]]] | None = None,
    thoughts: dict[int, str] | None = None, boundaries: dict[int, bool] | None = None,
    agreed: dict[int, bool] | None = None, memory: dict[int, str] | None = None,
    chain_break_at: int | None = None, checkpoints: dict[int, str] | None = None,
    goal_text: str = "Rename the invoice draft in the editor",
) -> Path:
    """A minimally valid published artifact, then the caller's one deviation."""
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    calls = calls or {}
    thoughts = thoughts or {}
    boundaries = {} if boundaries is None else boundaries
    checkpoints = checkpoints or {}
    events = []
    snapshots = []
    decisions = []
    memory_before = INITIAL_MEMORY
    for index in range(n_events):
        event_id = f"sem_{index:03d}"
        packet = calls.get(index) or [_call(action="left_click")]
        events.append({
            "semantic_event_id": event_id, "day_tag": DAY, "user_id": "u0",
            "date": "2026-01-01", "day_event_index": index, "segment_id": "seg",
            "image": f"ar:///store#{index}", "anchor_master_idx": index * 10,
            "tool_calls": packet, "assistant_action": _assistant_action(packet),
            "active_goal_path": ["goal_long", "goal_mid"],
        })
        memory_after = (memory or {}).get(index, f"State after event {index}.")
        thought = thoughts.get(index, "")
        snapshot: dict[str, Any] = {
            "memory_snapshot_id": f"memory_{index:03d}", "day_tag": DAY, "user_id": "u0",
            "anchor_semantic_event_id": event_id, "anchor_event_index": index,
            "visible_through_event_id": event_id, "segment_id": "seg",
            "active_goal_path": ["goal_long", "goal_mid"],
            "memory_before": ("chain is broken here" if index == chain_break_at
                              else memory_before),
            "memory_after": memory_after, "thought": thought,
            "references": [event_id], "upcoming_tool_calls": packet,
            "image": f"ar:///store#{index}", "checkpoint_id": None,
        }
        if index in boundaries:
            snapshot["is_decision_boundary"] = boundaries[index]
        if agreed and index in agreed:
            snapshot["agreed"] = agreed[index]
        snapshots.append(snapshot)
        memory_before = memory_after
        if thought:
            decisions.append({
                "decision_id": f"decision_{index:03d}", "day_tag": DAY, "user_id": "u0",
                "anchor_semantic_event_id": event_id, "anchor_event_index": index,
                "visible_through_event_id": event_id, "thought": thought,
                "references": [event_id], "gate": "divergence",
            })
    checkpoint_rows = []
    for index, completed in sorted(checkpoints.items()):
        checkpoint_rows.append({
            "checkpoint_id": f"checkpoint_{index:03d}", "day_tag": DAY, "user_id": "u0",
            "anchor_semantic_event_id": f"sem_{index:03d}", "anchor_event_index": index,
            "visible_through_event_id": f"sem_{index:03d}", "segment_id": "seg",
            "active_goal_path": ["goal_long", "goal_mid"],
            "text": _checkpoint_text(completed),
            "packing_config_hash": PACKING_HASH, "is_day_final": index == n_events - 1,
            "source_memory_snapshot_id": f"memory_{index:03d}",
        })
    goals = [
        {"goal_id": "goal_long", "parent_id": None, "level": "long",
         "text": "Finish the invoice paperwork", "provenance": "explicit",
         "start_semantic_event_id": "sem_000", "start_event_index": 0,
         "end_event_index": n_events - 1, "user_id": "u0", "date": "2026-01-01"},
        {"goal_id": "goal_mid", "parent_id": "goal_long", "level": "mid",
         "text": goal_text, "provenance": "explicit",
         "start_semantic_event_id": "sem_000", "start_event_index": 0,
         "end_event_index": n_events - 1, "user_id": "u0", "date": "2026-01-01"},
    ]
    _write(artifact / "semantic_events.jsonl", events)
    _write(artifact / "memory_snapshots.jsonl", snapshots)
    _write(artifact / "decision_thoughts.jsonl", decisions)
    _write(artifact / "checkpoints.jsonl", checkpoint_rows)
    _write(artifact / "goal_nodes.jsonl", goals)
    _write(artifact / "mission_links.jsonl", [])
    (artifact / "manifest.json").write_text(json.dumps({
        "artifact_type": "realigned_goals", "method": METHOD, "method_schema_version": 3,
        "prompt_versions": {"causal_replay": "causal_replay_v4"},
        "packing_config": {"capacity": 8, "hash": PACKING_HASH},
    }))
    return artifact


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _qa(artifact: Path, **kwargs: Any) -> dict[str, Any]:
    report, _draft = run_qa(artifact, stage04_chat=kwargs.pop("stage04_chat", None),
                            sample=kwargs.pop("sample", 20),
                            no_llm=kwargs.pop("no_llm", True),
                            labeler=kwargs.pop("labeler", None), **kwargs)
    return report


# ---------------------------------------------------------------------------
# clean baseline
# ---------------------------------------------------------------------------


def test_clean_artifact_reports_no_violations(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path, thoughts={0: "The invoice row is highlighted so renaming can start."},
        boundaries={0: True, 1: False, 2: False, 3: True, 4: False, 5: False},
        agreed={0: False, 3: True},
        checkpoints={3: "Opened the draft and highlighted the invoice row."},
    )
    report = _qa(artifact)
    assert report["ok"], report["violations"]
    checks = report["checks"]
    assert checks["parser"]["n_failures"] == 0
    assert checks["memory_chain"] == {**checks["memory_chain"], "n_breaks": 0, "n_links": 5}
    assert checks["references"]["n_future"] == 0
    assert checks["leaks"]["n_texts_with_leak"] == 0
    assert checks["thoughts"]["density"] == pytest.approx(1 / 6, abs=1e-6)
    assert checks["thoughts"]["motor_thought_density"] == 0.0
    assert checks["thoughts"]["divergence_rate"] == 0.5
    assert checks["checkpoints"]["n_folding_regressions"] == 0
    assert checks["stage04"]["status"] == "skipped"
    assert checks["resumability"]["status"] == "skipped"


def test_old_schema_artifact_reports_unknowns_instead_of_crashing(tmp_path: Path) -> None:
    """Rows predating is_decision_boundary/agreed/gate/packing fields."""
    artifact = _artifact(tmp_path, thoughts={2: "The dialog is open, so confirm it."},
                         checkpoints={3: "Opened the dialog."})
    for name in ("memory_snapshots.jsonl", "checkpoints.jsonl", "decision_thoughts.jsonl"):
        path = artifact / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        stripped = ["is_decision_boundary", "agreed", "gate", "packing_config_hash",
                    "is_day_final", "source_memory_snapshot_id"]
        _write(path, [{k: v for k, v in row.items() if k not in stripped} for row in rows])
    report = _qa(artifact)
    assert report["ok"], report["violations"]
    thoughts = report["checks"]["thoughts"]
    assert thoughts["n_unknown_boundary"] == 6
    assert thoughts["motor_thought_density"] is None
    assert thoughts["divergence_rate"] is None
    assert thoughts["gate_counts"] == {"unknown": 1}
    assert report["checks"]["checkpoints"]["n_unknown_day_final"] == 1
    missing = report["schema"]["rows_missing_new_fields"]
    assert missing["memory_snapshots"] == {"is_decision_boundary": 6, "agreed": 6}
    assert missing["checkpoints"]["packing_config_hash"] == 1
    assert missing["decisions"] == {"gate": 1}


# ---------------------------------------------------------------------------
# leak detector
# ---------------------------------------------------------------------------


def test_leak_detector_flags_typed_text_from_the_future(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path, n_events=4,
        calls={2: [_call(action="type", text="Quarterly-Zebrafish-Budget.xlsx")]},
        memory={0: "The user will save this as Quarterly-Zebrafish-Budget.xlsx later."},
    )
    report = _qa(artifact)
    leaks = report["checks"]["leaks"]
    assert leaks["n_texts_with_leak"] == 1
    assert leaks["leak_rate"] > 0
    assert leaks["worst"][0]["event_index"] == 0
    assert leaks["worst"][0]["kind"] == "memory_after"
    assert "quarterly-zebrafish-budget.xlsx" in leaks["worst"][0]["leaked"]
    assert any(row["check"] == "leaks" for row in report["violations"])
    assert not report["ok"]


def test_leak_detector_accepts_legitimately_past_and_goal_strings(tmp_path: Path) -> None:
    """The same string is fine once typed earlier, in the current packet, or when
    the active goal text already names it."""
    typed = [_call(action="type", text="Quarterly-Zebrafish-Budget.xlsx")]
    past = _artifact(
        tmp_path / "past", n_events=4, calls={1: typed},
        memory={2: "Saved the file as Quarterly-Zebrafish-Budget.xlsx."},
    )
    assert _qa(past)["checks"]["leaks"]["n_texts_with_leak"] == 0
    current = _artifact(
        tmp_path / "current", n_events=4, calls={2: typed},
        memory={2: "Typing Quarterly-Zebrafish-Budget.xlsx is the intended action."},
    )
    assert _qa(current)["checks"]["leaks"]["n_texts_with_leak"] == 0
    goal = _artifact(
        tmp_path / "goal", n_events=4, calls={3: typed},
        goal_text="Save the sheet as Quarterly-Zebrafish-Budget.xlsx",
        memory={0: "The target name Quarterly-Zebrafish-Budget.xlsx comes from the goal."},
    )
    assert _qa(goal)["checks"]["leaks"]["n_texts_with_leak"] == 0


def test_leak_detector_flags_future_key_combos_in_a_checkpoint(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path, n_events=5,
        calls={4: [_call(action="key", keys=["ctrl", "shift", "escape"])]},
        checkpoints={2: "Pressed nothing yet; ctrl+shift+escape is still to come."},
    )
    leaks = _qa(artifact)["checks"]["leaks"]
    assert leaks["n_leaks_by_kind"]["checkpoint"] == 1
    assert "ctrl+shift+escape" in leaks["worst"][0]["leaked"]


# ---------------------------------------------------------------------------
# memory chain + references
# ---------------------------------------------------------------------------


def test_memory_chain_break_is_detected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, chain_break_at=3)
    chain = _qa(artifact)["checks"]["memory_chain"]
    assert chain["n_breaks"] == 1
    assert chain["breaks"][0] == {**chain["breaks"][0], "kind": "chain_break",
                                 "event_index": 3}


def test_missing_snapshot_and_future_reference_are_detected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=4)
    path = artifact / "memory_snapshots.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[1]["references"] = ["sem_003", "sem_009"]
    _write(path, [rows[0], rows[1], rows[3]])
    report = _qa(artifact)
    chain = report["checks"]["memory_chain"]
    assert {row["kind"] for row in chain["breaks"]} >= {"event_without_snapshot",
                                                       "non_contiguous_anchor"}
    refs = report["checks"]["references"]
    assert (refs["n_future"], refs["n_unresolved"]) == (1, 1)
    assert not report["ok"]


# ---------------------------------------------------------------------------
# thought + checkpoint metrics
# ---------------------------------------------------------------------------


def test_motor_thought_density_violation_is_detected(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path, n_events=4,
        thoughts={1: "This drag continues the selection I already started."},
        boundaries={0: True, 1: False, 2: False, 3: False},
    )
    report = _qa(artifact)
    thoughts = report["checks"]["thoughts"]
    assert thoughts["n_motor_events"] == 3
    assert thoughts["n_motor_thoughts"] == 1
    assert thoughts["motor_thought_density"] == pytest.approx(1 / 3)
    assert thoughts["motor_thoughts"][0]["event_index"] == 1
    assert any(row["detail"].startswith("motor event") for row in report["violations"])


def test_thought_word_budget_and_parrot_score(tmp_path: Path) -> None:
    parrot = "Left click. Left click on the button. Click the button."
    artifact = _artifact(tmp_path, n_events=3, thoughts={
        0: " ".join(["word"] * 61),
        1: parrot,
    })
    thoughts = _qa(artifact)["checks"]["thoughts"]
    assert thoughts["n_over_budget"] == 1
    assert thoughts["over_budget"][0]["n_words"] == 61
    assert thoughts["word_counts"]["max"] == 61
    assert thoughts["parrot_worst"][0]["event_index"] == 1
    assert thoughts["parrot_worst"][0]["unigram"] > 0.5


def test_checkpoint_budget_violation_is_detected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=4,
                         checkpoints={2: " ".join(["completed"] * 200)})
    report = _qa(artifact)
    checkpoints = report["checks"]["checkpoints"]
    assert checkpoints["n_over_budget"] == 1
    assert checkpoints["over_budget"][0]["n_words"] > 180
    assert any("CHECKPOINT_MAX_WORDS" in row["detail"] for row in report["violations"])


def test_checkpoint_empty_field_is_detected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=4, checkpoints={2: "Opened the draft."})
    path = artifact / "checkpoints.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["values"] = {name: "Known." for name in CHECKPOINT_FIELDS}
    row["values"]["Next step"] = ""
    _write(path, [row])
    checkpoints = _qa(artifact)["checks"]["checkpoints"]
    assert checkpoints["n_incomplete_fields"] == 1
    assert checkpoints["incomplete_fields"][0]["empty"] == ["Next step"]


def test_folding_growth_regression_is_detected_and_folded_chain_is_not(tmp_path: Path) -> None:
    growing = _artifact(tmp_path / "growing", n_events=8, checkpoints={
        2: " ".join(["step"] * 10),
        4: " ".join(["step"] * 20),
        6: " ".join(["step"] * 40),
    })
    report = _qa(growing)
    checkpoints = report["checks"]["checkpoints"]
    kinds = {row["kind"] for row in checkpoints["folding_regressions"]}
    assert "monotonic_growth" in kinds
    assert checkpoints["chains"][0]["completed_word_counts"] == [10, 20, 40]
    assert checkpoints["chains"][0]["growth_ratio"] == 4.0
    assert not report["ok"]

    folded = _artifact(tmp_path / "folded", n_events=8, checkpoints={
        2: "Opened the draft.", 4: "Opened the draft, then renamed two rows.",
        6: "Renamed the rows and saved.",
    })
    assert _qa(folded)["checks"]["checkpoints"]["n_folding_regressions"] == 0


def test_completed_restating_the_previous_anchor_is_a_regression(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=8, checkpoints={
        2: "Opened the draft.",
        4: "Opened the draft. Then renamed the first row.",
    })
    regressions = _qa(artifact)["checks"]["checkpoints"]["folding_regressions"]
    assert [row["kind"] for row in regressions] == ["restates_previous"]


# ---------------------------------------------------------------------------
# stage-04 chat.jsonl round trip
# ---------------------------------------------------------------------------


def _record(conversation_id: str, *, episode: str, segment: int, images: int,
            goal_text: str, checkpoint_in: str | None, checkpoint_out: str | None,
            bad_action: bool = False) -> dict[str, Any]:
    action = ("<tool_call>\n{\"name\":\"computer_use\",\"arguments\":"
              "{\"action\":\"left_click\"}}\n</tool_call>")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [
            {"type": "text", "text": goal_conditioning(goal_text, checkpoint_in)},
            {"type": "image", "image": "ar:///store#0"}]},
    ]
    for index in range(images - 1):
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": "prose that cannot parse" if bad_action else action}]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": f"ar:///store#{index + 1}"}]})
    if checkpoint_out is not None:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": checkpoint_out}]})
    return {
        "conversation_id": conversation_id, "episode_id": episode,
        "segment_index": segment, "n_images": images, "messages": messages,
        "day_tag": DAY,
    }


def _chat(tmp_path: Path, records: list[dict[str, Any]], *,
          capacity: int | None = 8) -> Path:
    directory = tmp_path / "stage04"
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "chat.jsonl", records)
    if capacity is not None:
        (directory / "manifest.json").write_text(json.dumps(
            {"packing_config": {"capacity": capacity}}))
    return directory / "chat.jsonl"


def test_stage04_round_trip_handoff_and_capacity_pass(tmp_path: Path) -> None:
    text = _checkpoint_text()
    chat = _chat(tmp_path, [
        _record("e_s000", episode="e", segment=0, images=3, goal_text="Do the task",
                checkpoint_in=None, checkpoint_out=text),
        _record("e_s001", episode="e", segment=1, images=3, goal_text="Do the task",
                checkpoint_in=text, checkpoint_out=None),
    ])
    parse_reply, reason = load_parse_reply()
    assert parse_reply is not None, reason
    stage04 = check_stage04(chat, parse_reply, reason)
    assert stage04["parse"] == {**stage04["parse"], "n_failures": 0}
    assert (stage04["n_handoffs"], stage04["n_handoff_breaks"]) == (1, 0)
    assert stage04["n_over_capacity"] == 0
    assert stage04["n_image_count_mismatches"] == 0


def test_stage04_parser_round_trip_failure_is_detected(tmp_path: Path) -> None:
    chat = _chat(tmp_path, [
        _record("e_s000", episode="e", segment=0, images=3, goal_text="Do the task",
                checkpoint_in=None, checkpoint_out=None, bad_action=True),
    ])
    parse_reply, reason = load_parse_reply()
    stage04 = check_stage04(chat, parse_reply, reason)
    assert stage04["parse"]["n_failures"] == 2
    assert "conversation_id" in stage04["parse"]["failures"][0]


def test_stage04_handoff_break_and_capacity_overflow_are_detected(tmp_path: Path) -> None:
    chat = _chat(tmp_path, [
        _record("e_s000", episode="e", segment=0, images=5, goal_text="Do the task",
                checkpoint_in=None, checkpoint_out=_checkpoint_text("Opened the draft.")),
        _record("e_s001", episode="e", segment=1, images=5, goal_text="Do the task",
                checkpoint_in=_checkpoint_text("Opened the draft, edited later."),
                checkpoint_out=None),
    ], capacity=4)
    parse_reply, reason = load_parse_reply()
    stage04 = check_stage04(chat, parse_reply, reason)
    assert stage04["n_handoff_breaks"] == 1
    assert stage04["handoff_breaks"][0]["to"] == "e_s001"
    assert stage04["n_over_capacity"] == 2
    assert stage04["capacity"] == 4


def test_stage04_parser_skip_path_reports_a_reason(tmp_path: Path) -> None:
    chat = _chat(tmp_path, [
        _record("e_s000", episode="e", segment=0, images=2, goal_text="Do the task",
                checkpoint_in=None, checkpoint_out=None),
    ])
    stage04 = check_stage04(chat, None, "ModuleNotFoundError: no action_parser")
    assert stage04["parse"]["status"] == "skipped"
    assert "action_parser" in stage04["parse"]["reason"]
    assert stage04["status"] == "ran"


def test_stage04_check_feeds_parser_validity_gate(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=4, checkpoints={2: "Opened the draft."})
    chat = _chat(tmp_path, [
        _record("e_s000", episode="e", segment=0, images=3, goal_text="Do the task",
                checkpoint_in=None, checkpoint_out=None, bad_action=True),
    ])
    report, draft = run_qa(artifact, stage04_chat=chat, sample=20, no_llm=True, labeler=None)
    assert draft["parser_validity"] is False
    assert report["checks"]["parser"]["n_failures"] == 0
    assert any(row["check"] == "stage04" for row in report["violations"])


# ---------------------------------------------------------------------------
# resumability probe (fake labeler)
# ---------------------------------------------------------------------------


class _FakeLabeler:
    """Stands in for annotation/lib/labeler.Labeler: records the probe context so
    tests can assert the fresh context carries nothing but goal + checkpoint +
    one screenshot, and answers the judge from a fixed verdict script."""

    def __init__(self, verdicts: list[bool], *, raise_on: int | None = None) -> None:
        self.verdicts = list(verdicts)
        self.raise_on = raise_on
        self.n_text_calls = 0
        self.probes: list[dict[str, Any]] = []
        self.judges: list[str] = []

    def call_text(self, system: str, user: str, images: list[Any] | None = None,
                  **_kwargs: Any) -> str:
        self.n_text_calls += 1
        if self.raise_on is not None and self.n_text_calls - 1 == self.raise_on:
            raise RuntimeError("labeler exploded")
        self.probes.append({"system": system, "user": user,
                            "n_images": len(images or [])})
        return f"I would continue from checkpoint {len(self.probes)}."

    def call_json(self, _system: str, user: str, **_kwargs: Any) -> dict[str, Any]:
        self.judges.append(user)
        index = len(self.judges) - 1
        compatible = self.verdicts[index] if index < len(self.verdicts) else True
        return {"compatible": compatible, "reason": "scripted verdict"}


@pytest.fixture(autouse=True)
def _no_frame_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic artifacts point at ``ar://`` URIs with no store behind them;
    the probe only needs *a* payload per boundary screenshot."""
    monkeypatch.setattr(
        "realigned_pipeline.annotation.qa_sequential_goal_memory.frames_to_data_urls",
        lambda refs, _height, _quality: [f"data:image/jpeg;base64,{ref}" for ref in refs],
    )


def _probe_artifact(tmp_path: Path, anchors: list[int], n_events: int = 12) -> Path:
    return _artifact(tmp_path, n_events=n_events,
                     checkpoints={index: f"Reached anchor {index}." for index in anchors})


def test_resumability_probe_builds_a_fresh_context_and_scores_pass_rate(tmp_path: Path) -> None:
    artifact = _probe_artifact(tmp_path, [2, 5, 8])
    labeler = _FakeLabeler([True, True, False])
    report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=False,
                           labeler=labeler)
    resume = report["checks"]["resumability"]
    assert (resume["n_probed"], resume["n_pass"], resume["n_error"]) == (3, 2, 0)
    assert resume["pass_rate"] == pytest.approx(2 / 3)
    assert draft["checkpoints"] is False  # 0.67 < 0.8
    for probe in labeler.probes:
        assert probe["n_images"] == 1
        assert probe["user"].startswith("GOAL: Rename the invoice draft in the editor")
        assert "<checkpoint>" in probe["user"]
        assert "State after event" not in probe["user"]  # no rolling memory leaks in
    assert all("computer_use" not in probe["user"] for probe in labeler.probes)
    assert "left_click" in labeler.judges[0]  # the judge sees the true continuation


def test_resumability_probe_pass_rate_at_threshold_drafts_true(tmp_path: Path) -> None:
    artifact = _probe_artifact(tmp_path, [2, 4, 6, 8, 10])
    labeler = _FakeLabeler([True, True, True, True, False])
    _report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=False,
                            labeler=labeler)
    assert draft["checkpoints"] is True
    assert PROBE_PASS_RATE_MIN == 0.8


def test_resumability_probe_counts_labeler_errors_as_failures(tmp_path: Path) -> None:
    artifact = _probe_artifact(tmp_path, [2, 5, 8])
    labeler = _FakeLabeler([True, True, True], raise_on=1)
    report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=False,
                           labeler=labeler)
    resume = report["checks"]["resumability"]
    assert (resume["n_error"], resume["n_pass"]) == (1, 2)
    assert resume["pass_rate"] == pytest.approx(2 / 3)
    assert draft["checkpoints"] is False


def test_probe_skips_day_final_checkpoints_and_samples_deterministically(tmp_path: Path) -> None:
    artifact_dir = _artifact(tmp_path, n_events=6, checkpoints={
        1: "Anchor one.", 3: "Anchor three.", 5: "Day final."})
    from realigned_pipeline.annotation.qa_sequential_goal_memory import load_artifact

    artifact = load_artifact(artifact_dir)
    selected = sample_boundary_checkpoints(artifact, 20)
    # checkpoint_005 is the day-final anchor: nothing in this day resumes from it.
    assert {row["checkpoint_id"] for _day, row in selected} == {
        "checkpoint_001", "checkpoint_003"}
    # Hash-ordered, so the same artifact always probes the same subset — and a
    # smaller --sample is a prefix of a larger one.
    repeated = sample_boundary_checkpoints(load_artifact(artifact_dir), 20)
    assert [row["checkpoint_id"] for _day, row in repeated] == [
        row["checkpoint_id"] for _day, row in selected]
    assert [row["checkpoint_id"] for _day, row
            in sample_boundary_checkpoints(artifact, 1)] == [selected[0][1]["checkpoint_id"]]
    assert sample_boundary_checkpoints(artifact, 0) == []


def test_no_llm_leaves_the_checkpoint_gate_null(tmp_path: Path) -> None:
    artifact = _probe_artifact(tmp_path, [2, 5])
    labeler = _FakeLabeler([True, True])
    report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=True,
                           labeler=labeler)
    assert labeler.probes == []
    assert report["checks"]["resumability"] == {"status": "skipped", "reason": "--no-llm"}
    assert draft["checkpoints"] is None
    assert "probe skipped" in draft["basis"]["checkpoints"]


# ---------------------------------------------------------------------------
# review draft + full-run gate
# ---------------------------------------------------------------------------


def test_draft_gates_reflect_the_deterministic_checks(tmp_path: Path) -> None:
    clean = _artifact(tmp_path / "clean", n_events=4, checkpoints={2: "Opened."})
    _report, draft = run_qa(clean, stage04_chat=None, sample=20, no_llm=True, labeler=None)
    assert draft["action_provenance"] is True
    assert draft["parser_validity"] is True
    assert [draft[gate] for gate in ("goal_grounding", "causal_thoughts",
                                     "cross_day_links")] == [None, None, None]
    assert draft["reviewed_by"] is None
    assert set(FULL_RUN_GATES) <= set(draft)
    assert all(draft["basis"][gate] for gate in FULL_RUN_GATES)

    leaky = _artifact(
        tmp_path / "leaky", n_events=4,
        calls={3: [_call(action="type", text="Quarterly-Zebrafish-Budget.xlsx")]},
        memory={0: "Will type Quarterly-Zebrafish-Budget.xlsx eventually."})
    _report, leaky_draft = run_qa(leaky, stage04_chat=None, sample=20, no_llm=True,
                                  labeler=None)
    assert leaky_draft["action_provenance"] is False
    assert leaky_draft["parser_validity"] is True

    broken = _artifact(tmp_path / "broken", n_events=4, chain_break_at=2)
    _report, broken_draft = run_qa(broken, stage04_chat=None, sample=20, no_llm=True,
                                   labeler=None)
    assert broken_draft["action_provenance"] is False


def test_emitted_review_draft_never_passes_the_full_run_gate(tmp_path: Path) -> None:
    """The real gate is stage_annotate._require_pilot_review; a draft that made it
    through would silently unblock a corpus run."""
    artifact = _probe_artifact(tmp_path, [2, 5, 8])
    labeler = _FakeLabeler([True, True, True])
    _report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=False,
                            labeler=labeler)
    assert draft["checkpoints"] is True  # every auto-gate at its best
    assert draft["action_provenance"] is True
    assert draft["parser_validity"] is True
    assert not would_pass_full_run_gate(draft)
    path = artifact / "review_draft.json"
    assert json.loads(path.read_text()) == draft
    with pytest.raises(SystemExit, match="reviewed_by or passing gates"):
        _require_pilot_review(path)

    # Only a human filling in the nulls AND reviewed_by opens the gate.
    approved = {**draft, "reviewed_by": "human",
                **{gate: True for gate in FULL_RUN_GATES}}
    approved_path = artifact / "review_approved.json"
    approved_path.write_text(json.dumps(approved))
    _require_pilot_review(approved_path)
    assert would_pass_full_run_gate(approved)


def test_draft_writer_refuses_a_gate_satisfying_draft() -> None:
    report = {
        "artifact_dir": "/nowhere",
        "counts": {"n_goal_nodes": 1, "n_semantic_events": 1, "n_mission_links": 0,
                   "n_days": 1},
        "checks": {
            "parser": {"status": "ran", "n_failures": 0, "n_action_texts": 1,
                       "n_checkpoint_texts": 1},
            "memory_chain": {"status": "ran", "n_breaks": 0},
            "references": {"status": "ran", "n_future": 0, "n_unresolved": 0},
            "leaks": {"status": "ran", "n_texts_with_leak": 0, "leak_rate": 0.0,
                      "n_texts": 1},
            "thoughts": {"status": "ran", "density": 0.5, "motor_thought_density": 0.0,
                         "divergence_rate": 0.5, "parrot_mean_unigram": 0.0,
                         "n_over_budget": 0},
            "checkpoints": {"status": "ran", "n_over_budget": 0,
                            "n_folding_regressions": 0},
            "stage04": {"status": "skipped", "reason": "no --stage04-chat"},
            "resumability": {"status": "ran", "pass_rate": 1.0, "n_probed": 1,
                             "n_error": 0},
        },
    }
    draft = draft_review(report)
    assert draft["reviewed_by"] is None
    assert not would_pass_full_run_gate(draft)


def test_qa_writes_both_outputs_into_the_artifact_dir(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=4, checkpoints={2: "Opened."})
    report, draft = run_qa(artifact, stage04_chat=None, sample=20, no_llm=True,
                           labeler=None)
    assert json.loads((artifact / "qa_report.json").read_text()) == report
    assert json.loads((artifact / "review_draft.json").read_text()) == draft
    assert report["artifact_dir"] == str(artifact.resolve())


def test_wrong_method_artifact_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, n_events=3)
    (artifact / "manifest.json").write_text(json.dumps({"method": "describe_extract"}))
    with pytest.raises(SystemExit, match="not 'sequential_goal_memory'"):
        run_qa(artifact, stage04_chat=None, sample=20, no_llm=True, labeler=None)
