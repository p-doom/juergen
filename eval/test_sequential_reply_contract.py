"""Sequential goal-memory runtime/training contract tests.

Three halves. The reply grammar (``parse_sequential_reply`` on the exact action
and checkpoint shapes the recipe emits); the SERIALIZATION IDENTITY between what
``eval/freeroll.py`` assembles at rollout time and the Stage-04 record shape the
model was trained on, checked against the shape transcribed from the design spec;
and the same identity checked against what the REAL Stage-04 packer
(``build_sequential_conversations``) emits for the same flow. The last one is the
drift alarm between ``eval/`` and ``data_pipeline/``: the spec-derived test says
the runtime matches what we designed, the packer-derived test says it matches
what actually ships. A runtime whose turn sequence drifts from the record shape is
silently out of distribution in a way no parser check can catch.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
# freeroll.py puts data_pipeline on sys.path the same way so the runtime and the
# Stage-04 packer share one contract module; mirror it here rather than
# re-declaring any of the contract's constants in the test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))

from action_parser import parse_sequential_reply  # noqa: E402
from osworld_runtime import (  # noqa: E402
    ScreenshotCheckpointController, append_turn, compact_to_current,
    step_messages, validate_single_eviction,
)
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402
from realigned_pipeline.lib.conversations import image_block, text_block  # noqa: E402
from realigned_pipeline.lib.sequential_conversations import (  # noqa: E402
    build_sequential_conversations,
)
from realigned_pipeline.lib.sequential_goal_memory_contract import (  # noqa: E402
    CHECKPOINT_CONTROL_REQUEST, RECIPE, RESUME_UPWEIGHT_TURNS, goal_conditioning,
    render_checkpoint, system_prompt,
)
from realigned_pipeline.lib.sequential_packing import (  # noqa: E402
    PackingConfig, boundary_events, packing_config_hash, segments_from_boundaries,
)


def _call(arguments: str) -> str:
    return (
        '<tool_call>\n{"name":"computer_use","arguments":'
        + arguments + "}\n</tool_call>"
    )


def test_strict_action_reply_and_normalized_range() -> None:
    reply = parse_sequential_reply(
        "<think>I need to move toward the visible target.</think>\n"
        + _call('{"action":"mouse_move_rel","delta":[-1000,1000]}'),
        expected="action",
    )
    assert reply.kind == "action"
    assert reply.thought == "I need to move toward the visible target."
    assert [(p.kind, p.dx, p.dy) for p in reply.action.primitives] == [
        ("move", -1000, 1000)
    ]
    with pytest.raises(ValueError, match="normalized delta"):
        parse_sequential_reply(
            _call('{"action":"mouse_move_rel","delta":[1001,0]}'),
            expected="action",
        )
    with pytest.raises(ValueError, match="non-negative"):
        parse_sequential_reply(
            _call('{"action":"wait","time":-0.1}'), expected="action")
    with pytest.raises(ValueError, match="outside"):
        parse_sequential_reply(
            _call('{"action":"wait","time":0}') + " trailing prose",
            expected="action",
        )


def test_exact_checkpoint_reply_only() -> None:
    fields = (
        "Long-term goal", "Mid-term objective", "Short-term objective",
        "Completed", "Current state", "Next step", "Critical details",
    )
    lines = ["<checkpoint>"]
    for field in fields:
        lines.extend([f"## {field}", "Known.", ""])
    lines[-1] = "</checkpoint>"
    text = "\n".join(lines)
    assert parse_sequential_reply(text, expected="checkpoint").checkpoint["Completed"] == "Known."
    with pytest.raises(ValueError, match="exact required"):
        parse_sequential_reply(text + "\nextra", expected="checkpoint")
    with pytest.raises(ValueError, match="expected an action"):
        parse_sequential_reply(text, expected="action")


def test_checkpoint_controller_fires_at_seventy_percent_and_resets() -> None:
    controller = ScreenshotCheckpointController(10, 0.7)
    assert controller.threshold == 7
    for _ in range(5):
        controller.note_screenshot()
    assert not controller.due
    controller.note_screenshot()
    assert controller.due
    controller.reset_to_current()
    assert controller.screenshots == 1
    assert not controller.due


# ---------------------------------------------------------------------------
# Single-eviction guarantee
# ---------------------------------------------------------------------------


def test_single_eviction_guard_rejects_window_smaller_than_capacity() -> None:
    """A window narrower than the capacity lets block eviction fire first."""
    controller = ScreenshotCheckpointController(8, 0.7)
    with pytest.raises(ValueError, match="single-eviction violation"):
        validate_single_eviction(n_history_frames=7, controller=controller)
    with pytest.raises(ValueError, match="n_history_frames=1"):
        validate_single_eviction(n_history_frames=1, controller=controller)
    # Equality is the wired case (freeroll derives both from the same
    # expression) and a wider window is trivially safe.
    validate_single_eviction(n_history_frames=8, controller=controller)
    validate_single_eviction(n_history_frames=64, controller=controller)


def test_freeroll_calls_the_eviction_guard_at_rollout_setup() -> None:
    """The guard is only worth anything if the sequential path actually runs it.

    ``_run_rollout`` needs a booted VM, so assert the call site from the source:
    the guard must be invoked inside ``_run_rollout`` itself, not left as an
    importable-but-unused helper.
    """
    tree = ast.parse((Path(__file__).resolve().parent / "freeroll.py").read_text())
    rollout = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_rollout"
    )
    called = {
        node.func.id for node in ast.walk(rollout)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_single_eviction" in called


def test_block_eviction_never_precedes_the_controller() -> None:
    """With the guard satisfied, the window still holds every frame at the fire.

    Walks the runtime bookkeeping (``append_turn`` + the controller) for a full
    segment and checks the controller becomes due while the window is still
    complete — i.e. no frame since the last compaction has been dropped, which
    is what a training segment guarantees.
    """
    for capacity, fraction in ((4, 0.7), (10, 0.7), (16, 1.0), (3, 0.5)):
        controller = ScreenshotCheckpointController(capacity, fraction)
        validate_single_eviction(n_history_frames=capacity, controller=controller)
        frames = ["step_000.png"]
        actions: list[str] = []
        step = 0
        while not controller.due:
            step += 1
            append_turn(frames, actions, f"step_{step:03d}.png", f"a{step}",
                        n_history_frames=capacity)
            controller.note_screenshot()
        # Every screenshot the controller counted is still in the window, and
        # the window never overflowed into a block eviction.
        assert len(frames) == controller.screenshots == controller.threshold
        assert frames == [f"step_{i:03d}.png" for i in range(controller.threshold)]
        assert len(actions) == len(frames) - 1


# ---------------------------------------------------------------------------
# Serialization identity: runtime turn sequence vs Stage-04 record shape
# ---------------------------------------------------------------------------

# capacity 4 @ 0.7 -> threshold 3: the control turn interrupts at the third
# screenshot, giving a two-action segment, a boundary, and a resumed segment —
# the smallest flow that exercises every structural feature.
_CAPACITY = 4
_FRACTION = 0.7
_GOAL = "Rename the exported invoice and move it into the Q3 folder."
# One source for the flow's content, consumed twice: the runtime replay feeds
# these back as model replies, and the synthetic annotation day below publishes
# the same calls/thoughts/frames as its semantic events. Any difference the
# identity assertions find is therefore structural, never a content mismatch.
_CALLS = (
    '{"action":"left_click"}',
    '{"action":"mouse_move_rel","delta":[120,-40]}',
    '{"action":"type","text":"invoice_q3.pdf"}',
    '{"action":"key","keys":["enter"]}',
)
_THOUGHTS = {
    0: "The file list is visible, so target the export row.",
    2: "The rename field is focused now.",
}
_FRAMES = tuple(f"step_{index:03d}.png" for index in range(len(_CALLS)))
_ACTIONS = tuple(
    (f"<think>{_THOUGHTS[index]}</think>\n" if index in _THOUGHTS else "") + _call(call)
    for index, call in enumerate(_CALLS)
)
_CHECKPOINT = render_checkpoint({
    "Long-term goal": "Tidy the Q3 invoice exports.",
    "Mid-term objective": "Rename the newly exported invoice.",
    "Short-term objective": "Open the rename field on the export row.",
    "Completed": "Selected the exported invoice row in the file manager.",
    "Current state": "File manager focused, export row selected.",
    "Next step": "Trigger rename on the selected row.",
    "Critical details": "The export is the newest row, dated today.",
})


def _frame_id(image: str) -> str:
    """``<image step_002.png>`` (runtime) and ``step_002.png`` (record) unify."""
    return image[len("<image "):-1] if image.startswith("<image ") else image


def _canonical(messages: list[dict[str, Any]]) -> list[tuple[str, tuple]]:
    """``(role, blocks)`` per turn, images reduced to their frame identity.

    The only permitted difference between the two serializations: the runtime
    references a frame by ``<image step_NNN.png>`` placeholder (the pixels live
    beside the trace) while a record references it by path, and the runtime's
    system turn is a bare string where the record wraps it in one text block.
    Text bytes are compared verbatim.
    """
    canonical: list[tuple[str, tuple]] = []
    for message in messages:
        content = message["content"]
        blocks = (
            [("text", content)] if isinstance(content, str)
            else [
                ("text", block["text"]) if block["type"] == "text"
                else ("image", _frame_id(block["image"]))
                for block in content
            ]
        )
        canonical.append((message["role"], tuple(blocks)))
    return canonical


def _stage04_record(
    *,
    goal_text: str,
    checkpoint_in: str | None,
    events: list[tuple[str, str]],
    checkpoint_out: tuple[str, str] | None,
) -> list[dict[str, Any]]:
    """The spec's Phase-2A "Record for segment k" turn sequence, verbatim.

    system turn; user turn = ``goal_conditioning(goal_text, checkpoint_in)`` +
    image of the span's first event; alternating assistant action / user image
    turns for the rest of the span; then, when the next event is a boundary, a
    user turn of ``CHECKPOINT_CONTROL_REQUEST`` + the boundary image and an
    assistant checkpoint turn.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [text_block(system_prompt())]}
    ]
    for offset, (image, assistant) in enumerate(events):
        content = (
            [text_block(goal_conditioning(goal_text, checkpoint_in))]
            if offset == 0 else []
        )
        content.append(image_block(image))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [text_block(assistant)]})
    if checkpoint_out is not None:
        boundary_image, checkpoint_text = checkpoint_out
        messages.append({"role": "user", "content": [
            text_block(CHECKPOINT_CONTROL_REQUEST), image_block(boundary_image)]})
        messages.append(
            {"role": "assistant", "content": [text_block(checkpoint_text)]})
    return messages


def _replay_runtime() -> list[list[tuple[list[dict[str, Any]], str]]]:
    """Replay freeroll's sequential rollout loop over scripted model replies.

    Mirrors ``_run_rollout``'s sequential path exactly, through the same helpers
    it uses: ``step_messages`` for the turn assembly, ``append_turn`` for the
    rolling window, ``compact_to_current`` for the post-checkpoint compaction,
    ``ScreenshotCheckpointController`` for the fire decision,
    ``goal_conditioning`` for the instruction text. ``persist_instruction`` is
    not a parameter here because the sequential path now requires it, so the
    instruction is simply the current ``instr_text`` on every turn.

    Frames are their own filenames — ``append_turn`` treats them as opaque, and
    the identity we care about is which frame lands in which turn.

    Returns one list of ``(messages, reply)`` per SEGMENT: the turns assembled
    between compactions, in order.
    """
    controller = ScreenshotCheckpointController(_CAPACITY, _FRACTION)
    validate_single_eviction(n_history_frames=_CAPACITY, controller=controller)
    system_text = SYSTEM_PROMPTS[RECIPE]
    instr_text = goal_conditioning(_GOAL)
    frames = ["step_000.png"]
    actions: list[str] = []
    segments: list[list[tuple[list[dict[str, Any]], str]]] = [[]]
    pending = list(_ACTIONS)
    step = 0
    while pending:
        step += 1
        if step > 1 and controller.due:
            _, control_messages = step_messages(
                system_prompt=system_text, instruction=instr_text, step=step,
                n_frames=len(frames), recent_actions=actions,
                current_text=CHECKPOINT_CONTROL_REQUEST,
            )
            parse_sequential_reply(_CHECKPOINT, expected="checkpoint")
            segments[-1].append((control_messages, _CHECKPOINT))
            instr_text = goal_conditioning(_GOAL, _CHECKPOINT)
            compact_to_current(frames, actions)
            controller.reset_to_current()
            segments.append([])
        _, messages = step_messages(
            system_prompt=system_text, instruction=instr_text, step=step,
            n_frames=len(frames), recent_actions=actions,
        )
        reply = pending.pop(0)
        parse_sequential_reply(reply, expected="action")
        segments[-1].append((messages, reply))
        append_turn(frames, actions, f"step_{step:03d}.png", reply,
                    n_history_frames=_CAPACITY)
        controller.note_screenshot()
    return segments


def _runtime_records() -> list[list[dict[str, Any]]]:
    """One completed record per segment.

    Within a segment the runtime prompt is append-only (asserted separately), so
    the LAST message list of a segment already contains every earlier turn; the
    record is that list plus the reply the model gave to it.
    """
    return [
        [*turns[-1][0], {"role": "assistant", "content": turns[-1][1]}]
        for turns in _replay_runtime()
    ]


def test_runtime_turn_sequence_is_identical_to_the_stage04_record_shape() -> None:
    runtime = _runtime_records()
    assert len(runtime) == 2, "capacity 4 @ 0.7 must yield exactly one boundary"
    expected = [
        _stage04_record(
            goal_text=_GOAL,
            checkpoint_in=None,
            events=[("step_000.png", _ACTIONS[0]), ("step_001.png", _ACTIONS[1])],
            checkpoint_out=("step_002.png", _CHECKPOINT),
        ),
        _stage04_record(
            goal_text=_GOAL,
            checkpoint_in=_CHECKPOINT,
            events=[("step_002.png", _ACTIONS[2]), ("step_003.png", _ACTIONS[3])],
            checkpoint_out=None,
        ),
    ]
    for index, (got, want) in enumerate(zip(runtime, expected, strict=True)):
        assert _canonical(got) == _canonical(want), f"segment {index} diverged"


def test_runtime_record_roles_and_text_placement() -> None:
    """Spell out the structure the identity assertion above compares against."""
    first, second = _runtime_records()
    assert [m["role"] for m in first] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert [m["role"] for m in second] == [
        "system", "user", "assistant", "user", "assistant"]

    # Goal conditioning rides the opening user turn only, ahead of the image.
    first_blocks = _canonical(first)[1][1]
    assert first_blocks == (
        ("text", goal_conditioning(_GOAL)), ("image", "step_000.png"))
    assert _canonical(first)[3][1] == (("image", "step_001.png"),)

    # The control request is its OWN user turn carrying the boundary frame.
    assert _canonical(first)[5][1] == (
        ("text", CHECKPOINT_CONTROL_REQUEST), ("image", "step_002.png"))
    assert first[-1]["role"] == "assistant"
    assert parse_sequential_reply(
        first[-1]["content"], expected="checkpoint").checkpoint["Next step"] == (
            "Trigger rename on the selected row.")

    # No stray control text leaks into the resumed record.
    assert not any(
        CHECKPOINT_CONTROL_REQUEST in block
        for _, blocks in _canonical(second) for kind, block in blocks
        if kind == "text"
    )


def test_boundary_screenshot_is_duplicated_across_the_handoff() -> None:
    """The frame the control turn shows reopens the next record.

    ``reset_to_current`` keeps the boundary screenshot, so it is both the last
    image of the compacted record and the first image of the resumed one — the
    same duplication ``segments_from_boundaries`` encodes by giving the boundary
    event's action to the NEXT segment.
    """
    first, second = _runtime_records()
    first_images, second_images = (
        [block for _, blocks in _canonical(record) for kind, block in blocks
         if kind == "image"]
        for record in (first, second)
    )
    assert first_images == ["step_000.png", "step_001.png", "step_002.png"]
    assert second_images == ["step_002.png", "step_003.png"]
    assert first_images[-1] == second_images[0]
    # Capacity is respected on both sides of the handoff (spec invariant 3).
    assert len(first_images) <= _CAPACITY
    assert len(second_images) <= _CAPACITY


def test_checkpoint_bytes_are_carried_verbatim_into_the_resume_conditioning() -> None:
    """Handoff byte-identity (spec invariant 2), measured on the runtime path."""
    first, second = _runtime_records()
    checkpoint_out = first[-1]["content"]
    assert checkpoint_out == _CHECKPOINT
    resume_text = _canonical(second)[1][1][0][1]
    assert resume_text == goal_conditioning(_GOAL, checkpoint_out)
    assert resume_text.endswith(checkpoint_out.strip())
    assert resume_text == f"GOAL: {_GOAL}\n\n{checkpoint_out}"


def test_runtime_prompt_is_append_only_within_a_segment() -> None:
    """Each turn's prompt extends the previous one — no rewrite, no eviction.

    This is what makes the final message list of a segment a faithful stand-in
    for the whole Stage-04 record (and, incidentally, what keeps sglang's
    RadixAttention prefix cache warm across a segment).
    """
    for index, turns in enumerate(_replay_runtime()):
        for (earlier, _), (later, _) in zip(turns, turns[1:], strict=False):
            assert _canonical(later)[:len(earlier)] == _canonical(earlier), (
                f"segment {index} rewrote an earlier turn")
            assert len(later) > len(earlier)


def test_system_prompt_bytes_are_shared_with_stage04() -> None:
    """One versioned file, no strip on either side — same bytes in both paths."""
    assert RECIPE == "sequential_goal_memory_v1"
    assert SYSTEM_PROMPTS[RECIPE] == system_prompt()


# ---------------------------------------------------------------------------
# Drift alarm: runtime turn sequence vs the REAL Stage-04 packer output
# ---------------------------------------------------------------------------

_DAY_TAG = "u0_2026-03-04"
# fraction_low == fraction_high == the runtime's fixed fraction, so the packer's
# per-segment jitter draw collapses onto the single value the controller uses and
# the boundary lands exactly where ScreenshotCheckpointController fires. This is
# the ONLY config under which the two are expected to cut at the same event —
# with real jitter, training deliberately brackets [0.5, 0.85] around it.
_PACKING = PackingConfig(
    capacity=_CAPACITY, fraction_low=_FRACTION, fraction_high=_FRACTION, seed=0)
# Pin the goal rendering: the packer draws a mode per segment, and only
# explicit_long renders a stable GOAL across both segments of this day.
_LONG_ONLY = {"explicit_long": 1.0}
_DAY_FINAL_CHECKPOINT = render_checkpoint({
    "Long-term goal": "Tidy the Q3 invoice exports.",
    "Mid-term objective": "Rename the newly exported invoice.",
    "Short-term objective": "Confirm the rename took effect.",
    "Completed": "Renamed the exported invoice to invoice_q3.pdf.",
    "Current state": "File manager shows the renamed row.",
    "Next step": "Move the file into the Q3 folder.",
    "Critical details": "The Q3 folder is one level up in Documents.",
})


def _event_id(index: int) -> str:
    return f"{_DAY_TAG}_sem{index:03d}"


def _synthetic_day() -> dict[str, Any]:
    """A four-event annotated day whose packing IS the runtime flow, event for event.

    Shaped like the 03b/03c artifact Stage 04 consumes: positional
    ``day_event_index``, one rolling-memory snapshot per event, sparse
    ``decisions`` thoughts, and pass-03c ``checkpoints`` rows stamped with this
    config's ``packing_config_hash``.

    Frames carry the runtime's own ``step_NNN.png`` filenames. Both
    serializations treat the image string as an opaque token — the packer copies
    ``event["image"]``, the runtime emits an ``<image LABEL>`` placeholder — so
    naming them identically is what makes "which frame lands in which turn"
    directly comparable instead of merely positionally comparable.
    """
    events = [
        {
            "semantic_event_id": _event_id(index),
            "day_event_index": index,
            "segment_id": "s0",
            "image": frame,
            "t_day_s": float(index),
            "tool_calls": [{"name": "computer_use", "arguments": json.loads(call)}],
            "assistant_action": _call(call),
        }
        for index, (frame, call) in enumerate(zip(_FRAMES, _CALLS, strict=True))
    ]
    last = len(events) - 1
    return {
        "day_tag": _DAY_TAG, "user_id": "u0", "date": "2026-03-04",
        "semantic_events": events,
        "goal_nodes": [
            {"goal_id": f"{_DAY_TAG}_long", "parent_id": None, "level": "long",
             "text": _GOAL, "provenance": "explicit",
             "start_event_index": 0, "end_event_index": last},
            {"goal_id": f"{_DAY_TAG}_mid0", "parent_id": f"{_DAY_TAG}_long",
             "level": "mid", "text": "Open the rename field on the export row.",
             "provenance": "explicit", "start_event_index": 0, "end_event_index": 1},
            {"goal_id": f"{_DAY_TAG}_mid1", "parent_id": f"{_DAY_TAG}_long",
             "level": "mid", "text": "Type the new invoice file name.",
             "provenance": "proactive", "start_event_index": 2, "end_event_index": last},
        ],
        "decisions": [
            {"anchor_semantic_event_id": _event_id(index), "thought": thought}
            for index, thought in sorted(_THOUGHTS.items())
        ],
        "memory_snapshots": [
            {"memory_snapshot_id": f"mem_{_event_id(index)}",
             "anchor_semantic_event_id": _event_id(index),
             "anchor_event_index": index,
             "memory_after": f"State after {_event_id(index)}."}
            for index in range(len(events))
        ],
        "event_dispositions": [
            {"raw_event_id": f"{_event_id(index)}:r0",
             "semantic_event_id": _event_id(index), "disposition": "emitted"}
            for index in range(len(events))
        ],
        "checkpoints": [
            # The boundary anchor the runtime's controller fires at, carrying the
            # SAME bytes the replay's model returns for the control turn.
            {"checkpoint_id": f"cp_{_event_id(2)}", "day_tag": _DAY_TAG,
             "anchor_semantic_event_id": _event_id(2), "anchor_event_index": 2,
             "text": _CHECKPOINT,
             "packing_config_hash": packing_config_hash(_PACKING),
             "is_day_final": False,
             "source_memory_snapshot_id": f"mem_{_event_id(2)}"},
            # 03c also projects the day-final anchor (for cross-day resume);
            # unused here, present so the artifact is the real shape.
            {"checkpoint_id": f"cp_{_event_id(last)}", "day_tag": _DAY_TAG,
             "anchor_semantic_event_id": _event_id(last), "anchor_event_index": last,
             "text": _DAY_FINAL_CHECKPOINT,
             "packing_config_hash": packing_config_hash(_PACKING),
             "is_day_final": True,
             "source_memory_snapshot_id": f"mem_{_event_id(last)}"},
        ],
    }


def _packer_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_sequential_conversations(
        [_synthetic_day()],
        system_prompt=SYSTEM_PROMPTS[RECIPE],
        parse_reply=parse_sequential_reply,
        cfg=_PACKING,
        mode_weights=_LONG_ONLY,
    )


def test_packer_cuts_the_day_where_the_runtime_controller_fires() -> None:
    """The premise of the identity test: same threshold, same boundary event."""
    controller = ScreenshotCheckpointController(_CAPACITY, _FRACTION)
    assert controller.threshold == 3
    boundaries = boundary_events(len(_FRAMES), day_tag=_DAY_TAG, cfg=_PACKING)
    assert boundaries == [2]
    assert segments_from_boundaries(len(_FRAMES), boundaries) == [(0, 1), (2, 3)]


def test_runtime_turns_are_identical_to_the_real_packer_records() -> None:
    """The drift alarm: eval/ assembly vs data_pipeline/ packer, turn by turn.

    Same normalization as the spec-derived test — image placeholder vs path, and
    the runtime's bare-string system turn vs the record's one-element text block.
    Everything else, including every text byte, must match exactly.
    """
    records, summary = _packer_records()
    assert summary["n_segments"] == len(records) == 2
    assert [row["mode"] for row in records] == ["explicit_long", "explicit_long"]
    assert [row["instruction"] for row in records] == [_GOAL, _GOAL]

    runtime = _runtime_records()
    assert len(runtime) == len(records)
    for index, (got, want) in enumerate(zip(runtime, records, strict=True)):
        assert _canonical(got) == _canonical(want["messages"]), (
            f"segment {index} diverged between the runtime and the packer")


def test_packer_and_runtime_agree_on_image_accounting_and_handoff() -> None:
    """The metadata the packer records about a segment describes the runtime too."""
    records, _ = _packer_records()
    runtime = _runtime_records()
    for record, turns in zip(records, runtime, strict=True):
        runtime_images = [
            block for _, blocks in _canonical(turns) for kind, block in blocks
            if kind == "image"
        ]
        assert record["n_images"] == len(runtime_images) <= _CAPACITY
    # 3 = two action frames + the shared boundary frame; 2 = boundary + one more.
    assert [row["n_images"] for row in records] == [3, 2]
    assert records[0]["checkpoint_out_id"] == records[1]["checkpoint_in_id"]
    assert records[0]["messages"][-1]["content"][0]["text"] == _CHECKPOINT
    # The resumed record is the one whose opening turn carries the checkpoint, and
    # its upweighted turns are the assistant actions right after the resume.
    assert records[0]["resume_upweight_turns"] == []
    assert records[1]["resume_upweight_turns"] == [2, 4]
    assert len(records[1]["resume_upweight_turns"]) <= RESUME_UPWEIGHT_TURNS
    assert all(records[1]["messages"][turn]["role"] == "assistant"
               for turn in records[1]["resume_upweight_turns"])


def test_spec_derived_and_packer_derived_expectations_agree() -> None:
    """Belt and braces: the two independent expectations must describe one shape.

    If Phase 2A ever changes the record shape, this fails alongside the packer
    identity test while the spec-derived test still passes — which localizes the
    drift to the packer rather than to eval/.
    """
    records, _ = _packer_records()
    spec = [
        _stage04_record(
            goal_text=_GOAL, checkpoint_in=None,
            events=[(_FRAMES[0], _ACTIONS[0]), (_FRAMES[1], _ACTIONS[1])],
            checkpoint_out=(_FRAMES[2], _CHECKPOINT),
        ),
        _stage04_record(
            goal_text=_GOAL, checkpoint_in=_CHECKPOINT,
            events=[(_FRAMES[2], _ACTIONS[2]), (_FRAMES[3], _ACTIONS[3])],
            checkpoint_out=None,
        ),
    ]
    for index, (got, want) in enumerate(zip(records, spec, strict=True)):
        assert _canonical(got["messages"]) == _canonical(want), (
            f"segment {index} diverged between the packer and the spec")
