#!/usr/bin/env python3
"""Pure fail-closed transition contract for the roadmap stage-2 design.

This module does not launch a model, VM, or job.  It makes on-policy cursor,
target, retry, and render transitions executable in CPU tests before any runner
or recipe exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


Point = tuple[int, int]
BBox = tuple[int, int, int, int]


class TransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClosedLoopState:
    episode_id: str
    target_index: int
    cursor: Point
    attempts_on_target: int = 0
    attempts_total: int = 0
    target_hit_attempts: tuple[int, ...] = ()
    terminated: bool = False
    success: bool = False


@dataclass(frozen=True)
class AttemptEvidence:
    raw_output: str
    parse_ok: bool
    schema_ok: bool
    unit_range_ok: bool
    dispatched: bool
    endpoint: Point | None
    actual_cursor_after: Point | None
    guest_hit: bool | None


@dataclass(frozen=True)
class Transition:
    before: ClosedLoopState
    after: ClosedLoopState
    attempt_number: int
    valid_output: bool
    dispatched: bool
    hit: bool
    target_advanced: bool
    render_changed: bool
    terminal_reason: str | None


def initial_state(episode_id: str, cursor: Point) -> ClosedLoopState:
    return ClosedLoopState(episode_id=episode_id, target_index=0, cursor=cursor)


def _inside(point: Point, bbox: BBox) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _validate_targets(targets: Sequence[BBox]) -> None:
    if not targets:
        raise TransitionError("episode has no targets")
    for bbox in targets:
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise TransitionError(f"invalid target bbox: {bbox}")


def advance(
    state: ClosedLoopState,
    evidence: AttemptEvidence,
    targets: Sequence[BBox],
    *,
    max_attempts_per_target: int = 3,
) -> Transition:
    """Apply one attempt, preserving actual on-policy state after a miss."""
    _validate_targets(targets)
    if state.terminated:
        raise TransitionError("cannot advance a terminated episode")
    if not (0 <= state.target_index < len(targets)):
        raise TransitionError("active target index is out of range")
    if not (0 <= state.attempts_on_target < max_attempts_per_target):
        raise TransitionError("attempt counter is out of range")
    if max_attempts_per_target <= 0:
        raise TransitionError("retry budget must be positive")
    valid_output = evidence.parse_ok and evidence.schema_ok and evidence.unit_range_ok
    attempt_number = state.attempts_on_target + 1
    attempts_total = state.attempts_total + 1
    bbox = targets[state.target_index]

    if not valid_output:
        if evidence.dispatched:
            raise TransitionError("invalid output was dispatched")
        if evidence.actual_cursor_after is not None or evidence.guest_hit is not None:
            raise TransitionError("invalid no-op has guest actuation evidence")
        exhausted = attempt_number == max_attempts_per_target
        after = ClosedLoopState(
            episode_id=state.episode_id,
            target_index=state.target_index,
            cursor=state.cursor,
            attempts_on_target=attempt_number,
            attempts_total=attempts_total,
            target_hit_attempts=state.target_hit_attempts,
            terminated=exhausted,
            success=False,
        )
        return Transition(
            before=state,
            after=after,
            attempt_number=attempt_number,
            valid_output=False,
            dispatched=False,
            hit=False,
            target_advanced=False,
            render_changed=False,
            terminal_reason="retry_budget_exhausted" if exhausted else None,
        )

    if not evidence.dispatched:
        raise TransitionError("valid output was not dispatched")
    if evidence.endpoint is None or evidence.actual_cursor_after is None:
        raise TransitionError("dispatched output lacks endpoint/cursor readback")
    if evidence.actual_cursor_after != evidence.endpoint:
        raise TransitionError("physical cursor differs from registered endpoint")
    geometric_hit = _inside(evidence.actual_cursor_after, bbox)
    if evidence.guest_hit is not geometric_hit:
        raise TransitionError("guest hit state differs from endpoint geometry")

    cursor = evidence.actual_cursor_after
    if geometric_hit:
        target_hit_attempts = state.target_hit_attempts + (attempt_number,)
        next_index = state.target_index + 1
        completed = next_index == len(targets)
        after = ClosedLoopState(
            episode_id=state.episode_id,
            target_index=(state.target_index if completed else next_index),
            cursor=cursor,
            attempts_on_target=0,
            attempts_total=attempts_total,
            target_hit_attempts=target_hit_attempts,
            terminated=completed,
            success=completed,
        )
        return Transition(
            before=state,
            after=after,
            attempt_number=attempt_number,
            valid_output=True,
            dispatched=True,
            hit=True,
            target_advanced=True,
            render_changed=(cursor != state.cursor or not completed),
            terminal_reason="all_targets_reached" if completed else None,
        )

    exhausted = attempt_number == max_attempts_per_target
    after = ClosedLoopState(
        episode_id=state.episode_id,
        target_index=state.target_index,
        cursor=cursor,
        attempts_on_target=attempt_number,
        attempts_total=attempts_total,
        target_hit_attempts=state.target_hit_attempts,
        terminated=exhausted,
        success=False,
    )
    return Transition(
        before=state,
        after=after,
        attempt_number=attempt_number,
        valid_output=True,
        dispatched=True,
        hit=False,
        target_advanced=False,
        render_changed=cursor != state.cursor,
        terminal_reason="retry_budget_exhausted" if exhausted else None,
    )


def reference_png(contract: Any, state: ClosedLoopState, targets: Sequence[BBox]) -> bytes:
    """Render the next observation from the active target and actual cursor."""
    _validate_targets(targets)
    if state.terminated:
        raise TransitionError("terminated episode has no next model observation")
    return contract.render_png(targets[state.target_index], state.cursor)


def request_seed(condition: str, episode_id: str, target_index: int, attempt: int) -> int:
    if condition not in {"single_step_sentinel", "multi_step_closed_loop"}:
        raise TransitionError(f"unknown condition: {condition}")
    if target_index < 0 or attempt < 1:
        raise TransitionError("invalid seed slot")
    key = f"proper-vm-roadmap-stage2-v1|{condition}|{episode_id}|{target_index}|{attempt}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
