"""Ordered action projection for Stage-05 event records."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

AGGREGATE_ACTION_SCHEMA = "aggregate_delta_keys_v1"
ORDERED_ACTION_SCHEMA = "ordered_events_v2"
ACTION_SCHEMAS = (AGGREGATE_ACTION_SCHEMA, ORDERED_ACTION_SCHEMA)
DEFAULT_ACTION_SCHEMA = ORDERED_ACTION_SCHEMA
DEFAULT_CONTINUOUS_ACTION_HZ = 10.0

PrimitiveKind = Literal["move", "scroll", "down", "up"]
_INPUT_NAME_RE = re.compile(r"^[^\s(),;]+$")


@dataclass(frozen=True)
class ActionPrimitive:
    kind: PrimitiveKind
    dx: int | None = None
    dy: int | None = None
    input_name: str | None = None

    def render(self) -> str:
        if self.kind in {"move", "scroll"}:
            return f"{self.kind}({self.dx},{self.dy})"
        return f"{self.kind}({self.input_name})"


@dataclass(frozen=True)
class ProjectedAction:
    text: str
    primitives: tuple[ActionPrimitive, ...]


@dataclass
class HeldStateDiagnostics:
    duplicate_down: int = 0
    dangling_up: int = 0
    non_neutral_trajectory: int = 0
    held_at_trajectory_end: int = 0

    def finish_trajectory(self, held: set[str]) -> None:
        if held:
            self.non_neutral_trajectory += 1
            self.held_at_trajectory_end += len(held)

    def update(self, other: HeldStateDiagnostics) -> None:
        self.duplicate_down += other.duplicate_down
        self.dangling_up += other.dangling_up
        self.non_neutral_trajectory += other.non_neutral_trajectory
        self.held_at_trajectory_end += other.held_at_trajectory_end

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicate_down": self.duplicate_down,
            "dangling_up": self.dangling_up,
            "non_neutral_trajectory": self.non_neutral_trajectory,
            "held_at_trajectory_end": self.held_at_trajectory_end,
        }


def _continuous_primitive(kind: str, dx: float, dy: float) -> ActionPrimitive | None:
    rounded_dx = round(dx)
    rounded_dy = round(dy)
    if rounded_dx == 0 and rounded_dy == 0:
        return None
    return ActionPrimitive(
        kind=cast("PrimitiveKind", kind),
        dx=rounded_dx,
        dy=rounded_dy,
    )


def _input_primitive(kind: str, value: Any) -> ActionPrimitive:
    name = str(value)
    if not _INPUT_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid input name: {name!r}")
    projected_kind: PrimitiveKind = "down" if kind == "press" else "up"
    return ActionPrimitive(kind=projected_kind, input_name=name)


def project_ordered_action(
    events: list[dict[str, Any]],
    *,
    interval_start_s: float,
    continuous_action_hz: float,
) -> ProjectedAction:
    if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
        raise ValueError("continuous_action_hz must be finite and positive")
    if not math.isfinite(interval_start_s):
        raise ValueError("interval_start_s must be finite")

    primitives: list[ActionPrimitive] = []
    pending: tuple[int, str, float, float] | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        _tick, kind, dx, dy = pending
        primitive = _continuous_primitive(kind, dx, dy)
        if primitive is not None:
            primitives.append(primitive)
        pending = None

    for event in events:
        kind = str(event["kind"])
        if kind in {"move", "scroll"}:
            event_time_s = float(event["local_time_s"])
            dx = float(event["dx"])
            dy = float(event["dy"])
            if not all(math.isfinite(value) for value in (event_time_s, dx, dy)):
                raise ValueError(f"Non-finite continuous event: {event!r}")
            tick = math.floor((event_time_s - interval_start_s) * continuous_action_hz)
            if pending is not None and pending[0] == tick and pending[1] == kind:
                pending = (tick, kind, pending[2] + dx, pending[3] + dy)
            else:
                flush()
                pending = (tick, kind, dx, dy)
        elif kind in {"press", "release"}:
            flush()
            primitives.append(_input_primitive(kind, event["key"]))
        else:
            raise ValueError(f"Unsupported action event kind: {kind!r}")
    flush()

    frozen = tuple(primitives)
    return ProjectedAction(
        text="; ".join(primitive.render() for primitive in frozen) if frozen else "NO_OP",
        primitives=frozen,
    )


def update_held_state(
    primitives: tuple[ActionPrimitive, ...],
    *,
    held: set[str],
    diagnostics: HeldStateDiagnostics,
) -> None:
    for primitive in primitives:
        if primitive.kind == "down":
            assert primitive.input_name is not None
            if primitive.input_name in held:
                diagnostics.duplicate_down += 1
            held.add(primitive.input_name)
        elif primitive.kind == "up":
            assert primitive.input_name is not None
            if primitive.input_name not in held:
                diagnostics.dangling_up += 1
            held.discard(primitive.input_name)
