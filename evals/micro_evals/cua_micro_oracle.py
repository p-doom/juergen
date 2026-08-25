"""Scripted expert policies ("oracles") for the CUA micro-eval suite.

The micro-eval suite is fully deterministic: every target is a ``fixed_norm``
bbox written into the task JSON, the cursor start is specified, the fixture is
a program we wrote, and success is a programmatic verifier. So the correct
action for every turn is knowable in closed form *before* looking at a pixel --
no demonstrator and no teacher model required.

This module turns that observation into a policy. An oracle is a generator of
action lines in the same wire format the model emits (``cua_ordered_typing_v1``
/ ordered_events_v3), driven by a declarative ``oracle`` block on the task:

    "oracle": {"plan": [{"op": "approach", "click": "LMB"}]}

``cua_micro_eval.py --mode harvest`` runs the ordinary multiturn loop with the
oracle substituted for the model call, so the frames, the dispatch path and the
verifier are all the real ones -- only the decision-maker changes. The
(prompt window, action line) pairs that fall out are training data whose format
cannot drift from the eval's, because it *is* the eval's.

Two design points worth keeping:

* **Deltas are recomputed from the live cursor every turn.** A plan says "get
  onto the Writer icon", not "emit these three deltas" -- so rounding, pointer
  acceleration and screen clipping self-correct instead of accumulating. This
  is the whole reason the oracle runs inside the loop rather than generating
  data offline.
* **Approaches are staged and jittered.** The system prompt promises the model
  that "it may take several turns to arrive -- the cursor moves visibly between
  turns, so correct your aim as you close in". A teleport-and-click demo would
  contradict that. ``approach`` lands 2-3 progressively smaller moves, with the
  residual offsets and the step count drawn per trajectory, so the model learns
  error *reduction* rather than three memorised deltas.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from evals.micro_evals.action_parser import OrderedPrimitive

# Fraction-of-remaining-distance residuals for each approach shape. Entry i is
# how far from the target centre waypoint i lands, as a fraction of the total
# start->centre distance; the last entry is always 0.0 (land on centre). The
# ranges are sampled per trajectory -- see _approach_waypoints.
_APPROACH_SHAPES: dict[int, tuple[tuple[float, float], ...]] = {
    2: ((0.10, 0.22), (0.0, 0.0)),
    3: ((0.18, 0.32), (0.04, 0.10), (0.0, 0.0)),
}
# A waypoint this close to the centre (VM px) is indistinguishable from landing,
# so an "intermediate" step there would teach nothing. Pushed outward instead.
_MIN_INTERMEDIATE_PX = 12.0
_NO_OP = "NO_OP"

_OPS = frozenset(
    {"approach", "click", "key", "type", "scroll", "wait", "wait_title", "no_op"}
)


class OracleError(RuntimeError):
    """The oracle cannot produce a correct action (bad plan, or a fixture the
    plan's assumptions no longer hold for). Fails the trajectory rather than
    emitting a guess -- a wrong label is worse than a missing one."""


@dataclass
class OracleEnv:
    """Guest probes the oracle needs, injected by the caller.

    Passed in rather than imported so this module stays free of
    ``cua_micro_eval``'s import graph (which imports *this* module).
    """

    active_title: Callable[[], str]


@dataclass
class OracleRuntime:
    """Live per-turn state, refreshed by the harvest loop before each action.

    The plan generator reads these attributes *after* resuming from a yield, so
    it always sees the state produced by its own previous action.
    """

    env: OracleEnv
    screen: tuple[int, int]
    model_resolution: tuple[int, int] | None
    rng: random.Random
    cursor: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    turn_index: int = 0
    verifier_ok: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _in_bbox(point: tuple[float, float], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _clip(point: tuple[float, float], screen: tuple[int, int]) -> tuple[int, int]:
    return (
        max(0, min(screen[0] - 1, round(point[0]))),
        max(0, min(screen[1] - 1, round(point[1]))),
    )


def vm_to_model_delta(
    delta: tuple[int, int],
    screen: tuple[int, int],
    model_resolution: tuple[int, int] | None,
) -> tuple[int, int]:
    """Convert a VM-pixel delta into the model-resolution pixels an action line
    carries.

    Exact inverse of ``cua_micro_eval.denormalize_native_ordered_action``, which
    scales the other way right before dispatch. Getting this backwards is the
    single easiest way to produce data that looks right and trains a model to
    miss every target by the resolution ratio.
    """
    if not model_resolution:
        return delta
    mw, mh = model_resolution
    sw, sh = screen
    return (round(delta[0] * mw / sw), round(delta[1] * mh / sh))


def _approach_waypoints(
    start: tuple[int, int],
    bbox: tuple[int, int, int, int],
    screen: tuple[int, int],
    rng: random.Random,
    n_steps: int | None,
) -> list[tuple[int, int]]:
    """Waypoints for a staged, human-shaped approach from ``start`` onto ``bbox``.

    The last waypoint is always the bbox centre ("land the cursor tip on the
    CENTER of the target element", per the system prompt). Earlier ones sit at a
    jittered offset from the centre, in a random direction, at a decreasing
    fraction of the original distance -- an overshoot as often as an undershoot,
    so the data does not teach a single approach bearing.
    """
    center = _bbox_center(bbox)
    if n_steps is None:
        n_steps = rng.choice(sorted(_APPROACH_SHAPES))
    if n_steps not in _APPROACH_SHAPES:
        raise OracleError(f"approach steps must be one of {sorted(_APPROACH_SHAPES)}, got {n_steps}")
    distance = math.hypot(center[0] - start[0], center[1] - start[1])
    waypoints: list[tuple[int, int]] = []
    for low, high in _APPROACH_SHAPES[n_steps]:
        if high <= 0.0:
            waypoints.append(center)
            continue
        residual = rng.uniform(low, high) * distance
        angle = rng.uniform(0.0, 2.0 * math.pi)
        point = (
            center[0] + residual * math.cos(angle),
            center[1] + residual * math.sin(angle),
        )
        # An intermediate waypoint that already sits on the target makes the
        # remaining steps meaningless -- push it out along its own bearing until
        # it clears both the bbox and the "too close to matter" radius.
        for _ in range(24):
            if not _in_bbox(point, bbox) and residual >= _MIN_INTERMEDIATE_PX:
                break
            residual = max(residual * 1.6, _MIN_INTERMEDIATE_PX * 1.6)
            point = (
                center[0] + residual * math.cos(angle),
                center[1] + residual * math.sin(angle),
            )
        waypoints.append(_clip(point, screen))
    return waypoints


# ---------------------------------------------------------------------------
# action-line rendering
# ---------------------------------------------------------------------------


def render(primitives: list[OrderedPrimitive]) -> str:
    """Render primitives as one ordered_events_v3 action line.

    Uses ``OrderedPrimitive.render`` (the parser's own round-trip helper) rather
    than hand-formatting, so a harvested target is parseable by construction.
    """
    if not primitives:
        return _NO_OP
    return "; ".join(primitive.render() for primitive in primitives)


def _move(delta: tuple[int, int]) -> list[OrderedPrimitive]:
    """A move primitive, or nothing when the delta rounds away.

    The system prompt forbids ``move(0,0)`` outright ("leave the primitive out
    instead"), and a sub-pixel final correction legitimately rounds to zero.
    """
    if delta == (0, 0):
        return []
    return [OrderedPrimitive(kind="move", dx=delta[0], dy=delta[1])]


def _click(button: str, count: int) -> list[OrderedPrimitive]:
    if button not in ("LMB", "MMB", "RMB"):
        raise OracleError(f"click button must be LMB/MMB/RMB, got {button!r}")
    out: list[OrderedPrimitive] = []
    for _ in range(count):
        out.append(OrderedPrimitive(kind="down", input_name=button))
        out.append(OrderedPrimitive(kind="up", input_name=button))
    return out


def _chord(keys: list[str]) -> list[OrderedPrimitive]:
    """Press in order, release in reverse -- the prompt's key-chord recipe."""
    if not keys:
        raise OracleError("key op needs a non-empty 'keys' list")
    return [OrderedPrimitive(kind="down", input_name=key) for key in keys] + [
        OrderedPrimitive(kind="up", input_name=key) for key in reversed(keys)
    ]


# ---------------------------------------------------------------------------
# plan ops
# ---------------------------------------------------------------------------


def _op_approach(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    bbox = _resolve_op_bbox(rt, op)
    waypoints = _approach_waypoints(rt.cursor, bbox, rt.screen, rt.rng, op.get("steps"))
    button = op.get("click")
    count = int(op.get("count", 1))
    for index, waypoint in enumerate(waypoints):
        # Recomputed against the LIVE cursor, so any drift from the previous
        # turn is absorbed here instead of compounding.
        delta = vm_to_model_delta(
            (waypoint[0] - rt.cursor[0], waypoint[1] - rt.cursor[1]),
            rt.screen,
            rt.model_resolution,
        )
        primitives = _move(delta)
        is_last = index == len(waypoints) - 1
        if is_last and button:
            primitives += _click(str(button), count)
        if not primitives:
            # Already exactly on the waypoint and nothing else to do; skipping
            # keeps a degenerate NO_OP turn out of the trajectory.
            continue
        yield render(primitives)


def _op_click(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    yield render(_click(str(op.get("button", "LMB")), int(op.get("count", 1))))


def _op_key(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    yield render(_chord([str(key) for key in op.get("keys", [])]))


def _op_type(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    text = op.get("text")
    if not isinstance(text, str) or not text:
        raise OracleError("type op needs a non-empty 'text'")
    primitives: list[OrderedPrimitive] = [OrderedPrimitive(kind="type", text=text)]
    # Non-character keys end the typed run and are their own primitives -- the
    # prompt's "type then confirm" recipe.
    for key in op.get("then", []) or []:
        primitives += _chord([str(key)])
    yield render(primitives)


def _op_scroll(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    dx, dy = int(op.get("dx", 0)), int(op.get("dy", 0))
    if (dx, dy) == (0, 0):
        raise OracleError("scroll op must have a nonzero dx or dy")
    yield render([OrderedPrimitive(kind="scroll", dx=dx, dy=dy)])


def _op_wait(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    for _ in range(max(1, int(op.get("turns", 1)))):
        yield _NO_OP


def _op_wait_title(rt: OracleRuntime, op: dict[str, Any]) -> Iterator[str]:
    """NO_OP until a window title matches, then fall through.

    Apps take an unpredictable number of turns to appear, and the prompt tells
    the model to NO_OP and re-check rather than mash the action again -- so the
    oracle demonstrates exactly that, with the real number of waits the real
    fixture needed on this run.
    """
    pattern = re.compile(str(op["pattern"]))
    budget = max(1, int(op.get("max_turns", 6)))
    for _ in range(budget):
        if pattern.search(rt.env.active_title()):
            return
        yield _NO_OP
    if not pattern.search(rt.env.active_title()):
        raise OracleError(f"wait_title: {op['pattern']!r} never appeared in {budget} turn(s)")


_OP_TABLE: dict[str, Callable[[OracleRuntime, dict[str, Any]], Iterator[str]]] = {
    "approach": _op_approach,
    "click": _op_click,
    "key": _op_key,
    "type": _op_type,
    "scroll": _op_scroll,
    "wait": _op_wait,
    "wait_title": _op_wait_title,
    "no_op": lambda rt, op: iter([_NO_OP]),
}


def _resolve_op_bbox(rt: OracleRuntime, op: dict[str, Any]) -> tuple[int, int, int, int]:
    """The op's target in VM pixels.

    Defaults to the turn's own target (``rt.bbox``, already resolved by the
    harness). An explicit ``bbox_norm`` lets one plan aim at something the task
    JSON does not name as its target -- e.g. the search result inside the page,
    when the task's target is the Chrome dock icon.
    """
    target = op.get("target", "turn")
    if target == "turn":
        return rt.bbox
    if isinstance(target, dict) and "bbox_norm" in target:
        raw = target["bbox_norm"]
        if not isinstance(raw, list) or len(raw) != 4:
            raise OracleError(f"bbox_norm needs [x1,y1,x2,y2], got {raw!r}")
        sw, sh = rt.screen
        return (
            round(raw[0] * sw / 1000),
            round(raw[1] * sh / 1000),
            round(raw[2] * sw / 1000),
            round(raw[3] * sh / 1000),
        )
    raise OracleError(f"unsupported oracle target {target!r}")


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------


def validate_plan(spec: Any, *, where: str) -> list[dict[str, Any]]:
    """Structural check of a task's ``oracle`` block, run at suite-load time.

    Deliberately strict and eager: a typo'd op discovered halfway through a
    90-minute harvest run costs a whole VM-hour, and the resulting trajectory is
    silently dropped rather than loudly wrong.
    """
    if not isinstance(spec, dict) or set(spec) - {"plan"} or "plan" not in spec:
        raise ValueError(f"{where}: oracle must be an object with exactly a 'plan' key")
    plan = spec["plan"]
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"{where}: oracle.plan must be a non-empty list")
    for index, op in enumerate(plan):
        if not isinstance(op, dict) or "op" not in op:
            raise ValueError(f"{where}: oracle.plan[{index}] must be an object with an 'op'")
        name = op["op"]
        if name not in _OPS:
            raise ValueError(
                f"{where}: oracle.plan[{index}] unknown op {name!r}; expected one of {sorted(_OPS)}"
            )
        if name == "approach":
            steps = op.get("steps")
            if steps is not None and steps not in _APPROACH_SHAPES:
                raise ValueError(
                    f"{where}: oracle.plan[{index}] steps must be null or one of "
                    f"{sorted(_APPROACH_SHAPES)}"
                )
        if name == "wait_title" and not isinstance(op.get("pattern"), str):
            raise ValueError(f"{where}: oracle.plan[{index}] wait_title needs a 'pattern' string")
        if name == "type" and not isinstance(op.get("text"), str):
            raise ValueError(f"{where}: oracle.plan[{index}] type needs a 'text' string")
    return [dict(op) for op in plan]


def run_plan(rt: OracleRuntime, plan: list[dict[str, Any]]) -> Iterator[str]:
    """Yield one action line per turn until the plan is exhausted.

    The harvest loop refreshes ``rt`` between yields and stops pulling as soon
    as the verifier passes, so a plan that would have gone on longer simply ends
    early -- which is the common case, since most plans carry a trailing wait.
    """
    for op in plan:
        yield from _OP_TABLE[op["op"]](rt, op)
