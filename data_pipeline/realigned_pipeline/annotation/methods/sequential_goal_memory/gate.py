"""Deterministic decision-boundary pre-gate for causal thought sparsity.

Pure and side-effect free: no I/O, no labeler, no randomness. Thought sparsity
in the annotated data is STRUCTURAL — a thought is only ever considered where
the trajectory actually turns — instead of being begged for in prose ("write a
thought only when useful"), which every labeler over-serves.

A semantic event is a decision boundary when the human plausibly had to decide
something rather than continue an in-flight motor plan:

* the day's first event (nothing is oriented yet),
* an event where any goal node (long/mid/short) starts,
* the first event of a new recording segment,
* an event that follows a real pause (> ``gap_s``),
* an event whose PREVIOUS action handed control to the machine — a submit key,
  a modifier shortcut, or an explicit ``wait`` — so the outcome must now be
  read off the screen,
* a scroll that reverses the previous scroll's direction (overshoot / search).

Everything else is motor execution and is annotated memory-only.
"""

from __future__ import annotations

from typing import Any, Sequence

DECISION_GAP_S = 5.0
# Generic computer_use key names (action_format.RDEV_TO_COMPUTER_USE_KEY
# collapses the left/right variants before they reach a tool call). Shift is
# deliberately NOT a modifier here: shift+letter is capitalization, i.e. typing.
SUBMIT_KEYS = frozenset({"enter", "return", "kpenter"})
MODIFIER_KEYS = frozenset({"ctrl", "control", "alt", "option", "command", "cmd",
                           "meta", "super", "win", "fn"})


def _calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    calls = event.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _action(call: dict[str, Any]) -> str:
    return str(call.get("action") or "")


def _keys(call: dict[str, Any]) -> list[str]:
    raw = call.get("keys")
    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, (list, tuple)):
        raw = []
    return [str(key).strip().casefold() for key in raw]


def _hands_over_control(event: dict[str, Any]) -> bool:
    """Whether this event's action makes the machine, not the human, act next."""
    for call in _calls(event):
        action = _action(call)
        if action == "wait":
            return True
        if action == "key" and any(key in SUBMIT_KEYS or key in MODIFIER_KEYS
                                   for key in _keys(call)):
            return True
    return False


def _number(value: Any) -> float:
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if numeric else 0.0


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _scroll_sign(event: dict[str, Any]) -> int | None:
    """Net scroll direction of the event, or None when it does not scroll."""
    total = 0.0
    scrolls = False
    for call in _calls(event):
        if _action(call) in ("scroll", "hscroll"):
            total += _number(call.get("pixels"))
            scrolls = True
    return _sign(total) if scrolls else None


def is_decision_boundary(events: Sequence[dict[str, Any]], index: int,
                         goal_nodes: Sequence[dict[str, Any]], *,
                         gap_s: float = DECISION_GAP_S) -> bool:
    """Whether semantic event ``index`` is a decision boundary (see module doc)."""
    if not 0 <= index < len(events):
        raise IndexError(f"event index {index} outside a {len(events)}-event day")
    if index == 0:
        return True
    if any(int(node["start_event_index"]) == index for node in goal_nodes):
        return True
    event, previous = events[index], events[index - 1]
    if str(event.get("segment_id") or "") != str(previous.get("segment_id") or ""):
        return True
    if float(event["t_day_s"]) - float(previous["t_day_s"]) > gap_s:
        return True
    if _hands_over_control(previous):
        return True
    before, now = _scroll_sign(previous), _scroll_sign(event)
    return bool(before and now and before != now)
