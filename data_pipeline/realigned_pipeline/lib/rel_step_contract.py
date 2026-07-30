"""Canonical ``computer_use_rel_step_v1`` contract.

The JSON file next to the realigned pipeline is the source of truth shared by
stage 04, eval parsing, and the system prompt.  Keeping the small loader here
lets data-pipeline code use typed constants without duplicating the contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "action_specs/computer_use_rel_step_v1.json"
)


def load_rel_step_spec() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text())
    if spec.get("format") != "computer_use_rel_step_v1":
        raise ValueError(f"wrong relative-step spec at {SPEC_PATH}")
    return spec


SPEC = load_rel_step_spec()
MOVEMENT_SCALES = tuple(int(v) for v in SPEC["movement_scales"])
DIRECTIONS = tuple(tuple(int(x) for x in v) for v in SPEC["directions"])
VALID_MOVE_DELTAS = frozenset(
    (scale * direction[0], scale * direction[1])
    for scale in MOVEMENT_SCALES
    for direction in DIRECTIONS
)
SCROLL_STEPS = frozenset(int(v) for v in SPEC["scroll_steps"])
MAX_TOOL_CALLS = int(SPEC["max_tool_calls"])
TYPING_GAP_S = float(SPEC["typing_gap_s"])
TYPING_MAX_CHARS = int(SPEC["typing_max_chars"])


def quantize_direction(dx: float, dy: float) -> tuple[int, int] | None:
    """Nearest of the eight screen-relative directions; magnitude is ignored."""
    if dx == 0 and dy == 0:
        return None
    norm = (dx * dx + dy * dy) ** 0.5
    return max(
        DIRECTIONS,
        key=lambda d: (dx * d[0] + dy * d[1]) / ((d[0] ** 2 + d[1] ** 2) ** 0.5 * norm),
    )


def rel_step_delta(dx: float, dy: float, scale: int) -> tuple[int, int] | None:
    if scale not in MOVEMENT_SCALES:
        raise ValueError(f"invalid relative-step scale {scale}")
    direction = quantize_direction(dx, dy)
    if direction is None:
        return None
    return scale * direction[0], scale * direction[1]
