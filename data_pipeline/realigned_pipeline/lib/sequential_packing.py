"""Deterministic packing geometry for the sequential goal-memory recipe.

Pure, deterministic, no I/O and no LLM: this is the shared arithmetic that lets
annotation pass 03c and Stage 04 agree on *where* a training record ends, by
simulating the eval runtime's ``ScreenshotCheckpointController``
(``eval/osworld_runtime.py``) over the annotated semantic-event stream.

Counting mirrors that controller exactly: a segment opens with
``screenshots = 1`` at its first event (one screenshot per semantic event,
i.e. one ``note_screenshot`` per new frame), the count is compared against
``max(_, ceil(capacity * fraction))``, and a fired checkpoint does
``reset_to_current`` — the boundary frame carries over into the next segment.
The only deliberate difference is the trigger fraction: the runtime holds a
fixed 0.7 while training jitters per segment over ``[fraction_low,
fraction_high]`` so the model sees compactions at many context fills. The
threshold floor here is 2 rather than the controller's 1 because a segment
must contain at least one action to be trainable.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

MODES = ("explicit_mid", "explicit_long", "proactive")
DEFAULT_MODE_WEIGHTS = {"explicit_mid": 0.45, "explicit_long": 0.25, "proactive": 0.30}
_MODE_LEVEL = {"explicit_mid": "mid", "explicit_long": "long"}

# mouse_move_rel agreement tolerances: a delta smaller than MOVE_ZERO_DELTA
# (normalized 0-1000 units, i.e. 4% of the screen) is treated as "no motion on
# that axis" and matches either sign; overall travel may differ by up to the
# ratio band. The band is reciprocal-closed so the comparison is symmetric.
MOVE_ZERO_DELTA = 40
MOVE_RATIO_LOW = 0.4
MOVE_RATIO_HIGH = 2.5


@dataclass(frozen=True)
class PackingConfig:
    capacity: int                 # runtime screenshot capacity
    fraction_low: float = 0.5     # per-segment trigger fraction jitter range
    fraction_high: float = 0.85
    seed: int = 0
    n_packings: int = 1

    def __post_init__(self) -> None:
        if self.capacity < 3:
            raise ValueError(f"packing capacity must be >= 3, got {self.capacity!r}")
        if not 0 < self.fraction_low <= self.fraction_high <= 1:
            raise ValueError(
                "packing fractions must satisfy 0 < fraction_low <= fraction_high <= 1, "
                f"got ({self.fraction_low!r}, {self.fraction_high!r})"
            )
        if self.n_packings < 1:
            raise ValueError(f"n_packings must be >= 1, got {self.n_packings!r}")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _rng(key: list[Any]) -> random.Random:
    """A private generator keyed by stable ids only — never global state."""
    return random.Random(int(_hash(key), 16))


def packing_config_hash(cfg: PackingConfig) -> str:
    return _hash(asdict(cfg))


def boundary_events(n_events: int, *, day_tag: str, cfg: PackingConfig,
                    packing_index: int = 0) -> list[int]:
    """Semantic-event indices at which the runtime would compact.

    One jitter draw per segment, in order, so a day's boundary chain is fully
    determined by ``(cfg.seed, day_tag, packing_index)``. The event that trips
    the threshold is the anchor and also the first event of the next segment
    (its screenshot is shared). A boundary on the day's final event is dropped:
    there would be no continuation to train. Result is strictly increasing
    within ``[1, n_events - 2]``.
    """
    if n_events < 0:
        raise ValueError(f"n_events must be non-negative, got {n_events!r}")
    if packing_index < 0:
        raise ValueError(f"packing_index must be non-negative, got {packing_index!r}")
    rng = _rng([cfg.seed, day_tag, packing_index])
    boundaries: list[int] = []
    start = 0
    while start < n_events:
        fraction = rng.uniform(cfg.fraction_low, cfg.fraction_high)
        threshold = max(2, math.ceil(cfg.capacity * fraction))
        anchor = start + threshold - 1
        if anchor > n_events - 2:
            break
        boundaries.append(anchor)
        start = anchor
    return boundaries


def segments_from_boundaries(n_events: int, boundaries: list[int]) -> list[tuple[int, int]]:
    """Inclusive ACTION spans per segment, one per boundary plus a tail.

    The boundary event's action belongs to the NEXT segment; its screenshot
    appears in both records (control turn of the earlier one, opening turn of
    the later), mirroring the runtime, where the control request interrupts
    before the action for the current screenshot is taken.
    """
    if n_events <= 0:
        if boundaries:
            raise ValueError(f"boundaries {boundaries} given for a {n_events}-event day")
        return []
    previous = 0
    for boundary in boundaries:
        if not previous < boundary <= n_events - 1:
            raise ValueError(
                f"boundary {boundary} is not strictly increasing within "
                f"[1, {n_events - 1}] (previous {previous})"
            )
        previous = boundary
    starts = [0, *boundaries]
    ends = [*(boundary - 1 for boundary in boundaries), n_events - 1]
    return list(zip(starts, ends))


def eligible_modes(span: tuple[int, int], goal_nodes: list[dict[str, Any]]) -> list[str]:
    """Goal-rendering modes available for one action span, in ``MODES`` order.

    An explicit mode needs ONE node of that level covering every event in the
    span — a chain of sibling nodes does not qualify, since the record carries
    a single ``GOAL:``. Proactive (hindsight relabeling) is always available.
    """
    start, end = int(span[0]), int(span[1])
    if start > end:
        raise ValueError(f"empty action span {span!r}")
    covering = {
        str(node["level"]) for node in goal_nodes
        if int(node["start_event_index"]) <= start <= end <= int(node["end_event_index"])
    }
    return [mode for mode in MODES
            if mode == "proactive" or _MODE_LEVEL[mode] in covering]


def sample_mode(eligible: list[str], weights: dict[str, float], *, seed: int,
                day_tag: str, packing_index: int, segment_index: int) -> str:
    """Seeded categorical draw over ``eligible`` with weights renormalized.

    Keyed by the segment's stable identity, so re-running Stage 04 with the
    same config reproduces every mode. Independent of the order of
    ``eligible``.
    """
    unknown = sorted(set(eligible) - set(MODES))
    if unknown:
        raise ValueError(f"unknown packing mode(s): {unknown}")
    ordered = [mode for mode in MODES if mode in eligible]
    if not ordered:
        raise ValueError("sample_mode needs at least one eligible mode")
    masses = [float(weights.get(mode) or 0.0) for mode in ordered]
    if any(mass < 0 for mass in masses):
        raise ValueError(f"negative mode weight in {weights}")
    total = sum(masses)
    if total <= 0:
        raise ValueError(f"mode weights {weights} sum to zero over eligible {ordered}")
    draw = _rng([seed, day_tag, packing_index, segment_index]).random() * total
    cumulative = 0.0
    for mode, mass in zip(ordered, masses):
        cumulative += mass
        if draw < cumulative:
            return mode
    return ordered[-1]


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments") if isinstance(call, dict) else None
    return arguments if isinstance(arguments, dict) else {}


def _lower(value: Any) -> str:
    return str(value or "").strip().casefold()


def _keys(arguments: dict[str, Any]) -> list[str]:
    raw = arguments.get("keys")
    if isinstance(raw, str):  # a predictor that wrote one key unwrapped
        raw = [raw]
    elif not isinstance(raw, (list, tuple)):
        raw = []
    return [_lower(key) for key in raw]


def _text(arguments: dict[str, Any]) -> str:
    return " ".join(str(arguments.get("text") or "").split())


def _number(value: Any) -> float:
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if numeric else 0.0


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _delta(arguments: dict[str, Any]) -> tuple[float, float]:
    raw = arguments.get("delta")
    raw = list(raw) if isinstance(raw, (list, tuple)) else []
    raw = [*raw, 0, 0][:2]
    return _number(raw[0]), _number(raw[1])


def _axis_agree(predicted: float, actual: float) -> bool:
    return (abs(predicted) < MOVE_ZERO_DELTA or abs(actual) < MOVE_ZERO_DELTA
            or _sign(predicted) == _sign(actual))


def _move_agree(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
    px, py = _delta(predicted)
    ax, ay = _delta(actual)
    if not (_axis_agree(px, ax) and _axis_agree(py, ay)):
        return False
    predicted_travel, actual_travel = math.hypot(px, py), math.hypot(ax, ay)
    if predicted_travel < MOVE_ZERO_DELTA and actual_travel < MOVE_ZERO_DELTA:
        return True
    if predicted_travel <= 0 or actual_travel <= 0:
        return False
    ratio = predicted_travel / actual_travel
    return MOVE_RATIO_LOW <= ratio <= MOVE_RATIO_HIGH


def _scroll_agree(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
    return _sign(_number(predicted.get("pixels"))) == _sign(_number(actual.get("pixels")))


def _button_agree(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
    return _lower(predicted.get("button")) == _lower(actual.get("button"))


def _key_agree(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
    return _lower(predicted.get("key")) == _lower(actual.get("key"))


_CALL_RULES: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {
    "key": lambda predicted, actual: _keys(predicted) == _keys(actual),
    "key_down": _key_agree,
    "key_up": _key_agree,
    "type": lambda predicted, actual: _text(predicted) == _text(actual),
    "mouse_move_rel": _move_agree,
    "scroll": _scroll_agree,
    "hscroll": _scroll_agree,
    "wait": lambda predicted, actual: True,
    "terminate": lambda predicted, actual: (
        _lower(predicted.get("status")) == _lower(actual.get("status"))),
    **{name: _button_agree for name in (
        "left_click", "right_click", "middle_click", "double_click", "triple_click",
        "button_down", "button_up",
    )},
}


def actions_agree(predicted: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    """Whether two ordered ``computer_use`` tool-call lists mean the same thing.

    Used by the annotation agreement gate: a thought is only revealed where the
    predictor diverges from the recorded human action, so tolerance is on the
    motor details (exact pixel travel, wait duration) and strict on intent
    (which keys, which text, which direction). A malformed or unrecognized
    predicted call simply disagrees — for the gate that is the safe direction.
    """
    if len(predicted) != len(actual):
        return False
    for predicted_call, actual_call in zip(predicted, actual):
        want = _arguments(predicted_call)
        got = _arguments(actual_call)
        action = str(want.get("action") or "")
        if action != str(got.get("action") or ""):
            return False
        rule = _CALL_RULES.get(action)
        if rule is None or not rule(want, got):
            return False
    return True
