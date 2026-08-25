"""Strict action parsers for the CUA micro-eval suite.

Ported from yll/cua-micro-evals (juergen). Deliberately NOT merged into
action_parser.py: that module's ``OrderedPrimitive``/``OrderedAction`` are
this branch's OWN independent evolution of the shared ordered-mini-program
vocabulary (``input_name``, ``render()``, ``as_key_event()``, no ``keys``/
``count``/``status`` fields) for the ``ordered_events_v2/v3`` formats
freeroll/grounding dispatch against. yll's branch evolved the SAME class
names in a different, incompatible direction (``name`` instead of
``input_name``, plus ``keys``/``count``/``status`` for the richer
``computer_use_rel_step_v1`` and native-Qwen3VL-cua vocabularies: click,
button_down/up, key_combo, key_down/up, wait, terminate). Reusing this
branch's ``OrderedPrimitive``/``OrderedAction`` here would either crash on
the first unexpected constructor kwarg or silently miscarry state -- so
this module defines its own ``RelStepPrimitive``/``RelStepAction`` instead.
``cua_micro_eval.py`` imports them aliased back to ``OrderedAction``/
``OrderedPrimitive`` so its body (copied from yll's branch) needed no
further changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from action_parser import ComputerUseCall

# --------------------------------------------------------------------------
# Shared action vocabulary (rel-step + native-Qwen3VL-cua superset).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelStepPrimitive:
    """One primitive of an ordered/native action program.

    ``computer_use_rel_step_v1`` tool calls reuse ``move`` (mouse_move_rel),
    ``scroll``/``hscroll`` (dy/dx in pixels; positive up/right, matching
    pyautogui), and ``type``, and add:

    - ``"click"``        ``name`` in left/right/middle, ``count`` 1/2/3
                         (left/right/middle_click, double_click, triple_click)
    - ``"button_down"``/``"button_up"``  ``name`` in left/right/middle
    - ``"key_combo"``    ``keys``: pyautogui names pressed in order,
                         released in reverse (the `key` action)
    - ``"key_down"``/``"key_up"``        ``name``: one pyautogui key name
    - ``"wait"``         no fields; NEVER dispatched (behaves like NO_OP)
    - ``"terminate"``    ``status`` in success/failure; NEVER dispatched
                         (the rollout loop's stop condition)
    """

    kind: str
    dx: int | None = None
    dy: int | None = None
    name: str | None = None
    mouse_button: int | None = None  # 1/2/3 if a mouse button, else None
    text: str | None = None
    keys: tuple[str, ...] | None = None  # key_combo only
    count: int | None = None  # click only: 1/2/3
    status: str | None = None  # terminate only: "success" | "failure"


@dataclass(frozen=True)
class RelStepAction:
    """Parsed ordered action: primitives in emission order (empty for NO_OP)."""

    primitives: tuple[RelStepPrimitive, ...]
    no_op: bool

    @property
    def has_left_click_press(self) -> bool:
        return any(
            (p.kind == "down" and p.name == "LMB")
            or (p.kind in ("click", "button_down") and p.name == "left")
            for p in self.primitives
        )


# --------------------------------------------------------------------------
# <think> stripping, shared by both parsers below.
# --------------------------------------------------------------------------


def _strip_leading_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block (mirrors freeroll's
    ``_strip_think``): opener present -> cut through the FIRST closer;
    unterminated opener -> the whole reply is thought (return "");
    dangling closer with no opener -> cut through it; else unchanged."""
    s = text.lstrip()
    if s.startswith("<think>"):
        end = s.find("</think>")
        if end == -1:
            return ""
        return s[end + len("</think>") :]
    end = s.find("</think>")
    if end != -1 and "<think>" not in s[:end]:
        return s[end + len("</think>") :]
    return text


_CUA_V4_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)

_CUA_V4_BUTTONS = ("left", "right", "middle")
_CUA_V4_STATUSES = ("success", "failure")

# click-family action -> (pyautogui button, click count).
_CUA_V4_CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}

# --------------------------------------------------------------------------
# Official Qwen3-VL computer-use cookbook tool calls.
# --------------------------------------------------------------------------

_QWEN3VL_CUA_ACTIONS = frozenset(
    {
        "key",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "scroll",
        "hscroll",
        "wait",
        "terminate",
        "answer",
    }
)
_QWEN3VL_CUA_ARGUMENTS = frozenset(
    {"action", "keys", "text", "coordinate", "pixels", "time", "status"}
)


def _validate_qwen3vl_grid_coordinate(coordinate: object, *, where: str) -> None:
    """A cookbook ``coordinate`` argument: two numbers on the 0..1000 grid."""
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinate
        )
        or any(not 0 <= float(value) <= 1000 for value in coordinate)
    ):
        raise ValueError(f"{where}: coordinate must be two numbers on the 0..1000 grid")


def _validate_qwen3vl_cua_arguments(arguments: dict, *, where: str) -> None:
    """Validate the executable portion of Qwen3-VL's cookbook schema."""
    unknown = set(arguments) - _QWEN3VL_CUA_ARGUMENTS
    if unknown:
        raise ValueError(f"{where}: unknown argument(s) {sorted(unknown)}")
    action = arguments.get("action")
    if action not in _QWEN3VL_CUA_ACTIONS:
        raise ValueError(f"{where}: unknown or missing action {action!r}")

    if action == "key":
        keys = arguments.get("keys")
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key.strip() for key in keys)
        ):
            raise ValueError(f"{where}: keys must be a non-empty string array")
    if action in {"type", "answer"} and not isinstance(arguments.get("text"), str):
        raise ValueError(f"{where}: text must be a string for action {action!r}")
    if action in {"mouse_move", "left_click_drag"}:
        _validate_qwen3vl_grid_coordinate(arguments.get("coordinate"), where=where)
    elif action in _CUA_V4_CLICKS and arguments.get("coordinate") is not None:
        # The cookbook schema documents every click as happening "at a
        # specified (x, y) pixel coordinate", and off-the-shelf Qwen3-VL
        # overwhelmingly emits the click that way -- one call carrying the
        # target -- rather than a separate mouse_move first. The coordinate is
        # not *required* (a bare click at the current cursor is also legal, and
        # is what computer_use_rel_step_v1-style replies look like), but when
        # present it is executed, so it has to be validated like any other.
        _validate_qwen3vl_grid_coordinate(arguments.get("coordinate"), where=where)
    if action in {"scroll", "hscroll"}:
        pixels = arguments.get("pixels")
        if isinstance(pixels, bool) or not isinstance(pixels, (int, float)):
            raise ValueError(f"{where}: pixels must be a number for action {action!r}")
    if action == "wait":
        wait_s = arguments.get("time")
        if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)):
            raise ValueError(f"{where}: time must be a number for action 'wait'")
    if action == "terminate" and arguments.get("status") not in _CUA_V4_STATUSES:
        raise ValueError(
            f"{where}: status must be one of {_CUA_V4_STATUSES}, got {arguments.get('status')!r}"
        )


def parse_qwen3vl_computer_use_action(text: str) -> tuple[ComputerUseCall, ...]:
    """Strictly parse official Qwen3-VL native ``computer_use`` calls.

    The official cookbook uses the Nous function-call prompt: an optional
    leading Thinking block followed only by one or more ``<tool_call>`` JSON
    blocks. Prose outside those blocks, malformed JSON, and non-computer tools
    are rejected transactionally.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_qwen3vl_computer_use_action expects str, got {type(text)!r}")
    stripped = _strip_leading_think(text)
    matches = list(_CUA_V4_TOOL_CALL_RE.finditer(stripped))
    if not matches:
        raise ValueError(f"no <tool_call> blocks found in {stripped.strip()!r}")
    previous_end = 0
    for match in matches:
        fragment = stripped[previous_end : match.start()].strip()
        if fragment:
            raise ValueError(f"unexpected content outside <tool_call> blocks: {fragment!r}")
        previous_end = match.end()
    tail = stripped[previous_end:].strip()
    if tail:
        raise ValueError(f"unexpected content outside <tool_call> blocks: {tail!r}")

    calls: list[ComputerUseCall] = []
    for index, match in enumerate(matches, start=1):
        where = f"tool call {index}"
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{where}: payload must be a JSON object")
        unknown_top = set(payload) - {"name", "arguments"}
        if unknown_top:
            raise ValueError(f"{where}: unknown top-level key(s) {sorted(unknown_top)}")
        if payload.get("name") != "computer_use":
            raise ValueError(f"{where}: name must be 'computer_use', got {payload.get('name')!r}")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{where}: arguments must be a JSON object")
        _validate_qwen3vl_cua_arguments(arguments, where=where)
        calls.append(ComputerUseCall(name="computer_use", arguments=arguments))
    return tuple(calls)


# --------------------------------------------------------------------------
# computer_use_rel_step_v1 -- strict, transactional, relative-mouse.
# Binding contract: data_pipeline/realigned_pipeline/action_specs/
# computer_use_rel_step_v1.json (movement scales/directions, scroll steps,
# max tool calls, typing char cap).
# --------------------------------------------------------------------------

_REL_STEP_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline/realigned_pipeline/action_specs/computer_use_rel_step_v1.json"
)
_REL_STEP_SPEC = json.loads(_REL_STEP_SPEC_PATH.read_text())
if _REL_STEP_SPEC.get("format") != "computer_use_rel_step_v1":  # pragma: no cover
    raise RuntimeError(f"wrong relative-step spec at {_REL_STEP_SPEC_PATH}")

_REL_STEP_VALID_DELTAS = frozenset(
    (int(scale) * int(direction[0]), int(scale) * int(direction[1]))
    for scale in _REL_STEP_SPEC["movement_scales"]
    for direction in _REL_STEP_SPEC["directions"]
)
_REL_STEP_SCROLL_STEPS = frozenset(int(v) for v in _REL_STEP_SPEC["scroll_steps"])
_REL_STEP_MAX_CALLS = int(_REL_STEP_SPEC["max_tool_calls"])
_REL_STEP_MAX_TYPE_CHARS = int(_REL_STEP_SPEC["typing_max_chars"])
_REL_STEP_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "mouse_move_rel": ("delta",),
    "left_click": (),
    "right_click": (),
    "middle_click": (),
    "double_click": (),
    "triple_click": (),
    "button_down": ("button",),
    "button_up": ("button",),
    "key": ("keys",),
    "type": ("text",),
    "scroll": ("steps",),
    "hscroll": ("steps",),
    "wait": (),
    "terminate": ("status",),
}


def _rel_step_primitive(  # noqa: PLR0911
    arguments: dict, *, where: str
) -> RelStepPrimitive:
    """Validate one rel-step arguments object without coercion or repair."""
    action = arguments.get("action")
    if not isinstance(action, str) or action not in _REL_STEP_REQUIRED_ARGS:
        raise ValueError(f"{where}: unknown or missing action {action!r}")
    required = set(_REL_STEP_REQUIRED_ARGS[action])
    extra = set(arguments) - {"action"} - required
    missing = required - set(arguments)
    if extra:
        raise ValueError(f"{where}: unknown argument(s) {sorted(extra)} for {action!r}")
    if missing:
        raise ValueError(f"{where}: missing argument(s) {sorted(missing)} for {action!r}")

    if action == "mouse_move_rel":
        delta = arguments["delta"]
        if (
            not isinstance(delta, list)
            or len(delta) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in delta)
            or tuple(delta) not in _REL_STEP_VALID_DELTAS
        ):
            raise ValueError(f"{where}: delta must be an exact fixed relative step, got {delta!r}")
        return RelStepPrimitive(kind="move", dx=delta[0], dy=delta[1])
    if action == "key":
        keys = arguments["keys"]
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(k, str) or not k.strip() for k in keys)
        ):
            raise ValueError(f"{where}: keys must be a non-empty string array")
        return RelStepPrimitive(kind="key_combo", keys=tuple(keys))
    if action == "type":
        value = arguments["text"]
        if not isinstance(value, str) or not value or len(value) > _REL_STEP_MAX_TYPE_CHARS:
            raise ValueError(f"{where}: text must contain 1..{_REL_STEP_MAX_TYPE_CHARS} characters")
        return RelStepPrimitive(kind="type", text=value)
    if action in _CUA_V4_CLICKS:
        button, count = _CUA_V4_CLICKS[action]
        return RelStepPrimitive(kind="click", name=button, count=count)
    if action in ("button_down", "button_up"):
        button = arguments["button"]
        if button not in _CUA_V4_BUTTONS:
            raise ValueError(f"{where}: invalid button {button!r}")
        return RelStepPrimitive(kind=action, name=button)
    if action in ("scroll", "hscroll"):
        steps = arguments["steps"]
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps not in _REL_STEP_SCROLL_STEPS
        ):
            raise ValueError(
                f"{where}: steps must be one of {sorted(_REL_STEP_SCROLL_STEPS)}, got {steps!r}"
            )
        return RelStepPrimitive(
            kind="scroll",
            dx=steps if action == "hscroll" else 0,
            dy=steps if action == "scroll" else 0,
        )
    if action == "wait":
        return RelStepPrimitive(kind="wait")
    if action == "terminate":
        if arguments["status"] != "success":
            raise ValueError(f"{where}: terminate status must be 'success'")
        return RelStepPrimitive(kind="terminate", status="success")
    raise AssertionError(f"unhandled relative-step action {action!r}")  # pragma: no cover


def _parse_rel_step_block(body: str) -> RelStepPrimitive:
    where = f"tool call {body.strip()!r}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{where}: payload must be an object")
    if set(payload) != {"name", "arguments"}:
        raise ValueError(f"{where}: top-level keys must be exactly name and arguments")
    if payload["name"] != _REL_STEP_SPEC["tool_name"]:
        raise ValueError(f"{where}: name must be {_REL_STEP_SPEC['tool_name']!r}")
    arguments = payload["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError(f"{where}: arguments must be an object")
    return _rel_step_primitive(arguments, where=where)


def _validate_rel_step_program(primitives: tuple[RelStepPrimitive, ...]) -> None:
    """Enforce response-level atomicity before any primitive is dispatched."""
    if len(primitives) > _REL_STEP_MAX_CALLS:
        raise ValueError(f"too many tool calls: {len(primitives)} > {_REL_STEP_MAX_CALLS}")
    terminations = [i for i, p in enumerate(primitives) if p.kind == "terminate"]
    if terminations:
        if len(primitives) != 1:
            raise ValueError("terminate must be the only tool call in a reply")
        return

    state_calls = [i for i, p in enumerate(primitives) if p.kind in ("button_down", "button_up")]
    move_calls = [i for i, p in enumerate(primitives) if p.kind == "move"]
    if state_calls:
        if len(state_calls) != 2:
            raise ValueError("drag must contain exactly one button_down and one button_up")
        down_i, up_i = state_calls
        down, up = primitives[down_i], primitives[up_i]
        if (
            down.kind != "button_down"
            or up.kind != "button_up"
            or down.name != up.name
            or down_i != 0
            or up_i != len(primitives) - 1
            or not move_calls
            or any(not (down_i < i < up_i) for i in move_calls)
            or any(p.kind not in ("button_down", "move", "button_up") for p in primitives)
        ):
            raise ValueError(
                "drag must be button_down, one or more fixed moves, button_up for the same button"
            )
        return

    if len(move_calls) > 1:
        raise ValueError("normal replies may contain at most one mouse move")
    if move_calls and any(p.kind != "move" for p in primitives):
        raise ValueError("a normal mouse move must be the only action in its reply")


def parse_computer_use_rel_step_action(text: str) -> RelStepAction:
    """Strict, transactional parser for ``computer_use_rel_step_v1``.

    An optional leading think block is accepted. Everything after it must be
    valid tool-call blocks, and the whole program is validated before the
    returned action can reach the executor. There is deliberately no tolerant
    parser for this safety-critical format.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_computer_use_rel_step_action expects str, got {type(text)!r}")
    s = _strip_leading_think(text)
    matches = list(_CUA_V4_TOOL_CALL_RE.finditer(s))
    if not matches:
        raise ValueError(f"no <tool_call> blocks found in {s.strip()!r}")
    prev_end = 0
    for match in matches:
        fragment = s[prev_end : match.start()].strip()
        if fragment:
            raise ValueError(f"unexpected content outside tool calls: {fragment!r}")
        prev_end = match.end()
    tail = s[prev_end:].strip()
    if tail:
        raise ValueError(f"unexpected content outside tool calls: {tail!r}")
    primitives = tuple(_parse_rel_step_block(m.group(1)) for m in matches)
    _validate_rel_step_program(primitives)
    return RelStepAction(primitives=primitives, no_op=all(p.kind == "wait" for p in primitives))
