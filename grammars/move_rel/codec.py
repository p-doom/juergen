"""The ``move_rel`` computer_use grammar: explicit relative move, normalized.

Two corrections over the ``native_rel_v1`` dialect, both enforced here by
construction:

1. A relative move is its own action. v1 folded the delta into a click's
   ``coordinate``, which native computer_use / pyautogui read as an absolute
   target. Here a move-and-click is two calls — ``move_rel
   {coordinate:[dx,dy]}`` then a coordinate-less click — and a click that
   carries a coordinate is a parse error naming v1.
2. Coordinates are normalized to thousandths of the screen, per axis.
   ``compile`` is the only place that grid is converted back to pixels
   (``px = norm * dim / 1000``, the exact inverse of the encoder's
   ``norm = round(px / dim * 1000)``).

Collapses into one codec:

* ``osworld_parity/vendor/move_rel_format.py`` (``norm_axis``,
  ``split_and_normalize``, ``convert_turn_v2``, and its inlined SYSTEM_PROMPT),
* ``osworld_parity/scripts/convert_abs_to_moverel.py::split_already_normalized``
  (the same split for input that is already normalized),
* ``eval/osworld_system_prompts.py`` ``native_rel_v2`` and
  ``osworld_parity/split/moverel_system_prompt.txt`` (three byte-identical
  copies of one prompt, kept in sync by hand),
* ``rl/grounding/parsing.py::parse_first_move`` (the RL rollout reader, which
  already keyed on ``coordinate`` and had ``move_rel`` bolted onto its move set).

Semantics chosen where the sources disagreed:

* ``coordinate`` — not ``delta`` — carries the relative offset. That is what
  ``rl/grounding/parsing.py`` reads and what the trained checkpoints emit;
  ``rl/computer_use/actions.py`` requires ``delta``, which is rejected here with
  a message saying so.
* A coordinate on a click is rejected: that is the absolute / relative confusion
  this grammar exists to remove. ``rl/grounding/parsing.py`` accepted it because
  it also handles v1.
* Out-of-range normalized values are accepted, not clamped at parse. The prompt
  states the [-999, 999] range, and resolution clamps to the display anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pixeldesk.geometry import DisplayGeometry
from pixeldesk.ir import Operation

from .. import _support

#: Thousandths of the screen. The eval harness's ``--rel_coord_grid 1000``.
GRID = 1000

PRODUCER = {"prompt_id": "native_rel_v2"}

CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
BUTTON_NAMES = ("left", "right", "middle")
MAX_WAIT_SECONDS = 10.0


class MoveRelError(ValueError):
    """Malformed move_rel computer_use action."""


def pixels_from_norm(norm: int, dimension: int) -> int:
    """Thousandths of one axis -> pixels on that axis."""
    return int(round(float(norm) * float(dimension) / float(GRID)))


def norm_from_pixels(pixels: int, dimension: int) -> int:
    """Pixels on one axis -> thousandths of that axis. The encoder's direction."""
    if not dimension:
        return 0
    return int(round(float(pixels) / float(dimension) * float(GRID)))


@dataclass(frozen=True)
class MoveRelCall:
    """One validated call. ``delta`` is a normalized relative offset."""

    action: str
    delta: tuple[int, int] | None = None
    button: str = ""
    keys: tuple[str, ...] = ()
    text: str = ""
    scroll: int = 0
    seconds: float = 0.0
    status: str = ""

    def arguments(self) -> dict[str, Any]:
        value: dict[str, Any] = {"action": self.action}
        if self.delta is not None:
            value["coordinate"] = [self.delta[0], self.delta[1]]
        if self.button:
            value["button"] = self.button
        if self.keys:
            value["keys"] = list(self.keys)
        if self.action == "type" and self.text:
            value["text"] = self.text
        if self.action == "scroll":
            value["pixels"] = self.scroll
        if self.action == "wait":
            value["time"] = self.seconds
        if self.action == "terminate":
            value["status"] = self.status
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.arguments()


@dataclass(frozen=True)
class MoveRelAction:
    """Every call the turn emitted, in order."""

    calls: tuple[MoveRelCall, ...] = ()
    prompt_digest: str = field(default="", compare=False)

    @property
    def terminate(self) -> bool:
        return any(call.action == "terminate" for call in self.calls)

    @property
    def status(self) -> str:
        for call in self.calls:
            if call.action == "terminate":
                return call.status
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {"calls": [call.to_dict() for call in self.calls]}


class MoveRelCodec:
    """You operate a desktop computer using the computer_use tool.

    The first user turn shows the initial screen and the user's goal; each
    subsequent user turn shows the current screen. Reply with one or more
    computer_use tool calls that advance toward the goal.

    Mouse movement is RELATIVE and NORMALIZED. To move the cursor, emit a
    `move_rel` action whose `coordinate` is a [dx, dy] offset from the CURRENT
    cursor position, expressed in thousandths of the screen (each axis in
    [-999, 999]; dx = 1000 spans the full width, dy = 1000 the full height;
    positive dx = right, positive dy = down). It is NOT an absolute screen
    coordinate. Look at the visible cursor in the screenshot to judge how far
    and in which direction to move. To click a target, FIRST `move_rel` by the
    relative offset, THEN issue a click with NO coordinate — the click lands at
    the current cursor position.
    """

    name = "move_rel"

    #: Empty by design: a move_rel and its follow-on click are separate calls in
    #: the same turn, so no token sequence marks the end of a turn.
    stop_sequences: tuple[str, ...] = ()

    def tool_description(self) -> None:
        """Use a mouse and keyboard to interact with a computer, and take screenshots.
        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * The mouse cannot jump to an absolute (x, y). Every move is a relative offset from where the cursor already is, in thousandths of the screen.
        * It may take several turns to arrive: the cursor moves visibly between turns, so correct your aim as you close in.
        * Land the cursor tip on the CENTER of the target element before clicking, not on its edge.
        """

    @_support.production("move_rel")
    def _move_rel(self) -> None:
        """Move the cursor by the relative normalized offset `coordinate` = [dx, dy] (pyautogui.moveRel)."""

    @_support.production("left_click")
    def _left_click(self) -> None:
        """Click the left button at the CURRENT cursor position. Takes no coordinate; move first with move_rel."""

    @_support.production("right_click")
    def _right_click(self) -> None:
        """Click the right button at the current cursor position. Takes no coordinate."""

    @_support.production("middle_click")
    def _middle_click(self) -> None:
        """Click the middle button at the current cursor position. Takes no coordinate."""

    @_support.production("double_click")
    def _double_click(self) -> None:
        """Double click at the current cursor position. Takes no coordinate."""

    @_support.production("triple_click")
    def _triple_click(self) -> None:
        """Triple click at the current cursor position. Takes no coordinate."""

    @_support.production("mouse_down")
    def _mouse_down(self) -> None:
        """Press and HOLD a mouse button (`button` = 'left', 'right', 'middle'). A drag is move_rel, mouse_down, one or more move_rel, then mouse_up."""

    @_support.production("mouse_up")
    def _mouse_up(self) -> None:
        """Release a mouse button."""

    @_support.production("key")
    def _key(self) -> None:
        """Press a key or chord, e.g. ['ctrl','a'], ['enter'], ['tab']. Presses in order, releases in reverse."""

    @_support.production("key_down")
    def _key_down(self) -> None:
        """Hold keys across steps."""

    @_support.production("key_up")
    def _key_up(self) -> None:
        """Release held keys."""

    @_support.production("type")
    def _type(self) -> None:
        """Type a string of text. The text cannot contain a newline; press Return with `key`."""

    @_support.production("scroll")
    def _scroll(self) -> None:
        """Scroll the wheel: positive `pixels` scroll up, negative scroll down."""

    @_support.production("wait")
    def _wait(self) -> None:
        """Do nothing this step, for `time` seconds."""

    @_support.production("terminate")
    def _terminate(self) -> None:
        """The goal is complete (`status` = 'success' or 'failure')."""

    def notes(self) -> None:
        """For each action, return a JSON object within <tool_call></tool_call> tags. To move the cursor 12 right / 8 up (normalized) and left-click there:
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [12, -8]}}
        </tool_call>
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "left_click"}}
        </tool_call>
        """

    def describe(self) -> str:
        return _support.render_tool_prompt(
            self,
            properties={
                "coordinate": {
                    "description": "[dx, dy]: the RELATIVE offset from the current cursor, in thousandths of the screen. Required only by `action=move_rel`.",
                    "type": "array",
                },
                "button": {
                    "description": "'left', 'right' or 'middle'. Required only by `action=mouse_down` and `action=mouse_up`.",
                    "type": "string",
                },
                "keys": {
                    "description": "Required only by `action=key`, `action=key_down` and `action=key_up`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type`.",
                    "type": "string",
                },
                "pixels": {
                    "description": "The amount of scrolling. Required only by `action=scroll`.",
                    "type": "number",
                },
                "time": {
                    "description": "The seconds to wait. Required only by `action=wait`.",
                    "type": "number",
                },
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
        )

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def report(self) -> dict[str, Any]:
        return _support.drift_report(self, producer=PRODUCER)

    def parse(self, text: str) -> MoveRelAction:
        calls = tuple(
            self.validate_call(arguments)
            for arguments in _support.iter_tool_calls(text)
        )
        if not calls:
            raise MoveRelError("no valid computer_use tool call found")
        return MoveRelAction(calls=calls, prompt_digest=self.digest)

    def format(self, action: MoveRelAction) -> str:
        return _support.render_tool_calls([call.arguments() for call in action.calls])

    def compile(
        self,
        text: str,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        return self.compile_action(self.parse(text), geometry, cursor)

    def compile_action(
        self,
        action: MoveRelAction,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Denormalize each delta, fold it onto ``cursor``, emit absolute pixels."""
        width, height = _support.screen_size(geometry)
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        for call in action.calls:
            name = call.action
            if name == "move_rel":
                assert call.delta is not None
                target = _support.clamp(
                    (
                        here[0] + pixels_from_norm(call.delta[0], width),
                        here[1] + pixels_from_norm(call.delta[1], height),
                    ),
                    geometry,
                )
                if target != here:
                    operations.append(_support.move_to(target))
                here = target
            elif name in CLICKS:
                button, repeats = CLICKS[name]
                for _ in range(repeats):
                    operations.append(_support.mouse_down(button))
                    operations.append(_support.mouse_up(button))
            elif name in ("mouse_down", "mouse_up"):
                button = call.button
                assert button, "validate_call and the lift both set button"
                operations.append(
                    _support.mouse_down(button)
                    if name == "mouse_down"
                    else _support.mouse_up(button)
                )
            elif name == "key":
                operations.extend(_support.key_down(key) for key in call.keys)
                operations.extend(_support.key_up(key) for key in reversed(call.keys))
            elif name == "key_down":
                operations.extend(_support.key_down(key) for key in call.keys)
            elif name == "key_up":
                operations.extend(_support.key_up(key) for key in call.keys)
            elif name == "type":
                operations.append(
                    _support.lower_typing(call.text, error=MoveRelError)
                )
            elif name == "scroll":
                if call.scroll:
                    operations.append(_support.scroll(0, call.scroll))
            elif name == "wait":
                operations.append(_support.wait(call.seconds))
            elif name == "terminate":
                continue
            else:  # pragma: no cover - validate_call fixes the set
                raise MoveRelError(f"unsupported action: {name!r}")
        return tuple(operations)

    def validate_call(self, arguments: dict[str, Any]) -> MoveRelCall:
        name = str(arguments.get("action", "")).strip()
        known = {item.syntax for item in _support.productions(self)}
        if name not in known:
            raise MoveRelError(f"unsupported move_rel action: {name!r}")
        if "delta" in arguments:
            raise MoveRelError(
                "move_rel carries its relative offset in `coordinate`; `delta` is "
                "the rl/computer_use schema and is not this grammar"
            )
        if name == "move_rel":
            return MoveRelCall(name, delta=self._delta(arguments))
        if "coordinate" in arguments:
            raise MoveRelError(
                f"{name!r} takes no coordinate: a relative offset in a click's "
                "coordinate is the native_rel_v1 defect this grammar removes. "
                "Emit move_rel, then the coordinate-less action."
            )
        if name in ("key", "key_down", "key_up"):
            return MoveRelCall(name, keys=self._keys(arguments))
        if name == "type":
            text = arguments.get("text")
            if not isinstance(text, str) or not text:
                raise MoveRelError("type requires non-empty text")
            return MoveRelCall(name, text=text)
        if name == "scroll":
            if "clicks" in arguments:
                raise MoveRelError(
                    "move_rel spells the scroll magnitude `pixels`; `clicks` is "
                    "rung1's native_absolute spelling and is not this grammar"
                )
            try:
                return MoveRelCall(name, scroll=int(round(float(arguments.get("pixels", 0)))))
            except (TypeError, ValueError) as exc:
                raise MoveRelError("scroll pixels must be numeric") from exc
        if name == "wait":
            try:
                seconds = float(arguments.get("time", 1.0))
            except (TypeError, ValueError) as exc:
                raise MoveRelError("wait time must be numeric") from exc
            return MoveRelCall(name, seconds=max(0.0, min(MAX_WAIT_SECONDS, seconds)))
        if name == "terminate":
            return MoveRelCall(name, status=str(arguments.get("status", "success")))
        if name in ("mouse_down", "mouse_up"):
            button = str(arguments.get("button", "left"))
            if button not in BUTTON_NAMES:
                raise MoveRelError(f"unsupported mouse button: {button!r}")
            return MoveRelCall(name, button=button)
        return MoveRelCall(name)

    def _delta(self, arguments: dict[str, Any]) -> tuple[int, int]:
        raw = arguments.get("coordinate")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise MoveRelError("move_rel requires a two-item coordinate [dx, dy]")
        try:
            dx, dy = (int(round(float(value))) for value in raw)
        except (TypeError, ValueError) as exc:
            raise MoveRelError("coordinate values must be numeric") from exc
        return dx, dy

    def _keys(self, arguments: dict[str, Any]) -> tuple[str, ...]:
        keys = arguments.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list) or not keys:
            raise MoveRelError(
                f"{arguments.get('action')!r} requires a non-empty keys array"
            )
        return tuple(_support.normalize_key(key, error=MoveRelError) for key in keys)

    def action_from_operations(
        self,
        operations: Sequence[Operation],
        *,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> MoveRelAction:
        """Absolute Operations -> calls. The inverse of ``compile_action``.

        Every absolute endpoint is differenced against the running cursor and
        normalized, so this is the one lift that is not exact: the thousandths
        grid quantises, and re-compiling can land a pixel or two off. That is the
        grammar's own ceiling — the same one training pays — not a defect of the
        lift, and it is why the vectors use deltas that round-trip exactly.

        A ``glide_to`` becomes a plain ``move_rel``, which PRESERVES the drag
        (the button is held across it) and drops only the stroke's duration; this
        schema expresses a drag as move, mouse_down, move, mouse_up and has no
        single drag action. Termination is a ``terminate`` call, not a flag.
        """
        status = _support.terminate_status(terminate, error=MoveRelError)
        groups = _support.group_operations(
            operations, geometry=geometry, cursor=cursor, error=MoveRelError
        )
        width, height = _support.screen_size(geometry)
        calls: list[MoveRelCall] = []
        here = _support.clamp(cursor, geometry)
        for group in groups:
            kind = group.kind
            if kind in ("move", "stroke"):
                assert group.target is not None
                pixels = (group.target[0] - here[0], group.target[1] - here[1])
                delta = (
                    norm_from_pixels(pixels[0], width),
                    norm_from_pixels(pixels[1], height),
                )
                # Guarded per axis: testing only ``delta == (0, 0)`` let a mixed
                # move through, because on a 4000-wide display a (1, 100) pixel
                # move normalises to (0, 50) and the horizontal component
                # vanishes silently.
                vanished = tuple(
                    "xy"[axis]
                    for axis in (0, 1)
                    if delta[axis] == 0 and pixels[axis] != 0
                )
                if vanished:
                    where = (
                        "both axes"
                        if len(vanished) == 2
                        else f"the {vanished[0]} axis"
                    )
                    raise MoveRelError(
                        f"a {pixels} pixel move is finer than the "
                        f"{GRID}ths grid and would vanish on {where}"
                    )
                if delta != (0, 0):
                    calls.append(MoveRelCall("move_rel", delta=delta))
                here = group.target
            elif kind == "click":
                named = _support.click_action_name(group.button, group.repeats)
                if named is not None:
                    calls.append(MoveRelCall(named))
                else:
                    for _ in range(group.repeats):
                        calls.append(MoveRelCall("mouse_down", button=group.button))
                        calls.append(MoveRelCall("mouse_up", button=group.button))
            elif kind in ("button_down", "button_up"):
                calls.append(
                    MoveRelCall(
                        "mouse_down" if kind == "button_down" else "mouse_up",
                        button=group.button,
                    )
                )
            elif kind == "chord":
                calls.append(MoveRelCall("key", keys=group.keys))
            elif kind in ("key_down", "key_up"):
                calls.append(MoveRelCall(kind, keys=group.keys))
            elif kind == "type":
                if "\n" in group.text or "\r" in group.text:
                    raise MoveRelError(
                        "type() cannot embed a newline; press Return with `key`"
                    )
                calls.append(MoveRelCall("type", text=group.text))
            elif kind == "scroll":
                if group.dx:
                    raise MoveRelError(
                        "horizontal scroll cannot be expressed: this schema has "
                        "only a vertical `scroll` with `pixels`"
                    )
                if group.dy:
                    calls.append(MoveRelCall("scroll", scroll=group.dy))
            elif kind == "wait":
                calls.append(
                    MoveRelCall(
                        "wait",
                        seconds=max(0.0, min(MAX_WAIT_SECONDS, group.seconds)),
                    )
                )
            else:  # pragma: no cover - group_operations fixes the set
                raise MoveRelError(f"cannot lift group kind: {kind!r}")
        if status is not None:
            calls.append(MoveRelCall("terminate", status=status))
        if not calls:
            raise MoveRelError(
                "an empty operation stream has no representation: this grammar "
                "has no idle action, only an explicit wait"
            )
        return MoveRelAction(calls=tuple(calls), prompt_digest=self.digest)

    def from_pixel_delta(
        self,
        delta: tuple[int, int],
        geometry: DisplayGeometry,
        *,
        then: MoveRelCall | None = None,
    ) -> tuple[MoveRelCall, ...]:
        """A raw pixel delta plus an optional follow-on call -> v2 calls.

        This is ``move_rel_format.split_and_normalize``'s rule, on the codec: the
        delta becomes an explicit normalized ``move_rel``, the follow-on action
        stays coordinate-less, and a ZERO delta emits only the follow-on call.
        """
        width, height = _support.screen_size(geometry)
        normalized = (
            norm_from_pixels(delta[0], width),
            norm_from_pixels(delta[1], height),
        )
        calls: list[MoveRelCall] = []
        if normalized != (0, 0):
            calls.append(MoveRelCall("move_rel", delta=normalized))
        if then is not None:
            calls.append(then)
        return tuple(calls)


def action_from_dict(value: dict[str, Any]) -> MoveRelAction:
    return MoveRelAction(
        calls=tuple(CODEC.validate_call(call) for call in value.get("calls", ()))
    )


CODEC = MoveRelCodec()
