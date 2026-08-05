"""The native ``computer_use`` grammar with ABSOLUTE pixel coordinates.

This is the dialect an off-the-shelf Qwen3-VL already speaks — the reference
grammar the 33.9% OSWorld-Verified baseline was measured in — so its surface is
kept exactly: ``<tool_call>{"name":"computer_use","arguments":{…}}</tool_call>``,
one or more calls per turn, ``coordinate`` is an absolute screen pixel.

Collapses into one codec:

* ``osworld_parity/sign_of_life_v2/actions.py::compile_native_absolute``,
* ``osworld_parity/proper_vm_capability_ladder/rung1/executor.py::
  NativeAbsoluteExecutor`` (which added ``mouse_down`` / ``mouse_up`` /
  ``drag_to`` and accepted ``clicks`` as well as ``pixels`` for scroll),
* ``eval/action_parser.py::parse_computer_use_tool_call`` and
  ``parse_computer_use_tool_calls`` (last-call-wins vs all-calls-in-order — this
  codec keeps ALL calls in order, because a turn legitimately contains
  mouse_down, mouse_move, mouse_up),
* the RL rollout extractor in ``rl/computer_use/parsing.py``.

Semantics chosen where the sources disagreed:

* ``drag_to`` is accepted as an alias of ``left_click_drag`` (see
  ``ACTION_ALIASES``) rather than rejected. It was rung1's spelling for the same
  gesture; a checkpoint that emits it should not fail to parse.
* ``coordinate`` on a click is CORRECT here (that is what absolute means).
  ``delta`` — the key ``rl/computer_use/actions.py`` requires — is rejected with
  a message naming it, because a relative offset has no meaning in this grammar.
* ``triple_click`` dispatches three press/release pairs. The historical tool
  description said "simulated as double-click since it's the closest action";
  the sign-of-life executor already did three, which is strictly more faithful.
* A zero scroll emits no operation instead of raising. The sign-of-life codec
  raised "zero scroll is not an action"; a no-op is not a format error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from desktop_env.geometry import DisplayGeometry
from desktop_env.ir import Operation

from .. import _support

#: Provenance of the producer prompt (``eval/osworld_system_prompts.py``
#: ``computer_use_v1``, byte-equal in shape to ``split/eval_system_prompt.txt``).
#: ``describe()`` regenerates that envelope from docstrings, so the digests are
#: not expected to match; ``report()` says so rather than raising.
PRODUCER = {"prompt_id": "computer_use_v1"}

CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
DRAGS = {"left_click_drag"}

#: ACCEPTED ON INPUT, NOT ADVERTISED IN THE PROMPT.
#:
#: ``rung1``'s ``NativeAbsoluteExecutor`` exposed ``drag_to`` for exactly the
#: gesture ``left_click_drag`` names, so a checkpoint trained or evaluated through
#: it may still emit that spelling. Rejecting it turns a semantically correct
#: completion into a parse error at eval time, which is the expensive direction of
#: the trade. The alias is therefore normalised to the advertised name here rather
#: than declared as a production: adding a production would put a second name for
#: one gesture into the tool schema the model reads, and teaching the model two
#: spellings is not the goal — surviving one it already knows is.
#:
#: ``format`` re-emits the canonical name, so the alias canonicalises on the round
#: trip in the same way ``"; "`` canonicalises to ``" ; "`` in the bare-token arms.
ACTION_ALIASES = {"drag_to": "left_click_drag"}

BUTTON_NAMES = ("left", "right", "middle")
MAX_WAIT_SECONDS = 10.0

#: ``left_click_drag`` lowers to a TIMED stroke, not an instant jump. The
#: historical lowering (sol_v2 and rung1 alike) was
#: ``mouse_down, move_to, mouse_up``, which many toolkits do not register as a
#: drag at all — the same class of defect as degrading a drag into a stationary
#: click. ``glide_to`` is an optional backend capability whose documented
#: fallback is ``move_to``, so a backend without it reproduces the old behaviour
#: explicitly instead of the codec choosing it silently.
DRAG_SECONDS = 0.5


class NativeAbsoluteError(ValueError):
    """Malformed native computer_use action."""


@dataclass(frozen=True)
class NativeCall:
    """One validated ``computer_use`` call. ``coordinate`` is absolute pixels."""

    action: str
    coordinate: tuple[int, int] | None = None
    button: str = ""
    keys: tuple[str, ...] = ()
    text: str = ""
    scroll: int = 0
    seconds: float = 0.0
    status: str = ""

    def arguments(self) -> dict[str, Any]:
        value: dict[str, Any] = {"action": self.action}
        if self.coordinate is not None:
            value["coordinate"] = [self.coordinate[0], self.coordinate[1]]
        if self.button:
            value["button"] = self.button
        if self.keys:
            value["keys"] = list(self.keys)
        if self.action in ("type", "answer") and self.text:
            value["text"] = self.text
        if self.action in ("scroll", "hscroll"):
            value["pixels"] = self.scroll
        if self.action == "wait":
            value["time"] = self.seconds
        if self.action == "terminate":
            value["status"] = self.status
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.arguments()


@dataclass(frozen=True)
class NativeAbsoluteAction:
    """Every call the turn emitted, in order."""

    calls: tuple[NativeCall, ...] = ()
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


class NativeAbsoluteCodec:
    """You are a helpful assistant."""

    name = "native_absolute"

    #: Empty by design: several tool calls per turn are legal, so no token
    #: sequence marks the end of a turn.
    stop_sequences: tuple[str, ...] = ()

    def tool_description(self) -> None:
        """Use a mouse and keyboard to interact with a computer, and take screenshots.
        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * Coordinates are ABSOLUTE screen pixels: (0, 0) is the top-left corner, x grows right and y grows down. Read the target's position off the screenshot before moving.
        * Whenever you intend to move the cursor to click on an element like an icon, consult the screenshot to determine the coordinates of the element before moving the cursor.
        * Make sure to click any buttons, links, icons with the cursor tip in the CENTER of the element, not on its edge.
        """

    @_support.production("key")
    def _key(self) -> None:
        """Performs key down presses on the arguments passed in order, then performs key releases in reverse order."""

    @_support.production("key_down")
    def _key_down(self) -> None:
        """Presses and HOLDS the keys passed, so a modifier can span several steps."""

    @_support.production("key_up")
    def _key_up(self) -> None:
        """Releases the keys passed, in the order given."""

    @_support.production("type")
    def _type(self) -> None:
        """Type a string of text on the keyboard. The text cannot contain a newline; press Return with `key`."""

    @_support.production("mouse_move")
    def _mouse_move(self) -> None:
        """Move the cursor to a specified (x, y) absolute pixel coordinate on the screen."""

    @_support.production("left_click")
    def _left_click(self) -> None:
        """Click the left mouse button, at `coordinate` if given, otherwise at the current cursor position."""

    @_support.production("right_click")
    def _right_click(self) -> None:
        """Click the right mouse button, at `coordinate` if given, otherwise at the current cursor position."""

    @_support.production("middle_click")
    def _middle_click(self) -> None:
        """Click the middle mouse button, at `coordinate` if given, otherwise at the current cursor position."""

    @_support.production("double_click")
    def _double_click(self) -> None:
        """Double-click the left mouse button, at `coordinate` if given, otherwise at the current cursor position."""

    @_support.production("triple_click")
    def _triple_click(self) -> None:
        """Triple-click the left mouse button, at `coordinate` if given, otherwise at the current cursor position."""

    @_support.production("left_click_drag")
    def _drag(self) -> None:
        """Press the left button at the current position, move to `coordinate`, and release."""

    @_support.production("mouse_down")
    def _mouse_down(self) -> None:
        """Press and HOLD a mouse button (`button` = 'left', 'right', 'middle') after optionally moving to `coordinate`."""

    @_support.production("mouse_up")
    def _mouse_up(self) -> None:
        """Release a mouse button after optionally moving to `coordinate`."""

    @_support.production("scroll")
    def _scroll(self) -> None:
        """Performs a scroll of the mouse scroll wheel. Positive `pixels` scroll up, negative scroll down."""

    @_support.production("hscroll")
    def _hscroll(self) -> None:
        """Performs a horizontal scroll. Positive `pixels` scroll right, negative scroll left."""

    @_support.production("wait")
    def _wait(self) -> None:
        """Wait `time` seconds for the change to happen."""

    @_support.production("terminate")
    def _terminate(self) -> None:
        """Terminate the current task and report its completion status (`status` = 'success' or 'failure')."""

    @_support.production("answer")
    def _answer(self) -> None:
        """Answer a question, with the answer in `text`."""

    def notes(self) -> None:
        """For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """

    # -- interface ---------------------------------------------------------

    def describe(self) -> str:
        return _support.render_tool_prompt(
            self,
            properties={
                "keys": {
                    "description": "Required only by `action=key`, `action=key_down` and `action=key_up`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type` and `action=answer`.",
                    "type": "string",
                },
                "coordinate": {
                    "description": "(x, y): the absolute x (pixels from the left edge) and y (pixels from the top edge) coordinate.",
                    "type": "array",
                },
                "pixels": {
                    "description": "The amount of scrolling. Required only by `action=scroll` and `action=hscroll`.",
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

    def parse(self, text: str) -> NativeAbsoluteAction:
        calls = tuple(
            self.validate_call(arguments) for arguments in _support.iter_tool_calls(text)
        )
        if not calls:
            raise NativeAbsoluteError("no valid computer_use tool call found")
        return NativeAbsoluteAction(calls=calls, prompt_digest=self.digest)

    def format(self, action: NativeAbsoluteAction) -> str:
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
        action: NativeAbsoluteAction,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Clamp each absolute coordinate onto the display and lower in order."""
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        for call in action.calls:
            target = (
                _support.clamp(call.coordinate, geometry)
                if call.coordinate is not None
                else None
            )
            name = call.action
            if name == "mouse_move":
                assert target is not None
                operations.append(_support.move_to(target))
                here = target
            elif name in CLICKS:
                if target is not None and target != here:
                    operations.append(_support.move_to(target))
                    here = target
                button, repeats = CLICKS[name]
                for _ in range(repeats):
                    operations.append(_support.mouse_down(button))
                    operations.append(_support.mouse_up(button))
            elif name in DRAGS:
                assert target is not None
                operations.append(_support.mouse_down("left"))
                operations.append(_support.glide_to(target, DRAG_SECONDS))
                operations.append(_support.mouse_up("left"))
                here = target
            elif name in ("mouse_down", "mouse_up"):
                if target is not None and target != here:
                    operations.append(_support.move_to(target))
                    here = target
                button = call.button or "left"
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
                    _support.lower_typing(call.text, error=NativeAbsoluteError)
                )
            elif name == "scroll":
                if call.scroll:
                    operations.append(_support.scroll(0, call.scroll))
            elif name == "hscroll":
                if call.scroll:
                    operations.append(_support.scroll(call.scroll, 0))
            elif name == "wait":
                operations.append(_support.wait(call.seconds))
            elif name in ("terminate", "answer"):
                continue
            else:  # pragma: no cover - validate_call fixes the set
                raise NativeAbsoluteError(f"unsupported action: {name!r}")
        return tuple(operations)

    # -- training-target construction --------------------------------------

    def action_from_operations(
        self,
        operations: Sequence[Operation],
        *,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> NativeAbsoluteAction:
        """Absolute Operations -> calls. The inverse of ``compile_action``.

        Emits the idiomatic native form, which is what the base model speaks: a
        move followed by a click collapses into one ``left_click`` carrying the
        coordinate, a held left button around a stroke collapses into one
        ``left_click_drag``, and a palindromic key run collapses into one
        ``key``. Every collapse is exactly invertible by ``compile_action``.
        Termination is a ``terminate`` call inside the turn, not a flag.
        """
        status = _support.terminate_status(terminate, error=NativeAbsoluteError)
        groups = _support.group_operations(
            operations, geometry=geometry, cursor=cursor, error=NativeAbsoluteError
        )
        calls: list[NativeCall] = []
        index = 0
        while index < len(groups):
            group = groups[index]
            following = groups[index + 1] if index + 1 < len(groups) else None
            kind = group.kind
            if kind == "move":
                absorbed = self._absorb_move(group, following, groups, index)
                if absorbed is not None:
                    call, step = absorbed
                    calls.append(call)
                    index += step
                    continue
                calls.append(NativeCall("mouse_move", coordinate=group.target))
                index += 1
            elif kind == "click":
                calls.extend(self._click_calls(group, coordinate=None))
                index += 1
            elif kind == "button_down":
                drag = self._drag_call(groups, index)
                if drag is not None:
                    calls.append(drag)
                    index += 3
                    continue
                calls.append(NativeCall("mouse_down", button=group.button))
                index += 1
            elif kind == "button_up":
                calls.append(NativeCall("mouse_up", button=group.button))
                index += 1
            elif kind == "stroke":
                raise NativeAbsoluteError(
                    "a timed stroke outside a held left button cannot be "
                    "expressed: the only stroke in this schema is left_click_drag"
                )
            elif kind == "chord":
                calls.append(NativeCall("key", keys=group.keys))
                index += 1
            elif kind in ("key_down", "key_up"):
                calls.append(NativeCall(kind, keys=group.keys))
                index += 1
            elif kind == "type":
                if "\n" in group.text or "\r" in group.text:
                    raise NativeAbsoluteError(
                        "type() cannot embed a newline; press Return with `key`"
                    )
                calls.append(NativeCall("type", text=group.text))
                index += 1
            elif kind == "scroll":
                if group.dy:
                    calls.append(NativeCall("scroll", scroll=group.dy))
                if group.dx:
                    calls.append(NativeCall("hscroll", scroll=group.dx))
                index += 1
            elif kind == "wait":
                calls.append(
                    NativeCall(
                        "wait",
                        seconds=max(0.0, min(MAX_WAIT_SECONDS, group.seconds)),
                    )
                )
                index += 1
            else:  # pragma: no cover - group_operations fixes the set
                raise NativeAbsoluteError(f"cannot lift group kind: {kind!r}")
        if status is not None:
            calls.append(NativeCall("terminate", status=status))
        if not calls:
            raise NativeAbsoluteError(
                "an empty operation stream has no representation: this grammar "
                "has no idle action, only an explicit wait"
            )
        return NativeAbsoluteAction(calls=tuple(calls), prompt_digest=self.digest)

    def _absorb_move(
        self,
        group: _support.Group,
        following: _support.Group | None,
        groups: Sequence[_support.Group],
        index: int,
    ) -> tuple[NativeCall, int] | None:
        """A move a following action can carry as its own ``coordinate``."""
        if following is None:
            return None
        if following.kind == "click":
            named = _support.click_action_name(following.button, following.repeats)
            if named is not None:
                return NativeCall(named, coordinate=group.target), 2
            return None
        if following.kind == "button_down":
            if self._drag_call(groups, index + 1) is not None:
                # A drag follows. Its schema action presses where the cursor
                # already is and names only the endpoint, so the approach move
                # stays a separate mouse_move and the drag collapses next.
                return None
            return (
                NativeCall(
                    "mouse_down", coordinate=group.target, button=following.button
                ),
                2,
            )
        if following.kind == "button_up":
            return (
                NativeCall(
                    "mouse_up", coordinate=group.target, button=following.button
                ),
                2,
            )
        return None

    def _drag_call(
        self, groups: Sequence[_support.Group], index: int
    ) -> NativeCall | None:
        """``button_down(left), stroke, button_up(left)`` -> ``left_click_drag``."""
        if index + 2 >= len(groups):
            return None
        down, stroke, up = groups[index], groups[index + 1], groups[index + 2]
        if down.kind != "button_down" or down.button != "left":
            return None
        if stroke.kind != "stroke" or up.kind != "button_up" or up.button != "left":
            return None
        return NativeCall("left_click_drag", coordinate=stroke.target)

    def _click_calls(
        self, group: _support.Group, *, coordinate: tuple[int, int] | None
    ) -> list[NativeCall]:
        named = _support.click_action_name(group.button, group.repeats)
        if named is not None:
            return [NativeCall(named, coordinate=coordinate)]
        calls: list[NativeCall] = []
        for repeat in range(group.repeats):
            calls.append(
                NativeCall(
                    "mouse_down",
                    button=group.button,
                    coordinate=coordinate if repeat == 0 else None,
                )
            )
            calls.append(NativeCall("mouse_up", button=group.button))
        return calls

    # -- validation --------------------------------------------------------

    def validate_call(self, arguments: dict[str, Any]) -> NativeCall:
        name = str(arguments.get("action", "")).strip()
        name = ACTION_ALIASES.get(name, name)
        known = {item.syntax for item in _support.productions(self)}
        if name not in known:
            raise NativeAbsoluteError(f"unsupported native absolute action: {name!r}")
        if "delta" in arguments:
            raise NativeAbsoluteError(
                "native_absolute coordinates are absolute; `delta` belongs to the "
                "relative computer_use dialects (see the move_rel grammar)"
            )
        coordinate = self._coordinate(
            arguments,
            required=name in DRAGS or name == "mouse_move",
            allowed=name in DRAGS
            or name in CLICKS
            or name in ("mouse_move", "mouse_down", "mouse_up"),
        )
        if name in ("key", "key_down", "key_up"):
            return NativeCall(name, keys=self._keys(arguments))
        if name in ("type", "answer"):
            text = arguments.get("text")
            if not isinstance(text, str) or not text:
                raise NativeAbsoluteError(f"{name} requires non-empty text")
            return NativeCall(name, text=text)
        if name in ("scroll", "hscroll"):
            raw = arguments.get("pixels", arguments.get("clicks", 0))
            try:
                return NativeCall(name, scroll=int(round(float(raw))))
            except (TypeError, ValueError) as exc:
                raise NativeAbsoluteError("scroll pixels must be numeric") from exc
        if name == "wait":
            try:
                seconds = float(arguments.get("time", 1.0))
            except (TypeError, ValueError) as exc:
                raise NativeAbsoluteError("wait time must be numeric") from exc
            return NativeCall(name, seconds=max(0.0, min(MAX_WAIT_SECONDS, seconds)))
        if name == "terminate":
            status = str(arguments.get("status", "success"))
            return NativeCall(name, status=status)
        if name in ("mouse_down", "mouse_up"):
            button = str(arguments.get("button", "left"))
            if button not in BUTTON_NAMES:
                raise NativeAbsoluteError(f"unsupported mouse button: {button!r}")
            return NativeCall(name, coordinate=coordinate, button=button)
        return NativeCall(name, coordinate=coordinate)

    def _coordinate(
        self, arguments: dict[str, Any], *, required: bool, allowed: bool
    ) -> tuple[int, int] | None:
        raw = arguments.get("coordinate")
        if raw is None:
            if required:
                raise NativeAbsoluteError(
                    f"{arguments.get('action')!r} requires coordinate [x, y]"
                )
            return None
        if not allowed:
            raise NativeAbsoluteError(
                f"{arguments.get('action')!r} does not take a coordinate"
            )
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise NativeAbsoluteError("coordinate must be [x, y]")
        try:
            x, y = (int(round(float(value))) for value in raw)
        except (TypeError, ValueError) as exc:
            raise NativeAbsoluteError("coordinate values must be numeric") from exc
        return x, y

    def _keys(self, arguments: dict[str, Any]) -> tuple[str, ...]:
        keys = arguments.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list) or not keys:
            raise NativeAbsoluteError(
                f"{arguments.get('action')!r} requires a non-empty keys array"
            )
        return tuple(
            _support.normalize_key(key, error=NativeAbsoluteError) for key in keys
        )


def action_from_dict(value: dict[str, Any]) -> NativeAbsoluteAction:
    return NativeAbsoluteAction(
        calls=tuple(CODEC.validate_call(call) for call in value.get("calls", ()))
    )


CODEC = NativeAbsoluteCodec()
