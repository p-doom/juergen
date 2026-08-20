"""The diff-of-absolute bare-token grammar.

Surface: ``dx dy scroll`` optionally followed by ``; +KEY -KEY`` events, or a
control token. The deltas are raw screen pixels obtained as the difference of
two absolute positions — the teacher's intended target minus the cursor before
the action — which is what "diffabs" names and what ``from_absolute`` below
implements. Behaviour is identical to the absolute action it was derived from;
only the encoding is relative.

Collapses into one codec:

* ``eval/action_parser.py::parse_action`` (the base bare-token parser) and
  ``parse_action_tolerant`` (its prose-tolerant twin),
* the prompt, which existed as five byte-identical or near-identical copies:
  ``osworld_parity/split/diffabs_system_prompt.txt``,
  ``eval/osworld_system_prompts.py`` ``psai_v1`` (identical), ``training_v1``
  and ``yll_v1`` (the same grammar, one adding TERMINATE),
  ``pipeline/lib/config.py::SYSTEM_PROMPT``, and
  ``CanonicalFormatter.reply_contract``,
* the absolute-to-relative rewrite rule in
  ``osworld_parity/scripts/convert_abs_to_relative.py``.

Semantics chosen where the sources disagreed:

* The action is the last ``Action: <body>`` marker if present, otherwise the
  last non-empty line; every prompt in this family puts reasoning before the
  action. ``parse_action`` cut the text at the first newline while
  ``parse_action_tolerant`` fell back to the last line.
* Termination is not in this grammar at all — it is the harness's control
  channel (``_support.CONTROL_SPEC``). ``psai_v1`` documents only ``NO_OP``;
  ``yll_v1`` added a bare ``TERMINATE`` action line and no ``FAIL``, and the
  checkpoints trained on it emit that line, which no longer parses here.
* There is no ``type()`` element. Literal text is spelled as key transitions
  here; the coalesced ``type()`` element is what ``deltatype_v2`` adds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from desktop.geometry import DisplayGeometry
from desktop.ir import Operation

from .. import _support

PRODUCER = {
    "prompt_ids": "psai_v1 == split/diffabs_system_prompt.txt; training_v1; yll_v1"
}

#: Retained from ``eval/action_parser.py`` for consumers that scored on X11
#: button codes. The IR itself carries button names.
X11_BUTTON_CODES = {"LMB": 1, "MMB": 2, "RMB": 3}


class DiffabsError(ValueError):
    """Malformed diffabs action text."""


@dataclass(frozen=True)
class DiffabsAction:
    """A diff-of-absolute move plus ordered key/button transitions."""

    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[_support.Element, ...] = ()
    no_op: bool = False
    #: The episode-control status this turn declares, from ``_support.CONTROL_SPEC``.
    #: Set by the lift only; ``parse`` never sees a control line.
    terminate: str | None = None
    prompt_digest: str = field(default="", compare=False)

    @property
    def has_left_click_press(self) -> bool:
        return any(
            item.kind == "event" and item.name == "LMB" and item.pressed
            for item in self.elements
        )

    @property
    def has_left_click_release(self) -> bool:
        return any(
            item.kind == "event" and item.name == "LMB" and item.pressed is False
            for item in self.elements
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "scroll": self.scroll,
            "elements": [element.to_dict() for element in self.elements],
            "no_op": self.no_op,
            "terminate": self.terminate,
        }


def action_from_dict(value: dict[str, Any]) -> DiffabsAction:
    return DiffabsAction(
        dx=int(value.get("dx", 0)),
        dy=int(value.get("dy", 0)),
        scroll=int(value.get("scroll", 0)),
        elements=tuple(
            _support.element_from_dict(item) for item in value.get("elements", ())
        ),
        no_op=bool(value.get("no_op", False)),
        terminate=_support.terminate_status(value.get("terminate"), error=DiffabsError),
    )


class DiffabsCodec:
    """You operate a desktop computer.

    The first user turn shows the initial screen and the user's goal; subsequent
    user turns show the current screen, with the cursor visible as a small arrow.
    Reply with the next action toward that goal as one line. Reasoning may
    precede it: the action is the last `Action:` line if you write one, otherwise
    the last line of your reply.
    """

    name = "diffabs"

    #: Empty by design: reasoning and an `Action:` marker legally precede the
    #: action line, so no token sequence marks the end of a turn.
    stop_sequences: tuple[str, ...] = ()

    @_support.production("dx dy scroll")
    def _mouse(self) -> None:
        """Three integers. dx and dy move the cursor RELATIVE to where it is, in
        screen pixels (dx > 0 right, dx < 0 left; dy > 0 down, dy < 0 up); scroll
        is wheel ticks (positive scrolls up). Judge the offset from the cursor to
        the target off the screenshot.
        """

    @_support.production("dx dy scroll ; +KEY -KEY")
    def _with_events(self) -> None:
        """The same move, then the space-separated transitions after `;`, applied
        in order. To click, move then click: `dx dy 0 ; +LMB -LMB`.
        """

    @_support.production("+NAME")
    def _press(self) -> None:
        """Press NAME. Mouse buttons are LMB, RMB, MMB. Keyboard keys use rdev
        names: KeyA, Return, Escape, Tab, Space, Backspace, ShiftLeft,
        ControlLeft, ArrowUp, and so on.
        """

    @_support.production("-NAME")
    def _release(self) -> None:
        """Release NAME. A chord presses in order and releases in reverse:
        `+ControlLeft +KeyC -KeyC -ControlLeft`.
        """

    @_support.production("NO_OP")
    def _no_op(self) -> None:
        """No action this turn; wait for the screen to settle."""

    def notes(self) -> None:
        """Emit one action per turn."""

    def describe(self) -> str:
        return _support.render_spec(self)

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def report(self) -> dict[str, Any]:
        return _support.drift_report(self, producer=PRODUCER)

    def parse(self, text: str) -> DiffabsAction:
        line = _support.action_line(text, error=DiffabsError)
        digest = self.digest
        if line == "NO_OP":
            return DiffabsAction(no_op=True, prompt_digest=digest)
        mouse, _, tail = line.partition(";")
        dx, dy, scroll = _support.parse_mouse_triple(mouse, error=DiffabsError)
        elements = _support.scan_elements(
            tail, allow_type=False, allow_move=False, error=DiffabsError
        )
        return DiffabsAction(dx, dy, scroll, elements, prompt_digest=digest)

    def format(self, action: DiffabsAction) -> str:
        body = (
            "NO_OP"
            if action.no_op
            else f"{action.dx} {action.dy} {action.scroll}"
            + _support.render_elements(action.elements)
        )
        return _support.with_control(body, action.terminate, error=DiffabsError)

    def compile(
        self,
        text: str,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        return self.compile_action(self.parse(text), geometry, cursor)

    def compile_action(
        self,
        action: DiffabsAction,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Add the delta back onto ``cursor``, recovering the absolute target."""
        if action.no_op:
            return ()
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        target = _support.clamp((here[0] + action.dx, here[1] + action.dy), geometry)
        if target != here:
            operations.append(_support.move_to(target))
        if action.scroll:
            operations.append(_support.scroll(0, action.scroll))
        operations.extend(
            _support.lower_transitions(action.elements, error=DiffabsError)
        )
        return tuple(operations)

    def action_from_operations(
        self,
        operations: Sequence[Operation],
        *,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> DiffabsAction:
        """Absolute Operations -> an action. The inverse of ``compile_action``.

        One visible ceiling: a ``coalesced_type`` raises, because this grammar
        spells literal text as key transitions. That is a real limit of the
        grammar, not of the lift.
        """
        status = _support.terminate_status(terminate, error=DiffabsError)
        groups = _support.group_operations(
            operations, geometry=geometry, cursor=cursor, error=DiffabsError
        )
        here = _support.clamp(cursor, geometry)
        plan = _support.bare_token_plan(
            groups,
            cursor=here,
            allow_type=False,
            allow_stroke=False,
            error=DiffabsError,
        )
        if plan.idle:
            return DiffabsAction(
                no_op=True, terminate=status, prompt_digest=self.digest
            )
        dx, dy = (
            (0, 0)
            if plan.target is None
            else (plan.target[0] - here[0], plan.target[1] - here[1])
        )
        return DiffabsAction(
            dx,
            dy,
            plan.scroll,
            plan.elements,
            terminate=status,
            prompt_digest=self.digest,
        )

    def from_absolute(
        self,
        cursor: tuple[int, int],
        target: tuple[int, int],
        *,
        scroll: int = 0,
        elements: tuple[_support.Element, ...] = (),
    ) -> DiffabsAction:
        """The diff-of-absolute rule: the delta that reproduces ``target``.

        ``target`` is the absolute pixel the teacher's coordinate resolved to
        (post-scale, post-clip) and ``cursor`` is the position before the action,
        so ``compile`` recovers exactly the same absolute endpoint. That identity
        is why the encoding change is behaviour preserving.
        """
        return DiffabsAction(
            dx=target[0] - cursor[0],
            dy=target[1] - cursor[1],
            scroll=scroll,
            elements=elements,
        )


CODEC = DiffabsCodec()
