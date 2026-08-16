"""The bare-line ABSOLUTE grammar: paired_eval's absolute control arm.

``x y scroll`` or ``x y scroll ; EVENTS``, where x and y are absolute integer
screen-pixel coordinates. Everything else — the one-sentence prose preamble, the
element tail, the ``type("…")`` payload, the button and key names — is identical
to ``compact_raw``: the two are a matched pair whose only difference is whether
the two integers name a position or an offset.

"control" is inherited from its paired-eval arm role
(``paired_runtime`` ``arm.name == "native_absolute_control"``,
``prompts/native_absolute_control.txt``); the name is kept so receipts,
recipes and prompt digests still line up with what is on disk.

Semantics, chosen so the arm is never weaker than its relative twin:

* Keyboard transitions are accepted. Lowering straight to the IR gives
  ``key_down`` / ``key_up``, so an element that is neither a mouse button nor
  ``type()`` is not a parse error.
* The coordinates are unconditional. x and y always name a position, even in
  ``0 0 0 ; type("x")``; the move is skipped only when that position is the
  current cursor, which is exactly when ``compact_raw`` skips its own move. So
  the matched action for "type without moving" is
  ``<cursor_x> <cursor_y> 0 ; type("x")``, and both arms lower to the same
  operations.
* ``0 0 0`` is the top-left corner, not "do not move" — the semantic difference
  from ``compact_raw``.
* Like ``compact_raw``: the action is the last non-empty line, there is no
  line-count cap, the canonical separator is ``" ; "``, and there are no control
  tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from desktop.geometry import DisplayGeometry
from desktop.ir import Operation

from .. import _support

PRODUCER = {
    "arm": "native_absolute_control",
    "prompt_file": "paired_runtime/prompts/native_absolute_control.txt",
    "prompt_sha256": (
        "248c161d3b63dcbf5bffab979e34c7f11cdab4eef527e8e2f2cd71fe66afdc3f"
    ),
}

#: Its matched relative twin. The two differ only in whether the leading
#: integers name a position or an offset; a change to one must be made to the
#: other.
PAIRED_WITH = "compact_raw"


class NativeAbsoluteControlError(ValueError):
    """Malformed bare-line absolute action text."""


@dataclass(frozen=True)
class NativeAbsoluteControlAction:
    """An absolute pixel position plus ordered key/button/typing elements."""

    x: int = 0
    y: int = 0
    scroll: int = 0
    elements: tuple[_support.Element, ...] = ()
    prompt_digest: str = field(default="", compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "scroll": self.scroll,
            "elements": [element.to_dict() for element in self.elements],
        }


def action_from_dict(value: dict[str, Any]) -> NativeAbsoluteControlAction:
    return NativeAbsoluteControlAction(
        x=int(value.get("x", 0)),
        y=int(value.get("y", 0)),
        scroll=int(value.get("scroll", 0)),
        elements=tuple(
            _support.element_from_dict(item) for item in value.get("elements", ())
        ),
    )


class NativeAbsoluteControlCodec:
    # The class docstring is the prompt preamble. It is assigned below from
    # ``_support.MATCHED_ARM_PREAMBLE`` so that ``compact_raw`` renders
    # byte-identical text.

    name = "native_absolute_control"

    #: Empty by design: the prose sentence precedes the action line, so no token
    #: sequence marks the end of a turn.
    stop_sequences: tuple[str, ...] = ()

    @_support.production("x y scroll")
    def _mouse(self) -> None:
        """Three integers. x and y are the ABSOLUTE screen-pixel coordinates to
        put the cursor on — (0, 0) is the top-left corner, x grows right and y
        grows down — and scroll is wheel ticks (positive scrolls up). Read the
        target's position off the screenshot. To act without moving, name the
        cursor's current position.
        """

    @_support.production("x y scroll ; EVENTS")
    def _with_events(self) -> None:
        """The same position, then the space-separated elements after `;`,
        applied in order. To move and click, use `x y 0 ; +LMB -LMB`.
        """

    # The three element productions and the notes are shared verbatim with the
    # relative arm, so their docstrings come from the same constants as the
    # preamble. Only the mouse-triple productions above differ.

    @_support.production("+NAME")
    def _press(self) -> None: ...

    @_support.production("-NAME")
    def _release(self) -> None: ...

    @_support.production('type("TEXT")')
    def _type(self) -> None: ...

    def notes(self) -> None: ...

    def describe(self) -> str:
        return _support.render_spec(self)

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def report(self) -> dict[str, Any]:
        report = _support.drift_report(self, producer=PRODUCER)
        report["paired_with"] = PAIRED_WITH
        return report

    def parse(self, text: str) -> NativeAbsoluteControlAction:
        line = _support.final_line(text, error=NativeAbsoluteControlError)
        mouse, _, tail = line.partition(";")
        x, y, scroll = _support.parse_mouse_triple(
            mouse, error=NativeAbsoluteControlError
        )
        elements = _support.scan_elements(
            tail, allow_type=True, allow_move=False, error=NativeAbsoluteControlError
        )
        return NativeAbsoluteControlAction(
            x, y, scroll, elements, prompt_digest=self.digest
        )

    def format(self, action: NativeAbsoluteControlAction) -> str:
        head = f"{action.x} {action.y} {action.scroll}"
        return head + _support.render_elements(action.elements)

    def compile(
        self,
        text: str,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        return self.compile_action(self.parse(text), geometry, cursor)

    def compile_action(
        self,
        action: NativeAbsoluteControlAction,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Clamp the named position onto the display. ``cursor`` only decides
        whether the move is redundant — it never contributes to the target."""
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        target = _support.clamp((action.x, action.y), geometry)
        if target != here:
            operations.append(_support.move_to(target))
        if action.scroll:
            operations.append(_support.scroll(0, action.scroll))
        operations.extend(
            _support.lower_transitions(
                action.elements, error=NativeAbsoluteControlError
            )
        )
        return tuple(operations)

    def action_from_operations(
        self,
        operations: Sequence[Operation],
        *,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> NativeAbsoluteControlAction:
        """Absolute Operations -> an action. The inverse of ``compile_action``.

        The one place this differs from its relative twin: an empty stream is
        ``<cursor_x> <cursor_y> 0``, not ``0 0 0`` — in an absolute grammar the
        origin is the top-left corner, so idling means naming where the cursor
        already is. Same stream, two lifts; that is why the lift belongs on the
        codec and not in a converter.
        """
        if (
            _support.terminate_status(terminate, error=NativeAbsoluteControlError)
            is not None
        ):
            raise NativeAbsoluteControlError(
                "this grammar has no control tokens, so termination cannot be "
                "expressed; its paired_eval arm ends episodes by semantic-step "
                "accounting"
            )
        groups = _support.group_operations(
            operations,
            geometry=geometry,
            cursor=cursor,
            error=NativeAbsoluteControlError,
        )
        here = _support.clamp(cursor, geometry)
        plan = _support.bare_token_plan(
            groups,
            cursor=here,
            allow_type=True,
            allow_stroke=False,
            error=NativeAbsoluteControlError,
        )
        target = plan.target if plan.target is not None else here
        return NativeAbsoluteControlAction(
            target[0],
            target[1],
            plan.scroll,
            plan.elements,
            prompt_digest=self.digest,
        )

    def from_target(
        self,
        target: tuple[int, int],
        *,
        scroll: int = 0,
        elements: tuple[_support.Element, ...] = (),
    ) -> NativeAbsoluteControlAction:
        """The label for an absolute ``target``. Takes NO cursor read.

        This is the asymmetry the paired comparison measures: the absolute arm's
        label is a function of the element's geometry alone, while
        ``compact_raw.from_target`` additionally needs one fresh cursor position
        and is wrong if that read is stale.
        """
        return NativeAbsoluteControlAction(
            x=target[0], y=target[1], scroll=scroll, elements=elements
        )


_support.apply_matched_arm_prose(NativeAbsoluteControlCodec)

CODEC = NativeAbsoluteControlCodec()
