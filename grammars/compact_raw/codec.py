"""The compact raw-relative bare-line grammar (the Phase-B paired-eval arm).

One sentence of prose, then one bare action line ``dx dy scroll`` or
``dx dy scroll ; EVENTS``, where dx and dy are RAW relative screen pixels.

Collapses into one codec:

* ``osworld_parity/proper_vm_capability_ladder/rung1/executor.py::
  parse_compact_raw`` and ``CompactRawExecutor``,
* ``osworld_parity/proper_vm_capability_ladder/rung2_sameapp/actions.py::
  compile_compact`` / ``compile_compact_with_cursor_evidence`` (the
  target-geometry-to-text direction, which is this codec's ``format``),
* ``osworld_parity/proper_vm_capability_ladder/paired_runtime/runtime.py::
  _compact_action`` and ``_action_line``,
* the prompt ``paired_runtime/prompts/compact_raw_phaseb.txt``.

Semantics chosen where the sources disagreed:

* The action is the LAST non-empty line. ``parse_compact_raw`` read
  ``splitlines()[0]`` — the FIRST line — while ``_action_line`` handed it the
  last of at most two lines, so the same completion could resolve to two
  different actions depending on which entry point saw it. The prompt says
  prose first, then the action line, so the last line is the action.
* ``_action_line`` also rejected any completion with more than two non-empty
  lines. That cap is gone: extra reasoning is not a format error.
* The canonical separator is ``" ; "``. ``compile_compact`` emitted ``"; "``;
  both parse, one is written.
* There are deliberately no control tokens. This arm spells idling as
  ``0 0 0`` and ends episodes by semantic-step accounting, not TERMINATE. Use
  ``deltatype_v2`` if the grammar needs NO_OP / TERMINATE / FAIL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pixeldesk.geometry import DisplayGeometry
from pixeldesk.ir import Operation

from .. import _support

PRODUCER = {"prompt_file": "paired_runtime/prompts/compact_raw_phaseb.txt"}

#: Its matched absolute twin. The two differ only in whether the leading
#: integers name a position or an offset, so anything changed here must be
#: changed there or the paired comparison measures the change instead.
PAIRED_WITH = "native_absolute_control"


class CompactRawError(ValueError):
    """Malformed compact raw action text."""


@dataclass(frozen=True)
class CompactRawAction:
    """A relative raw-pixel move plus ordered key/button/typing elements."""

    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[_support.Element, ...] = ()
    prompt_digest: str = field(default="", compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "scroll": self.scroll,
            "elements": [element.to_dict() for element in self.elements],
        }


def action_from_dict(value: dict[str, Any]) -> CompactRawAction:
    return CompactRawAction(
        dx=int(value.get("dx", 0)),
        dy=int(value.get("dy", 0)),
        scroll=int(value.get("scroll", 0)),
        elements=tuple(
            _support.element_from_dict(item) for item in value.get("elements", ())
        ),
    )


class CompactRawCodec:
    # The class docstring IS the prompt preamble, and it is assigned below from
    # ``_support.MATCHED_ARM_PREAMBLE`` rather than written here, because
    # ``native_absolute_control`` must render the byte-identical text. The two had
    # already drifted by a line-wrap when each held its own copy.

    name = "compact_raw"

    #: Empty by design: the prose sentence precedes the action line, so no token
    #: sequence marks the end of a turn.
    stop_sequences: tuple[str, ...] = ()

    @_support.production("dx dy scroll")
    def _mouse(self) -> None:
        """Three integers. dx and dy are a RELATIVE move in raw screen pixels
        from the current cursor (dx > 0 right, dy > 0 down); scroll is wheel
        ticks (positive scrolls up). `0 0 0` moves nothing.
        """

    @_support.production("dx dy scroll ; EVENTS")
    def _with_events(self) -> None:
        """The same move, then the space-separated elements after `;`, applied
        in order. To move and click, use `dx dy 0 ; +LMB -LMB`.
        """

    # The three element productions and the notes are shared verbatim with the
    # absolute arm, so their docstrings come from the same constants as the
    # preamble. Only the mouse-triple productions above differ, and that
    # difference IS the experiment.

    @_support.production("+NAME")
    def _press(self) -> None: ...

    @_support.production("-NAME")
    def _release(self) -> None: ...

    @_support.production('type("TEXT")')
    def _type(self) -> None: ...

    def notes(self) -> None: ...

    # -- interface ---------------------------------------------------------

    def describe(self) -> str:
        return _support.render_spec(self)

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def report(self) -> dict[str, Any]:
        report = _support.drift_report(self, producer=PRODUCER)
        report["paired_with"] = PAIRED_WITH
        return report

    def parse(self, text: str) -> CompactRawAction:
        line = _support.final_line(text, error=CompactRawError)
        mouse, _, tail = line.partition(";")
        dx, dy, scroll = _support.parse_mouse_triple(mouse, error=CompactRawError)
        elements = _support.scan_elements(
            tail, allow_type=True, allow_move=False, error=CompactRawError
        )
        return CompactRawAction(dx, dy, scroll, elements, prompt_digest=self.digest)

    def format(self, action: CompactRawAction) -> str:
        head = f"{action.dx} {action.dy} {action.scroll}"
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
        action: CompactRawAction,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Resolve the raw relative delta against ``cursor`` into absolute pixels."""
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        target = _support.clamp((here[0] + action.dx, here[1] + action.dy), geometry)
        if target != here:
            operations.append(_support.move_to(target))
        if action.scroll:
            operations.append(_support.scroll(0, action.scroll))
        operations.extend(
            _support.lower_transitions(action.elements, error=CompactRawError)
        )
        return tuple(operations)

    # -- training-target construction --------------------------------------

    def action_from_operations(
        self,
        operations: Sequence[Operation],
        *,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> CompactRawAction:
        """Absolute Operations -> an action. The inverse of ``compile_action``.

        An empty stream is ``0 0 0`` — this arm's idle action. A ``glide_to``
        raises: the grammar has no stroke primitive, and the converters this
        replaces degraded exactly that into a stationary ``+LMB -LMB``.
        """
        if _support.terminate_status(terminate, error=CompactRawError) is not None:
            raise CompactRawError(
                "this grammar has no control tokens, so termination cannot be "
                "expressed; deltatype_v2 has TERMINATE and FAIL"
            )
        groups = _support.group_operations(
            operations, geometry=geometry, cursor=cursor, error=CompactRawError
        )
        here = _support.clamp(cursor, geometry)
        plan = _support.bare_token_plan(
            groups,
            cursor=here,
            allow_type=True,
            allow_stroke=False,
            error=CompactRawError,
        )
        dx, dy = (
            (0, 0)
            if plan.target is None
            else (plan.target[0] - here[0], plan.target[1] - here[1])
        )
        return CompactRawAction(
            dx, dy, plan.scroll, plan.elements, prompt_digest=self.digest
        )

    def from_target(
        self,
        cursor: tuple[int, int],
        target: tuple[int, int],
        *,
        scroll: int = 0,
        elements: tuple[_support.Element, ...] = (),
    ) -> CompactRawAction:
        """The delta that carries ``cursor`` to an absolute ``target``.

        This is ``rung2_sameapp.compile_compact``'s job: a scripted trajectory
        knows the geometry of the element it wants and one fresh cursor read.
        Keeping it on the codec is what makes the label and the parse the same
        grammar.
        """
        return CompactRawAction(
            dx=target[0] - cursor[0],
            dy=target[1] - cursor[1],
            scroll=scroll,
            elements=elements,
        )


_support.apply_matched_arm_prose(CompactRawCodec)

CODEC = CompactRawCodec()
