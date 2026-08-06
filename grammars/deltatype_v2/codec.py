"""The Phase-B compact deltatype-v2 grammar.

Collapses four parsers of one grammar into one codec:

* ``experiments/phaseb_deltatype_raw_v2/action_v2.py`` (parse / format /
  ordered_plan / dispatch),
* ``osworld_parity/sign_of_life_v2/compact_relative.py`` (a near-clone of it:
  byte-identical private ``_scan_elements`` and ``_validate_ordered_move``),
* the ``parse_deltatype`` / ``format_deltatype`` half of ``eval/action_parser.py``
  (same grammar, coalesced ``type()``, TERMINATE / FAIL),
* the ``compile_compact_relative`` lowering that used to live beside the sealed
  sign-of-life prompt.

``compact_relative.verify_sealed_contract`` raised ``RuntimeError`` when the
inlined ``SYSTEM_PROMPT`` no longer hashed to its recorded digest — while that
prompt was inlined in the same module, so editing the grammar in place tripped
the check and forking a worktree was cheaper. ``report()`` below returns the
same three digests as data with a ``matches_producer`` boolean. It cannot raise,
and there is no replacement gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pixeldesk.geometry import DisplayGeometry
from pixeldesk.ir import Operation

from .. import _support

#: Provenance of the sealed Phase-B producer prompt, recorded so drift is
#: visible. ``describe()`` renders a superset of it (the sealed text omits
#: ``type()``, which the producer's parser has always accepted), so
#: ``matches_producer`` is expected to be False and that is not an error.
PRODUCER = {
    "prompt_sha256": (
        "57f7d0b230974068618b48151b73215d5517d5445a99dbf5abdc05557e3482e6"
    ),
    "action_v2_sha256": (
        "1ded3d5a7e51da71cf3082049fbdd404971ebf72a95d93f333ebb3ee3075ccb7"
    ),
    "dataset_manifest_sha256": (
        "77085ee3c2ea7d780e96ade76efbffc0746139c0c619a5d9cbcec8562a1a25d5"
    ),
}

DRAG_SECONDS = 0.5


class DeltatypeV2Error(ValueError):
    """Malformed deltatype-v2 action text."""


@dataclass(frozen=True)
class DeltatypeV2Action:
    """dx / dy / scroll plus ordered elements, or one control token."""

    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[_support.Element, ...] = ()
    no_op: bool = False
    terminate: bool = False
    fail: bool = False
    #: sha256 of the prompt the parse ran under. Reported, never enforced, and
    #: excluded from equality so a round-trip still compares equal.
    prompt_digest: str = field(default="", compare=False)

    @property
    def control(self) -> bool:
        return self.no_op or self.terminate or self.fail

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "scroll": self.scroll,
            "elements": [element.to_dict() for element in self.elements],
            "no_op": self.no_op,
            "terminate": self.terminate,
            "fail": self.fail,
        }


def action_from_dict(value: dict[str, Any]) -> DeltatypeV2Action:
    return DeltatypeV2Action(
        dx=int(value.get("dx", 0)),
        dy=int(value.get("dy", 0)),
        scroll=int(value.get("scroll", 0)),
        elements=tuple(
            _support.element_from_dict(item) for item in value.get("elements", ())
        ),
        no_op=bool(value.get("no_op", False)),
        terminate=bool(value.get("terminate", False)),
        fail=bool(value.get("fail", False)),
    )


class DeltatypeV2Codec:
    """You operate a desktop computer from screenshots.

    Return one bare action line. Reasoning before it is legal and expected —
    only the FINAL non-empty line is read as the action.

    Mouse values are RAW PIXEL deltas from the current cursor. Optional ordered
    elements follow ` ; ` and are executed left to right, after the move.
    """

    name = "deltatype_v2"

    #: Empty by design: prose before the action line is legal, so no token
    #: sequence marks the end of a turn earlier than the completion does.
    stop_sequences: tuple[str, ...] = ()

    # -- productions: each docstring is the only spec of that production ----

    @_support.production("dx dy scroll")
    def _mouse(self) -> None:
        """Three integers. dx and dy are a move in RAW SCREEN PIXELS relative to
        the current cursor (dx > 0 right, dy > 0 down); scroll is wheel ticks
        (positive scrolls up). The move is applied before any elements.
        """

    @_support.production("+NAME")
    def _press(self) -> None:
        """Press NAME. Mouse buttons are LMB, RMB, MMB. Keyboard keys use rdev
        names: Return, Tab, Backspace, Escape, Space, ShiftLeft, ControlLeft,
        AltLeft, ArrowUp, KeyA, and so on.
        """

    @_support.production("-NAME")
    def _release(self) -> None:
        """Release NAME. A left click is `+LMB -LMB`; a chord presses in order
        and releases in reverse: `+ControlLeft +KeyC -KeyC -ControlLeft`.
        """

    @_support.production('type("TEXT")')
    def _type(self) -> None:
        """Type TEXT as one coalesced burst. TEXT is a JSON string, so it may
        contain spaces, `;`, `+` and escaped quotes. It must NOT contain a
        newline — press Return as an event (`+Return -Return`) instead.
        """

    @_support.production("MOVE(dx,dy)")
    def _drag(self) -> None:
        """A second raw-pixel delta applied over 0.5 s. Legal ONLY in the
        left-button drag form:
          initial_dx initial_dy 0 ; +LMB MOVE(drag_dx,drag_dy) -LMB
        The initial delta moves to the drag start; use 0 0 to drag from the
        current cursor. MOVE(0,0) is a real zero-distance drag.
        """

    @_support.production("NO_OP")
    def _no_op(self) -> None:
        """Do nothing this turn; wait for the screen to settle."""

    @_support.production("TERMINATE")
    def _terminate(self) -> None:
        """The goal is complete. Stop."""

    @_support.production("FAIL")
    def _fail(self) -> None:
        """The goal is impossible. Stop."""

    def notes(self) -> None:
        """Emit exactly one action line per turn."""

    # -- interface ---------------------------------------------------------

    def describe(self) -> str:
        return _support.render_spec(self)

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def report(self) -> dict[str, Any]:
        """Prompt provenance as data. Never raises."""
        return _support.drift_report(self, producer=PRODUCER)

    def parse(self, text: str) -> DeltatypeV2Action:
        line = _support.final_line(text, error=DeltatypeV2Error)
        digest = self.digest
        if line == "NO_OP":
            return DeltatypeV2Action(no_op=True, prompt_digest=digest)
        if line == "TERMINATE":
            return DeltatypeV2Action(terminate=True, prompt_digest=digest)
        if line == "FAIL":
            return DeltatypeV2Action(fail=True, prompt_digest=digest)
        mouse, _, tail = line.partition(";")
        dx, dy, scroll = _support.parse_mouse_triple(mouse, error=DeltatypeV2Error)
        elements = _support.scan_elements(
            tail, allow_type=True, allow_move=True, error=DeltatypeV2Error
        )
        action = DeltatypeV2Action(dx, dy, scroll, elements, prompt_digest=digest)
        return self._validate_drag(action)

    def format(self, action: DeltatypeV2Action) -> str:
        if action.no_op:
            return "NO_OP"
        if action.terminate:
            return "TERMINATE"
        if action.fail:
            return "FAIL"
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
        action: DeltatypeV2Action,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Resolve raw relative deltas against ``cursor`` into absolute pixels."""
        if action.control:
            return ()
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        target = _support.clamp((here[0] + action.dx, here[1] + action.dy), geometry)
        if target != here:
            operations.append(_support.move_to(target))
        here = target
        if action.scroll:
            operations.append(_support.scroll(0, action.scroll))
        for element in action.elements:
            if element.kind == "move":
                assert element.delta is not None
                here = _support.clamp(
                    (here[0] + element.delta[0], here[1] + element.delta[1]), geometry
                )
                operations.append(_support.glide_to(here, DRAG_SECONDS))
                continue
            operations.extend(
                _support.lower_transitions((element,), error=DeltatypeV2Error)
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
    ) -> DeltatypeV2Action:
        """Absolute Operations -> an action. The inverse of ``compile_action``.

        This grammar spells termination three ways, so ``terminate`` resolves to
        TERMINATE for success and FAIL for failure. A ``glide_to`` inside a held
        left button becomes the MOVE drag form; the stroke's own duration is
        dropped, because the grammar fixes it at ``DRAG_SECONDS``.
        """
        status = _support.terminate_status(terminate, error=DeltatypeV2Error)
        groups = _support.group_operations(
            operations, geometry=geometry, cursor=cursor, error=DeltatypeV2Error
        )
        if status is not None:
            if groups:
                raise DeltatypeV2Error(
                    "a terminating turn carries no operations in this grammar: "
                    "TERMINATE and FAIL are whole action lines"
                )
            return DeltatypeV2Action(
                terminate=status == "success",
                fail=status == "failure",
                prompt_digest=self.digest,
            )
        here = _support.clamp(cursor, geometry)
        plan = _support.bare_token_plan(
            groups,
            cursor=here,
            allow_type=True,
            allow_stroke=True,
            error=DeltatypeV2Error,
        )
        if plan.idle:
            return DeltatypeV2Action(no_op=True, prompt_digest=self.digest)
        dx, dy = (
            (0, 0)
            if plan.target is None
            else (plan.target[0] - here[0], plan.target[1] - here[1])
        )
        return self._validate_drag(
            DeltatypeV2Action(
                dx, dy, plan.scroll, plan.elements, prompt_digest=self.digest
            )
        )

    # -- grammar rule ------------------------------------------------------

    def _validate_drag(self, action: DeltatypeV2Action) -> DeltatypeV2Action:
        moves = [
            element for element in action.elements if element.kind == "move"
        ]
        if not moves:
            return action
        expected = (
            _support.Element("event", name="LMB", pressed=True),
            moves[0],
            _support.Element("event", name="LMB", pressed=False),
        )
        if len(moves) != 1 or action.scroll != 0 or action.elements != expected:
            raise DeltatypeV2Error(
                "MOVE is reserved for `initial_dx initial_dy 0 ; "
                "+LMB MOVE(drag_dx,drag_dy) -LMB`"
            )
        return action


CODEC = DeltatypeV2Codec()
