"""Grammar registry: one place that knows how each action format is written.

Two defects came from grammar assumptions leaking into analysis code:

* **Defect #5** — a mouse-op detector that matched only ``computer_use`` op
  names (``mouse_move``, ``left_click``, ...) scored the *bare-line* grammar a
  fake 0% mouse-op rate. The real rate was 80.1%. Any per-grammar metric must
  dispatch through a registry, never pattern-match one grammar's vocabulary.
* **Defect #14** — 11-13% of ``deltatype`` completions omit the scroll token,
  emitting ``dx dy ; +LMB -LMB`` instead of ``dx dy scroll ; ...``. Strict
  three-token parsing counts those as *no-moves*, a penalty that lands only on
  that one grammar and makes it look worse than it is. The registry parses them
  in a tolerant mode that **recovers the move and counts the omission**, and the
  omission count is a reported diagnostic, never a silent repair.

The parsing itself is delegated to ``juergen/eval/action_parser.py`` — the exact
module the evaluation harness uses — so a round-trip audit through this registry
is an audit through the eval parser (see :mod:`rft.roundtrip`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rft.errors import SchemaError

# ---------------------------------------------------------------------------
# eval/action_parser.py import. It is a flat module in the sibling `eval`
# workspace member, not an installed package, so tests and CLIs add that
# directory to sys.path (see rft.evalparser).
from rft.evalparser import (
    Action,
    DeltaTypeAction,
    format_deltatype,
    have,
    parse_action,
    parse_computer_use_tool_calls,
    parse_deltatype,
)

#: computer_use op names that move the pointer or press a mouse button. Kept
#: here, in the registry, so that no analysis module ever hard-codes them again
#: (defect #5).
#: NOTE ``move_rel`` is here as well as ``mouse_move``: the relative
#: (``move_rel``/``native_rel``) prompt family names the pointer-move op
#: ``move_rel``, and omitting it makes the whole relative grammar read as
#: "no mouse ops" — defect #5 all over again, one vocabulary down.
COMPUTER_USE_MOUSE_OPS: frozenset[str] = frozenset(
    {
        "mouse_move",
        "move_rel",
        "mouse_move_rel",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
    }
)

#: Ops that move the pointer (as opposed to pressing a button or scrolling).
COMPUTER_USE_MOVE_OPS: frozenset[str] = frozenset(
    {"mouse_move", "move_rel", "mouse_move_rel", "left_click_drag"}
)


@dataclass(frozen=True)
class MouseOp:
    """A pointer-affecting operation extracted from one completion.

    ``dx``/``dy`` are the *relative POINTER* delta the op requests, when the grammar
    expresses one. For grammars that address absolute coordinates the deltas are
    ``None`` and ``absolute`` carries the target.

    ``scroll`` is a **separate axis** and is deliberately NOT expressed as a dy.
    Folding a scroll amount into the pointer delta makes a pure-scroll action look
    like a large pointer move (a ``0 0 -800`` scroll became a ``(0, -800)`` "move"),
    which corrupts every magnitude statistic that consumes ``net_delta``.
    """

    kind: str  # "move" | "button" | "scroll"
    dx: int | None = None
    dy: int | None = None
    absolute: tuple[int, int] | None = None
    scroll: int | None = None
    detail: str = ""


@dataclass
class ParsedCompletion:
    """Grammar-agnostic view of one model completion.

    ``mouse_ops`` is what a mouse-op detector must consume. ``anomalies`` is the
    list of tolerated-but-counted deviations (e.g. a missing scroll token); it
    is a reported diagnostic, never discarded.

    ``canonical`` is the completion re-serialised into its own grammar, so that
    ``parse -> serialise -> parse`` is a real round trip through the harness parser
    (:mod:`rft.roundtrip`). It is a **valid grammar string**, never a debug repr —
    a canonical form that cannot be re-parsed makes the audit vacuous.
    """

    grammar: str
    raw: str
    mouse_ops: tuple[MouseOp, ...] = ()
    terminate: bool = False
    fail: bool = False
    no_op: bool = False
    typed_text: tuple[str, ...] = ()
    anomalies: tuple[str, ...] = ()
    canonical: str | None = None
    #: Raw parsed payload, kept so the grammar's serialiser can rebuild the
    #: completion losslessly (tool-call argument dicts, bare-line triplet, ...).
    payload: object = None

    @property
    def has_mouse_op(self) -> bool:
        return bool(self.mouse_ops)

    @property
    def net_delta(self) -> tuple[int, int] | None:
        """Net relative POINTER delta across ops, or None if there is none.

        ``None`` means "this completion expresses no relative pointer motion" —
        an absolute-convention completion, a pure scroll, a pure keypress, a
        NO_OP/TERMINATE. It is not ``(0, 0)``: a caller that wants to know whether
        the policy stood still must distinguish "did not move" from "did not say".
        Scroll never contributes; see :class:`MouseOp`.
        """
        rel = [
            (o.dx, o.dy)
            for o in self.mouse_ops
            if o.kind != "scroll" and o.dx is not None and o.dy is not None
        ]
        if not rel:
            return None
        return (sum(d[0] for d in rel), sum(d[1] for d in rel))

    @property
    def net_scroll(self) -> int | None:
        """Net scroll amount, or None if the completion expresses no scroll."""
        amounts = [o.scroll for o in self.mouse_ops if o.scroll is not None]
        if not amounts:
            return None
        return sum(amounts)


# ---------------------------------------------------------------------------
# Thinking-tag handling (defect #15)
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> tuple[str, bool]:
    """Return ``(visible_text, had_thinking)`` for a possibly-thinking completion.

    **Defect #15.** The Qwen3-VL-4B-Thinking chat template injects the *opening*
    ``<think>`` tag into the prompt, so the completion carries only the closing
    ``</think>``. A parser that requires a balanced ``<think>...</think>`` pair
    rejects every completion and reports a false all-zeros.

    This handles all three shapes: balanced pair, closing tag only (the
    template-injected case), and no tags at all.
    """
    if not isinstance(text, str):
        raise TypeError(f"strip_thinking expects str, got {type(text).__name__}")
    close = text.find(_THINK_CLOSE)
    if close != -1:
        # Everything up to and including the close tag is reasoning, whether or
        # not an opening tag is present in the completion.
        return text[close + len(_THINK_CLOSE) :].strip(), True
    if _THINK_OPEN in text:
        # Opening tag present but never closed: the completion was truncated
        # mid-reasoning. That is a real failure, not a parse convention.
        raise SchemaError(
            "completion opens <think> but never closes it (truncated mid-reasoning); "
            "this is an unterminated generation, not a parseable action"
        )
    return text.strip(), False


# ---------------------------------------------------------------------------
# Per-grammar parsers
# ---------------------------------------------------------------------------

_THREE_INT_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
_TWO_INT_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s*$")


def _mouse_ops_from_bare(dx: int, dy: int, scroll: int, events: Sequence[Any]) -> list[MouseOp]:
    ops: list[MouseOp] = []
    if (dx, dy) != (0, 0):
        ops.append(MouseOp(kind="move", dx=dx, dy=dy, detail="bare-line delta"))
    if scroll:
        ops.append(MouseOp(kind="scroll", scroll=scroll, detail="bare-line scroll"))
    for ev in events:
        if getattr(ev, "mouse_button", None) is not None:
            ops.append(MouseOp(kind="button", detail=f"{ev.kind} {ev.what}"))
    return ops


def _format_bare_line(dx: int, dy: int, scroll: int, events: Sequence[Any], no_op: bool) -> str:
    """Serialise a bare-line action back to its canonical grammar string."""
    if no_op:
        return "NO_OP"
    label = f"{dx} {dy} {scroll}"
    toks = [("+" if e.kind == "press" else "-") + e.what for e in events]
    if toks:
        label += " ; " + " ".join(toks)
    return label


def _parse_bare_line(text: str) -> ParsedCompletion:
    """The crowd-cast-native bare-token grammar: ``dx dy scroll ; +K -K``.

    Uses ``eval.action_parser.parse_action`` verbatim.
    """
    visible, _ = strip_thinking(text)
    action: Action = parse_action(visible)
    ops = _mouse_ops_from_bare(action.dx, action.dy, action.scroll, action.events)
    return ParsedCompletion(
        grammar="bare_line",
        raw=text,
        mouse_ops=tuple(ops),
        no_op=action.no_op,
        canonical=_format_bare_line(
            action.dx, action.dy, action.scroll, action.events, action.no_op
        ),
        payload=action,
    )


def _split_mouse_segment(visible: str) -> tuple[str, str]:
    head, _, tail = visible.partition(";")
    return head, tail


def _parse_deltatype(text: str) -> ParsedCompletion:
    """``deltatype``: bare-token grammar + ``type("...")`` + TERMINATE/FAIL.

    Tolerates the defect-#14 two-token mouse segment (missing scroll) by
    inserting an explicit ``scroll=0`` and recording an anomaly. The move is
    recovered; the omission is counted.
    """
    visible, _ = strip_thinking(text)
    anomalies: list[str] = []
    candidate = visible
    first_line = visible.split("\n", 1)[0].strip()
    head, tail = _split_mouse_segment(first_line)
    if first_line not in {"NO_OP", "TERMINATE", "FAIL"} and _TWO_INT_RE.match(head):
        dx_s, dy_s = _TWO_INT_RE.match(head).groups()  # type: ignore[union-attr]
        repaired = f"{dx_s} {dy_s} 0" + (f" ;{tail}" if tail else "")
        anomalies.append("missing_scroll_token")
        candidate = repaired
    action: DeltaTypeAction = parse_deltatype(candidate)
    ops = _mouse_ops_from_bare(action.dx, action.dy, action.scroll, action.events)
    return ParsedCompletion(
        grammar="deltatype",
        raw=text,
        mouse_ops=tuple(ops),
        terminate=action.terminate,
        fail=action.fail,
        no_op=action.no_op,
        typed_text=action.type_texts,
        anomalies=tuple(anomalies),
        canonical=format_deltatype(action),
        payload=action,
    )


def _coord_pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict) and {"x", "y"} <= set(value):
        try:
            return int(value["x"]), int(value["y"])
        except (TypeError, ValueError):
            return None
    return None


def _parse_computer_use(text: str, *, relative: bool) -> ParsedCompletion:
    """The Qwen ``<tool_call>``/``computer_use`` grammar.

    ``relative=True`` is the ``move_rel`` variant, where ``coordinate`` carries a
    *delta*; ``relative=False`` is the absolute/pixel variant, where it carries
    a screen coordinate. Both are dispatched here so that no caller has to guess
    which convention a number is in — mixing them up is what produced the
    single-step "grounding wall" artifact.
    """
    visible, _ = strip_thinking(text)
    grammar = "computer_use_move_rel" if relative else "computer_use_absolute"
    if "TERMINATE" in visible and "<tool_call>" not in visible:
        return ParsedCompletion(
            grammar=grammar, raw=text, terminate=True, canonical="TERMINATE", payload=[]
        )
    calls = parse_computer_use_tool_calls(visible)
    ops: list[MouseOp] = []
    typed: list[str] = []
    terminate = False
    anomalies: list[str] = []
    for call in calls:
        args = call.arguments
        op = args.get("action")
        if op == "terminate":
            terminate = True
            continue
        if op == "type":
            value = args.get("text")
            if isinstance(value, str):
                typed.append(value)
            continue
        if op not in COMPUTER_USE_MOUSE_OPS:
            # keyboard ops etc. - not a mouse op, but note unknown names loudly.
            if op not in {"key", "wait", "screenshot", "hold_key"}:
                anomalies.append(f"unknown_op:{op}")
            continue
        coord = _coord_pair(args.get("coordinate"))
        if op == "scroll":
            amount = args.get("scroll_amount") or args.get("amount")
            ops.append(
                MouseOp(
                    kind="scroll",
                    scroll=int(amount) if isinstance(amount, (int, float)) else None,
                    detail="computer_use scroll",
                )
            )
            continue
        if coord is None:
            ops.append(MouseOp(kind="button", detail=f"computer_use {op}"))
            continue
        kind = "move" if op in COMPUTER_USE_MOVE_OPS else "button"
        if relative:
            ops.append(MouseOp(kind=kind, dx=coord[0], dy=coord[1],
                               detail=f"computer_use {op} (rel)"))
        else:
            ops.append(MouseOp(kind=kind, absolute=coord,
                               detail=f"computer_use {op} (abs)"))
    arguments = [dict(c.arguments) for c in calls]
    return ParsedCompletion(
        grammar=grammar,
        raw=text,
        mouse_ops=tuple(ops),
        terminate=terminate,
        typed_text=tuple(typed),
        anomalies=tuple(anomalies),
        canonical=_format_computer_use(arguments),
        payload=arguments,
    )


def _format_computer_use(arguments: Sequence[Mapping[str, Any]]) -> str:
    """Serialise computer_use tool calls back into re-parseable ``<tool_call>`` blocks."""
    blocks = [
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": dict(args)}, sort_keys=True,
                     ensure_ascii=False)
        + "\n</tool_call>"
        for args in arguments
    ]
    return "\n".join(blocks)


@dataclass(frozen=True)
class Grammar:
    """One registered action format."""

    name: str
    parse: Callable[[str], ParsedCompletion]
    #: True if pointer targets are expressed as relative deltas.
    relative: bool
    #: Human-readable note about the convention, printed in diagnostics so a
    #: reader never has to infer whether a number is a delta or a coordinate.
    convention: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: ``eval/action_parser.py`` symbols this grammar needs. If the loaded parser
    #: revision lacks any of them the grammar is registered but *unavailable*, and
    #: using it raises with the missing symbol named. See :mod:`rft.evalparser`.
    requires: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return all(have(sym) for sym in self.requires)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sym for sym in self.requires if not have(sym))


GRAMMARS: dict[str, Grammar] = {}


def register(grammar: Grammar) -> None:
    for key in (grammar.name, *grammar.aliases):
        if key in GRAMMARS:
            raise SchemaError(f"grammar name/alias {key!r} registered twice")
        GRAMMARS[key] = grammar


register(
    Grammar(
        name="bare_line",
        parse=_parse_bare_line,
        relative=True,
        convention="relative pointer delta, `dx dy scroll ; +K -K`",
        aliases=("diffabs", "bare", "crowdcast"),
    )
)
register(
    Grammar(
        name="deltatype",
        parse=_parse_deltatype,
        relative=True,
        convention="relative pointer delta + coalesced type(), TERMINATE/FAIL tokens",
        requires=("parse_deltatype", "format_deltatype", "DeltaTypeAction"),
    )
)
register(
    Grammar(
        name="computer_use_move_rel",
        parse=lambda t: _parse_computer_use(t, relative=True),
        relative=True,
        convention="computer_use tool calls whose `coordinate` is a RELATIVE delta",
        aliases=("move_rel", "native_rel"),
        requires=("parse_computer_use_tool_calls",),
    )
)
register(
    Grammar(
        name="computer_use_absolute",
        parse=lambda t: _parse_computer_use(t, relative=False),
        relative=False,
        convention="computer_use tool calls whose `coordinate` is an ABSOLUTE pixel",
        aliases=("absolute", "pixel", "computer_use"),
        requires=("parse_computer_use_tool_calls",),
    )
)


def get_grammar(name: str, *, require_available: bool = True) -> Grammar:
    """Look up a registered grammar, listing the alternatives on failure.

    Raises:
        SchemaError: the name is unknown, or (when ``require_available``) the
            loaded ``eval/action_parser.py`` revision lacks the symbols the grammar
            needs. The second case names the missing symbols — it never falls back
            to an approximate parse.
    """
    try:
        grammar = GRAMMARS[name]
    except KeyError:
        raise SchemaError(
            f"unknown action grammar {name!r}; registered: {sorted(set(GRAMMARS))!r}"
        ) from None
    if require_available and not grammar.available:
        raise SchemaError(
            f"grammar {name!r} needs eval/action_parser.py symbols {list(grammar.missing)}, "
            f"which the loaded parser does not provide. {_evalparser_hint()}"
        )
    return grammar


def _evalparser_hint() -> str:
    from rft.evalparser import describe

    return describe() + " -- set JUERGEN_EVAL_DIR to an eval/ that has them."


def available_grammars() -> dict[str, bool]:
    """Availability of every distinct registered grammar, for diagnostics."""
    return {g.name: g.available for g in dict.fromkeys(GRAMMARS.values())}


def parse_completion(text: str, *, grammar: str) -> ParsedCompletion:
    """Parse one completion under a *named* grammar.

    There is no auto-detection. A caller that does not know which grammar it is
    reading cannot produce a trustworthy metric — that is defect #5 restated.
    """
    return get_grammar(grammar).parse(text)


def has_mouse_op(text: str, *, grammar: str) -> bool:
    """Grammar-dispatched mouse-op detector (the defect-#5 replacement)."""
    return parse_completion(text, grammar=grammar).has_mouse_op
