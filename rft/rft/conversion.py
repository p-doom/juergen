"""Action-format conversion that can only touch the ACTION SPAN.

**The defect this module exists to make impossible.** In
``build_osworld_format_records.py::convert_response`` the absolute branch was::

    if fmt == "absolute":
        return resp_text          # verbatim: prose preamble AND tool schema kept

while every *relative* branch re-rendered the action from scratch and silently
dropped everything around it. Measured over the shipped datasets: a reasoning
preamble was present in **2383/2383 absolute records and 0/2441 relative** ones;
the canonical ``<tools>`` schema likewise 2383/2383 vs 0/2441.

The reasoning preamble is **format-independent natural language**. Only the action
span ever needed converting. So the pipeline deleted the visual-reasoning scratchpad
from exactly the arm that had to learn a new convention and kept it for the arm that
had nothing to learn — which invalidates every absolute-vs-relative comparison built
on it. Each output looked individually plausible, so it survived for weeks.

The structural fix is to make the conversion incapable of expressing that bug:

* :func:`split_response` separates a response into ``prefix`` / ``action_span`` /
  ``suffix``;
* :func:`convert_action_span` rebuilds the response as
  ``prefix + f(action_span) + suffix``, so the surrounding bytes are carried over by
  construction rather than by the converter remembering to;
* :func:`assert_only_action_span_changed` is the gate: it compares source and
  converted context byte-for-byte and raises on any difference.

A converter that wants to change the prose has to say so explicitly, and the reason
is recorded in the dataset manifest (see :mod:`rft.arms`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rft.errors import SchemaError


class ContextAlteredError(SchemaError):
    """A conversion changed something outside the action span."""


#: A ``<tool_call>`` block, including the tags. Multiple blocks in a row (plus the
#: whitespace between them) form one action span: a drag is ``mouse_down`` +
#: ``mouse_move`` + ``mouse_up`` and splitting them would be meaningless.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)

#: Bare-token action line: ``dx dy [scroll] [; +K -K]`` or a control token.
_BARE_ACTION_LINE = re.compile(
    r"^[ \t]*(?:NO_OP|TERMINATE|FAIL|-?\d+[ \t]+-?\d+(?:[ \t]+-?\d+)?(?:[ \t]*;.*)?)[ \t]*$"
)

#: Markers a prompt family uses to introduce the action ("Action:", "Answer:").
#: The marker itself is CONTEXT, not action: it is format-independent.
_ACTION_MARKER = re.compile(r"(?im)^[ \t]*(?:action|answer)[ \t]*:[ \t]*")


@dataclass(frozen=True)
class ResponseParts:
    """A response split into non-action context and the action span.

    ``prefix + action_span + suffix == original`` always holds exactly; the split is
    a partition of the original bytes, never a normalisation of them.
    """

    prefix: str
    action_span: str
    suffix: str
    #: ``"tool_call"`` / ``"bare_line"`` are RECOGNISED actions. ``"last_line"`` is the
    #: permissive fallback (the last non-blank line, used so that an unknown grammar
    #: still round-trips identically) and ``"none"`` means the text is entirely blank.
    #: Only recognised kinds may be converted by default — see
    #: :attr:`recognised` and :func:`convert_action_span`.
    kind: str

    def __post_init__(self) -> None:
        # The invariant that makes the whole module trustworthy.
        if self.prefix + self.action_span + self.suffix != self.original:
            raise SchemaError("ResponseParts is not a partition of the original text")

    @property
    def original(self) -> str:
        return self._original

    def rebuild(self, new_action: str) -> str:
        return self.prefix + new_action + self.suffix

    @property
    def has_prose_context(self) -> bool:
        """Whether any non-whitespace context surrounds the action."""
        return bool(self.prefix.strip() or self.suffix.strip())

    @property
    def recognised(self) -> bool:
        """Whether the action span matched a KNOWN action shape.

        ``False`` for the ``last_line`` fallback: that span is "whatever was on the
        last line", which is fine to copy through unchanged but must not be handed to
        a converter — rewriting it would be a wholesale rewrite of unrecognised text.
        """
        return self.kind in ("tool_call", "bare_line")


# ``ResponseParts`` needs the original for its invariant check but must stay frozen
# and hashable; store it via object.__setattr__ in a small factory instead of a field
# so the dataclass equality stays on the three visible parts.
def _make_parts(prefix: str, action: str, suffix: str, kind: str, original: str) -> ResponseParts:
    parts = ResponseParts.__new__(ResponseParts)
    object.__setattr__(parts, "prefix", prefix)
    object.__setattr__(parts, "action_span", action)
    object.__setattr__(parts, "suffix", suffix)
    object.__setattr__(parts, "kind", kind)
    object.__setattr__(parts, "_original", original)
    parts.__post_init__()
    return parts


def split_response(text: str) -> ResponseParts:
    """Split a response into ``prefix`` / ``action_span`` / ``suffix``.

    Recognises two action shapes:

    * one or more consecutive ``<tool_call>...</tool_call>`` blocks (the whole run is
      the action span, so a multi-call drag stays together);
    * a bare-token action line (``dx dy scroll ; +K -K``, ``NO_OP``, ``TERMINATE``,
      ``FAIL``) — the LAST such line, because a prompt may quote examples earlier.

    If no action is found, ``kind`` is ``"none"`` and the whole text is the prefix.
    That is not an error here: :func:`convert_action_span` will refuse to convert it,
    which is the right place to fail.
    """
    if not isinstance(text, str):
        raise TypeError(f"split_response expects str, got {type(text).__name__}")

    blocks = list(_TOOL_CALL_BLOCK.finditer(text))
    if blocks:
        start = blocks[0].start()
        end = blocks[-1].end()
        return _make_parts(text[:start], text[start:end], text[end:], "tool_call", text)

    # Bare-token action: prefer the last line that actually matches the action
    # grammar; fall back to the last non-blank line. The strict pass first means
    # trailing prose does not get mistaken for an action; the fallback means a
    # grammar this module has not been taught still round-trips identically.
    lines = text.splitlines(keepends=True)

    def _at(i: int, kind: str) -> ResponseParts:
        stripped = lines[i].rstrip("\r\n")
        prefix = "".join(lines[:i])
        tail = lines[i][len(stripped):]
        return _make_parts(prefix, stripped, tail + "".join(lines[i + 1:]), kind, text)

    for i in range(len(lines) - 1, -1, -1):
        if _BARE_ACTION_LINE.match(lines[i].rstrip("\r\n")):
            return _at(i, "bare_line")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            return _at(i, "last_line")
    return _make_parts(text, "", "", "none", text)


def convert_action_span(
    text: str,
    convert: Callable[[str], str],
    *,
    require_action: bool = True,
    keep_prose: bool = True,
) -> str:
    """Apply ``convert`` to the action span only; carry the rest over verbatim.

    This is the API every format converter must be expressed through. A converter
    written as ``str -> str`` over the whole response can delete the prose; a
    converter written as ``action_span -> action_span`` cannot.

    Args:
        keep_prose: contract item **C4**. Dropping prose is an *explicit* option, and
            when used it must be applied **symmetrically to every arm** — including
            the absolute one. The original defect was not "prose was dropped", it was
            "prose was dropped from some arms and not others, depending on which code
            branch you fell into". Use
            :func:`assert_prose_policy_symmetric` to enforce the symmetry across a
            multi-arm build.

    Raises:
        SchemaError: ``require_action`` and no action span was found.
    """
    parts = split_response(text)
    if not parts.recognised:
        if require_action:
            raise SchemaError(
                f"no RECOGNISED action span in response {text[:120]!r} "
                f"(split kind={parts.kind!r}); refusing to convert. A response whose "
                "action cannot be located must be handled explicitly - rewriting it "
                "wholesale is how format-independent content gets destroyed. Pass "
                "require_action=False to copy it through unchanged instead."
            )
        return text
    new_action = convert(parts.action_span)
    if not keep_prose:
        return new_action
    return parts.rebuild(new_action)


def assert_prose_policy_symmetric(policies: Mapping[str, bool]) -> None:
    """Raise unless every arm uses the SAME prose policy (contract item C4).

    Args:
        policies: ``{arm_name: keep_prose}``.

    Raises:
        SchemaError: the arms disagree. This is the defect in its most general form:
            it does not matter which policy is chosen, only that one arm cannot have
            a different one from another.
    """
    if not policies:
        raise SchemaError("no arms given")
    values = set(policies.values())
    if len(values) > 1:
        keep = sorted(k for k, v in policies.items() if v)
        drop = sorted(k for k, v in policies.items() if not v)
        raise SchemaError(
            "prose policy is ASYMMETRIC across arms (contract C4): "
            f"keep_prose=True for {keep!r} but False for {drop!r}. Prose is "
            "format-independent, so whichever policy you want must apply to every arm "
            "- an asymmetry here is exactly the confound that invalidated the "
            "absolute-vs-relative comparison (2383/2383 vs 0/2441)."
        )


def assert_only_action_span_changed(
    source: str, converted: str, *, context: str = ""
) -> None:
    """Raise unless every byte outside the action span is unchanged.

    Args:
        source: the original (teacher) response.
        converted: the response after format conversion.
        context: identifier for the error message (sample id, format name).

    Raises:
        ContextAlteredError: prefix or suffix differ. The message quotes both sides,
            because the historical failure was invisible in aggregate and obvious the
            moment two records were put side by side.
    """
    src = split_response(source)
    dst = split_response(converted)
    where = f" [{context}]" if context else ""
    problems: list[str] = []
    if src.prefix != dst.prefix:
        problems.append(
            f"PREFIX changed{where}:\n"
            f"  source ({len(src.prefix)} chars): {src.prefix!r}\n"
            f"  output ({len(dst.prefix)} chars): {dst.prefix!r}"
        )
    if src.suffix != dst.suffix:
        problems.append(
            f"SUFFIX changed{where}:\n"
            f"  source ({len(src.suffix)} chars): {src.suffix!r}\n"
            f"  output ({len(dst.suffix)} chars): {dst.suffix!r}"
        )
    if problems:
        raise ContextAlteredError(
            "conversion altered content OUTSIDE the action span. The reasoning "
            "preamble is format-INDEPENDENT text; only the action span may change. "
            "Dropping it from one arm and keeping it in another destroys the "
            "comparison those arms exist to make.\n" + "\n".join(problems)
        )


# ---------------------------------------------------------------------------
# context feature extraction (used by rft.arms for cross-arm parity)
# ---------------------------------------------------------------------------

#: A natural-language reasoning preamble: non-whitespace text before the action that
#: is not itself an action. This is what was deleted from every relative record.
_TOOLS_SCHEMA = re.compile(r"<tools>.*?</tools>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ContextFeatures:
    """Format-independent properties of a response, for cross-arm comparison."""

    has_reasoning_preamble: bool
    has_tools_schema: bool
    has_action_marker: bool
    prefix_chars: int
    suffix_chars: int
    action_kind: str

    @classmethod
    def of(cls, text: str) -> ContextFeatures:
        parts = split_response(text)
        around = parts.prefix + parts.suffix
        without_schema = _TOOLS_SCHEMA.sub("", around)
        without_marker = _ACTION_MARKER.sub("", without_schema)
        return cls(
            has_reasoning_preamble=bool(without_marker.strip()),
            has_tools_schema=bool(_TOOLS_SCHEMA.search(around)),
            has_action_marker=bool(_ACTION_MARKER.search(around)),
            prefix_chars=len(parts.prefix),
            suffix_chars=len(parts.suffix),
            action_kind=parts.kind,
        )

    def describe(self) -> str:
        return (
            f"preamble={self.has_reasoning_preamble} tools_schema={self.has_tools_schema} "
            f"action_marker={self.has_action_marker} prefix={self.prefix_chars}c "
            f"suffix={self.suffix_chars}c action_kind={self.action_kind}"
        )
