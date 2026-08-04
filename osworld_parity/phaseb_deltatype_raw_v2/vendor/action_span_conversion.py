#!/usr/bin/env python3
"""REFERENCE FIX + REGRESSION TESTS for the action-span conversion contract.

THE BUG (found 2026-07-30, `onpolicy_distill/scripts/build_osworld_format_records.py`):

    def convert_response(resp_text, fmt, per_step, step_k):
        if fmt == "absolute":
            return resp_text          # <-- verbatim teacher text, prose INCLUDED
        ...
        return _moverel_render(a, cb, tgt)   # <-- action ONLY, prose DISCARDED

Measured consequence: the canonical `<tools>` schema and a natural-language
reasoning preamble were present in 2383/2383 absolute training records and
0/2441 of every relative format's records. The visual-reasoning scratchpad was
deleted from precisely the arm that had a new convention to learn, and kept for
the arm that had nothing to learn. This invalidates every TRAINED
absolute-vs-relative comparison in the project.

THE CONTRACT (what conversion must guarantee):

  C1  Conversion rewrites ONLY the action span. Every other byte of the
      assistant turn -- prose, thinking blocks, whitespace, ordering -- is
      passed through unchanged.
  C2  Consequently, two format arms built from the same source are
      byte-identical OUTSIDE their action spans.
  C3  Prose is format-INDEPENDENT: it is natural language about the intent and
      is never a function of the action grammar.
  C4  Dropping prose, if ever wanted, is an EXPLICIT and SYMMETRIC option
      (`keep_prose=False` applied to ALL arms including absolute), never an
      accident of which branch you fell into.

This module implements the contract as a pure function plus the regression tests
that catch the defect class. Written to be lifted into the RFT infra as a stage;
NOT applied in place to the shared builder, because other owners' jobs read that
file and a unilateral edit would be exactly the kind of untested glue we are
trying to stop accumulating.
"""
from __future__ import annotations

import re
from typing import Callable

# An action span is either a <tool_call>...</tool_call> block or, in the
# bare-token grammars, the final action line. Prose is everything else.
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)


def split_assistant_turn(text: str) -> tuple[str, str, str]:
    """Split an assistant turn into (prose_before, action_span, prose_after).

    Tool-call grammar: the action span is the <tool_call> block (or the run of
    consecutive blocks). Bare-token grammar: the action span is the LAST
    non-blank line. Whitespace between the parts is preserved by returning the
    exact slices, so ``prose_before + action + prose_after == text``.
    """
    blocks = list(_TOOLCALL_RE.finditer(text))
    if blocks:
        start, end = blocks[0].start(), blocks[-1].end()
        return text[:start], text[start:end], text[end:]
    # bare-token: last non-blank line is the action
    lines = text.splitlines(keepends=True)
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            idx = i
            break
    if idx is None:
        return text, "", ""
    before = "".join(lines[:idx])
    line = lines[idx]
    stripped = line.rstrip("\n")
    trailing = line[len(stripped):]
    return before, stripped, trailing + "".join(lines[idx + 1:])


def convert_assistant_turn(
    text: str,
    render_action: Callable[[str], str],
    *,
    keep_prose: bool = True,
) -> str:
    """Contract-conforming conversion.

    ``render_action`` receives ONLY the action span and returns the action span
    in the target grammar. Prose is passed through untouched (C1/C3). Dropping
    prose is explicit and applies to every arm identically (C4).
    """
    before, action, after = split_assistant_turn(text)
    new_action = render_action(action)
    if not keep_prose:
        return new_action
    return before + new_action + after


# ---------------------------------------------------------------------------
# Regression tests -- the defect class, not just the one instance.
# ---------------------------------------------------------------------------
def _tests() -> int:
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name}: {detail}")

    TOOL = ('Action: Click the "X" button on the top-right corner of the '
            '"Can\'t update Chrome" pop-up to close it.\n'
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "left_click", "coordinate": [982, 127]}}\n</tool_call>')
    BARE = ('<think>\nI want to close the update popup.\n</think>\n'
            '925 -403 0 ; +LMB -LMB')

    # --- T1: prose survives conversion (the actual bug)
    out = convert_assistant_turn(TOOL, lambda a: "925 -403 0 ; +LMB -LMB")
    check("T1 prose preserved", "top-right corner" in out, repr(out))
    check("T1 action replaced", "925 -403 0 ; +LMB -LMB" in out, repr(out))
    check("T1 old action gone", "982" not in out, repr(out))

    # --- T2: C2, two arms byte-identical outside the action span
    a = convert_assistant_turn(TOOL, lambda s: "AAA")
    b = convert_assistant_turn(TOOL, lambda s: "BBBBBB")
    pa, aa, sa = split_assistant_turn(a)
    pb, ab, sb = split_assistant_turn(b)
    check("T2 prefix identical", pa == pb, f"{pa!r} != {pb!r}")
    check("T2 suffix identical", sa == sb, f"{sa!r} != {sb!r}")

    # --- T3: thinking blocks are prose and must survive
    out = convert_assistant_turn(BARE, lambda a: "1 2 0")
    check("T3 think preserved", "<think>" in out and "close the update popup" in out, repr(out))
    check("T3 action replaced", out.rstrip().endswith("1 2 0"), repr(out))

    # --- T4: round-trip identity when render_action is the identity
    for src in (TOOL, BARE, "NO_OP", "", "   ", "a\n\nb\n925 -403 0\n"):
        out = convert_assistant_turn(src, lambda a: a)
        check("T4 identity round-trip", out == src, f"{src!r} -> {out!r}")

    # --- T5: keep_prose=False is symmetric -- action-only for ANY grammar
    for src in (TOOL, BARE):
        out = convert_assistant_turn(src, lambda a: a, keep_prose=False)
        check("T5 action only", "<think>" not in out and "top-right" not in out, repr(out))

    # --- T6: the ORIGINAL buggy behaviour is detected by T2. Simulate it.
    def buggy(text, fmt):
        if fmt == "absolute":
            return text                      # prose kept
        _, action, _ = split_assistant_turn(text)
        return action                        # prose dropped
    ba, bb = buggy(TOOL, "absolute"), buggy(TOOL, "moverel")
    pa2, _, _ = split_assistant_turn(ba)
    pb2, _, _ = split_assistant_turn(bb)
    check("T6 buggy asymmetry IS detected", pa2 != pb2,
          "the regression test would not have caught the real bug")

    # --- T7: multiple consecutive tool_calls are ONE action span (move+click)
    MULTI = ('Move then click.\n<tool_call>\n{"a": 1}\n</tool_call>\n'
             '<tool_call>\n{"a": 2}\n</tool_call>')
    before, action, after = split_assistant_turn(MULTI)
    check("T7 multi-block span", action.count("<tool_call>") == 2, repr(action))
    check("T7 multi prose", before.strip() == "Move then click.", repr(before))

    print(f"action-span conversion regression tests: {7 - len({f.split(':')[0][:2] for f in fails})}/7 groups passing")
    for f in fails:
        print("  FAIL", f)
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if _tests() else 0)
