"""The `ordered_events_v3` prompts: with a stated goal, and without one.

The grammar's own spec is the no-goal prompt. Its preamble already says "Each
user turn shows the current screen" — an unconditioned rollout — so
`ordered_events_v3_no_goal` is the identity edit and its digest equals the
codec's own.

`ordered_events_v3_goal` replaces exactly that observation sentence. It is the
one sentence that differed between `cua_ordered_typing_v1` and
`cua_ordered_typing_v1_no_goal` in the retired `eval/osworld_system_prompts.py`
dict — the whole goal/no-goal distinction was those 72 bytes, and everything
else in both was the grammar, which is why they are an edit here and not two
files.

The `<think></think>` clause is carried into the goal wording deliberately: it
is a fact about how this codec parses (`_support.final_line` reads the last
non-empty line), so it holds whether or not a goal was stated. Dropping it in
one variant would make the two prompts disagree about the grammar.
"""

from __future__ import annotations

from prompts import Prompt, register

GRAMMAR = "ordered_events_v3"

#: The codec's own observation sentence, verbatim from its class docstring —
#: line breaks included, because `replace` matches the rendered spec exactly.
_UNCONDITIONED = (
    "Each user turn shows the current screen, with the cursor visible as a small\n"
    "arrow. Reply with exactly ONE action line per turn; only the final non-empty\n"
    "line is read, so a `<think></think>` block may precede it."
)

_GOAL_CONDITIONED = (
    "The first user turn states the goal alongside the current screen; each later\n"
    "user turn shows the updated screen, with the cursor visible as a small arrow.\n"
    "Reply with exactly ONE action line per turn; only the final non-empty line is\n"
    "read, so a `<think></think>` block may precede it."
)

NO_GOAL = register(
    Prompt(
        id="ordered_events_v3_no_goal",
        grammar=GRAMMAR,
        summary="the grammar's own spec, unedited — an unconditioned rollout",
    )
)

GOAL = register(
    Prompt(
        id="ordered_events_v3_goal",
        grammar=GRAMMAR,
        summary="the goal is stated in the first user turn alongside the screen",
        replace=((_UNCONDITIONED, _GOAL_CONDITIONED),),
    )
)

NO_GOAL_THINKING = register(
    Prompt(
        id="ordered_events_v3_no_goal_thinking",
        grammar=GRAMMAR,
        summary="unedited spec behind the shared thinking preamble",
        thinking=True,
    )
)

GOAL_THINKING = register(
    Prompt(
        id="ordered_events_v3_goal_thinking",
        grammar=GRAMMAR,
        summary="goal-conditioned, behind the shared thinking preamble",
        replace=((_UNCONDITIONED, _GOAL_CONDITIONED),),
        thinking=True,
    )
)
