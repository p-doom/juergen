"""The prompt registry composes over a grammar; it never restates one.

These pin the two properties that make that safe: an unedited prompt IS the
grammar's own spec byte for byte, and an edit that no longer matches its codec
docstring raises instead of silently rendering the base prompt.
"""

from __future__ import annotations

import hashlib

import pytest

import grammars
import prompts
from grammars import THINKING_PREAMBLE
from grammars._support import CONTROL_SPEC


def test_every_registered_prompt_names_a_real_grammar():
    known = set(grammars.available())
    for name in prompts.names():
        assert prompts.grammar_of(name) in known, name


def test_an_unedited_prompt_is_the_grammars_own_spec():
    """`ordered_events_v3_no_goal` declares no edits, so it must not move the
    digest a checkpoint was trained under."""
    prompt = prompts.get("ordered_events_v3_no_goal")
    assert prompt.replace == ()
    assert prompt.describe() == grammars.describe("ordered_events_v3")
    assert prompt.digest == grammars.load("ordered_events_v3").digest


def test_the_goal_edit_changes_one_sentence_and_nothing_else():
    base = prompts.describe("ordered_events_v3_no_goal").splitlines()
    goal = prompts.describe("ordered_events_v3_goal").splitlines()
    assert "states the goal" in "\n".join(goal)
    assert "states the goal" not in "\n".join(base)
    # Everything from the action-line paragraph onward is identical.
    anchor = "An action line is one or more primitives separated by `; ` and applied"
    assert base[base.index(anchor):] == goal[goal.index(anchor):]


def test_every_prompt_ends_with_the_control_block():
    """A prompt's epilogue must not land after the control line: the codecs'
    notes say "nothing else except the control line below"."""
    for name in prompts.names():
        assert prompts.describe(name).endswith(CONTROL_SPEC + "\n"), name


def test_every_prompt_has_exactly_one_control_block():
    for name in prompts.names():
        assert prompts.describe(name).count("Ending the episode") == 1, name


def test_digests_are_distinct_per_prompt():
    seen = {name: prompts.digest(name) for name in prompts.names()}
    assert len(set(seen.values())) == len(seen), seen


def test_report_carries_both_the_prompt_and_the_grammar_digest():
    report = prompts.report("ordered_events_v3_goal")
    assert report["system_prompt_sha256"] == prompts.digest("ordered_events_v3_goal")
    assert report["grammar_sha256"] == grammars.load("ordered_events_v3").digest
    assert report["system_prompt_sha256"] != report["grammar_sha256"]


def test_an_edit_that_no_longer_matches_raises_rather_than_vanishing():
    """The whole point of exact-match `replace`: a reworded codec docstring must
    fail loudly, not quietly render the base prompt under an edited prompt's id."""
    stale = prompts.Prompt(
        id="_stale",
        grammar="ordered_events_v3",
        summary="an edit whose anchor text is not in the spec",
        replace=(("a sentence this codec has never contained", "..."),),
    )
    with pytest.raises(RuntimeError, match="occurs 0 times"):
        stale.describe()


def test_digest_is_the_sha256_of_the_rendered_text():
    name = "ordered_events_v3_goal"
    assert prompts.digest(name) == hashlib.sha256(
        prompts.describe(name).encode()
    ).hexdigest()


def test_unknown_prompt_id_names_what_is_registered():
    with pytest.raises(LookupError, match="ordered_events_v3_goal"):
        prompts.get("no_such_prompt")


def test_an_unedited_thinking_prompt_is_the_form_the_harness_accepts():
    """`DesktopHarnessConfig` accepts the bare `describe()` and
    `THINKING_PREAMBLE + describe()` without an `expect_prompt_mismatch`
    justification. An unedited thinking prompt must render as the second, or it
    would need a written justification to evaluate a checkpoint trained on it."""
    rendered = prompts.describe("ordered_events_v3_no_goal_thinking")
    assert rendered == THINKING_PREAMBLE + grammars.describe("ordered_events_v3")


def test_thinking_does_not_disturb_the_control_block():
    rendered = prompts.describe("ordered_events_v3_goal_thinking")
    assert rendered.startswith(THINKING_PREAMBLE)
    assert rendered.endswith(CONTROL_SPEC + "\n")
    assert rendered.count("Ending the episode") == 1
