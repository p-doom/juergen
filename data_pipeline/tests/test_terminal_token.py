"""The terminal token is the only way the control channel enters training data.

A keylog records no intent to terminate — the demonstrator just stopped — so
`TERMINATE: success` can only come from `--terminal-token`. Spelling it wrong is
invisible until eval: the label parses as an ACTION, the episode never ends, and
the grammar's one control channel quietly has two spellings in the corpus.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE = REPO_ROOT / "pipeline" / "crowdcast" / "stage_04_build_conversations.py"


def run_with_token(token: str) -> tuple[int, str]:
    """Invoke the stage far enough to clear the config guards.

    The filter dir does not exist, so a token that passes validation fails later
    on I/O — which is how these tests tell "refused by the guard" from "accepted".
    """
    proc = subprocess.run(
        [
            sys.executable, str(STAGE),
            "--filter-dir", "/nonexistent", "--fps", "1",
            "--output-dir", "/nonexistent-out",
            "--action-format", "ordered_events_v3",
            "--terminal-token", token,
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


class TerminalTokenTest(unittest.TestCase):
    def assert_refused(self, token: str) -> None:
        _, out = run_with_token(token)
        self.assertIn("not a control line", out, f"{token!r} should be refused")

    def assert_accepted(self, token: str) -> None:
        _, out = run_with_token(token)
        self.assertNotIn("not a control line", out, f"{token!r} should be accepted")
        # Reached real I/O, i.e. got past every config guard.
        self.assertIn("not a filter artifact", out)

    def test_the_two_valid_spellings_are_accepted(self):
        self.assert_accepted("TERMINATE: success")
        self.assert_accepted("TERMINATE: failure")

    def test_a_bare_control_token_is_refused(self):
        """The pre-rearchitecture corpora wrote this. It parses as an action."""
        self.assert_refused("TERMINATE")

    def test_an_invented_status_is_refused(self):
        self.assert_refused("TERMINATE: done")
        self.assert_refused("TERMINATE: ok")

    def test_a_token_that_merely_contains_the_word_is_left_alone(self):
        """`<terminate>` is the vendor tool-call spelling and a legitimate token;
        the guard fires on tokens *trying* to be a control line, not on any
        mention of the word."""
        self.assert_accepted("<terminate>")

    def test_an_unrelated_token_is_left_alone(self):
        self.assert_accepted("<|im_end|>")

    def test_no_token_is_fine(self):
        proc = subprocess.run(
            [
                sys.executable, str(STAGE),
                "--filter-dir", "/nonexistent", "--fps", "1",
                "--output-dir", "/nonexistent-out",
                "--action-format", "ordered_events_v3",
            ],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        self.assertIn("not a filter artifact", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
