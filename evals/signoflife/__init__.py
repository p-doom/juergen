"""Crowd-Cast sign-of-life v2 — one fixed development gate.

Deterministic cells whose success is decided from realized guest state, in two
tiers. The scored tier is six cells — the original four:

  * run `ls` in an already-focused terminal, and observe both exact shell history
    and a unique directory-listing marker in the terminal output;
  * enter one exact supplied paragraph into a focused terminal capture;
  * click Chrome in the desktop dock and verify a Chrome process is the active
    foreground window;
  * focus a visible-but-unfocused terminal, enter an exact command, and verify the
    resulting file bytes and the shell history.

plus two promoted from the candidate tier once measured: click a small target no
single move reaches among confusable neighbours, and type into a field and click a
named button *without* submitting. Both are decided from a Tk fixture's own
runtime widget measurements (`evals/fixtures/tk.py`).

The candidate tier holds two more — submission as a key transition, and stopping
when the first stage looks done. Both are fully calibrated; they are held back
because the off-the-shelf reference passes them, so they can only dilute the
scored mean. See `suite.CANDIDATE_KINDS`.

Not a benchmark and not a train/dev/test split. The gate is calibrated: within a
tier, the scripted oracle arm must pass every cell and the negative arm must fail
every cell, per grammar, through the same parse/compile/executor path the model
arms use.

`verifiers`' plugin loader requires `__all__` to name exactly one `Taskset`
subclass and at most one `Harness` subclass; exporting `DesktopHarness` here makes
it this taskset's default harness.
"""

from evals.harness import DesktopHarness
from evals.signoflife.cells import (
    ARMS,
    CONTROL_ARMS,
    MODEL_ARMS,
    verify_phaseb_provenance,
)
from evals.signoflife.guest import SignOfLifePreparer, register_preparers
from evals.signoflife.oracle import evaluate_postcondition
from evals.signoflife.suite import (
    CANDIDATE_KINDS,
    NO_SUBMIT_CELLS,
    SCORED_KINDS,
    TIERS,
    DevelopmentSuite,
    DevelopmentTask,
    load_suite,
)
from evals.signoflife.taskset import (
    SignOfLifeTask,
    SignOfLifeTaskset,
    SignOfLifeTasksetConfig,
)

__all__ = [
    "ARMS",
    "CANDIDATE_KINDS",
    "CONTROL_ARMS",
    "MODEL_ARMS",
    "NO_SUBMIT_CELLS",
    "SCORED_KINDS",
    "TIERS",
    "DesktopHarness",
    "DevelopmentSuite",
    "DevelopmentTask",
    "SignOfLifePreparer",
    "SignOfLifeTask",
    "SignOfLifeTaskset",
    "SignOfLifeTasksetConfig",
    "evaluate_postcondition",
    "load_suite",
    "register_preparers",
    "verify_phaseb_provenance",
]
