"""Evaluation layer: one episode driver, tasks that own their own judgement.

`harness.py` is the single `Harness`. It replaces six episode drivers that were
the same loop six times:

    screenshot -> prompt from codec.describe() + history policy -> sample
    -> codec.parse -> codec.compile -> pixeldesk executes -> oracle -> repeat

`tasks.py` holds the task data/state and the tasksets that enumerate them.
`oracles.py` holds the state oracles as rewards that declare `runtime` and so
keep reading real VM state. `indicators.py` holds the diagnostics that used to be
recomputed by ad-hoc scripts over `result.json` trees and now ride the trace.
`fixtures/` holds the task-specific fixtures pixeldesk deliberately left out
because they are tasks, not VM plumbing: the browser pages and their host server, the
CDP client, Chrome launch, and the Writer / Calc / Files guest scripts. They attach to
tasks through the same `Preparer` seam, so a task that needs Chrome brings its own
Chrome and the session never learns Chrome exists.

`signoflife/` is the calibrated four-cell gate: **one** taskset, **four** harness
configs.

This package is a library, **not** a verifiers plugin id: its `__all__` names three
`Taskset` subclasses, and `loaders._plugin_class` requires exactly one. The plugin
packages are `evals.signoflife`, `rl.movebox`, `rl.grounding` and `rl.target_box`.
"""

import evals.fixtures.preparers  # noqa: F401  registers the fixture-backed preparers
from evals.harness import DesktopHarness, DesktopHarnessConfig
from evals.indicators import (
    DIGIT_LATTICE,
    SUBMIT_KEYS,
    FailureModeIndicators,
    MouseIndicators,
    delta_histogram,
    on_lattice,
)
from evals.oracles import (
    OracleOutcome,
    StateOracle,
    final_probe,
    probe_now,
)
from evals.tasks import (
    PREPARERS,
    DesktopState,
    DesktopTask,
    DesktopTaskData,
    FreerollTaskset,
    GroundingTaskset,
    OSWorldTaskset,
    Preparer,
    register_preparer,
)

__all__ = [
    "DIGIT_LATTICE",
    "PREPARERS",
    "SUBMIT_KEYS",
    "DesktopHarness",
    "DesktopHarnessConfig",
    "DesktopState",
    "DesktopTask",
    "DesktopTaskData",
    "FailureModeIndicators",
    "FreerollTaskset",
    "GroundingTaskset",
    "MouseIndicators",
    "OSWorldTaskset",
    "OracleOutcome",
    "Preparer",
    "StateOracle",
    "delta_histogram",
    "final_probe",
    "on_lattice",
    "probe_now",
    "register_preparer",
]
