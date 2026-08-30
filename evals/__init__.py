"""Evaluation layer: one episode driver, tasks that own their own judgement.

`harness.py` is the single `Harness`:

    screenshot -> prompt from codec.describe() + history policy -> sample
    -> codec.parse -> codec.compile -> desktop executes -> oracle -> repeat

`tasks.py` holds the task data/state and the tasksets that enumerate them.
`oracles.py` holds trace-only rewards over terminal evidence recorded by the
harness. `indicators.py` holds the diagnostics, computed in-process from the step
records the harness writes so they ride the trace.
`fixtures/` holds the task-specific fixtures: the browser pages and their host
server, the CDP client, Chrome launch, and the Writer / Calc / Files guest scripts.
They attach to tasks through the same `Preparer` seam, so a task that needs Chrome
brings its own Chrome and the session never learns Chrome exists.

`signoflife/` is the calibrated four-cell gate: one taskset, six harness configs.

This package is a library, not a verifiers plugin id: its `__all__` names two
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
)
from evals.tasks import (
    PREPARERS,
    DesktopState,
    DesktopTask,
    DesktopTaskData,
    FreerollTaskset,
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
    "MouseIndicators",
    "OSWorldTaskset",
    "OracleOutcome",
    "Preparer",
    "StateOracle",
    "delta_histogram",
    "final_probe",
    "on_lattice",
    "register_preparer",
]
