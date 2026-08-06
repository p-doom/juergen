"""The one sign-of-life taskset.

Four fixed cells, one row each. The six arms are six `DesktopHarnessConfig`s over
this one taskset (see `cells.py`), so an arm can never quietly redefine the suite.

Reset isolation comes from the pool: verifiers shards a taskset across `spawn`-ed
workers and each worker checks out its own desktop, so the isolation property
survives without a fan-out flag.
"""

from __future__ import annotations

from typing import Any, Iterable

import verifiers.v1 as vf

from evals.indicators import FailureModeIndicators, MouseIndicators, SamplingProvenance
from evals.oracles import OracleOutcome, StateOracle
import evals.signoflife.guest  # noqa: F401  import registers the four preparers
from evals.signoflife.oracle import evaluate_postcondition
from evals.signoflife.suite import NO_SUBMIT_CELLS, load_suite
from evals.tasks import DesktopTask, DesktopTaskData

__all__ = ["SignOfLifeTask", "SignOfLifeTaskset", "SignOfLifeTasksetConfig"]


class SignOfLifeTask(
    StateOracle,
    FailureModeIndicators,
    MouseIndicators,
    SamplingProvenance,
    DesktopTask,
):
    """A gate cell. Success is realized guest state; everything else is a metric.

    The mixin order is the reporting order and nothing more — `discover_decorated`
    walks the MRO and sorts by (priority, name), so no mixin can shadow another's
    signal by accident.

    `evals.oracles.PairedArmDivergence` is deliberately NOT mixed in: a
    `@vf.group_reward` makes the episode require `n >= 2` (`env.py:308-312`), which
    would break every single-sample gate run. Mix it in for an explicit two-arm
    comparison run and nowhere else.
    """

    def evaluate_state(
        self, task: DesktopTaskData, state: dict[str, Any]
    ) -> OracleOutcome:
        return evaluate_postcondition(
            task.name or "", task.kind, dict(task.expected), state
        )


class SignOfLifeTasksetConfig(vf.TasksetConfig):
    task_ids: list[str] = []
    """Restrict to named cells. Present for reproducing a single-cell rerun; the
    gate is only a gate when all four run."""


class SignOfLifeTaskset(vf.Taskset[SignOfLifeTask, SignOfLifeTasksetConfig]):
    def load(self) -> Iterable[SignOfLifeTask]:
        suite = load_suite()
        keep = set(self.config.task_ids)
        for idx, cell in enumerate(suite.tasks):
            if keep and cell.id not in keep:
                continue
            yield SignOfLifeTask(
                DesktopTaskData(
                    idx=idx,
                    name=cell.id,
                    prompt=cell.instruction,
                    instruction=cell.instruction,
                    kind=cell.kind,
                    max_steps=cell.max_steps,
                    expected=dict(cell.expected),
                    no_submit=cell.id in NO_SUBMIT_CELLS,
                    setup={
                        "suite_id": suite.suite_id,
                        "suite_manifest_sha256": suite.manifest_sha256,
                        "suite_role": suite.role,
                        "final_benchmark": suite.final_benchmark,
                    },
                ),
                self.config.task,
            )
