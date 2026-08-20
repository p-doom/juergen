"""The OSWorld benchmark family, as a runnable verifiers plugin.

Joins `evals/tasks.py`'s OSWorld rows to `evals/oracles.py`'s
`OSWorldEvaluateOracle`. The base `OSWorldTaskset` yields a bare `DesktopTask`,
which carries no rewards, so on its own that family produces episodes and no
score; `DesktopFacade.setup()` / `.evaluate()` (`evals/vm.py`, `evals/osworld.py`)
supply the scoring path and this module is the config that uses them.

Flat module at the repo root for the same reason `signoflife.py` is: `verifiers`
0.2.1 cannot resolve a dotted plugin id (`loaders._import_plugin` calls
`find_spec("verifiers.v1.tasksets.evals.…")`, whose parent import raises out of
the guard meant to catch it). The id is `osworld_bench`.

`__all__` is a verifiers contract: `loaders._plugin_class` requires exactly one
`Taskset` subclass and at most one `Harness` subclass in it.

    python -m verifiers.v1.cli eval \\
        --taskset.id osworld_bench --harness.id osworld_bench \\
        --taskset.osworld_root "$OSWORLD_ROOT" \\
        --taskset.split_path "$OSWORLD_ROOT/evaluation_examples/test_all.json" \\
        --harness.codec native_absolute \\
        --harness.artifacts.output_dir "$ARTIFACT_ROOT" \\
        --harness.pool.pool_target evals.vm:kvm_desktop_pool \\
        --harness.pool.session_kwargs '{"image": "...qcow2", "root_dir": "..."}'

NO-LEAK. `OSWorldTasksetConfig.split_path` must point at the held-out split for
any published number, and ideally at a benchmark that was never trained on at all.
The `resume_dir` field exists because a 369-task array does get interrupted.
"""

from __future__ import annotations

from typing import Iterable

from evals.harness import (
    ArtifactConfig,
    DesktopHarness,
    DesktopHarnessConfig,
    HistoryConfig,
    ImageBudgetConfig,
    SettleConfig,
)
from evals.oracles import OSWorldEvaluateOracle
from evals.indicators import FailureModeIndicators, MouseIndicators, SamplingProvenance
from evals.tasks import DesktopTask, OSWorldTaskset as _OSWorldRows

__all__ = ["DesktopHarness", "OSWorldBenchTaskset"]

PLUGIN_ID = "osworld_bench"


class OSWorldBenchTask(
    OSWorldEvaluateOracle,
    FailureModeIndicators,
    MouseIndicators,
    SamplingProvenance,
    DesktopTask,
):
    """One OSWorld benchmark episode. The reward is `DesktopEnv.evaluate()`.

    No shaping and no postcondition twin. `OSWorldEvaluateOracle.task_success`
    raises on a missing reward rather than returning 0.0, so this class and
    `evaluate_on_finish` must be set together — see `OSWORLD_BENCH_ARM`. The
    indicator mixins are metrics only and cannot remove the reward.
    """


class OSWorldBenchTaskset(_OSWorldRows):
    """`evals.tasks.OSWorldTaskset`'s rows, as tasks that can actually be scored.

    A subclass rather than a change to the base: the base's rows are also what the
    grounding family and any un-scored replay run consume, and those must keep
    their reward-free task, because one throwing reward inside `Task.score`'s
    `asyncio.gather` drops the whole group's rewards.
    """

    def load(self) -> Iterable[OSWorldBenchTask]:
        for task in super().load():
            yield OSWorldBenchTask(task.data, self.config.task)


OSWORLD_BENCH_ARM = DesktopHarnessConfig(
    id=PLUGIN_ID,
    codec="native_absolute",
    # `OSWorldBenchTask`'s only reward reads `task_reward`, and only this flag
    # publishes it.
    evaluate_on_finish=True,
    # OSWorld's own postcondition is the score, and a benchmark task can start in
    # a state some other task would call solved; refusing to score that would drop
    # tasks the benchmark counts.
    require_unsolved_start=False,
    max_steps=15,
    history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
    images=ImageBudgetConfig(max_images=8),
    settle=SettleConfig(min_delay_s=0.75, per_kind={}),
    artifacts=ArtifactConfig(save_frames=True, save_prompts=True, write_gif=False),
)
"""The one arm that exercises `DesktopFacade.evaluate()` end to end.

`native_absolute` because the only calibrated external reference is off-the-shelf
Qwen3-VL-8B = 33.9% on OSWorld-Verified in the native convention; a benchmark arm
in a custom grammar has nothing to be read against. For a grammar A/B, change the
codec and nothing else.
"""
