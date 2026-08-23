"""The CUA-Gym mini-eval, as a runnable verifiers plugin.

Joins `evals/cuagym`'s pinned 28-task rows to its `CuaGymRewardOracle`: the
reward is each bundle's own `reward.py`, run in the guest at scoring time. Flat
module at the repo root for the same reason `osworld_bench.py` is: verifiers
0.2.1 cannot resolve a dotted plugin id. The id is `cuagym_bench`.

`__all__` is a verifiers contract: exactly one `Taskset` subclass and at most
one `Harness` subclass.

    python -m verifiers.v1.cli eval \\
        --taskset.id cuagym_bench --harness.id cuagym_bench \\
        --taskset.bundles_root "$CUAGYM_BUNDLES" \\
        --harness.codec <the grammar the checkpoint was trained on> \\
        --harness.artifacts.output_dir "$ARTIFACT_ROOT" \\
        --harness.pool.pool_target evals.vm:kvm_desktop_pool \\
        --harness.pool.session_kwargs '{"image": "...qcow2", "root_dir": "..."}'

Each of the 28 tasks was solved end to end by a stronger teacher model (score
1.0 in 15 steps or fewer), and each grader was checked to pay nothing on a
freshly reset VM before any agent acted — so a competent policy has headroom on
every row, and a score is never handed out for the starting state. Selection
detail and file pins: `evals/cuagym/suite.json`.

NO-LEAK. The teacher's solutions to these 28 tasks were used as fine-tuning
data on the RL side. A number from this suite measures training progress, not
generalization. Keep the tasks out of any training set whose checkpoints are
scored here, or say so next to the number.
"""

from __future__ import annotations

from typing import Iterable

from evals.cuagym.oracle import CuaGymRewardOracle
from evals.cuagym.taskset import CuaGymTaskset as _CuaGymRows
from evals.harness import (
    ArtifactConfig,
    DesktopHarness,
    DesktopHarnessConfig,
    HistoryConfig,
    ImageBudgetConfig,
    SettleConfig,
)
from evals.indicators import FailureModeIndicators, MouseIndicators, SamplingProvenance
from evals.tasks import DesktopTask

__all__ = ["CuaGymBenchTaskset", "DesktopHarness"]

PLUGIN_ID = "cuagym_bench"


class CuaGymBenchTask(
    CuaGymRewardOracle,
    FailureModeIndicators,
    MouseIndicators,
    SamplingProvenance,
    DesktopTask,
):
    """One mini-eval episode. The reward is the bundle's `reward.py`.

    `task_success` raises when the verifier cannot run (lease gone, no REWARD
    line), so an infrastructure failure drops the episode instead of scoring
    0.0. The indicator mixins are metrics only.
    """


class CuaGymBenchTaskset(_CuaGymRows):
    """`evals.cuagym.taskset`'s rows, as tasks that can actually be scored.

    A subclass for the same reason `OSWorldBenchTaskset` is one: the base rows
    stay reward-free for un-scored replay runs.
    """

    def load(self) -> Iterable[CuaGymBenchTask]:
        for task in super().load():
            yield CuaGymBenchTask(task.data, self.config.task)


CUAGYM_BENCH_ARM = DesktopHarnessConfig(
    id=PLUGIN_ID,
    codec="native_absolute",
    # the reward runs in the oracle, not through `DesktopFacade.evaluate()`
    evaluate_on_finish=False,
    require_unsolved_start=False,
    max_steps=25,
    history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
    images=ImageBudgetConfig(max_images=8),
    settle=SettleConfig(min_delay_s=0.75, per_kind={}),
    artifacts=ArtifactConfig(save_frames=True, save_prompts=True, write_gif=False),
)
"""`native_absolute` for the same calibration rationale as `OSWORLD_BENCH_ARM`;
for scoring a checkpoint, set the codec to the grammar it was trained on and
change nothing else."""
