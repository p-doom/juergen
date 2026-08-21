"""The CUA-Gym mini-eval family: 28 revision-pinned desktop tasks.

A task here is one CUA-Gym bundle — `task.json` (OSWorld-shaped `config` steps
plus an `evaluator` naming `./reward.py`), the setup script it downloads, and
the reward script that is the task's own verifier. The family exists because
the `osworld` family cannot run these: CUA-Gym's `download` URLs are
`./`-relative bundle members, not HTTP, and its evaluator is an in-guest Python
script, not OSWorld's getter/metric machinery.

Selection provenance lives in `suite.json`; the taskset refuses a bundle whose
bytes differ from the suite's pins, so a number from this family is always a
number on the same 28 tasks.
"""

from evals.cuagym.bundles import load_suite, verify_bundle
from evals.cuagym.oracle import CuaGymRewardOracle, parse_reward_stdout
from evals.cuagym.taskset import CuaGymTaskset, CuaGymTasksetConfig

__all__ = [
    "CuaGymRewardOracle",
    "CuaGymTaskset",
    "CuaGymTasksetConfig",
    "load_suite",
    "parse_reward_stdout",
    "verify_bundle",
]
