"""The reward is the bundle's own `reward.py`, run in the guest at scoring time.

Every reward script in the suite reports `REWARD: <x>` on stdout — the suite
generator verified that — so the parser accepts nothing else: a bare number, a
malformed prefix, or two REWARD lines that disagree all raise, because a wrong
parse here silently rescores a task. A raise is an infrastructure verdict, not
a 0.0: one throwing reward drops the group (`Task.score`'s `asyncio.gather`),
which is the correct fate for an episode whose verifier could not run.

The oracle needs the live lease — reward.py reads realized guest state, and 5
of the 28 graders carry a `postconfig` (ctrl-s + settle) that must run first.
There is no recorded fallback on purpose: a reward computed from anything but
the guest would be a different measurement wearing the same name. Size the
pool's `scoring_grace_s` so the lease survives scoring (the reward run itself
is a few guest commands).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from agent.desktop import lease_for_trace
from evals.cuagym.guest import run_reward
from evals.tasks import RESULT_KEY, DesktopTaskData, valid_result

__all__ = ["CuaGymRewardOracle", "parse_reward_stdout"]

_LOGGER = logging.getLogger(__name__)

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_REWARD_RE = re.compile(rf"^\s*REWARD:\s*({_NUMBER})\s*$", re.IGNORECASE)


def parse_reward_stdout(stdout: str) -> float:
    """One finite reward in [0, 1] from `REWARD:`-prefixed stdout, strictly."""

    if not isinstance(stdout, str):
        raise ValueError("reward stdout must be text")
    values: list[float] = []
    malformed = False
    for line in stdout.splitlines():
        match = _REWARD_RE.fullmatch(line)
        if match is not None:
            values.append(float(match.group(1)))
        elif "REWARD:" in line.upper():
            malformed = True
    if malformed:
        raise ValueError(f"malformed REWARD line in reward stdout:\n{stdout[-500:]}")
    if not values:
        raise ValueError(f"reward stdout has no REWARD line:\n{stdout[-500:]}")
    if len(set(values)) > 1:
        raise ValueError(f"reward stdout has conflicting REWARD lines: {sorted(set(values))}")
    reward = values[0]
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"reward must be finite and in [0, 1], got {reward!r}")
    return reward


class CuaGymRewardOracle:
    """Mixin: `task_success` is the bundle verifier's number, live or nothing."""

    full_success_threshold: float = 0.999

    @vf.reward
    async def task_success(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        del runtime  # declared so offline replay skips this reward, not scores it
        valid_result(trace, "cuagym")
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        lease = lease_for_trace(trace.id)
        if lease is None:
            raise RuntimeError(
                "cuagym reward needs the live guest but the lease's grace window "
                "closed before scoring — raise the pool's scoring_grace_s"
            )
        bundle_dir = Path(str(task.setup["bundle_dir"]))
        evaluator = dict((task.setup.get("task_config") or {}).get("evaluator") or {})
        stdout = run_reward(
            lease.session,
            (bundle_dir / "reward.py").read_text(encoding="utf-8"),
            list(evaluator.get("postconfig") or []),
            bundle_dir,
        )
        reward = parse_reward_stdout(stdout)
        trace.info.setdefault("oracle", {})["cuagym"] = {
            "task_id": task.name,
            "reward": reward,
            "stdout_tail": stdout[-500:],
        }
        return reward

    @vf.metric
    async def full_success(self, trace: vf.Trace) -> dict[str, float]:
        verdict = (trace.info.get("oracle") or {}).get("cuagym") or {}
        raw = verdict.get("reward")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return {"full_success": 0.0, "cuagym_reward_missing": 1.0}
        result = trace.info.get(RESULT_KEY) or {}
        return {
            "full_success": 1.0 if float(raw) >= self.full_success_threshold else 0.0,
            "cuagym_reward": float(raw),
            "parse_errors": float(result.get("parse_errors", 0)),
        }
