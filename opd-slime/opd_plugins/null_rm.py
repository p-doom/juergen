"""Zero-reward RM for pure on-policy distillation runs.

The OPD KL penalty (``--opd-kl-coef``) is the entire learning signal — we
don't want any task reward biasing the advantage estimator. Returning a
constant 0 makes the GRPO baseline subtract to zero, so advantages equal
``-opd_kl_coef * (student_logp - teacher_logp)`` after
``apply_opd_kl_to_advantages``.

Use via slime's ``--custom-rm-path opd_plugins.null_rm.reward_func``. Note
the module path is ``opd_plugins`` (not ``slime_plugins``) to avoid
shadowing slime's own ``slime_plugins`` package when both are on
PYTHONPATH simultaneously.
"""

from __future__ import annotations


async def reward_func(args, sample, **kwargs) -> float:  # noqa: ARG001
    return 0.0
