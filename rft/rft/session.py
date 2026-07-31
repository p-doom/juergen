"""Pooled-VM session hygiene: guest state must not leak between tasks.

**Defect #18, and it is LIVE.** ``DesktopEnv.reset()`` reverts to the clean snapshot
only when ``self.is_environment_used`` is True (OSWorld ``desktop_env/desktop_env.py``
:298-307 at commit ``b7db4d8``), and that flag is set by setup only when the task
config was non-empty::

    # desktop_env.py:316-320
    success = self.setup_controller.setup(self.config, ...)
    if success:
        if self.config:          # &lt;-- only marked used if there WERE setup ops
            self.is_environment_used = True

So a task whose ``config`` is ``[]``/absent leaves the flag False, and the **next**
task's ``reset()`` skips the revert entirely and inherits the previous guest state:
open windows, modified files, a moved cursor. Worse, for the ``apptainer`` provider
this stack uses, ``is_environment_used`` initialises to ``False``
(``desktop_env.py:155-156``), so even the **first** ``reset()`` of a fresh pooled
session skips the revert.

Any pooled consumer inherits this. ``rl/osworld/harness_task.py:130`` sets
``max_rollouts_per_session=50`` and ``pool.py:283-324`` reuses a ready session with
no state scrubbing, so one guest VM serves up to 50 consecutive different tasks with
a single ``env.reset()`` between them and **no mitigation**. The fast-reset drivers
do mitigate it (``osworld_fastreset/qemu_fast_reset.py:540-547`` and
``collect_fast.py:523-530`` force the base snapshot and clear the flag); the RL
harness does not.

This matters more for RFT than for a one-shot eval: a rollout that only succeeds
because the previous rollout left the right window open becomes an *accepted
training record*, and the model learns a precondition that will not exist at
inference.

:class:`SessionGuard` makes the revert decision depend on the task **about to run**
rather than the one that just finished, and refuses to hand out a session that has
not been reverted since the last task.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rft.errors import SchemaError


class SessionStateError(SchemaError):
    """A pooled session would have been reused without a revert."""


@dataclass
class SessionGuard:
    """Track one pooled VM session and enforce a revert before every task.

    Usage::

        guard = SessionGuard(max_rollouts_per_session=50, revert=env.reset_to_snapshot)
        for task in tasks:
            guard.begin_task(task["id"], task.get("config"))
            ...run the rollout...
            guard.end_task()

    :meth:`begin_task` reverts unconditionally when the session has been used
    before. That is the whole fix: correctness cannot depend on whether the
    *previous* task happened to declare a config.
    """

    max_rollouts_per_session: int
    revert: Callable[[], None]
    #: Set True only for a deliberately stateful sequence (e.g. a multi-task
    #: scenario meant to share state). Must be named explicitly.
    allow_dirty_reuse: bool = False

    n_rollouts_this_session: int = 0
    n_reverts: int = 0
    dirty: bool = False
    current_task: str | None = None
    history: list[tuple[str, bool, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_rollouts_per_session < 1:
            raise SchemaError(
                f"max_rollouts_per_session must be >= 1, got {self.max_rollouts_per_session}"
            )

    def begin_task(self, task_id: str, task_config: Any = None) -> bool:
        """Prepare the session for ``task_id``. Returns whether a revert happened.

        Raises:
            SessionStateError: a previous task is still open, or the session is
                dirty and ``allow_dirty_reuse`` was not set while a revert is
                impossible.
        """
        if self.current_task is not None:
            raise SessionStateError(
                f"begin_task({task_id!r}) while {self.current_task!r} is still open; "
                "call end_task() first"
            )
        if self.n_rollouts_this_session >= self.max_rollouts_per_session:
            raise SessionStateError(
                f"session already ran {self.n_rollouts_this_session} rollouts "
                f"(max {self.max_rollouts_per_session}); recycle the VM before continuing"
            )
        reverted = False
        if self.dirty:
            if self.allow_dirty_reuse:
                pass
            else:
                # Unconditional: the decision does NOT consult the previous task's
                # config, which is exactly the defect-#18 hole.
                self.revert()
                self.n_reverts += 1
                self.dirty = False
                reverted = True
        self.current_task = task_id
        self.history.append((task_id, reverted, bool(_config_nonempty(task_config))))
        return reverted

    def end_task(self) -> None:
        if self.current_task is None:
            raise SessionStateError("end_task() with no task open")
        self.current_task = None
        self.n_rollouts_this_session += 1
        self.dirty = True

    def describe(self) -> str:
        empty_cfg = sum(1 for _, _, nonempty in self.history if not nonempty)
        return (
            f"session: {self.n_rollouts_this_session} rollout(s), {self.n_reverts} revert(s), "
            f"{empty_cfg} task(s) with an EMPTY config "
            f"(each of which a config-conditional revert would have skipped - defect #18)"
        )


def _config_nonempty(task_config: Any) -> bool:
    if task_config is None:
        return False
    if isinstance(task_config, (list, tuple, dict, str)):
        return bool(task_config)
    return True


def audit_revert_policy(
    task_configs: list[Any], *, reverts_on_previous_config: bool
) -> dict[str, Any]:
    """Quantify how many tasks would run on dirty state under a given policy.

    ``reverts_on_previous_config=True`` models ``DesktopEnv``'s actual behaviour
    (revert iff the *previous* task had a non-empty config).
    ``False`` models :class:`SessionGuard` (always revert).

    Returns a dict with ``n_tasks``, ``n_dirty`` and ``dirty_task_indices`` so a
    test can assert the fixed policy leaves zero dirty tasks and the historical
    policy does not.
    """
    n_dirty = 0
    dirty_idx: list[int] = []
    for i in range(1, len(task_configs)):
        if reverts_on_previous_config:
            reverted = _config_nonempty(task_configs[i - 1])
        else:
            reverted = True
        if not reverted:
            n_dirty += 1
            dirty_idx.append(i)
    return {
        "n_tasks": len(task_configs),
        "n_dirty": n_dirty,
        "dirty_task_indices": dirty_idx,
        "policy": "previous-config-conditional" if reverts_on_previous_config else "always",
    }
