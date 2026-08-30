"""Revision-locked reading of a measured CUA-Gym task blocklist.

The document is a measurement of the dataset, not a fact about this code, so it
lives with the dataset and a run names the file it was measured with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .errors import SnapshotValidationError
from .manifest import PINNED_REVISION
from .models import TaskId

__all__ = ["TaskBlocklist", "load_blocklist"]

BLOCKLIST_VERSION = 1


@dataclass(frozen=True)
class TaskBlocklist:
    """Tasks a reset probe measured as useless for GRPO, for one dataset revision."""

    revision: str
    blocked: MappingProxyType[TaskId, tuple[str, ...]]
    probed: frozenset[TaskId]

    def reasons(self, task_id: TaskId) -> tuple[str, ...]:
        return self.blocked.get(task_id, ())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TaskBlocklist:
        revision = payload.get("revision")
        if revision != PINNED_REVISION:
            raise SnapshotValidationError(
                f"Blocklist is for revision {revision!r}, "
                f"but the pinned dataset is {PINNED_REVISION!r}"
            )
        version = payload.get("blocklist_version")
        if version != BLOCKLIST_VERSION:
            raise SnapshotValidationError(
                f"Unsupported blocklist_version {version!r}; "
                f"this build reads {BLOCKLIST_VERSION}"
            )
        blocked_tasks = payload.get("blocked_tasks")
        if not isinstance(blocked_tasks, dict):
            raise SnapshotValidationError("blocklist blocked_tasks must be an object")
        measured = payload.get("measured")
        if not isinstance(measured, dict):
            raise SnapshotValidationError("blocklist measured must be an object")
        probed = measured.get("reset_probe_task_ids")
        if not isinstance(probed, list):
            raise SnapshotValidationError(
                "blocklist measured.reset_probe_task_ids must be a list"
            )
        blocked: dict[TaskId, tuple[str, ...]] = {}
        for task_id, entry in blocked_tasks.items():
            if not isinstance(entry, dict):
                raise SnapshotValidationError(
                    f"blocklist entry for {task_id} must be an object"
                )
            reasons = entry.get("reasons")
            if not isinstance(reasons, list) or not reasons:
                raise SnapshotValidationError(
                    f"blocklist entry for {task_id} must list at least one reason"
                )
            blocked[TaskId(str(task_id))] = tuple(str(reason) for reason in reasons)
        return cls(
            revision=str(revision),
            blocked=MappingProxyType(blocked),
            probed=frozenset(TaskId(str(task_id)) for task_id in probed),
        )


def load_blocklist(path: Path) -> TaskBlocklist:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotValidationError(
            f"Could not read CUA-Gym task blocklist {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SnapshotValidationError("Task blocklist root must be an object")
    return TaskBlocklist.from_dict(payload)
