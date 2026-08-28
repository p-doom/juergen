from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.cua_gym.errors import SnapshotValidationError
from evals.cua_gym.manifest import PINNED_REVISION
from evals.cua_gym.models import TaskId
from evals.cua_gym.task_blocklist import (
    BLOCKLIST_VERSION,
    TaskBlocklist,
    load_blocklist,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "revision": PINNED_REVISION,
        "blocklist_version": BLOCKLIST_VERSION,
        "blocked_tasks": {"task-a": {"reasons": ["grader_pays_before_agent_acts"]}},
        "measured": {"reset_probe_task_ids": ["task-a", "task-b"]},
    }
    payload.update(overrides)
    return payload


def test_a_blocklist_is_read_from_the_path_a_run_names(tmp_path: Path) -> None:
    path = tmp_path / "blocklist.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    blocklist = load_blocklist(path)

    assert blocklist.revision == PINNED_REVISION
    assert blocklist.reasons(TaskId("task-a")) == ("grader_pays_before_agent_acts",)


def test_an_unreadable_blocklist_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(SnapshotValidationError, match="Could not read"):
        load_blocklist(tmp_path / "absent.json")


def test_a_blocklist_for_another_revision_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="pinned dataset"):
        TaskBlocklist.from_dict(_payload(revision="0" * 40))


def test_an_unsupported_blocklist_version_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="blocklist_version"):
        TaskBlocklist.from_dict(_payload(blocklist_version=BLOCKLIST_VERSION + 1))


def test_an_entry_without_a_reason_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="at least one reason"):
        TaskBlocklist.from_dict(_payload(blocked_tasks={"task-a": {"reasons": []}}))


def test_reasons_are_empty_for_a_task_that_is_not_blocked() -> None:
    blocklist = TaskBlocklist.from_dict(_payload())

    assert blocklist.reasons(TaskId("task-a")) == ("grader_pays_before_agent_acts",)
    assert blocklist.reasons(TaskId("task-b")) == ()
