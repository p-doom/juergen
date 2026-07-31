"""Task-id and content-digest contamination guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from stage5_rft.schema import EpisodeTrace, TaskSpec
from stage5_rft.util import ContractError, read_json


@dataclass(frozen=True)
class ContaminationBlocklist:
    task_ids: frozenset[str]
    task_content_sha256: frozenset[str]
    label: str
    testing_only: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "ContaminationBlocklist":
        value = read_json(path)
        return cls(
            task_ids=frozenset(str(x) for x in value.get("task_ids", [])),
            task_content_sha256=frozenset(
                str(x) for x in value.get("task_content_sha256", [])
            ),
            label=str(value.get("label", Path(path).name)),
            testing_only=bool(value.get("testing_only", False)),
        )


@dataclass(frozen=True)
class ContaminationReport:
    checked: int
    task_id_overlap: tuple[str, ...]
    content_digest_overlap: tuple[str, ...]
    unauthorized_splits: tuple[str, ...]
    blocklist_label: str
    blocklist_usable: bool

    @property
    def clean(self) -> bool:
        return self.blocklist_usable and not (
            self.task_id_overlap or self.content_digest_overlap or self.unauthorized_splits
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "task_id_overlap": list(self.task_id_overlap),
            "content_digest_overlap": list(self.content_digest_overlap),
            "unauthorized_splits": list(self.unauthorized_splits),
            "blocklist_label": self.blocklist_label,
            "blocklist_usable": self.blocklist_usable,
            "clean": self.clean,
        }


def audit_resets(
    resets: Iterable[Mapping[str, object]], blocklist: ContaminationBlocklist
) -> ContaminationReport:
    rows = list(resets)
    ids = sorted(
        {str(r["task_id"]) for r in rows if str(r["task_id"]) in blocklist.task_ids}
    )
    digests = sorted(
        {
            str(r["task_content_sha256"])
            for r in rows
            if str(r["task_content_sha256"]) in blocklist.task_content_sha256
        }
    )
    bad_splits = sorted(
        {
            str(r["source_split"])
            for r in rows
            if str(r["source_split"]) not in {"train", "train_adjacent", "validation"}
        }
    )
    return ContaminationReport(
        checked=len(rows),
        task_id_overlap=tuple(ids),
        content_digest_overlap=tuple(digests),
        unauthorized_splits=tuple(bad_splits),
        blocklist_label=blocklist.label,
        blocklist_usable=bool(
            blocklist.task_ids
            or blocklist.task_content_sha256
            or blocklist.testing_only
        ),
    )


def audit_tasks(
    tasks: Iterable[TaskSpec], blocklist: ContaminationBlocklist
) -> ContaminationReport:
    return audit_resets((task.reset.__dict__ for task in tasks), blocklist)


def audit_episodes(
    episodes: Iterable[EpisodeTrace], blocklist: ContaminationBlocklist
) -> ContaminationReport:
    return audit_resets((episode.reset.__dict__ for episode in episodes), blocklist)


def assert_clean(report: ContaminationReport) -> None:
    if not report.clean:
        raise ContractError(
            "CONTAMINATION GUARD FAILED: "
            f"task_ids={list(report.task_id_overlap)}, "
            f"content_digests={list(report.content_digest_overlap)}, "
            f"splits={list(report.unauthorized_splits)}, "
            f"blocklist_usable={report.blocklist_usable} against {report.blocklist_label}"
        )
