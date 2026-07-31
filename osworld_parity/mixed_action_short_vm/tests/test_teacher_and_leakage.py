from __future__ import annotations

import json
from pathlib import Path

from osworld_parity.mixed_action_short_vm.dataset import build_teacher_pairs
from osworld_parity.mixed_action_short_vm.manifest import (
    SEALED_EVALUATION_MANIFEST,
    load_authorized_tasks,
    load_manifest,
)
from osworld_parity.mixed_action_short_vm.runtime import Episode
from osworld_parity.mixed_action_short_vm.teacher import (
    NativeTeacherCollector,
    collect_compact_derivative,
    native_gold_actions,
)
from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    parse_compact_raw,
)


def _collect(task):
    episode = Episode(task, "native_absolute_control")
    receipt = episode.reset()
    collector = NativeTeacherCollector(task, receipt)
    observation = receipt.observation
    for action in native_gold_actions(task):
        collector.record(observation, action)
        observation = episode.step(action).observation
    return collector.finish()


def test_native_teacher_conversion_is_deterministic_and_explicit() -> None:
    task = next(
        item
        for item in load_authorized_tasks("development")
        if item.sequence_id == "focus_type_drag"
    )
    native = _collect(task)
    compact_a = collect_compact_derivative(task, native)
    compact_b = collect_compact_derivative(task, native)
    assert compact_a == compact_b
    assert compact_a.source_native_trace_sha256 == native.trace_sha256
    assert len(compact_a.actions) <= task.horizon
    assert all(isinstance(action, str) for action in compact_a.actions)
    assert all(parse_compact_raw(action) for action in compact_a.actions)
    assert any('type("München μ-' in action for action in compact_a.actions)

    actions = list(compact_a.actions)
    press = next(
        index for index, action in enumerate(actions) if action.endswith("+LMB")
    )
    assert actions[press + 1].split(";", 1)[0].strip() != "0 0 0"
    assert actions[press + 2].endswith("-LMB")
    assert "+LMB -LMB" not in actions[press]


def test_teacher_artifacts_are_split_isolated_and_contain_no_sealed_metadata(
    tmp_path: Path,
) -> None:
    train_output = tmp_path / "train"
    development_output = tmp_path / "development"
    train_report = build_teacher_pairs(train_output, split="train")
    development_report = build_teacher_pairs(
        development_output, split="development"
    )
    assert train_report["sealed_evaluation_payload_accessed"] is False
    assert development_report["sealed_evaluation_payload_accessed"] is False
    assert train_report["model_executed"] is False
    assert train_report["gpu_used"] is False

    train_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(train_output.iterdir())
    )
    development_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(development_output.iterdir())
    )
    sealed = load_manifest(SEALED_EVALUATION_MANIFEST)
    for cell in sealed.cells:
        assert cell.task_id not in train_text
        assert cell.task_id not in development_text
        assert cell.slot_sha256 not in train_text
        assert cell.slot_sha256 not in development_text
    assert sealed.manifest_payload_sha256 not in train_text
    assert sealed.manifest_payload_sha256 not in development_text

    train_rows = [
        json.loads(line)
        for line in (train_output / "native_absolute.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    development_rows = [
        json.loads(line)
        for line in (development_output / "native_absolute.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["task_sha256"] for row in train_rows}.isdisjoint(
        row["task_sha256"] for row in development_rows
    )
