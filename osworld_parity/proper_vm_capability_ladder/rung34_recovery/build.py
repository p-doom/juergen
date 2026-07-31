from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from .actions import ARMS
from .env import DeterministicRecoveryBackend, RecoveryTrainingEnv
from .rollouts import (
    SCHEMA_PATH,
    public_on_policy_record,
    scripted_recovery_records,
    write_jsonl,
)
from .spec import load_recovery_tasks, load_sealed_commitment


def assert_cpu_only() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {None, "", "-1"}:
        raise RuntimeError("ROADMAP 3.4 construction jobs are CPU-only")


def build_contract_artifacts(output: Path) -> dict[str, object]:
    assert_cpu_only()
    output.mkdir(parents=True, exist_ok=True)
    demo_count = 0
    for split in ("train", "development"):
        for arm in ARMS:
            rows = scripted_recovery_records(split, arm)
            write_jsonl(output / f"scripted_{split}_{arm}.jsonl", rows)
            demo_count += len(rows)

    contract_rows = []
    for arm in ARMS:
        backend = DeterministicRecoveryBackend()
        env = RecoveryTrainingEnv(backend, split="development", arm=arm)
        env.reset(task_index=0)
        failure = {"test": "executor_failure"} if arm == ARMS[0] else "TEST_EXECUTOR_FAILURE"
        ineffective = {"action": "wait", "time": 0.0} if arm == ARMS[0] else "0 0 0"
        gold = {"test": "gold"} if arm == ARMS[0] else "TEST_GOLD"
        env.step(failure)
        env.step(ineffective)
        env.step(gold)
        task = load_recovery_tasks("development")[0]
        contract_rows.append(
            public_on_policy_record(
                task,
                arm=arm,
                events=(asdict(event) for event in env.public_events()),
            )
        )
    write_jsonl(output / "on_policy_schema_contract.jsonl", contract_rows)
    shutil.copyfile(SCHEMA_PATH, output / SCHEMA_PATH.name)
    sealed = load_sealed_commitment()
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "mode": "construction_contract_only",
        "base_commit": "48a54e8585eb9d6abff31e2ba6ea857c946a7d3d",
        "scripted_demonstration_count": demo_count,
        "on_policy_schema_contract_count": len(contract_rows),
        "train_task_count": len(load_recovery_tasks("train")),
        "development_task_count": len(load_recovery_tasks("development")),
        "sealed_evaluation_task_count_committed": sealed["task_count"],
        "sealed_evaluation_opened": 0,
        "models_run": 0,
        "gpu_count": 0,
        "trainer_only_values_exported": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_contract_artifacts(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
