from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU, KvmFixtureSession
from ..trajectory import build_trajectory
from .conversion import assert_round_trip
from .demonstrations import write_jsonl
from .env import Rung1bTrainingEnv, VmEnvironmentBackend
from .splits import materialize_tasks


@dataclass(frozen=True)
class TeacherCandidate:
    native_actions: tuple[dict[str, Any], ...]
    initial_cursor: tuple[int, int]
    screenshot_sha256: tuple[str, ...]
    accepted_reward: float
    terminated: bool
    executor_failures: int = 0


def rejection_sample(
    produce: Callable[[int], TeacherCandidate], *, max_attempts: int
) -> tuple[TeacherCandidate, int]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        candidate = produce(attempt)
        if (
            candidate.accepted_reward == 1.0
            and candidate.terminated
            and candidate.executor_failures == 0
        ):
            return candidate, attempt + 1
    raise RuntimeError(f"native teacher rejection sampling exhausted {max_attempts} attempts")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def collect_vm_teacher(
    *,
    output: Path,
    split: str,
    qcow: Path,
    qemu: Path,
    provider: Path,
    max_attempts: int,
) -> list[dict[str, Any]]:
    fixtures = materialize_tasks(split)
    rows: list[dict[str, Any]] = []
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=output / "vm_logs",
    ) as session:
        backend = VmEnvironmentBackend(session)
        env = Rung1bTrainingEnv(
            backend, split=split, arm="native_absolute_control"
        )
        for index, fixture in enumerate(fixtures):
            def produce(attempt: int) -> TeacherCandidate:
                observation, _ = env.reset(task_index=index)
                initial_cursor = backend.transport.cursor_position()  # trainer-only executor state
                trajectory = build_trajectory(
                    fixture,
                    arm="native_absolute_control",
                    cursor=initial_cursor,
                    geometry=backend.geometry,
                )
                actions = tuple(
                    action for action in trajectory.actions if isinstance(action, dict)
                )
                screenshot_hashes = [_sha(observation.screenshot_png)]
                final_reward = 0.0
                terminated = False
                failures = 0
                for action in actions:
                    try:
                        observation, final_reward, terminated, truncated, _ = env.step(action)
                    except Exception:
                        failures += 1
                        break
                    screenshot_hashes.append(_sha(observation.screenshot_png))
                    if terminated or truncated:
                        break
                return TeacherCandidate(
                    actions,
                    initial_cursor,
                    tuple(screenshot_hashes),
                    final_reward,
                    terminated,
                    failures,
                )

            candidate, attempts = rejection_sample(produce, max_attempts=max_attempts)
            compact = assert_round_trip(
                candidate.native_actions, initial_cursor=candidate.initial_cursor
            )
            rows.append(
                {
                    "schema_version": 1,
                    "source": "native_absolute_scripted_teacher_vm_rejection_sampled",
                    "task_id": fixture.id,
                    "fixture_sha256": fixture.fixture_sha256,
                    "split": split,
                    "instruction": fixture.instruction,
                    "initial_cursor": list(candidate.initial_cursor),
                    "native_absolute_actions": list(candidate.native_actions),
                    "compact_raw_actions": list(compact),
                    "screenshot_sha256": list(candidate.screenshot_sha256),
                    "rejection_attempts": attempts,
                    "accepted": True,
                    "hidden_reward_in_record": False,
                    "oracle_state_in_record": False,
                }
            )
    return rows


def collect_contract_teacher(split: str, *, max_attempts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in materialize_tasks(split):
        initial_cursor = (73, 91)
        trajectory = build_trajectory(
            fixture, arm="native_absolute_control", cursor=initial_cursor
        )
        actions = tuple(action for action in trajectory.actions if isinstance(action, dict))

        def produce(attempt: int) -> TeacherCandidate:
            # The first candidate deterministically exercises rejection; the
            # second is the scripted-gold contract. No VM/model result is invented.
            return TeacherCandidate(
                actions,
                initial_cursor,
                (hashlib.sha256(f"contract-{fixture.id}-{attempt}".encode()).hexdigest(),),
                0.0 if attempt == 0 else 1.0,
                attempt > 0,
            )

        candidate, attempts = rejection_sample(produce, max_attempts=max_attempts)
        compact = assert_round_trip(actions, initial_cursor=initial_cursor)
        rows.append(
            {
                "schema_version": 1,
                "source": "native_absolute_teacher_collection_contract_only",
                "task_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "split": split,
                "instruction": fixture.instruction,
                "initial_cursor": list(candidate.initial_cursor),
                "native_absolute_actions": list(actions),
                "compact_raw_actions": list(compact),
                "screenshot_sha256": list(candidate.screenshot_sha256),
                "rejection_attempts": attempts,
                "accepted": True,
                "hidden_reward_in_record": False,
                "oracle_state_in_record": False,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "vm"), required=True)
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--max-attempts", "--max_attempts", type=int, default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = (
        collect_contract_teacher(args.split, max_attempts=args.max_attempts)
        if args.mode == "contract"
        else collect_vm_teacher(
            output=args.output,
            split=args.split,
            qcow=args.qcow,
            qemu=args.qemu,
            provider=args.provider,
            max_attempts=args.max_attempts,
        )
    )
    write_jsonl(args.output / "teacher_rollouts.jsonl", rows)
    result = {
        "schema_version": 1,
        "status": "passed",
        "mode": args.mode,
        "split": args.split,
        "record_count": len(rows),
        "accepted_count": sum(bool(row["accepted"]) for row in rows),
        "evaluation_opened": 0,
        "hidden_reward_exported": False,
        "oracle_state_exported": False,
        "gpu_count": 0,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
