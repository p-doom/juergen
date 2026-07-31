from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU, KvmFixtureSession
from .actions import ARMS, build_recovery_policy_actions
from .build import assert_cpu_only
from .env import RecoveryTrainingEnv
from .gates import require_earlier_gate_evidence
from .rollouts import public_on_policy_record, write_jsonl
from .spec import load_recovery_tasks
from .vm_backend import VmRecoveryBackend


def run_vm_replay(
    *,
    output: Path,
    gate_evidence: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
    limit: int | None = None,
) -> dict[str, object]:
    assert_cpu_only()
    require_earlier_gate_evidence(gate_evidence)
    output.mkdir(parents=True, exist_ok=True)
    tasks = load_recovery_tasks("development")
    if limit is not None:
        if limit <= 0:
            raise ValueError("VM replay limit must be positive")
        tasks = tasks[:limit]
    rows = []
    reset_proofs = 0
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=output / "vm_logs",
    ) as session:
        backend = VmRecoveryBackend(session)
        for arm in ARMS:
            env = RecoveryTrainingEnv(
                backend, split="development", arm=arm
            )
            for index, task in enumerate(tasks):
                env.reset(task_index=index)
                first_reset_hash = env.trainer_hidden_state_sha256()
                env.reset(task_index=index)
                if env.trainer_hidden_state_sha256() != first_reset_hash:
                    raise RuntimeError(f"post-perturbation reset drift: {task.id}/{arm}")
                reset_proofs += 1
                policy_actions = build_recovery_policy_actions(
                    task,
                    arm=arm,
                    cursor_after_injection=backend.transport.cursor_position(),
                    geometry=backend.geometry,
                )
                final_reward = 0.0
                terminated = False
                truncated = False
                for action in policy_actions:
                    _, final_reward, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                if final_reward != 1.0 or not terminated or truncated:
                    raise RuntimeError(f"scripted VM recovery failed: {task.id}/{arm}")
                rows.append(
                    public_on_policy_record(
                        task,
                        arm=arm,
                        events=(asdict(event) for event in env.public_events()),
                    )
                )
    write_jsonl(output / "vm_recovery_replays.jsonl", rows)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "mode": "cpu_kvm_scripted_replay",
        "episode_count": len(rows),
        "reset_proof_count": reset_proofs,
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
    parser.add_argument("--gate-evidence", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    run_vm_replay(
        output=args.output,
        gate_evidence=args.gate_evidence,
        qcow=args.qcow,
        qemu=args.qemu,
        provider=args.provider,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
