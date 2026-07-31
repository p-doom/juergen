"""Command-line entrypoints used by labctl recipes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.teacher_sft.collector import collect_rollouts
from experiments.teacher_sft.contracts import ContractError
from experiments.teacher_sft.conversion import convert_accepted
from experiments.teacher_sft.policy_eval import evaluate_policy
from experiments.teacher_sft.rejection import reject_rollouts
from experiments.teacher_sft.replay import replay_converted
from experiments.teacher_sft.sft import build_sft
from experiments.teacher_sft.task_sources import build_task_manifest
from experiments.teacher_sft.teacher import load_teacher_spec


def _path(value: str) -> Path:
    return Path(value).resolve()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-tasks")
    build.add_argument("--source-spec", type=_path, required=True)
    build.add_argument("--heldout-denylist", type=_path, required=True)
    build.add_argument("--output", type=_path, required=True)

    validate = commands.add_parser("validate-teacher")
    validate.add_argument("--teacher-spec", type=_path, required=True)

    collect = commands.add_parser("collect")
    collect.add_argument("--tasks", type=_path, required=True)
    collect.add_argument("--teacher-spec", type=_path, required=True)
    collect.add_argument("--output", type=_path, required=True)
    collect.add_argument("--env-command", required=True)
    collect.add_argument("--api-key-env", default="TEACHER_API_KEY")
    collect.add_argument("--candidates-per-task", type=int, default=1)
    collect.add_argument("--max-steps", type=int, default=30)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--success-reward", type=float, default=1.0)
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--shard-count", type=int, default=1)

    reject = commands.add_parser("reject")
    reject.add_argument("--tasks", type=_path, required=True)
    reject.add_argument("--rollouts", type=_path, required=True)
    reject.add_argument("--output", type=_path, required=True)
    reject.add_argument("--min-reward", type=float, default=1.0)
    reject.add_argument("--max-per-task", type=int, default=1)
    reject.add_argument("--allow-missing-success-termination", action="store_true")

    convert = commands.add_parser("convert")
    convert.add_argument("--rejection", type=_path, required=True)
    convert.add_argument("--output", type=_path, required=True)

    package = commands.add_parser("build-sft")
    package.add_argument("--converted", type=_path, required=True)
    package.add_argument("--heldout-denylist", type=_path, required=True)
    package.add_argument("--output", type=_path, required=True)

    replay = commands.add_parser("replay")
    replay.add_argument("--converted", type=_path, required=True)
    replay.add_argument("--output", type=_path, required=True)
    replay.add_argument("--env-command", required=True)
    replay.add_argument("--min-reward", type=float, default=1.0)
    replay.add_argument(
        "--split", choices=("train", "train_validation"), default="train_validation"
    )

    evaluate = commands.add_parser("eval-policy")
    evaluate.add_argument("--tasks", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True)
    evaluate.add_argument("--base-url", required=True)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--model-revision", required=True)
    evaluate.add_argument("--env-command", required=True)
    evaluate.add_argument("--api-key-env", default="POLICY_API_KEY")
    evaluate.add_argument(
        "--split", choices=("train", "train_validation"), default="train_validation"
    )
    evaluate.add_argument("--max-steps", type=int, default=30)
    evaluate.add_argument("--min-reward", type=float, default=1.0)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build-tasks":
        result = build_task_manifest(
            args.source_spec, args.heldout_denylist, args.output
        )
    elif args.command == "validate-teacher":
        spec = load_teacher_spec(args.teacher_spec)
        result = {
            key: spec[key] for key in ("model_id", "model_revision", "spec_sha256")
        }
    elif args.command == "collect":
        result = collect_rollouts(
            args.tasks,
            args.teacher_spec,
            args.output,
            env_command=args.env_command,
            api_key=os.environ.get(args.api_key_env, "none"),
            candidates_per_task=args.candidates_per_task,
            max_steps=args.max_steps,
            workers=args.workers,
            success_reward=args.success_reward,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "reject":
        result = reject_rollouts(
            args.tasks,
            args.rollouts,
            args.output,
            min_reward=args.min_reward,
            max_per_task=args.max_per_task,
            require_success_termination=not args.allow_missing_success_termination,
        )
    elif args.command == "convert":
        result = convert_accepted(args.rejection, args.output)
    elif args.command == "build-sft":
        result = build_sft(args.converted, args.heldout_denylist, args.output)
    elif args.command == "replay":
        result = replay_converted(
            args.converted,
            args.output,
            env_command=args.env_command,
            min_reward=args.min_reward,
            only_split=args.split,
        )
    else:
        result = evaluate_policy(
            args.tasks,
            args.output,
            base_url=args.base_url,
            model_id=args.model_id,
            model_revision=args.model_revision,
            env_command=args.env_command,
            api_key=os.environ.get(args.api_key_env, "none"),
            split=args.split,
            max_steps=args.max_steps,
            min_reward=args.min_reward,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FATAL teacher-sft contract: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
