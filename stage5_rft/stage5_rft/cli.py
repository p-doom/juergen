"""Command-line boundary for labctl and local CPU validation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from stage5_rft.bayes_launch import evaluate_relative_mouse_launch
from stage5_rft.collector import EpisodeCollector, EpisodeStore
from stage5_rft.contamination import ContaminationBlocklist
from stage5_rft.gates import construction_metrics, evaluate_gate_files
from stage5_rft.learner import build_learner_plan, write_learner_plan
from stage5_rft.metrics import matched_native_absolute_parity, summarize_separately
from stage5_rft.replay import (
    replay_episodes,
    validate_collection,
    validate_deterministic_reset,
)
from stage5_rft.relative_mouse_records import build_pure_relative_mouse_records
from stage5_rft.relative_mouse_batch import seal_relative_mouse_batch
from stage5_rft.relative_mouse_compare import compare_relative_mouse_eval
from stage5_rft.rft import RFTConfig, build_rft_dataset
from stage5_rft.schema import PolicyProvenance, TaskSpec
from stage5_rft.util import (
    ContractError,
    atomic_write_json,
    read_json,
    read_jsonl,
)


SELF_WRITING_COMMANDS = {
    "collect",
    "build-rft",
    "learner-plan",
    "build-relative-mouse-rft",
    "seal-relative-mouse-batch",
}


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ContractError("adapter must use module:symbol syntax")
    module_name, symbol_name = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(module_name), symbol_name)
    except (ImportError, AttributeError) as exc:
        raise ContractError(f"cannot load adapter {spec!r}: {exc}") from exc


def _write_or_print(value: dict[str, Any], path: str | None) -> None:
    if path:
        atomic_write_json(path, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    policy = PolicyProvenance.from_dict(read_json(args.policy))
    tasks = [TaskSpec.from_dict(row) for row in read_jsonl(args.tasks)]
    factory = _load_symbol(args.adapter)
    built = factory(policy=policy, rollout_root=args.out)
    if not isinstance(built, tuple) or len(built) != 2:
        raise ContractError("collector adapter factory must return (environment, actor)")
    environment, actor = built
    if actor.provenance.fingerprint != policy.fingerprint:
        raise ContractError("adapter actor provenance differs from requested policy")
    collector = EpisodeCollector(
        store=EpisodeStore(args.out),
        environment=environment,
        actor=actor,
        actor_id=args.actor_id,
        contamination_blocklist=ContaminationBlocklist.from_json(args.blocklist),
    )
    try:
        return collector.collect_many(tasks)
    finally:
        collector.close()


def _cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    return validate_collection(args.rollouts).as_dict()


def _cmd_replay_live(args: argparse.Namespace) -> dict[str, Any]:
    episodes = EpisodeStore(args.rollouts).load_all()
    environment = _load_symbol(args.adapter)(rollout_root=args.rollouts)
    return replay_episodes(episodes, environment).as_dict()


def _cmd_reset_live(args: argparse.Namespace) -> dict[str, Any]:
    episodes = EpisodeStore(args.rollouts).load_all()
    factory = _load_symbol(args.adapter)
    divergences: list[dict[str, Any]] = []
    checked = 0
    for episode in episodes:
        report = validate_deterministic_reset(
            episode,
            factory(rollout_root=args.rollouts),
            repeats=args.repeats,
        )
        checked += report.steps_checked
        divergences.extend(d.as_dict() for d in report.divergences)
    failed = len({(d["episode_id"], d["step_index"]) for d in divergences})
    return {
        "schema_version": "stage5.reset_report.v1",
        "episodes_checked": len(episodes),
        "resets_checked": checked,
        "divergences": divergences,
        "passed": not divergences,
        "pass_rate": 1.0 if checked == 0 else max(0.0, (checked - failed) / checked),
    }


def _cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    config = RFTConfig(
        mode=args.mode,
        minimum_return=args.minimum_return,
        val_fraction=args.val_fraction,
        split_salt=args.split_salt,
        reward_temperature=args.reward_temperature,
        maximum_weight=args.maximum_weight,
        enable_reward_weighting_experiment=args.enable_reward_weighting_experiment,
    )
    return build_rft_dataset(
        rollout_root=args.rollouts,
        output_dir=args.out,
        blocklist=ContaminationBlocklist.from_json(args.blocklist),
        config=config,
    )


def _cmd_metrics(args: argparse.Namespace) -> dict[str, Any]:
    candidate = EpisodeStore(args.candidate).load_all()
    baseline = EpisodeStore(args.native_absolute).load_all()
    return {
        "schema_version": "stage5.metrics.v1",
        "candidate": summarize_separately(candidate),
        "native_absolute": summarize_separately(baseline),
        "matched": matched_native_absolute_parity(candidate, baseline),
    }


def _cmd_gate(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_gate_files(args.metrics, args.config, phase=args.phase)


def _cmd_construction_report(args: argparse.Namespace) -> dict[str, Any]:
    return construction_metrics(
        rollout_root=args.rollouts,
        blocklist=ContaminationBlocklist.from_json(args.blocklist),
        live_replay_report=read_json(args.live_replay),
        deterministic_reset_report=read_json(args.deterministic_reset),
    )


def _cmd_learner_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_learner_plan(
        dataset_dir=args.dataset,
        output_checkpoint_dir=args.output_checkpoint,
        learner_run_id=args.learner_run_id,
        trainer_adapter=args.trainer_adapter,
        seed=args.seed,
    )
    write_learner_plan(plan, args.out)
    return plan.as_dict()


def _cmd_relative_mouse_launch_gate(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_relative_mouse_launch(
        probe_path=args.probe,
        attestation_path=args.attestation,
        config_path=args.config,
    )


def _cmd_build_relative_mouse_rft(args: argparse.Namespace) -> dict[str, Any]:
    return build_pure_relative_mouse_records(
        rollout_glob=args.rollouts_glob,
        output_dir=args.out,
        approved_background_root=args.approved_background_root,
        val_fraction=args.val_fraction,
        split_salt=args.split_salt,
        maximum_trajectories_per_task=args.maximum_trajectories_per_task,
    )


def _cmd_seal_relative_mouse_batch(args: argparse.Namespace) -> dict[str, Any]:
    return seal_relative_mouse_batch(
        shard_dirs=args.shard,
        output_dir=args.out,
        expected_tasks=args.expected_tasks,
        expected_k=args.expected_k,
        minimum_accepted=args.minimum_accepted,
        expected_train_seed=args.expected_train_seed,
    )


def _cmd_compare_relative_mouse_eval(args: argparse.Namespace) -> dict[str, Any]:
    return compare_relative_mouse_eval(
        baseline_probe_path=args.baseline_probe,
        candidate_probe_path=args.candidate_probe,
        baseline_sacct_path=args.baseline_sacct,
        candidate_sacct_path=args.candidate_sacct,
        candidate_manifest_path=args.candidate_manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage5-rft")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="collect/resume complete task-level episodes")
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--policy", required=True)
    collect.add_argument("--blocklist", required=True)
    collect.add_argument("--adapter", required=True)
    collect.add_argument("--actor-id", required=True)
    collect.add_argument("--out", required=True)
    collect.set_defaults(func=_cmd_collect)

    validate = sub.add_parser("validate", help="offline replay and artifact audit")
    validate.add_argument("--rollouts", required=True)
    validate.add_argument("--out")
    validate.set_defaults(func=_cmd_validate)

    replay = sub.add_parser("replay-live", help="re-execute actions after deterministic reset")
    replay.add_argument("--rollouts", required=True)
    replay.add_argument("--adapter", required=True)
    replay.add_argument("--out")
    replay.set_defaults(func=_cmd_replay_live)

    reset = sub.add_parser("reset-live", help="repeat deterministic VM resets")
    reset.add_argument("--rollouts", required=True)
    reset.add_argument("--adapter", required=True)
    reset.add_argument("--repeats", type=int, default=2)
    reset.add_argument("--out")
    reset.set_defaults(func=_cmd_reset_live)

    build = sub.add_parser("build-rft", help="build task-level rejection/RFT records")
    build.add_argument("--rollouts", required=True)
    build.add_argument("--blocklist", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--mode", choices=("rejection", "reward_weighted"), default="rejection")
    build.add_argument("--minimum-return", type=float, default=0.0)
    build.add_argument("--val-fraction", type=float, default=0.1)
    build.add_argument("--split-salt", default="stage5-rft-v1")
    build.add_argument("--reward-temperature", type=float, default=1.0)
    build.add_argument("--maximum-weight", type=float, default=4.0)
    build.add_argument("--enable-reward-weighting-experiment", action="store_true")
    build.set_defaults(func=_cmd_build)

    metrics = sub.add_parser("metrics", help="compute separated and matched parity metrics")
    metrics.add_argument("--candidate", required=True)
    metrics.add_argument("--native-absolute", required=True)
    metrics.add_argument("--out")
    metrics.set_defaults(func=_cmd_metrics)

    gate = sub.add_parser("gate", help="evaluate a preregistered gate phase")
    gate.add_argument("--metrics", required=True)
    gate.add_argument("--config", required=True)
    gate.add_argument("--phase", required=True)
    gate.add_argument("--out")
    gate.set_defaults(func=_cmd_gate)

    construction = sub.add_parser(
        "construction-report", help="assemble construction-gate metrics"
    )
    construction.add_argument("--rollouts", required=True)
    construction.add_argument("--blocklist", required=True)
    construction.add_argument("--live-replay", required=True)
    construction.add_argument("--deterministic-reset", required=True)
    construction.add_argument("--out")
    construction.set_defaults(func=_cmd_construction_report)

    learner = sub.add_parser("learner-plan", help="emit a launch-disabled learner handoff")
    learner.add_argument("--dataset", required=True)
    learner.add_argument("--output-checkpoint", required=True)
    learner.add_argument("--learner-run-id", required=True)
    learner.add_argument("--trainer-adapter", required=True)
    learner.add_argument("--seed", type=int, default=0)
    learner.add_argument("--out", required=True)
    learner.set_defaults(func=_cmd_learner_plan)

    launch_gate = sub.add_parser(
        "relative-mouse-launch-gate",
        help="Bayesian pass@1/4/8 and accepted/GPU-hour feasibility gate",
    )
    launch_gate.add_argument("--probe", required=True)
    launch_gate.add_argument("--attestation", required=True)
    launch_gate.add_argument("--config", required=True)
    launch_gate.add_argument("--out")
    launch_gate.set_defaults(func=_cmd_relative_mouse_launch_gate)

    relative_build = sub.add_parser(
        "build-relative-mouse-rft",
        help="build pure rejection-SFT records from verified move_rel successes",
    )
    relative_build.add_argument("--rollouts-glob", required=True)
    relative_build.add_argument("--approved-background-root", required=True)
    relative_build.add_argument("--out", required=True)
    relative_build.add_argument("--val-fraction", type=float, default=0.05)
    relative_build.add_argument("--split-salt", default="stage5-relative-mouse-pure-v1")
    relative_build.add_argument("--maximum-trajectories-per-task", type=int, default=4)
    relative_build.set_defaults(func=_cmd_build_relative_mouse_rft)

    seal_batch = sub.add_parser(
        "seal-relative-mouse-batch",
        help="verify and combine fixed-budget zero-error rollout shards",
    )
    seal_batch.add_argument("--shard", action="append", required=True)
    seal_batch.add_argument("--out", required=True)
    seal_batch.add_argument("--expected-tasks", type=int, required=True)
    seal_batch.add_argument("--expected-k", type=int, required=True)
    seal_batch.add_argument("--minimum-accepted", type=int, required=True)
    seal_batch.add_argument("--expected-train-seed", type=int, required=True)
    seal_batch.set_defaults(func=_cmd_seal_relative_mouse_batch)

    compare_eval = sub.add_parser(
        "compare-relative-mouse-eval",
        help="compare the pilot with the frozen synthetic-validation baseline",
    )
    compare_eval.add_argument("--baseline-probe", required=True)
    compare_eval.add_argument("--candidate-probe", required=True)
    compare_eval.add_argument("--baseline-sacct", required=True)
    compare_eval.add_argument("--candidate-sacct", required=True)
    compare_eval.add_argument("--candidate-manifest", required=True)
    compare_eval.add_argument("--out")
    compare_eval.set_defaults(func=_cmd_compare_relative_mouse_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        _write_or_print(
            result,
            getattr(args, "out", None) if args.command not in SELF_WRITING_COMMANDS else None,
        )
    except ContractError as exc:
        print(f"stage5-rft gate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
