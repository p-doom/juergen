"""``python -m rft.cli <stage>`` — the entry points the labctl recipes call.

One subcommand per stage, plus the cross-cutting audits. Every subcommand prints a
human-readable report to stdout **and** writes the machine-readable version next to
its outputs, so a slurm log alone is enough to see what happened (defect #13's lesson,
generalised).

Domain logic is injected, never embedded: ``--rollout-fn`` and ``--convert`` take
``module:function`` references. That keeps the per-format converters and the rollout
backends where they belong while this package owns the plumbing that kept breaking.

Exit codes: ``0`` success, ``2`` a gate failed (leak, parity, round trip, preflight),
``1`` an unexpected error. A gate failure is deliberately distinguishable so a labctl
recipe can tell "the data is wrong" from "the job crashed".
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rft.errors import RftError

GATE_FAILURE_EXIT = 2


def _load_ref(ref: str) -> Callable[..., Any]:
    """Resolve a ``module:function`` reference."""
    if ":" not in ref:
        raise RftError(f"expected module:function, got {ref!r}")
    module_name, func_name = ref.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise RftError(f"{module_name} has no attribute {func_name!r}") from exc


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[rft] wrote {path}")


# ---------------------------------------------------------------------------
# stage 1: preflight + sample
# ---------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    from rft.serving import preflight_chat_completion, validate_export_config

    if args.export_dir:
        audit = validate_export_config(args.export_dir)
        print(f"[rft] export OK: {audit.describe()}")
        if args.base_model_dir:
            from rft.serving import assert_export_differs_from_base

            assert_export_differs_from_base(args.export_dir, args.base_model_dir)
            print("[rft] export weights differ from the base model")
    result = preflight_chat_completion(
        base_url=args.base_url, model=args.model, timeout_s=args.timeout_s
    )
    print("[rft] " + result.describe())
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from rft.sampling import SamplingConfig, run_sampling
    from rft.splits import load_task_ids

    task_ids = sorted(t.split("/")[-1] for t in load_task_ids(args.tasks))
    cfg = SamplingConfig(
        task_ids=task_ids,
        k=args.k,
        grammar=args.grammar,
        out_path=Path(args.out),
        base_url=args.base_url,
        model=args.model,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_failure_rate=args.max_failure_rate,
        salt=args.salt,
    )
    report = run_sampling(cfg, _load_ref(args.rollout_fn))
    print("[rft] " + report.describe())
    return 0


# ---------------------------------------------------------------------------
# stage 2: score / filter
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    from rft.sampling import RolloutStore
    from rft.scoring import group_stats, score_rollouts, threshold_predicate, write_accepted

    payloads = list(RolloutStore(args.rollouts).read_all())
    scored, report = score_rollouts(
        payloads,
        accept=threshold_predicate(args.min_reward),
        reward_path=args.reward_path,
    )
    print("[rft] " + report.describe())
    print("[rft] " + group_stats(scored).describe())
    report.assert_healthy(
        max_unscored_fraction=args.max_unscored_fraction,
        max_malformed_fraction=args.max_malformed_fraction,
    )
    n = write_accepted(scored, args.out, report)
    print(f"[rft] wrote {n} accepted rollouts -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# stage 3: build records
# ---------------------------------------------------------------------------


def cmd_build_records(args: argparse.Namespace) -> int:
    from rft.records import build_records
    from rft.sampling import RolloutStore

    rollouts = list(RolloutStore(args.rollouts).read_all())
    report, split = build_records(
        rollouts,
        grammar=args.grammar,
        convert=_load_ref(args.convert),
        out_dir=Path(args.out),
        heldout_tasks_path=args.heldout,
        val_fraction=args.val_fraction,
        split_salt=args.salt,
        source_text_key=args.source_text_key or None,
        context_opt_out_reason=args.context_opt_out_reason or None,
    )
    print("[rft] " + report.describe())
    print(f"[rft] split: {len(split.train)} train tasks / {len(split.val)} val tasks")
    return 0


def cmd_audit_arms(args: argparse.Namespace) -> int:
    from rft.arms import audit_written_arms

    arm_dirs: dict[str, str] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise RftError(f"--arm expects name=dir, got {spec!r}")
        name, path = spec.split("=", 1)
        arm_dirs[name] = path
    report = audit_written_arms(
        arm_dirs, dimension=args.dimension, split=args.split, tolerance=args.tolerance
    )
    print("[rft] " + report.describe())
    if args.out:
        _write(Path(args.out), report.as_dict())
    if not report.controlled and not args.opt_out_reason:
        raise RftError(
            "arms are not a controlled comparison; see the report above. Pass "
            "--opt-out-reason to record a deliberate deviation."
        )
    return 0


# ---------------------------------------------------------------------------
# stage 4: train
# ---------------------------------------------------------------------------


def cmd_train_command(args: argparse.Namespace) -> int:
    from rft.records import verify_written_split
    from rft.training import (
        assert_paths_on_project,
        build_omegalax_invocation,
        count_jsonl_records,
    )

    verified = verify_written_split(args.dataset, args.heldout)
    print(f"[rft] dataset verified: {verified}")
    n_val = count_jsonl_records(Path(args.dataset) / "_normalized" / "val" / "chat.jsonl")
    assert_paths_on_project([args.save_dir])
    flags = json.loads(Path(args.flags).read_text()) if args.flags else {}
    invocation = build_omegalax_invocation(
        omegalax_repo=args.omegalax_repo,
        flags=flags,
        total_steps=args.num_steps,
        save_interval=args.save_every,
        n_val_records=n_val,
        global_batch_size=args.global_batch_size,
        requested_val_steps=args.val_steps,
        allow_partial_val=args.allow_partial_val,
        min_checkpoints_for_selection=args.min_checkpoints,
    )
    print("[rft] " + invocation.describe())
    if args.emit:
        Path(args.emit).write_text(invocation.command() + "\n")
        print(f"[rft] wrote command -> {args.emit}")
    return 0


def cmd_val_curve(args: argparse.Namespace) -> int:
    from rft.training import ValLossTee

    tee = ValLossTee()
    with Path(args.log).open(errors="replace") as fh:
        for echo in tee.scan(fh):
            print(echo)
    step, value = tee.best()
    print(f"[rft] best val/loss = {value:.6f} at step {step}")
    if args.out:
        _write(Path(args.out), {"points": tee.points, "best_step": step, "best": value})
    return 0


# ---------------------------------------------------------------------------
# stage 5: evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    from rft.diagnostics import delta_diagnostics, deltas_from_completions
    from rft.evaluation import (
        build_eval_report,
        expected_tasks_from_split,
        load_buckets,
        load_gdrive_exclusions,
        read_result_tree,
        write_eval_report,
    )
    from rft.evalparser import ACTION_PARSER_PATH

    results = read_result_tree(args.results_dir, reward_path=args.reward_path)
    expected = expected_tasks_from_split(args.split)
    excluded = load_gdrive_exclusions(args.exclude) if args.exclude else ()
    buckets = load_buckets(args.buckets) if args.buckets else None

    diagnostics = None
    if args.completions:
        texts = [
            line for line in Path(args.completions).read_text().splitlines() if line.strip()
        ]
        deltas, errors = deltas_from_completions(
            texts, grammar=args.grammar, skip_unparseable=True
        )
        if errors:
            print(f"[rft] {len(errors)} completion(s) yielded no delta; first: {errors[0]}")
        if deltas:
            diagnostics = delta_diagnostics(deltas)

    report = build_eval_report(
        results,
        run_name=args.run_name,
        expected_task_ids=expected,
        buckets=buckets,
        excluded_task_ids=excluded,
        diagnostics=diagnostics,
        reward_path=args.reward_path,
        parser_path=str(ACTION_PARSER_PATH),
        require_diagnostics=not args.no_diagnostics,
    )
    write_eval_report(report, args.out)
    if args.anchor:
        from rft.anchors import check_anchor

        mean = report.overall.mean_over_written
        if mean is None:
            raise RftError("no written results; cannot check an anchor")
        check_anchor(args.anchor, mean)
        print(f"[rft] anchor {args.anchor!r} satisfied by {mean:.4f}")
    return 0


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def cmd_provenance(args: argparse.Namespace) -> int:
    from rft.provenance import assert_comparable, resolve_checkpoint

    provs = [resolve_checkpoint(c, labctl_root=args.labctl_root) for c in args.checkpoint]
    for prov in provs:
        print(prov.describe())
    if args.dimension:
        fields = assert_comparable(provs, dimension=args.dimension)
        print(f"[rft] checkpoints are comparable on {args.dimension!r} alone")
        if args.out:
            _write(Path(args.out), {"dimension": args.dimension, "fields": fields,
                                    "checkpoints": [p.as_dict() for p in provs]})
    elif args.out:
        _write(Path(args.out), {"checkpoints": [p.as_dict() for p in provs]})
    return 0


def cmd_anchors(_args: argparse.Namespace) -> int:
    from rft.anchors import describe_all
    from rft.evalparser import describe as parser_describe
    from rft.grammars import available_grammars

    print(describe_all())
    print()
    print(parser_describe())
    print(f"grammar availability: {available_grammars()}")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rft", description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    q = sub.add_parser("preflight", help="stage 1 gate: validate an export and serve-check it")
    q.add_argument("--base-url", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--timeout-s", type=float, default=900.0)
    q.add_argument("--export-dir", default="")
    q.add_argument("--base-model-dir", default="")
    q.set_defaults(func=cmd_preflight)

    q = sub.add_parser("sample", help="stage 1: draw k completions per task")
    q.add_argument("--tasks", required=True, help="task-id list (259 train tasks)")
    q.add_argument("--k", type=int, required=True)
    q.add_argument("--grammar", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--base-url", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--rollout-fn", required=True, help="module:function performing 1 rollout")
    q.add_argument("--num-shards", type=int, default=1)
    q.add_argument("--shard-index", type=int, default=0)
    q.add_argument("--max-failure-rate", type=float, default=0.05)
    q.add_argument("--salt", default="rft-v1")
    q.set_defaults(func=cmd_sample)

    q = sub.add_parser("score", help="stage 2: accept-reject filter")
    q.add_argument("--rollouts", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--min-reward", type=float, default=1e-6)
    q.add_argument("--reward-path", default="scores.reward")
    q.add_argument("--max-unscored-fraction", type=float, default=0.05)
    q.add_argument("--max-malformed-fraction", type=float, default=0.0)
    q.set_defaults(func=cmd_score)

    q = sub.add_parser("build-records", help="stage 3: convert, audit, leak-check, split")
    q.add_argument("--rollouts", required=True)
    q.add_argument("--grammar", required=True)
    q.add_argument("--convert", required=True, help="module:function converting 1 rollout")
    q.add_argument("--out", required=True)
    q.add_argument("--heldout", required=True, help="the 110-task held-out list")
    q.add_argument("--val-fraction", type=float, default=0.1)
    q.add_argument("--salt", default="rft-v1")
    q.add_argument("--source-text-key", default="source_response")
    q.add_argument("--context-opt-out-reason", default="")
    q.set_defaults(func=cmd_build_records)

    q = sub.add_parser("audit-arms", help="assert format arms differ in exactly one thing")
    q.add_argument("--arm", action="append", required=True, help="name=converted/dir")
    q.add_argument("--dimension", default="action_format")
    q.add_argument("--split", default="train")
    q.add_argument("--tolerance", type=float, default=0.0)
    q.add_argument("--opt-out-reason", default="")
    q.add_argument("--out", default="")
    q.set_defaults(func=cmd_audit_arms)

    q = sub.add_parser("train-command", help="stage 4: validate + emit the omegalax argv")
    q.add_argument("--omegalax-repo", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--heldout", required=True)
    q.add_argument("--save-dir", required=True)
    q.add_argument("--flags", default="", help="JSON file of omegalax flags")
    q.add_argument("--num-steps", type=int, required=True)
    q.add_argument("--save-every", type=int, required=True)
    q.add_argument("--global-batch-size", type=int, default=1)
    q.add_argument("--val-steps", type=int, default=None)
    q.add_argument("--allow-partial-val", action="store_true")
    q.add_argument("--min-checkpoints", type=int, default=3)
    q.add_argument("--emit", default="")
    q.set_defaults(func=cmd_train_command)

    q = sub.add_parser("val-curve", help="stage 4: surface val/loss from a training log")
    q.add_argument("--log", required=True)
    q.add_argument("--out", default="")
    q.set_defaults(func=cmd_val_curve)

    q = sub.add_parser("evaluate", help="stage 5: bucketed report over a harness result tree")
    q.add_argument("--results-dir", required=True)
    q.add_argument("--split", required=True, help="{app: [task_id]} JSON")
    q.add_argument("--run-name", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--buckets", default="", help="heldout_buckets.json")
    q.add_argument("--exclude", default="", help="gdrive_unscorable.txt")
    q.add_argument("--reward-path", default="scores.reward")
    q.add_argument("--completions", default="", help="one completion per line, for diagnostics")
    q.add_argument("--grammar", default="bare_line")
    q.add_argument("--no-diagnostics", action="store_true")
    q.add_argument("--anchor", default="", help="anchor name to validate the reading against")
    q.set_defaults(func=cmd_evaluate)

    q = sub.add_parser("provenance", help="what IS this checkpoint (no Postgres needed)")
    q.add_argument("--checkpoint", action="append", required=True)
    q.add_argument("--labctl-root",
                   default="/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl")
    q.add_argument("--dimension", default="", help="assert comparability on this alone")
    q.add_argument("--out", default="")
    q.set_defaults(func=cmd_provenance)

    q = sub.add_parser("anchors", help="print the reference anchors and parser status")
    q.set_defaults(func=cmd_anchors)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RftError as exc:
        print(f"[rft] GATE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return GATE_FAILURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
