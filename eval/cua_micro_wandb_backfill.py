"""Post-hoc W&B upload for cua-micro eval runs that ran without ``[tracking.wandb]``.

Why this exists
---------------
``recipes/eval/osworld_freeroll_v24/osworld_per_hf_checkpoint_cua_micro_mid.toml``
was copied from its ``_medium`` sibling without the ``[tracking.wandb]`` block.
labctl therefore exported no ``WANDB_PROJECT``, so :func:`cua_micro_wandb.init`
took its "no project; not logging" branch and a whole generation of evals
finished with results on disk and nothing in W&B.

Everything those jobs would have logged is still recoverable, because a labctl
run dir is self-describing::

    <run>/.lab/submit.sh      -> the exact argv, LABCTL_RUN_ID, LABCTL_CONTEXT
    <run>/.lab/<recipe>_<job>.log -> which recipe *file* dispatched it
    <output_dir>/result.json  -> scores, per_task, and the resolved `params`

``result.json["params"]`` is the resolved config the eval recorded for itself
(sampling, action_format, model_resolution, ...), so the W&B config is rebuilt
from the run's own output rather than re-derived from argv -- no chance of this
script and the eval disagreeing about what actually ran.

Identity is reused wholesale from :mod:`cua_micro_wandb` -- the same
:func:`resolve_lineage` walk, the same ``group``/``run_name`` -- so a backfilled
run lands in exactly the group a live run would have, next to the trainer's own
curve. The W&B run id is the labctl run id (what labctl exports as
``WANDB_RUN_ID``), and ``resume="allow"`` makes re-running this script update
that run in place instead of creating a duplicate.

    # look first, upload nothing
    uv run --project=eval python eval/cua_micro_wandb_backfill.py --dry_run
    # then actually upload
    uv run --project=eval python eval/cua_micro_wandb_backfill.py

Runs still missing ``result.json`` (crashed or killed before aggregation) are
skipped and listed: there is no progress to upload for them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_micro_wandb

_LOGGER = logging.getLogger("cua_micro_wandb_backfill")

_RUNS_BASE = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/labctl_runs/runs")

# The recipe whose [tracking.wandb] block was missing. The job log inside a run
# dir is named <recipe file stem>_<slurm job id>.log, and that is the *only*
# per-run record of which recipe file dispatched it -- `name` inside the TOML is
# shared with the _medium sibling, so it cannot tell the two apart.
_DEFAULT_RECIPE_STEM = "osworld_per_hf_checkpoint_cua_micro_mid"

# Recipe generations share one stem, so scope by the suite path the job was
# handed. Empty string means "every generation".
_DEFAULT_SUITE_FILTER = "osworld_freeroll_v24"

_ENTITY = "pdoom"
_PROJECT = "omegalax"


@dataclass(frozen=True)
class EvalRun:
    """One dispatched eval, as reconstructed from its labctl run dir."""

    run_id: str
    job_id: int
    log_path: Path
    context_path: Path | None
    args: dict[str, str]
    """``--flag=value`` pairs off the submit.sh command line."""

    @property
    def output_dir(self) -> Path | None:
        raw = self.args.get("output_dir")
        return Path(raw) if raw else None

    @property
    def result_path(self) -> Path | None:
        out = self.output_dir
        return out / "result.json" if out else None


def discover(
    runs_user_dir: Path, recipe_stem: str, suite_filter: str
) -> tuple[list[EvalRun], list[str]]:
    """Every run dispatched by ``recipe_stem`` whose suite path matches the filter."""
    found: list[EvalRun] = []
    problems: list[str] = []
    for log_path in sorted(runs_user_dir.glob(f"*/.lab/{recipe_stem}_*.log")):
        lab = log_path.parent
        run_id = lab.parent.name
        match = re.search(r"_(\d+)\.log$", log_path.name)
        job_id = int(match.group(1)) if match else -1
        submit = lab / "submit.sh"
        if not submit.is_file():
            problems.append(f"{run_id}: no submit.sh")
            continue
        text = submit.read_text(errors="replace")
        args = _parse_args(text)
        if suite_filter and suite_filter not in args.get("suite", ""):
            continue
        context = _parse_export(text, "LABCTL_CONTEXT")
        found.append(
            EvalRun(
                run_id=run_id,
                job_id=job_id,
                log_path=log_path,
                context_path=Path(context) if context else None,
                args=args,
            )
        )
    return found, problems


def _parse_args(submit_text: str) -> dict[str, str]:
    """``--flag=value`` pairs from the job's command line.

    The eval is invoked through ``bash -c '...' -- --flag=value ...``, and
    labctl renders every recipe ``[args]`` entry in that long-option form, so a
    flat scan over the whole file is enough -- no need to find the exact line.
    """
    args: dict[str, str] = {}
    for raw in re.findall(r"--([A-Za-z_][A-Za-z0-9_]*)=(\S+)", submit_text):
        flag, value = raw
        args[flag] = value.rstrip("'\"")
    return args


def _parse_export(submit_text: str, name: str) -> str | None:
    match = re.search(rf"^export {re.escape(name)}=(.*)$", submit_text, re.MULTILINE)
    if not match:
        return None
    parts = shlex.split(match.group(1))
    return parts[0] if parts else None


def _namespace(run: EvalRun) -> argparse.Namespace:
    """The subset of the eval's parsed args that W&B identity depends on."""
    return argparse.Namespace(
        model_path=run.args.get("model_path"),
        suite=run.args.get("suite"),
        suite_level=run.args.get("suite_level"),
    )


def _resolution(raw: str) -> list[int] | None:
    match = re.fullmatch(r"(\d+)x(\d+)", raw.strip())
    return [int(match.group(1)), int(match.group(2))] if match else None


def _param(
    params: dict[str, Any], run: EvalRun, key: str, cast: Any = str
) -> Any:
    """``result.json["params"][key]``, falling back to the dispatched argv.

    A run that died before aggregating writes a *reduced* ``params`` (the 8
    incomplete v24 runs omit ``n_history_frames``), so the recipe's own value is
    the backstop -- it is what the job was launched with either way.
    """
    if key in params and params[key] is not None:
        return params[key]
    raw = run.args.get(key)
    if raw is None:
        return None
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return None


def build_payload(
    run: EvalRun, result: dict[str, Any], *, entity: str, project: str
) -> dict[str, Any]:
    """Rebuild the ``wandb.init`` kwargs the live eval would have used.

    Mirrors the ``cua_micro_wandb.init`` call in ``cua_micro_eval.py`` field for
    field; the values come from ``result.json`` (which is what the eval resolved
    at runtime) rather than being recomputed here.
    """
    namespace = _namespace(run)
    level = cua_micro_wandb.resolve_suite_level(namespace)

    # resolve_lineage reads $LABCTL_CONTEXT to locate <runs>/<user>/, exactly as
    # it would inside the job. Restored afterwards so runs cannot leak into each
    # other when a context path is missing.
    previous = os.environ.get("LABCTL_CONTEXT")
    if run.context_path is not None:
        os.environ["LABCTL_CONTEXT"] = str(run.context_path)
    else:
        os.environ.pop("LABCTL_CONTEXT", None)
    try:
        lineage = cua_micro_wandb.resolve_lineage(namespace.model_path)
    finally:
        if previous is None:
            os.environ.pop("LABCTL_CONTEXT", None)
        else:
            os.environ["LABCTL_CONTEXT"] = previous

    params = result.get("params") if isinstance(result.get("params"), dict) else {}
    action_format = _param(params, run, "action_format")
    suite_name = result.get("task")
    config = {
        "suite": suite_name,
        "suite_path": namespace.suite,
        "model_path": params.get("model_path", namespace.model_path),
        "attempts": _param(params, run, "attempts", int),
        "n_tasks": result.get("n_tasks"),
        "system_prompt_id": _param(params, run, "system_prompt_id"),
        "action_format": action_format,
        "model_resolution": _param(params, run, "model_resolution", _resolution),
        "n_history_frames": _param(params, run, "n_history_frames", int),
        "sampling": params.get("sampling"),
        "suite_level": level,
        "producer_recipe": lineage.producer_recipe,
        "producer_run_id": lineage.producer_run_id,
        "checkpoint_step": lineage.step,
        "lineage_chain": list(lineage.chain),
        "labctl_run_id": run.run_id,
        # The one field a live run has no reason to carry: how this run got here.
        "backfilled": True,
    }
    tags = sorted({t for t in (suite_name, action_format, level) if t})
    return {
        "lineage": lineage,
        "level": level,
        "init_kwargs": {
            "entity": entity,
            "project": project,
            # = labctl's WANDB_RUN_ID, so the W&B URL stays derivable from the
            # labctl run id and re-running this script resumes instead of
            # duplicating.
            "id": run.run_id,
            "resume": "allow",
            "name": lineage.run_name(run.run_id, level),
            "group": lineage.group,
            "job_type": "eval",
            "config": config,
            "tags": tags or None,
        },
    }


def upload(run: EvalRun, result: dict[str, Any], payload: dict[str, Any], wandb_dir: Path) -> str:
    """Start (or resume) the run and log its aggregate. Returns the run URL."""
    import wandb  # noqa: PLC0415

    init_kwargs = dict(payload["init_kwargs"])
    init_kwargs["dir"] = str(wandb_dir)
    wandb_run = wandb.init(**init_kwargs)
    # Reuse the live logger verbatim: same metric names, same `step=<checkpoint
    # step>`, same per_task table. A backfilled point is indistinguishable from
    # a live one on the chart, which is the whole objective.
    wrapper = cua_micro_wandb.WandbRun(wandb_run, payload["lineage"], wandb)
    wrapper.log_aggregate(result)
    url = wrapper.url or ""
    wrapper.finish(exit_code=0 if result.get("completed") else 1)
    return url


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs_dir",
        default=str(_RUNS_BASE / os.environ.get("USER", "alfred.nguyen")),
        help="labctl <runs_base>/runs/<user> dir to scan.",
    )
    parser.add_argument(
        "--recipe_stem",
        default=_DEFAULT_RECIPE_STEM,
        help="Recipe file stem, matched against <stem>_<job id>.log in each run's .lab/.",
    )
    parser.add_argument(
        "--suite_filter",
        default=_DEFAULT_SUITE_FILTER,
        help="Only runs whose --suite path contains this (scopes to one recipe "
        "generation, since the stem is shared across them). Empty = all.",
    )
    parser.add_argument(
        "--entity", default=_ENTITY, help="W&B entity (the missing [tracking.wandb] value)."
    )
    parser.add_argument("--project", default=_PROJECT, help="W&B project.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve and print what would be uploaded; touch W&B not at all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even for runs that already have a local wandb/ dir (i.e. "
        "ones that did log live). Off by default so a partially-tracked "
        "generation is safe to re-scan.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N uploads.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs_user_dir = Path(args.runs_dir)
    if not runs_user_dir.is_dir():
        parser.error(f"--runs_dir does not exist: {runs_user_dir}")
    runs, problems = discover(runs_user_dir, args.recipe_stem, args.suite_filter)
    for problem in problems:
        _LOGGER.warning("skipping %s", problem)
    _LOGGER.info(
        "found %d run(s) for recipe stem %r (suite filter %r)",
        len(runs),
        args.recipe_stem,
        args.suite_filter or "<none>",
    )

    uploaded, skipped_no_result, skipped_logged, failed = [], [], [], []
    for run in sorted(runs, key=lambda r: r.job_id):
        if args.limit is not None and len(uploaded) >= args.limit:
            _LOGGER.info("--limit %d reached; stopping", args.limit)
            break
        result_path = run.result_path
        if result_path is None or not result_path.is_file():
            skipped_no_result.append(run)
            continue
        output_dir = run.output_dir
        assert output_dir is not None
        if (output_dir / "wandb").is_dir() and not args.force:
            skipped_logged.append(run)
            continue
        try:
            result = json.loads(result_path.read_text())
            payload = build_payload(
                run, result, entity=args.entity, project=args.project
            )
        except Exception:
            _LOGGER.exception("%s: could not rebuild payload", run.run_id)
            failed.append(run)
            continue

        kwargs = payload["init_kwargs"]
        _LOGGER.info(
            "%s job=%s step=%s group=%s name=%s%s",
            run.run_id,
            run.job_id,
            payload["lineage"].step,
            kwargs["group"],
            kwargs["name"],
            " [dry run]" if args.dry_run else "",
        )
        if payload["lineage"].degraded:
            _LOGGER.warning("  degraded lineage: %s", payload["lineage"].degraded)
        if args.dry_run:
            uploaded.append(run)
            continue
        try:
            url = upload(run, result, payload, output_dir)
        except Exception:
            _LOGGER.exception("%s: upload failed", run.run_id)
            failed.append(run)
            continue
        _LOGGER.info("  -> %s", url)
        uploaded.append(run)

    verb = "would upload" if args.dry_run else "uploaded"
    _LOGGER.info(
        "%s=%d skipped_no_result=%d skipped_already_logged=%d failed=%d",
        verb,
        len(uploaded),
        len(skipped_no_result),
        len(skipped_logged),
        len(failed),
    )
    for run in skipped_no_result:
        _LOGGER.info("no result.json (nothing to upload): %s job=%s", run.run_id, run.job_id)
    for run in failed:
        _LOGGER.error("FAILED: %s job=%s", run.run_id, run.job_id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
