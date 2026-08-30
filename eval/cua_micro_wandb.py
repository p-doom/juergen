"""W&B logging for the cua-micro eval, keyed on the *training* run's identity.

Why this module exists
----------------------
A cua-micro eval job is dispatched per exported HF checkpoint, so on its own it
knows only ``--model_path``. That is useless as a W&B identity: the interesting
axis is "how does training run X score as it trains", i.e. one W&B *group* per
training recipe with one *run* per checkpoint step. So the naming contract is::

    group = <producer_recipe>_<training run id>   # one group per TRAINING run
    name  = eval_<level>_<step>_<EVAL run id>    # one run per eval job in it

Note the two run ids are different runs. ``group`` carries the *training*
run's, which is what makes it one group per training run -- and byte-identical
to the ``wandb_group`` omegalax's trainer uses for that same run, so training
curves and per-checkpoint eval scores share a group. ``name`` carries this
*eval job's* own labctl run id (= ``WANDB_RUN_ID``), so re-evaluating a
checkpoint yields a distinct run rather than a name collision.

None of these values is available as a labctl template token (``{inputs.X.path}`` and
``{inputs.X.id}`` are all we get, see labctl docs/RECIPE_CONTRACT.md), and the
checkpoint we are handed was produced by the *export* recipe, not by training.
:func:`resolve_lineage` recovers both by walking the artifact lineage on shared
storage -- no Postgres, which compute nodes cannot reach:

    <model_path>/.meta.json                  -> step, producer_run_id
    <runs>/<user>/<producer_run_id>/.lab/context.json
                                             -> inputs[role=checkpoint].resolved_path
    <that path>/.meta.json                   -> producer_recipe (+ repeat)

The walk repeats while the producing run itself consumed a checkpoint, so it
resolves the orbax->HF export hop today and would survive another hop being
added upstream. The *most upstream* checkpoint's ``producer_recipe`` is the
training recipe.

Entity/project come from ``[tracking.wandb]`` in the recipe, which labctl turns
into ``WANDB_ENTITY`` / ``WANDB_PROJECT`` / ``WANDB_RUN_ID`` (= the labctl run
id, making the W&B URL derivable from the run id alone). We deliberately do NOT
pass entity/project to ``wandb.init`` so that stays labctl's job -- but we DO
pass ``name``/``group`` explicitly, because labctl's own defaults
(``<recipe>-<run suffix>``, and a static group string) carry no checkpoint
identity. Explicit ``init`` kwargs outrank the env vars.

Nothing here is allowed to fail the eval. Telemetry that takes a 40-minute GPU
job down with it is worse than no telemetry, so every step degrades to a
warning and a coarser label.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Depth cap on the lineage walk. Two hops (HF export -> orbax training) is
# today's shape; the cap only exists so a cyclic/corrupt sidecar cannot spin.
_MAX_LINEAGE_HOPS = 6


@dataclass(frozen=True)
class Lineage:
    """Where the evaluated checkpoint came from."""

    producer_recipe: str | None
    """Recipe name of the most upstream checkpoint producer (the trainer)."""

    step: int | None
    """Training step of the evaluated checkpoint."""

    producer_run_id: str | None = None
    """labctl run id of that same training run (already ``run_``-prefixed).

    The *training* run, not this eval run: it is what makes one group per
    training run. Keyed on the eval's own run id instead, every eval job would
    land in a group of one.
    """

    chain: tuple[str, ...] = ()
    """producer_recipe of each artifact walked, nearest-first. Diagnostics."""

    degraded: str | None = None
    """Why the walk stopped short, if it did. Diagnostics."""

    @property
    def group(self) -> str | None:
        """``<producer_recipe>_<training run id>``.

        Deliberately byte-identical to the string omegalax's trainer passes as
        its own ``wandb_group`` (and to the orbax checkpoint's ``stream_alias``),
        so a training run and every eval of its checkpoints share one group.
        """
        if self.producer_recipe is None:
            return None
        if self.producer_run_id is None:
            return self.producer_recipe
        return f"{self.producer_recipe}_{self.producer_run_id}"

    def run_name(self, eval_run_id: str | None, level: str | None = None) -> str | None:
        """``eval_<level>_<step>_<eval run id>`` -- unique per eval job.

        ``level`` is the suite difficulty ("easy"/"mid", see
        :func:`resolve_suite_level`). It belongs in the name and NOT in the
        group: easy and mid evals of one training run are different views of
        the same run and should stay grouped together.

        ``eval_run_id`` is this job's ``$LABCTL_RUN_ID``, i.e. the same id W&B
        uses for the run itself, NOT :attr:`producer_run_id`. Re-evaluating one
        checkpoint therefore produces two distinctly named runs in the group
        instead of two runs called the same thing.
        """
        parts = ["eval"]
        if level:
            parts.append(level)
        if self.step is not None:
            parts.append(str(self.step))
        if eval_run_id:
            parts.append(eval_run_id)
        return "_".join(parts) if len(parts) > 1 else None


def add_wandb_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the W&B flags. Logging is off unless a project is resolvable."""
    g = parser.add_argument_group(
        "wandb",
        "Off unless WANDB_PROJECT is set (labctl exports it from the recipe's "
        "[tracking.wandb]) or --wandb_project is passed. See eval/cua_micro_wandb.py.",
    )
    g.add_argument(
        "--wandb_project",
        default=None,
        help="W&B project. Overrides WANDB_PROJECT; setting it also enables logging.",
    )
    g.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B entity. Overrides WANDB_ENTITY (which labctl sets from the recipe).",
    )
    g.add_argument(
        "--wandb_name",
        default=None,
        help="Override the run name. Default: eval_<suite level>_<checkpoint "
        "step>_<this eval job's labctl run id>; the step comes from the "
        "checkpoint's artifact lineage.",
    )
    g.add_argument(
        "--wandb_group",
        default=None,
        help="Override the run group. Default: <producer_recipe>_<training run "
        "id>, so every checkpoint of one training run groups together (and "
        "matches the group the trainer itself logs under).",
    )
    g.add_argument(
        "--suite_level",
        default=None,
        help="Suite difficulty ('easy'/'mid') to put in the run name, so the "
        "two suites of one checkpoint are distinguishable. Deliberately absent "
        "from the group: both are views of the same training run. Defaults to "
        "the trailing _<segment> of the --suite filename "
        "(cua_micro_tasks_easy.json -> easy).",
    )
    g.add_argument(
        "--wandb_tags",
        nargs="+",
        default=None,
        help="Extra W&B tags. The suite name and action_format are tagged anyway.",
    )
    g.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Passed through to wandb.init. 'disabled' turns logging off even "
        "when a project is set; 'offline' buffers to <output_dir>/wandb for a "
        "later `wandb sync`.",
    )
    return parser


# --------------------------------------------------------------------------
# lineage
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        _LOGGER.warning("wandb lineage: cannot read %s: %s", path, error)
        return None
    return payload if isinstance(payload, dict) else None


def _runs_user_dir() -> Path | None:
    """The ``<runs_base>/runs/<user>/`` dir holding every run of this user.

    Derived from our own ``$LABCTL_CONTEXT``
    (``<runs_base>/runs/<user>/<run_id>/.lab/context.json``) so we never have
    to parse the labctl cluster config, which lives outside the job's world.
    """
    raw = os.environ.get("LABCTL_CONTEXT")
    if not raw:
        return None
    # context.json -> .lab -> <run_id> -> <user>
    candidate = Path(raw).parent.parent.parent
    return candidate if candidate.is_dir() else None


def _upstream_checkpoint_path(runs_user_dir: Path | None, producer_run_id: str) -> Path | None:
    """Resolve the checkpoint that ``producer_run_id`` consumed, if any."""
    if runs_user_dir is None:
        return None
    context = _read_json(runs_user_dir / producer_run_id / ".lab" / "context.json")
    if context is None:
        return None
    inputs = context.get("inputs")
    if not isinstance(inputs, list):
        return None
    for entry in inputs:
        if not isinstance(entry, dict) or entry.get("role") != "checkpoint":
            continue
        resolved = entry.get("resolved_path")
        if isinstance(resolved, str) and resolved:
            return Path(resolved)
    return None


def resolve_lineage(model_path: str | None) -> Lineage:
    """Recover ``(producer_recipe, producer_run_id, step)`` for ``model_path``.

    Walks ``.meta.json`` -> producing run's ``context.json`` -> upstream
    checkpoint, and keeps the last (most upstream) ``producer_recipe`` and its
    matching ``producer_run_id``.
    Returns a best-effort :class:`Lineage`; missing pieces come back as
    ``None`` with ``degraded`` set rather than raising.
    """
    if not model_path:
        return Lineage(None, None, degraded="no --model_path")

    current = Path(model_path)
    runs_user_dir = _runs_user_dir()
    chain: list[str] = []
    # Parallel to `chain`: the producer_run_id of the artifact each recipe came
    # from, so the trainer's recipe and run id are read off the same hop.
    run_ids: list[str | None] = []
    step: int | None = None
    degraded: str | None = None
    if runs_user_dir is None:
        degraded = "LABCTL_CONTEXT unset or unusable; cannot walk past the exported checkpoint"

    for _ in range(_MAX_LINEAGE_HOPS):
        meta = _read_json(current / ".meta.json")
        if meta is None:
            if not chain:
                # Not a labctl artifact at all (a bare local checkpoint dir).
                # The step is still often recoverable from the dir name.
                degraded = f"no .meta.json at {current}"
                step = _step_from_dirname(current)
            break
        metadata = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        if step is None:
            raw_step = metadata.get("step")
            step = raw_step if isinstance(raw_step, int) else _step_from_dirname(current)
        recipe = metadata.get("producer_recipe")
        producer_run_id = meta.get("producer_run_id")
        has_run_id = isinstance(producer_run_id, str) and bool(producer_run_id)
        if isinstance(recipe, str) and recipe:
            chain.append(recipe)
            run_ids.append(producer_run_id if has_run_id else None)
        if not has_run_id:
            break
        upstream = _upstream_checkpoint_path(runs_user_dir, producer_run_id)
        if upstream is None:
            break
        current = upstream
    else:
        degraded = f"lineage deeper than {_MAX_LINEAGE_HOPS} hops; stopped early"

    producer_recipe = chain[-1] if chain else None
    if producer_recipe is None and degraded is None:
        degraded = "no producer_recipe in any .meta.json on the lineage"
    return Lineage(
        producer_recipe=producer_recipe,
        step=step,
        producer_run_id=run_ids[-1] if run_ids else None,
        chain=tuple(chain),
        degraded=degraded,
    )


def resolve_suite_level(args: argparse.Namespace) -> str | None:
    """The suite's difficulty label for the run name.

    Explicit ``--suite_level`` wins. Otherwise it is inferred from the suite
    filename's trailing ``_<segment>`` -- every suite in this repo is named
    ``cua_micro_tasks_<level>.json``, and the ``"suite"`` field *inside* those
    files is the same string ("cua_micro_tasks") for easy and mid alike, so the
    filename is the only place the level actually lives.
    """
    explicit = getattr(args, "suite_level", None)
    if explicit:
        return str(explicit)
    suite = getattr(args, "suite", None)
    if suite is None:
        return None
    _, separator, tail = Path(suite).stem.rpartition("_")
    return tail if separator and tail else None


def _step_from_dirname(path: Path) -> int | None:
    """``.../003000`` -> 3000. Checkpoint dirs are zero-padded step numbers."""
    name = path.name
    return int(name) if name.isdigit() else None


# --------------------------------------------------------------------------
# init / log
# --------------------------------------------------------------------------


class WandbRun:
    """Thin wrapper so callers never branch on "is W&B on?".

    A disabled instance swallows every call, which keeps the eval's happy path
    free of ``if wandb_run is not None`` noise.
    """

    def __init__(self, run: Any, lineage: Lineage, module: Any = None) -> None:
        self._run = run
        self._wandb = module
        self.lineage = lineage

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def log_aggregate(self, payload: dict[str, Any]) -> None:
        """Log the final ``result.json`` payload: scores, run-level counters, tasks.

        Everything is logged at ``step=<checkpoint step>`` so a group of
        per-checkpoint runs reads as a training-progress curve on one chart.
        """
        if self._run is None:
            return
        try:
            scores = payload.get("scores")
            metrics: dict[str, Any] = dict(scores) if isinstance(scores, dict) else {}
            for key in ("n_samples", "n_tasks", "elapsed_s"):
                if key in payload:
                    metrics[f"run/{key}"] = payload[key]
            metrics["run/completed"] = int(bool(payload.get("completed")))
            if self.lineage.step is not None:
                metrics["checkpoint/step"] = self.lineage.step
                self._run.log(metrics, step=self.lineage.step)
            else:
                self._run.log(metrics)
            # Summary duplicates the headline numbers so the runs *table* is
            # sortable without opening a chart.
            self._run.summary.update(metrics)

            per_task = payload.get("per_task")
            if isinstance(per_task, dict) and per_task:
                columns = ["task_id", *sorted({k for row in per_task.values() for k in row})]
                table = self._wandb.Table(columns=columns)
                for task_id, row in sorted(per_task.items()):
                    table.add_data(task_id, *[row.get(c) for c in columns[1:]])
                self._run.log({"per_task": table})
        except Exception:
            _LOGGER.exception("wandb: failed to log aggregate results (eval unaffected)")

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception:
            _LOGGER.exception("wandb: finish() failed (eval unaffected)")


def init(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    config: dict[str, Any],
    tags: list[str] | None = None,
    level: str | None = None,
) -> WandbRun:
    """Start the W&B run, or return a no-op :class:`WandbRun`.

    Logging is enabled iff a project is resolvable (``--wandb_project`` or
    ``WANDB_PROJECT``) and ``--wandb_mode`` is not ``disabled``.
    """
    lineage = resolve_lineage(getattr(args, "model_path", None))
    if lineage.degraded:
        _LOGGER.warning("wandb: degraded checkpoint lineage: %s", lineage.degraded)
    _LOGGER.info(
        "wandb: lineage producer_recipe=%s producer_run_id=%s step=%s chain=%s",
        lineage.producer_recipe,
        lineage.producer_run_id,
        lineage.step,
        list(lineage.chain),
    )

    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if args.wandb_mode == "disabled":
        _LOGGER.info("wandb: disabled by --wandb_mode")
        return WandbRun(None, lineage)
    if not project:
        _LOGGER.info(
            "wandb: no project (neither --wandb_project nor WANDB_PROJECT); not logging"
        )
        return WandbRun(None, lineage)

    eval_run_id = os.environ.get("LABCTL_RUN_ID")
    # The caller resolves this once and shares it with aggregate_results so the
    # run name and the score-key suffix agree; falling back keeps `init` usable
    # on its own.
    if level is None:
        level = resolve_suite_level(args)
    name = args.wandb_name or lineage.run_name(eval_run_id, level)
    group = args.wandb_group or lineage.group
    if name is None:
        # Neither a step nor a run id to build a name from (a bare local
        # checkpoint outside labctl). Anything beats a W&B animal name.
        name = "cua_micro_eval"
        _LOGGER.warning(
            "wandb: no suite level, step or LABCTL_RUN_ID; falling back to name=%s", name
        )

    try:
        # Deferred: this module must stay importable (and the eval must still
        # run) in a venv without wandb, and wandb's own import is not cheap.
        import wandb  # noqa: PLC0415
    except ImportError:
        _LOGGER.warning("wandb: package not installed; not logging")
        return WandbRun(None, lineage)

    init_kwargs: dict[str, Any] = {
        "project": project,
        # `dir` keeps wandb's scratch inside the eval's own output dir. Without
        # it wandb writes ./wandb/ into the CWD, which for a labctl job is the
        # repo source checkout.
        "dir": str(output_dir),
        "name": name,
        "group": group,
        "job_type": "eval",
        "config": {
            **config,
            "suite_level": level,
            "producer_recipe": lineage.producer_recipe,
            "producer_run_id": lineage.producer_run_id,
            "checkpoint_step": lineage.step,
            "lineage_chain": list(lineage.chain),
            "labctl_run_id": eval_run_id,
        },
        "tags": sorted(
            {*(tags or []), *(args.wandb_tags or []), *([level] if level else [])}
        )
        or None,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    try:
        run = wandb.init(**init_kwargs)
    except Exception:
        _LOGGER.exception("wandb: init failed; continuing without logging")
        return WandbRun(None, lineage)
    _LOGGER.info("wandb: logging to %s (group=%s, name=%s)", getattr(run, "url", "?"), group, name)
    return WandbRun(run, lineage, wandb)
