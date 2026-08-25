"""Realigned-pipeline training chain STUB: 03 filter -> 04 conversations @x fps
-> 05 measure -> 06 records.

Each stage is nested as its parent's on_complete child. The filter is shared
with chain_annotate (same master + realigned
manifest family); the training fps X is independent of the annotation fps.
Goal-conditioned training: point ``GOALS_DIR`` at a finished chain_annotate
artifact (leave None for the goal-free path).

STUB status — adjust before launching:
  * OMEGALAX_REPO is a placeholder for your omegalax checkout (stage 05
    subprocess-wraps its measure script). The master store + realigned clips
    manifest are not named here at all: stage 03 is imported from
    chain_annotate, so its MASTER_DIR / CLIPS_MANIFEST are the single place to
    point at a dataset family.
  * PROJECT_REPO is derived from this file's own location and validated (see
    below), so it follows whichever checkout you launch from instead of rotting
    into a stale absolute path.
  * pmanager injects ``--output_dir`` (and ``--source_path`` for 05/06); the
    03/04 input dirs are derived statically from the datasets root + version
    (the argparse stages accept ``--foo_bar=value`` via
    ``common.normalize_dashed_argv``).
"""

import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task


def _resolve_project_repo() -> str:
    """Repo root whose tree pmanager stages for the job.

    Both configs in this package derive the same thing -- the juergen checkout,
    from this file's location or from ``JUERGEN_REPO`` -- and stage the checkout
    itself, because their entrypoints are ``pipeline/crowdcast/...``.

    Duplicated verbatim in chain_annotate.py (as ``_as_child`` is) so each config
    stays standalone-launchable with no cross-import at module scope.
    """
    root = Path(os.environ.get("JUERGEN_REPO") or Path(__file__).resolve().parents[2])
    if not (root / "pipeline" / "crowdcast").is_dir():
        raise RuntimeError(
            f"PROJECT_REPO={root} has no 'pipeline/crowdcast/' directory, so "
            "every cfg.entrypoint.path in this config would fail at dispatch. "
            "Pre-rearchitecture checkouts keep the stages under 'data_pipeline/"
            "realigned_pipeline/', and checkouts from before the per-corpus "
            "split keep them at 'pipeline/'. Set JUERGEN_REPO to a checkout "
            "with the 'pipeline/crowdcast/' layout."
        )
    return str(root)


def _entrypoint(rel_path: str) -> str:
    """Return ``rel_path`` after asserting it exists under ``PROJECT_REPO``.

    Turns a stale entrypoint into an import-time error (``labctl validate`` /
    ``get_config()``) instead of a scheduled job that dies on a missing file.
    """
    if not (Path(PROJECT_REPO) / rel_path).is_file():
        raise RuntimeError(
            f"entrypoint {rel_path!r} does not exist under PROJECT_REPO={PROJECT_REPO}"
        )
    return rel_path


def _source(path: str) -> str:
    """Return ``path`` after asserting it exists.

    Covers the inputs this chain does not itself produce. The 03->04->05->06
    handoffs are deliberately not checked: the parent stage creates those
    directories after this config has been read, so statting them here would
    refuse every legitimate dispatch.
    """
    if not Path(path).exists():
        raise RuntimeError(
            f"input path does not exist: {path}. GOALS_DIR must name a finished "
            "chain_annotate artifact (or be None for the goal-free path); a "
            "missing one is scheduled and then dies on the node."
        )
    return path


def _omegalax_repo(path: str) -> str:
    """Return ``path`` after asserting it is an omegalax checkout on disk.

    Same contract as :func:`_entrypoint`, applied to the other path this config
    hands a stage: without it a stale OMEGALAX_REPO passes ``labctl validate``
    and the job dies on a node after being scheduled.
    """
    if not (Path(path) / "pyproject.toml").is_file():
        raise RuntimeError(
            f"OMEGALAX_REPO={path} is not an omegalax checkout (no pyproject.toml). "
            "The stage subprocess-wraps a script from this tree, so dispatching "
            "would schedule a job that dies on the node."
        )
    return path


PROJECT_REPO = _resolve_project_repo()
DATASETS_ROOT = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu"
OMEGALAX_REPO = "/fast/project/HFMI_SynergyUnit/yll/omegalax"  # needs the measure script

TAG = "ccast0618d_v2"
TRAIN_FPS = 1.0
GOALS_DIR = None  # e.g. f"{DATASETS_ROOT}/{TAG}_stage_03b_goals_describe_extract_fps_0.5"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

FILTER_VERSION = f"{TAG}_stage_03_filter"
CONV_VERSION = f"{TAG}_stage_04_conversations_fps_{TRAIN_FPS}" + ("_goals" if GOALS_DIR else "")
MEASURE_VERSION = f"{TAG}_stage_05_measure_fps_{TRAIN_FPS}"
RECORDS_VERSION = f"{TAG}_stage_06_records_fps_{TRAIN_FPS}"


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def stage_03_filter():
    # Shared with chain_annotate; imported lazily so each config stays
    # standalone-launchable.
    from configs import chain_annotate  # noqa: PLC0415

    # Two independent constants that have to name the same tree: the filter
    # writes under chain_annotate's root and stage 04 reads it under this one.
    if chain_annotate.DATASETS_ROOT != DATASETS_ROOT:
        raise RuntimeError(
            f"DATASETS_ROOT disagrees with chain_annotate's "
            f"({DATASETS_ROOT} vs {chain_annotate.DATASETS_ROOT}), so stage 04 "
            f"would read a filter_dir the shared stage 03 never writes."
        )
    return chain_annotate.stage_03_filter()


def stage_04_conversations():
    cfg = pipeline_task()
    cfg.name = CONV_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "4:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("pipeline/crowdcast/stage_04_build_conversations.py")
    cfg.entrypoint.args.filter_dir = f"{DATASETS_ROOT}/{FILTER_VERSION}"
    cfg.entrypoint.args.fps = TRAIN_FPS
    cfg.entrypoint.args.action_format = "canonical"
    cfg.entrypoint.args.num_workers = 32
    if GOALS_DIR:
        cfg.entrypoint.args.goals_dir = _source(GOALS_DIR)
        cfg.entrypoint.args.use_plans = True
        cfg.entrypoint.args.include_variants = True
        cfg.entrypoint.args.terminal_token = "<terminate>"
    cfg.inputs.source = {"kind": "dataset", "version": FILTER_VERSION}
    cfg.output.dataset_version = CONV_VERSION
    cfg.dataset = ""
    return cfg


def stage_05_measure():
    cfg = pipeline_task()
    cfg.name = MEASURE_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("pipeline/crowdcast/stage_05_measure_lengths.py")
    cfg.entrypoint.args.source_path = f"{DATASETS_ROOT}/{CONV_VERSION}"
    cfg.entrypoint.args.omegalax_repo = _omegalax_repo(OMEGALAX_REPO)
    cfg.entrypoint.args.model_id = MODEL_ID
    cfg.entrypoint.args.processor = MODEL_ID
    cfg.entrypoint.args.num_workers = 32
    cfg.inputs.source = {"kind": "dataset", "version": CONV_VERSION}
    cfg.output.dataset_version = MEASURE_VERSION
    cfg.dataset = ""
    return cfg


def stage_06_records():
    cfg = pipeline_task()
    cfg.name = RECORDS_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("pipeline/crowdcast/stage_06_training_records.py")
    cfg.entrypoint.args.source_path = f"{DATASETS_ROOT}/{CONV_VERSION}"
    cfg.entrypoint.args.message_lengths_path = f"{DATASETS_ROOT}/{MEASURE_VERSION}"
    cfg.entrypoint.args.omegalax_repo = _omegalax_repo(OMEGALAX_REPO)
    cfg.entrypoint.args.model_id = MODEL_ID
    cfg.entrypoint.args.processor = MODEL_ID
    cfg.entrypoint.args.max_length = 32768
    cfg.entrypoint.args.records_per_shard = 1024
    cfg.entrypoint.args.num_workers = 32
    cfg.entrypoint.args.val_fraction = 0.02
    cfg.inputs.source = {"kind": "dataset", "version": MEASURE_VERSION}
    cfg.output.dataset_version = RECORDS_VERSION
    cfg.dataset = ""
    return cfg


def get_config():
    # Build inside-out: 06 nested in 05, 05 in 04, 04 in 03.
    stage_06 = stage_06_records()
    stage_05 = stage_05_measure()
    stage_05.children = [_as_child(stage_06)]
    stage_04 = stage_04_conversations()
    stage_04.children = [_as_child(stage_05)]
    stage_03 = stage_03_filter()
    stage_03.children = [_as_child(stage_04)]
    return stage_03
