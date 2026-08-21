"""Realigned-pipeline annotation chain STUB: 03 filter -> 03b annotate @k fps.

The downstream stage is nested as the upstream stage's on_complete child.
Both stages read the same master store + realigned clips manifest family; the
annotation fps K is independent of any training fps (bounded only by the
master fps, integer stride).

STUB status — adjust before launching:
  * MASTER_DIR / CLIPS_MANIFEST name the ccast0618d v3 generation and are
    asserted to exist at ``get_config()``; swap them for your dataset family.
  * PROJECT_REPO is derived from this file's own location and validated, so it
    follows whichever checkout you launch from instead of rotting into a stale
    absolute path.
  * pmanager injects ``--output_dir``; the downstream stage's ``--filter_dir``
    is derived statically from the datasets root + version here (the stages
    accept pmanager's ``--foo_bar=value`` arg form via
    ``common.normalize_dashed_argv``).
  * stage 03b spends LABELER tokens: it needs AZURE_OPENAI_ENDPOINT /
    AZURE_OPENAI_API_KEY in the job env and honest --target-tpm/--max-workers.
"""

import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task


def _resolve_project_repo() -> str:
    """Repo root whose tree pmanager stages for the job.

    Both configs in this package derive the same thing -- the juergen checkout,
    from this file's location or from ``JUERGEN_REPO`` -- and stage the checkout
    itself, because their entrypoints are ``pipeline/...``.
    """
    root = Path(os.environ.get("JUERGEN_REPO") or Path(__file__).resolve().parents[2])
    if not (root / "pipeline").is_dir():
        raise RuntimeError(
            f"PROJECT_REPO={root} has no 'pipeline/' directory, so every "
            "cfg.entrypoint.path in this config would fail at dispatch. Pre-"
            "rearchitecture checkouts keep the stages under 'data_pipeline/"
            "realigned_pipeline/'. Set JUERGEN_REPO to a checkout with the "
            "root-'pipeline/' layout."
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


PROJECT_REPO = _resolve_project_repo()
DATASETS_ROOT = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu"
MASTER_DIR = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/alfred.nguyen/"
    "ccast0618d_dataset_full_v3_stage_01_master_frames_fps_15"
)
CLIPS_MANIFEST = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/alfred.nguyen/"
    "ccast0618d_dataset_full_v3_stage_02_realign_manifest/clips_manifest.jsonl"
)

TAG = "ccast0618d_v2"
ANNOTATE_FPS = 0.5
FILTER_VERSION = f"{TAG}_stage_03_filter"
GOALS_VERSION = f"{TAG}_stage_03b_goals_describe_extract_fps_{ANNOTATE_FPS}"


def _source(path: str) -> str:
    """Return ``path`` after asserting it exists.

    Same contract as :func:`_entrypoint`, applied to the dataset paths.
    """
    if not Path(path).exists():
        raise RuntimeError(
            f"dataset path does not exist: {path}. Repoint MASTER_DIR / "
            "CLIPS_MANIFEST at the current generation of your dataset family."
        )
    return path


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def stage_03_filter():
    cfg = pipeline_task()
    cfg.name = FILTER_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "4:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("pipeline/stage_03_filter.py")
    cfg.entrypoint.args.frames_master_dir = _source(MASTER_DIR)
    cfg.entrypoint.args.clips_manifest = _source(CLIPS_MANIFEST)
    cfg.entrypoint.args.num_workers = 32
    # Idle knobs (seconds; identical semantics on any master fps).
    cfg.entrypoint.args.idle_activity = "rounded"
    cfg.entrypoint.args.idle_judgment_bin_s = 2.0
    cfg.entrypoint.args.idle_min_duration_s = 4.0
    cfg.entrypoint.args.idle_keep_head_s = 2.0
    cfg.entrypoint.args.idle_keep_tail_s = 2.0
    cfg.inputs.source = {"kind": "path", "path_per_cluster": {"berlin": _source(MASTER_DIR)}}
    cfg.output.dataset_version = FILTER_VERSION
    cfg.dataset = ""
    return cfg


def stage_03b_annotate():
    cfg = pipeline_task()
    cfg.name = GOALS_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "24:00:00"
    cfg.resources.mem = "64GB"
    cfg.resources.cpus = 16
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("pipeline/annotation/stage_annotate.py")
    cfg.entrypoint.args.filter_dir = f"{DATASETS_ROOT}/{FILTER_VERSION}"
    cfg.entrypoint.args.fps = ANNOTATE_FPS
    cfg.entrypoint.args.method = "describe_extract"
    cfg.entrypoint.args.models = "Kimi-K2.6"
    cfg.entrypoint.args.target_tpm = 1_800_000
    cfg.entrypoint.args.max_workers = 64
    cfg.inputs.source = {"kind": "dataset", "version": FILTER_VERSION}
    cfg.output.dataset_version = GOALS_VERSION
    cfg.dataset = ""
    return cfg


def get_config():
    stage_b = stage_03b_annotate()
    stage_a = stage_03_filter()
    stage_a.children = [_as_child(stage_b)]
    return stage_a
