"""Stage A — extract frames + actions from S3 sync. Standalone-launchable.

Output dataset version: ``v1_raw_event_stream_5fps_360p_2026_04_09``
"""

import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task


def _resolve_project_repo() -> str:
    """Repo subtree whose tree pmanager stages for the job.

    Every config in this package derives the same thing -- the juergen checkout,
    from this file's location or from ``JUERGEN_REPO`` -- and then names the
    subtree it stages. This one stages ``data_pipeline/`` because its
    ``cfg.entrypoint.path`` is relative to it; the chain_* configs stage the
    checkout itself and name ``pipeline/...`` entrypoints. One derivation, one
    meaning for ``JUERGEN_REPO``, the subtree spelled out per config rather than
    encoded as a different ``parents[]`` index.

    The hardcoded ``/fast/home/franz.srambical/data_pipeline`` this replaces was
    a standalone pre-rearchitecture checkout that no longer exists, so the
    entrypoint resolved silently at config-build time and only failed once the
    job was already scheduled on a node.

    Duplicated verbatim across the stage_[a-d]_v1_* configs and mirrored in
    chain_annotate.py / chain_train.py, matching the existing ``_as_child``
    duplication: each config stays standalone-launchable with no cross-import at
    module scope.
    """
    juergen = Path(os.environ.get("JUERGEN_REPO") or Path(__file__).resolve().parents[2])
    root = juergen / "data_pipeline"
    if not (root / "annotation_pipeline").is_dir():
        raise RuntimeError(
            f"PROJECT_REPO={root} has no 'annotation_pipeline/' directory, so it is "
            "not a data_pipeline root and cfg.entrypoint.path in this config would "
            "fail at dispatch. JUERGEN_REPO must name a juergen checkout (the "
            "'data_pipeline/' subtree is appended to it), not a data_pipeline root."
        )
    return str(root)


SOURCE_UPLOADS = "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-04-09/uploads"


def _source_path() -> str:
    """Return the raw-uploads source dir, asserting it exists.

    ESCALATED, deliberately not repointed: this recording day is gone from disk
    (only crowd-cast-2026-06-18 and crowd-cast-2026-07-27 remain). Swapping it
    changes both which recordings the dataset is built from and the
    ``_2026_04_09`` version name baked into SOURCE/DATASET_VERSION, so it is a
    dataset-lineage decision, not a cleanup. Until that call is made this fails
    loudly at ``get_config()`` -- which is what ``labctl validate`` runs -- rather
    than dispatching a job that would find an empty source at stage-A runtime.
    """
    if not Path(SOURCE_UPLOADS).is_dir():
        raise RuntimeError(
            f"cfg.inputs.source path does not exist: {SOURCE_UPLOADS}. This "
            "recording day was removed; only crowd-cast-2026-06-18 and "
            "crowd-cast-2026-07-27 remain. Repointing changes the dataset lineage "
            "AND the _2026_04_09 version name in SOURCE_VERSION / DATASET_VERSION, "
            "so it needs an explicit decision -- pick the replacement day and bump "
            "both version strings together."
        )
    return SOURCE_UPLOADS


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
DATASET_VERSION = "v1_raw_event_stream_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "200GB"
    cfg.resources.cpus = 32

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("stage_a_prepare.py")
    cfg.entrypoint.args.target_fps = 5
    cfg.entrypoint.args.target_height = 360
    cfg.entrypoint.args.jpeg_quality = 85
    cfg.entrypoint.args.train_ratio = 0.8
    cfg.entrypoint.args.val_ratio = 0.1
    cfg.entrypoint.args.seed = 0
    cfg.entrypoint.args.num_workers = 32
    cfg.entrypoint.args.max_segments = 0

    cfg.inputs.source = {
        "kind": "path",
        "path_per_cluster": {"berlin": _source_path()},
    }

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""  # pipelines don't reference an input dataset version
    return cfg
