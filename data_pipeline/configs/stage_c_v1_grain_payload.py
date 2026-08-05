"""Stage C — compile chat.jsonl into Grain payload shards. Standalone.

Reads ``v1_run_length_capped_k0p4_5fps_360p_2026_04_09``.
Output dataset version: ``v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09``.
"""

import os
from pathlib import Path

from pmanager.configs.schema import pipeline_task


def _resolve_project_repo() -> str:
    """Repo subtree whose tree pmanager stages for the job.

    Derived from this file's own location (``<juergen>/data_pipeline/configs/``)
    rather than hardcoded: this config's ``cfg.entrypoint.path`` is named
    relative to ``data_pipeline/``, which is ``parents[1]`` (the chain_* configs
    name ``pipeline/...`` entrypoints and therefore use ``parents[2]``). The
    hardcoded ``/fast/home/franz.srambical/data_pipeline`` this replaces was a
    standalone pre-rearchitecture checkout that no longer exists, so the
    entrypoint resolved silently at config-build time and only failed once the
    job was already scheduled on a node. Fail here instead, and let
    ``JUERGEN_REPO`` name a different juergen checkout (its ``data_pipeline/``
    subtree is what gets staged) for anyone dispatching a tree other than this
    one.

    Duplicated verbatim across the stage_[a-d]_v1_* configs and mirrored in
    chain_annotate.py / chain_train.py, matching the existing ``_as_child``
    duplication: each config stays standalone-launchable with no cross-import at
    module scope.
    """
    override = os.environ.get("JUERGEN_REPO")
    root = Path(override) / "data_pipeline" if override else Path(__file__).resolve().parents[1]
    if not (root / "annotation_pipeline").is_dir():
        raise RuntimeError(
            f"PROJECT_REPO={root} has no 'annotation_pipeline/' directory, so it is "
            "not a data_pipeline root and cfg.entrypoint.path in this config would "
            "fail at dispatch. JUERGEN_REPO must name a juergen checkout (the "
            "'data_pipeline/' subtree is appended to it), not a data_pipeline root."
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
OMEGALAX_REPO = "/fast/home/franz.srambical/omegalax"
SOURCE_VERSION = "v1_run_length_capped_k0p4_5fps_360p_2026_04_09"
DATASET_VERSION = "v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "2:00:00"
    cfg.resources.mem = "64GB"
    cfg.resources.cpus = 8

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("stage_c_grain_payload.py")
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.messages_per_record = 128
    cfg.entrypoint.args.records_per_shard = 10_000

    cfg.inputs.source = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
