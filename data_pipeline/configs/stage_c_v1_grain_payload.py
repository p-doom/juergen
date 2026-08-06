"""Stage C — compile chat.jsonl into Grain payload shards. Standalone.

Reads ``v1_run_length_capped_k0p4_5fps_360p_2026_04_09``.
Output dataset version: ``v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09``.
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


def _omegalax_repo(path: str) -> str:
    """Return ``path`` after asserting it is an omegalax checkout on disk.

    Same contract as :func:`_entrypoint`, applied to the other path this config
    hands a stage. Without it a stale OMEGALAX_REPO passes ``labctl validate``
    and the job dies on a node after being scheduled -- which is exactly what
    the entrypoint check exists to prevent, so leaving this one unchecked made
    the guarantee only half true.
    """
    if not (Path(path) / "pyproject.toml").is_file():
        raise RuntimeError(
            f"OMEGALAX_REPO={path} is not an omegalax checkout (no pyproject.toml). "
            "The stage subprocess-wraps a script from this tree, so dispatching "
            "would schedule a job that dies on the node."
        )
    return path


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
    cfg.entrypoint.args.omegalax_repo = _omegalax_repo(OMEGALAX_REPO)
    cfg.entrypoint.args.messages_per_record = 128
    cfg.entrypoint.args.records_per_shard = 10_000

    cfg.inputs.source = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
