"""Stage D — build offline chunk index for the grain payload. Standalone.

Reads ``v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09``.
Output dataset version:
``v1_chunk_index_qwen3vl2b_len4096_msgs128_k0p4_5fps_360p_2026_04_09``.
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
# BROKEN, needs a decision -- deliberately NOT silently repointed.
#
# The old value "/fast/home/franz.srambical/omegalax-main" does not exist, so
# this stage fails at dispatch. Its comment also had the provenance wrong: the
# four flags stage D forwards (num_workers, overflow_mode, system_message_text,
# message_lengths_path) are NOT a main feature. Verified 2026-08-05 across every
# ref in the omegalax repo, all four appear together on exactly three:
#   remove-naive-split-overflow-mode  (+ its origin/ twin)
#   origin/feat/chunk-index-truncate-mode
# `main` has none of the four, and scripts/build_sft_chunk_index.py does not
# exist at all on `feat/extra-transforms-hook` -- which is the branch that
# /fast/home/franz.srambical/omegalax is on, so pointing this at the same tree
# stage C uses would NOT fix it (stage C is broken the same way: its
# compile_sft_dataset.py is also absent on that branch).
#
# No existing checkout is on any of the three viable refs, so repairing this
# needs a call that is not a cleanup. Left failing loudly rather than pointed
# somewhere that merely fails later and less clearly.
#
# This is PRE-EXISTING breakage, not refactor scope: both scripts were already
# absent on `feat/extra-transforms-hook` before the rearchitecture touched
# anything, so stages C and D were both broken independently of it.
#
# RECOMMENDATION (not a decision -- franz's call): port both stages to the
# current script API rather than pinning a worktree to one of those three refs.
# A worktree pinned to a stale non-`main` ref is exactly the fork pattern this
# rearchitecture exists to remove, and it would re-acquire the same rot the
# moment that branch diverges again. The two stages' forwarded flag sets are the
# concrete migration surface: stage C's compile_sft_dataset.py call, and stage
# D's four extra flags (num_workers, overflow_mode, system_message_text,
# message_lengths_path), none of which `main` accepts.
OMEGALAX_REPO = "/fast/home/franz.srambical/omegalax-main"
SOURCE_VERSION = "v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09"
DATASET_VERSION = "v1_chunk_index_qwen3vl2b_len4096_msgs128_k0p4_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "8:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 16

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = _entrypoint("stage_d_chunk_index.py")
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.model_id = "Qwen/Qwen3-VL-2B-Instruct"
    cfg.entrypoint.args.processor = "Qwen/Qwen3-VL-2B-Instruct"
    cfg.entrypoint.args.max_length = 4096
    cfg.entrypoint.args.records_per_shard = 100_000
    cfg.entrypoint.args.num_workers = 16

    cfg.inputs.payload = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
