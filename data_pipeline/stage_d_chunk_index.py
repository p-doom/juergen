"""Stage D: build the offline chunk index from compiled grain payload.

Wrapper around omegalax/scripts/build_sft_chunk_index.py. Same per-split
loop pattern as stage C; each call runs inside omegalax's uv venv.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# Filename of the per-message length cache written by the measure stage
# (mirrors omegalax grain_pipeline.MESSAGE_LENGTHS_FILENAME). The measure stage
# writes one per split under <message_lengths_path>/<split>/.
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Chunk-index output dir.", required=True)
flags.DEFINE_string("payload_path", None, "Grain payload root (stage C).", required=True)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_string("model_id", None, "Model id (resolves the tokenizer).", required=True)
flags.DEFINE_string(
    "processor", None, "HF repo for image processor config (defaults to model_id).", required=True
)
flags.DEFINE_integer("max_length", None, "Max sequence length.", required=True)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)
# --num_workers IS forwarded (see the cmd below); this comment used to claim the
# opposite. Verified 2026-08-05 by enumerating every ref in the omegalax repo:
# the four flags this stage forwards (num_workers, overflow_mode,
# system_message_text, message_lengths_path) are all present on exactly three
# refs -- `remove-naive-split-overflow-mode`, its origin/ twin, and
# `origin/feat/chunk-index-truncate-mode`. They are NOT on `main` (whose
# build_sft_chunk_index.py takes 9 flags and none of these four), and the script
# does not exist at all on `feat/extra-transforms-hook`. So --omegalax_repo must
# name a checkout of one of those three refs, or this stage dies on an
# unrecognized flag. See the config's OMEGALAX_REPO note.
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2). "
    "Forwarded to omegalax/scripts/build_sft_chunk_index.py.",
    required=True,
    lower_bound=2,
)
flags.DEFINE_string(
    "system_message_text",
    "",
    "If non-empty, prepend a text-only system message with this content to "
    "every emitted chunk. Forwarded verbatim to "
    "omegalax/scripts/build_sft_chunk_index.py --system_message_text.",
)
flags.DEFINE_enum(
    "overflow_mode",
    "split",
    ["split", "truncate", "drop"],
    "Behaviour for conversations longer than max_length. 'split' (default): "
    "pack into multiple consecutive chunks at turn boundaries (no turns "
    "dropped). 'truncate': keep only the first fitting chunk and drop the "
    "overflowing turn plus the rest of the conversation. 'drop': discard the "
    "whole conversation if it does not fit in a single chunk. Forwarded to "
    "omegalax/scripts/build_sft_chunk_index.py --overflow_mode; per-split "
    "truncation stats land in each split's truncation_stats.json.",
)
flags.DEFINE_string(
    "message_lengths_path",
    None,
    "Root of a measure-stage artifact holding per-split "
    f"<split>/{MESSAGE_LENGTHS_FILENAME} caches. When set, each split forwards "
    "its cache to build_sft_chunk_index.py so the tokenizer pass is skipped "
    "(per-message lengths are independent of max_length / overflow_mode, so "
    "one cache serves every sequence length). Optional: omit to tokenize "
    "in-line as before.",
)


#: Dedicated omegalax venv root -- see stage_c_grain_payload.py for the rationale
#: and the existing labctl UV_PROJECT_ENVIRONMENT convention this follows. Shared
#: default with stage C on purpose: both stages run the same project, so they can
#: reuse one synced environment.
_UV_ENV_DEFAULT = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/omegalax-datastages-venv"
)


def _uv_project_environment(omegalax_repo: Path) -> str:
    """Resolve UV_PROJECT_ENVIRONMENT for the omegalax subprocess.

    Duplicated from ``stage_c_grain_payload.py`` rather than shared, matching the
    existing ``_as_child`` / ``_resolve_project_repo`` duplication convention in
    this package: each stage stays standalone-runnable with no cross-stage import.
    Keep the two copies in step.
    """
    env_path = Path(os.environ.get("OMEGALAX_UV_PROJECT_ENVIRONMENT", _UV_ENV_DEFAULT))

    # Both the literal path and its symlink target, against both forms of the repo:
    # <repo>/.venv is a SYMLINK out to p-doom_shared/franz/venvs/omegalax-venv here,
    # so resolving first would make an inside-the-repo path look outside it.
    def _forms(p: Path) -> set[Path]:
        return {p.absolute(), Path(os.path.realpath(p))}

    for cand in _forms(env_path):
        for repo in _forms(omegalax_repo):
            if cand == repo or repo in cand.parents:
                raise RuntimeError(
                    f"UV_PROJECT_ENVIRONMENT={cand} is inside --omegalax_repo={repo}, "
                    "so syncing would mutate the training checkout's own environment. "
                    "Point OMEGALAX_UV_PROJECT_ENVIRONMENT at a path outside the repo "
                    f"(default: {_UV_ENV_DEFAULT})."
                )

    # A path comparison alone misses the symlink case: a path nowhere near the repo
    # can still BE the physical environment the trainer uses.
    checkout_venv = omegalax_repo / ".venv"
    if checkout_venv.exists() and Path(os.path.realpath(env_path)) == Path(
        os.path.realpath(checkout_venv)
    ):
        raise RuntimeError(
            f"UV_PROJECT_ENVIRONMENT={env_path} resolves to the same directory as "
            f"{checkout_venv} -> {os.path.realpath(checkout_venv)}, which IS the "
            "training checkout's environment. Syncing would mutate it. Pick a "
            f"different path (default: {_UV_ENV_DEFAULT})."
        )
    return str(env_path.absolute())


def _run_split(split: str, src_payload: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        # --locked, NOT --no-sync -- see the full note in stage_c_grain_payload.py
        # and _uv_project_environment() below. The venv is redirected out of the
        # trainer's checkout, so the sync is kept (it is what makes this stage
        # reproducible from omegalax's pyproject + uv.lock) and --locked turns a
        # lockfile mismatch into a loud failure. --no-sync was tried and REJECTED:
        # it would let this stage run mismatched code and environment with no
        # signal. Kept in lockstep with stage C so the two cannot drift again.
        "--locked",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/build_sft_chunk_index.py",
        f"--data_path={src_payload}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--max_length={FLAGS.max_length}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        f"--num_workers={FLAGS.num_workers}",
        f"--overflow_mode={FLAGS.overflow_mode}",
        "--overwrite",
    ]
    if FLAGS.system_message_text:
        cmd.append(f"--system_message_text={FLAGS.system_message_text}")
    if FLAGS.message_lengths_path:
        split_cache = Path(FLAGS.message_lengths_path) / split / MESSAGE_LENGTHS_FILENAME
        cmd.append(f"--message_lengths_path={split_cache}")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": _uv_project_environment(Path(FLAGS.omegalax_repo))}
    print(f"[stage_d] {split}: {' '.join(cmd)}", flush=True)
    print(f"[stage_d] UV_PROJECT_ENVIRONMENT={env['UV_PROJECT_ENVIRONMENT']}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, env=env, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"build_sft_chunk_index.py failed (rc={rc}) for {split}")
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    return {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    payload_path = Path(FLAGS.payload_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src = payload_path / split
        if not src.is_dir():
            print(f"[stage_d] no payload for split {split}, skipping")
            continue
        if not (src / "metadata.json").is_file():
            print(f"[stage_d] incomplete/empty payload for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src, out_split_dir))

    write_manifest(
        output_dir,
        stage="chunk_index",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "max_length": FLAGS.max_length,
            "records_per_shard": FLAGS.records_per_shard,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
            "system_message_text": FLAGS.system_message_text,
            "overflow_mode": FLAGS.overflow_mode,
            "message_lengths_path": FLAGS.message_lengths_path,
        },
        inputs={"payload": str(payload_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
