"""Stage C: compile per-split chat.jsonl into Grain ArrayRecord shards.

Wrapper around omegalax/scripts/compile_sft_dataset.py. omegalax's script
processes one split per invocation; this entrypoint loops over the three
splits in the source dataset and orchestrates the calls. Each subprocess
runs inside omegalax's own uv-managed venv via ``uv run --project``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Grain payload output dir.", required=True)
flags.DEFINE_string("source_path", None, "Filtered dataset root (stage B).", required=True)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_integer(
    "messages_per_record", None, "Maximum contiguous messages per payload block.", required=True
)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)


#: Where dedicated omegalax venvs live. Matches the existing convention: the
#: labctl submit.sh scripts already point UV_PROJECT_ENVIRONMENT at persistent
#: per-purpose venvs under this root (omegalax-qwen35-venv,
#: omegalax-qwen35-fullft-fix-venv, omegalax-gbs16/.venv). Stages C and D were the
#: outliers that synced into the checkout's own .venv.
_UV_ENV_DEFAULT = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/omegalax-datastages-venv"
)


def _uv_project_environment(omegalax_repo: Path) -> str:
    """Resolve UV_PROJECT_ENVIRONMENT for the omegalax subprocess.

    ``--project <omegalax_repo>`` makes uv sync that project before running. By
    default it would sync ``<omegalax_repo>/.venv`` -- i.e. mutate a LIVE training
    checkout's environment. Redirecting the environment keeps the sync (so the run
    stays reproducible from omegalax's pyproject + uv.lock, enforced by --locked)
    while leaving the trainer's tree untouched.

    Overridable per-run via ``OMEGALAX_UV_PROJECT_ENVIRONMENT`` rather than
    hardcoded, so a job can direct it without editing this file. Deliberately
    STAGE-DEDICATED AND PERSISTENT, not per-run: an omegalax venv is multi-GB, so
    building a fresh one per job is untenable; reusing one is fast under --locked
    when it is already correct and re-syncs when it is not.
    """
    env_path = Path(os.environ.get("OMEGALAX_UV_PROJECT_ENVIRONMENT", _UV_ENV_DEFAULT))

    # Check BOTH the literal path and its symlink target, and compare against both
    # forms of the repo. Checking only the resolved path is not enough: in this
    # deployment <repo>/.venv is a SYMLINK out to
    # p-doom_shared/franz/venvs/omegalax-venv, so resolving first makes an
    # inside-the-repo path look outside it. Checking only the literal path is not
    # enough either, for the mirror-image reason below.
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

    # And the case a path comparison alone misses: the checkout's .venv is a
    # symlink, so a path that is nowhere near the repo can still BE the physical
    # environment the trainer uses. Compare real paths.
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


def _run_split(split: str, src_chat: Path, out_split_dir: Path) -> dict:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        # --locked, NOT --no-sync. See _uv_project_environment() below: the venv is
        # redirected out of the trainer's checkout, so we WANT the sync -- it is
        # what makes this stage reproducible from omegalax's pyproject + uv.lock,
        # and --locked makes a lockfile mismatch fail loudly instead of silently
        # re-resolving.
        #
        # Do NOT "simplify" this to --no-sync. It was tried and REJECTED: it
        # suppresses the sync, so the stage runs against whatever happens to be in
        # that venv. If the venv had been built from a different omegalax commit
        # than --omegalax_repo is checked out at, the stage would run mismatched
        # code and environment with NO signal at all.
        #
        # Mechanism, measured on uv 0.7.19 (2026-08-05) in a scratch project:
        #   uv run --project X          -> "Installed 1 package", dep lands in the venv
        #   uv run --locked --project X -> same, it still installs
        #   uv run --frozen --project X -> same, it still installs
        #   uv run --no-sync --project X-> runs, installs nothing
        # So project-mode `uv run` INSTALLS; it was not observed to prune. Pruning
        # is `uv sync`'s behaviour (that is what reports `- msgpack` for the
        # goal-timeline viewer and what could strip torch / transformers /
        # flashinfer). The fix for that hazard is environment REDIRECTION, not sync
        # suppression -- redirect where it installs, don't stop it installing.
        "--locked",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/compile_sft_dataset.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--messages_per_record={FLAGS.messages_per_record}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        "--overwrite",
    ]
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": _uv_project_environment(Path(FLAGS.omegalax_repo))}
    print(f"[stage_c] {split}: {' '.join(cmd)}", flush=True)
    print(f"[stage_c] UV_PROJECT_ENVIRONMENT={env['UV_PROJECT_ENVIRONMENT']}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, env=env, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"compile_sft_dataset.py failed (rc={rc}) for {split}")
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    return {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}


def _has_jsonl_rows(path: Path) -> bool:
    with path.open() as f:
        return any(line.strip() for line in f)


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    for split in ("train", "val", "test"):
        src_chat = source_path / split / "chat.jsonl"
        if not src_chat.is_file():
            print(f"[stage_c] no chat.jsonl for split {split}, skipping")
            continue
        if not _has_jsonl_rows(src_chat):
            print(f"[stage_c] empty chat.jsonl for split {split}, skipping")
            continue
        out_split_dir = output_dir / split
        per_split.append(_run_split(split, src_chat, out_split_dir))

    write_manifest(
        output_dir,
        stage="grain_payload",
        params={
            "messages_per_record": FLAGS.messages_per_record,
            "records_per_shard": FLAGS.records_per_shard,
            "omegalax_repo": FLAGS.omegalax_repo,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
