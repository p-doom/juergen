"""Roundtrip eval: omegalax export → HF dir completion → SGLang+inspect.

Two modes (controlled by --checkpoint_path):
  1. Off-the-shelf round-trip (checkpoint_path == ""): exports the pretrained
     weights via omegalax to validate the export pipeline against the HF
     baseline. Used as a sanity check.
  2. Trained checkpoint (checkpoint_path = orbax dir): restores trained
     weights from the orbax checkpoint, exports to HF, evaluates. Used as
     the per-checkpoint eval during/after training.

The HF directory omegalax produces is incomplete — we patch it post-export
via ``hf_complete.complete_export_dir`` (copies tokenizer sidecars +
patches a few config.json keys from the cached HF snapshot for
``model_id``). The completed dir is what SGLang serves.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from absl import app, flags

import inspect_ai_patches  # noqa: F401  imported for side effects (monkey-patches)
from hf_complete import complete_export_dir, find_hf_snapshot
from inspect_runner import run_inspect_eval
from result import write_result
from sglang_runner import sglang_server

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Eval task dir.", required=True)
# When the eval is a child of a training run, ``cfg.inputs.checkpoint =
# {"kind": "parent_step_path"}`` resolves to the parent's step dir at fire
# time and pmanager injects --checkpoint_path. For off-the-shelf
# round-trip, the config sets cfg.entrypoint.args.checkpoint_path = "".
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Orbax checkpoint dir to restore. Empty string = export "
    "pretrained weights only (off-the-shelf roundtrip).",
    required=True,
)

# Model + omegalax export params:
flags.DEFINE_string(
    "model_id",
    None,
    "HF model_id (e.g. 'Qwen/Qwen3-VL-2B-Instruct'); selects "
    "the architecture and the snapshot we patch from.",
    required=True,
)
flags.DEFINE_string(
    "omegalax_repo", None, "omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_string(
    "hf_home", None, "HF cache root (must contain hub/<model_id>/snapshots/...).", required=True
)
flags.DEFINE_integer("tp_size", None, "Export tensor parallelism.", required=True)
flags.DEFINE_integer("fsdp_size", None, "Export FSDP size.", required=True)
flags.DEFINE_integer("dp_size", None, "Export DP size.", required=True)
# Optimizer-state-shape flags (must match training run; only used for orbax restore).
flags.DEFINE_float("max_grad_norm", None, "Optimizer-state shape: training run's.", required=True)
flags.DEFINE_integer(
    "grad_accum_steps", None, "Optimizer-state shape: training run's.", required=True
)

# inspect_ai params:
flags.DEFINE_string(
    "eval_task", None, "inspect_ai task spec, e.g. 'inspect_evals/ifeval'.", required=True
)
flags.DEFINE_float("temperature", None, "Generation temperature.", required=True)
flags.DEFINE_integer("max_tokens", None, "Max tokens per generation.", required=True)
flags.DEFINE_integer("seed", None, "RNG seed for inspect_ai.", required=True)
flags.DEFINE_integer("limit", None, "Cap on samples (0 = unlimited).", required=True)

# SGLang params:
flags.DEFINE_integer(
    "sglang_port",
    None,
    "SGLang port. 0 = derive from $SLURM_JOB_ID at runtime "
    "(30000 + jid % 10000) so concurrent evals on the same "
    "node don't collide.",
    required=True,
)
flags.DEFINE_string("sglang_api_key", None, "SGLang api key.", required=True)
flags.DEFINE_float("mem_fraction_static", None, "SGLang mem fraction.", required=True)
flags.DEFINE_integer("chunked_prefill_size", None, "SGLang chunked prefill.", required=True)


def main(_):
    output_dir = Path(FLAGS.output_dir)
    export_dir = output_dir / "exported_hf"
    sglang_log = output_dir / "sglang_server.log"
    inspect_log_dir = output_dir / "inspect_logs"
    limit = FLAGS.limit if FLAGS.limit > 0 else None
    orbax_path = Path(FLAGS.checkpoint_path) if FLAGS.checkpoint_path else None
    if FLAGS.sglang_port == 0:
        # Avoid same-node port collisions when multiple eval jobs land together.
        jid = int(os.environ.get("SLURM_JOB_ID", "0"))
        sglang_port = 30000 + (jid % 10000)
        print(f"[roundtrip_ifeval] auto-derived sglang_port={sglang_port} from SLURM_JOB_ID={jid}")
    else:
        sglang_port = FLAGS.sglang_port

    t_start = time.time()

    # --- Stage 1: omegalax export → HF safetensors ----------------------------
    # srun required: export_to_hf calls jax.distributed.initialize() which
    # needs SLURM_PROCID/SLURM_NODELIST — only populated inside srun context.
    # JAX_PLATFORMS=cpu: orbax restore needs ~36 GB optimizer state which
    # OOM-loops on the H100; export has no GPU compute so CPU is fine.
    export_dir.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "srun",
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/export_to_hf.py",
        f"--model_id={FLAGS.model_id}",
        f"--out_dir={export_dir}",
        f"--tp_size={FLAGS.tp_size}",
        f"--fsdp_size={FLAGS.fsdp_size}",
        f"--dp_size={FLAGS.dp_size}",
    ]
    if orbax_path is not None:
        export_cmd += [
            f"--checkpoint_path={orbax_path}",
            f"--max_grad_norm={FLAGS.max_grad_norm}",
            f"--grad_accum_steps={FLAGS.grad_accum_steps}",
        ]
    print(f"[roundtrip_ifeval] export: {' '.join(export_cmd)}", flush=True)
    t_export = time.time()
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    rc = subprocess.run(export_cmd, cwd=FLAGS.omegalax_repo, check=False, env=env).returncode
    export_elapsed = int(time.time() - t_export)
    if rc != 0:
        raise RuntimeError(f"omegalax export failed (rc={rc})")

    # --- Stage 2: complete the HF dir (tokenizer files + config patches) -----
    snapshot_dir = find_hf_snapshot(FLAGS.model_id, Path(FLAGS.hf_home))
    completion = complete_export_dir(export_dir, snapshot_dir)
    print(f"[hf_complete] copied={completion['copied']} patched={completion['patched']}")

    # --- Stage 3: SGLang serve + inspect_ai eval -----------------------------
    with sglang_server(
        model_path=str(export_dir),
        port=sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
    ) as server_url:
        scores, n_samples, inspect_elapsed = run_inspect_eval(
            task=FLAGS.eval_task,
            model=str(export_dir),
            server_url=server_url,
            api_key=FLAGS.sglang_api_key,
            temperature=FLAGS.temperature,
            max_tokens=FLAGS.max_tokens,
            seed=FLAGS.seed,
            log_dir=inspect_log_dir,
            limit=limit,
        )
    total_elapsed = int(time.time() - t_start)

    write_result(
        output_dir / "result.json",
        task=FLAGS.eval_task,
        scores=scores,
        params={
            "model_id": FLAGS.model_id,
            "tp_size": FLAGS.tp_size,
            "fsdp_size": FLAGS.fsdp_size,
            "dp_size": FLAGS.dp_size,
            "max_grad_norm": FLAGS.max_grad_norm,
            "grad_accum_steps": FLAGS.grad_accum_steps,
            "temperature": FLAGS.temperature,
            "max_tokens": FLAGS.max_tokens,
            "seed": FLAGS.seed,
            "limit": FLAGS.limit,
            "mem_fraction_static": FLAGS.mem_fraction_static,
            "chunked_prefill_size": FLAGS.chunked_prefill_size,
        },
        inputs={"checkpoint_path": FLAGS.checkpoint_path or "(pretrained)"},
        n_samples=n_samples,
        elapsed_s=total_elapsed,
        extra={
            "export_elapsed_s": export_elapsed,
            "inspect_elapsed_s": inspect_elapsed,
            "hf_completion": completion,
            "exported_hf_dir": str(export_dir),
        },
    )
    print(f"[roundtrip_ifeval] scores: {scores}")
    print(f"[roundtrip_ifeval] result: {output_dir / 'result.json'}")


if __name__ == "__main__":
    app.run(main)
