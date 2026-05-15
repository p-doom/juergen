"""IFEval against an HF model already in SGLang-loadable form.

Pure SGLang+inspect step — no omegalax export. Use this for off-the-shelf
baselines (model_path = HF model_id) or to evaluate an externally-prepared
HF directory.

For the "load orbax checkpoint, export, and eval" flow, use
``roundtrip_ifeval.py`` instead.
"""

from __future__ import annotations

import time
from pathlib import Path

from absl import app, flags

import inspect_ai_patches  # noqa: F401  imported for side effects (monkey-patches)
from inspect_runner import run_inspect_eval
from result import write_result
from sglang_runner import sglang_server

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Eval task dir.", required=True)

# Model identity (string flag, NOT an artifact input — SGLang accepts either an
# HF model_id like "Qwen/Qwen3-VL-2B-Instruct" or an absolute path to a
# local HF dir):
flags.DEFINE_string("model_path", None, "HF model_id or local HF dir.", required=True)

# inspect_ai params:
flags.DEFINE_string(
    "eval_task", None, "inspect_ai task spec, e.g. 'inspect_evals/ifeval'.", required=True
)
flags.DEFINE_float("temperature", None, "Generation temperature.", required=True)
flags.DEFINE_integer("max_tokens", None, "Max tokens per generation.", required=True)
flags.DEFINE_integer("seed", None, "RNG seed for inspect_ai.", required=True)
flags.DEFINE_integer(
    "limit", None, "Cap on samples (for smoke tests). 0 = unlimited.", required=True
)

# SGLang server params:
flags.DEFINE_integer("sglang_port", None, "SGLang server port.", required=True)
flags.DEFINE_string("sglang_api_key", None, "SGLang server API key.", required=True)
flags.DEFINE_float("mem_fraction_static", None, "SGLang --mem-fraction-static.", required=True)
flags.DEFINE_integer("chunked_prefill_size", None, "SGLang --chunked-prefill-size.", required=True)


def main(_):
    output_dir = Path(FLAGS.output_dir)
    sglang_log = output_dir / "sglang_server.log"
    inspect_log_dir = output_dir / "inspect_logs"
    limit = FLAGS.limit if FLAGS.limit > 0 else None

    t0 = time.time()
    with sglang_server(
        model_path=FLAGS.model_path,
        port=FLAGS.sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
    ) as server_url:
        scores, n_samples, inspect_elapsed = run_inspect_eval(
            task=FLAGS.eval_task,
            model=FLAGS.model_path,
            server_url=server_url,
            api_key=FLAGS.sglang_api_key,
            temperature=FLAGS.temperature,
            max_tokens=FLAGS.max_tokens,
            seed=FLAGS.seed,
            log_dir=inspect_log_dir,
            limit=limit,
        )
    elapsed = int(time.time() - t0)

    write_result(
        output_dir / "result.json",
        task=FLAGS.eval_task,
        scores=scores,
        params={
            "temperature": FLAGS.temperature,
            "max_tokens": FLAGS.max_tokens,
            "seed": FLAGS.seed,
            "limit": FLAGS.limit,
            "mem_fraction_static": FLAGS.mem_fraction_static,
            "chunked_prefill_size": FLAGS.chunked_prefill_size,
        },
        inputs={"model_path": FLAGS.model_path},
        n_samples=n_samples,
        elapsed_s=elapsed,
        extra={"inspect_elapsed_s": inspect_elapsed},
    )
    print(f"[ifeval] scores: {scores}")
    print(f"[ifeval] result: {output_dir / 'result.json'}")


if __name__ == "__main__":
    app.run(main)
