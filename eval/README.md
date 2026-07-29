# crowdcast-eval

Evaluation harnesses for crowd-cast supervised fine-tuning runs:

- **IFEval** — instruction-following benchmark via [`inspect-ai`][inspect-ai] over an [sglang][sglang] OpenAI-compatible server.
- **OSWorld** — desktop-agent benchmark using the upstream [`OSWorld`][osworld] `DesktopEnv` + `mm_agents.qwen3vl_agent` pointed at the same sglang server.

This repo is consumed by `pmanager`/`labctl` recipes that inject task params via `absl` flags and read back `result.json` for metric aggregation.

[inspect-ai]: https://github.com/UKGovernmentBEIS/inspect_ai
[sglang]: https://github.com/sgl-project/sglang
[osworld]: https://github.com/xlang-ai/OSWorld

## Layout

| File | Role |
| --- | --- |
| `ifeval.py` | IFEval entrypoint against a model already in HF/SGLang-loadable form. |
| `roundtrip_ifeval.py` | IFEval over the full pipeline: orbax → HF export → SGLang → inspect. |
| `osworld_one_task_runner.py` | Run a single OSWorld task end-to-end. |
| `osworld_fullbench_runner.py` | Run one OSWorld task per SLURM array index; designed for `--array` jobs. |
| `osworld_score.py` | Aggregate per-task `result.json` files into a benchmark score. |
| `hf_complete.py` | Patch tokenizer sidecars + missing `config.json` keys onto omegalax HF exports. |
| `inspect_ai_patches.py` | In-process monkey-patches for two `inspect-ai` perf bugs; import for side effects. |
| `inspect_runner.py` | Subprocess wrapper around the `inspect eval` CLI; parses eval logs. |
| `result.py` | `write_result()` — atomic writer for `result.json` (pmanager metric contract). |
| `sglang_runner.py` | `sglang_server()` context manager. Spawns SGLang, polls readiness, tears down. |
| `sampling.py` | **Single source of truth for Qwen-recommended sampling params** (Instruct vs Thinking). Every rollout harness wires to it. |
| `patches/` | Patches for vendored (`$OSWORLD_ROOT`) code — currently the `qwen3vl_agent.py` dead-flag fix. See `patches/README.md`. |

## Sampling (Qwen-recommended, enforced)

All closed-loop / rollout harnesses (`freeroll.py`, `osworld_grounding_runner.py`,
`osworld_one_task_runner.py`, `osworld_fullbench_runner.py`) decode with the
Qwen-recommended sampling tuple from `sampling.py` — **never greedy by default,
never a partial param set.** The regime is auto-detected (Instruct vs Thinking)
from the checkpoint id / system prompt, overridable with `--sampling_mode`:

| regime | temperature | top_p | top_k | repetition_penalty | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| **Instruct-VL** (current) | 0.7 | 0.8 | 20 | 1.0 | 1.5 |
| **Thinking-VL** | 1.0 | 0.95 | 20 | 1.0 | 0.0 |

Each param has an override flag (`--temperature`, `--top_p`, `--top_k`,
`--repetition_penalty`, `--presence_penalty`, `--max_tokens`); `--greedy` opts
out of sampling entirely (discouraged — the Qwen cards ship `greedy=false`; kept
for deterministic monitors like `bc_roundtrip.py`).

**presence_penalty:** defaults to Qwen's recommended **1.5** to "match Qwen
exactly", but our own closed-loop A/B found 1.5 does **not** cut our OSWorld
repetition (near no-op: no-terminate 0.31→0.34, repeat 0.53→0.56 — the
repetition is structural covariate-shift, not a decoding artifact). Pass
`--presence_penalty 0` for our OSWorld runs.

The stock OpenAI schema rejects `top_k` / `repetition_penalty` /
`presence_penalty`, so the OpenAI-client path (native runners, via the vendored
`Qwen3VLAgent`) routes them through `extra_body`; the raw-`requests` harnesses
(freeroll / grounding) send the full tuple flat to sglang. The native runners
additionally require the `patches/qwen3vl_agent_sampling.patch` checkout patch
(the vendored agent otherwise ignores sampling params — see `patches/README.md`).

## Setup

```bash
uv sync  # creates .venv with sglang, inspect-ai, OSWorld evaluator deps
```

For the OSWorld harnesses, point `OSWORLD_ROOT` at your OSWorld checkout:

```bash
export OSWORLD_ROOT=/path/to/OSWorld
```

The runners insert `$OSWORLD_ROOT` into `sys.path` to resolve `desktop_env` / `mm_agents` without an editable install.

## Running

### IFEval (off-the-shelf or local HF dir)

```bash
uv run python ifeval.py \
    --output_dir=/tmp/ifeval_run \
    --model_path=Qwen/Qwen3-VL-2B-Instruct \
    --eval_task=inspect_evals/ifeval \
    --temperature=0.0 --max_tokens=2048 --seed=42 --limit=0 \
    --sglang_port=30000 --sglang_api_key=test \
    --mem_fraction_static=0.80 --chunked_prefill_size=2048
```

### Roundtrip IFEval (orbax → HF → SGLang)

```bash
uv run python roundtrip_ifeval.py \
    --output_dir=/tmp/roundtrip_run \
    --checkpoint_path=/path/to/orbax/step_13500 \
    --model_id=Qwen/Qwen3-VL-2B-Instruct \
    --omegalax_repo=/path/to/omegalax \
    --hf_home=$HF_HOME \
    --tp_size=2 --fsdp_size=2 --dp_size=1 \
    --max_grad_norm=1.0 --grad_accum_steps=8 \
    --eval_task=inspect_evals/ifeval \
    --temperature=0.0 --max_tokens=2048 --seed=42 --limit=0 \
    --sglang_port=0 --sglang_api_key=test \
    --mem_fraction_static=0.80 --chunked_prefill_size=2048
```

Pass `--checkpoint_path=""` to export pretrained weights instead (sanity check of the export pipeline).

### OSWorld

```bash
# Single task:
uv run python osworld_one_task_runner.py --output_dir=... --model_path=... \
    --task_path=$OSWORLD_ROOT/evaluation_examples/examples/chrome/<task>.json \
    --path_to_vm=/path/to/Ubuntu.qcow2 --provider_name=apptainer ...

# Full benchmark (one task per SLURM_ARRAY_TASK_ID):
uv run python osworld_fullbench_runner.py --base_output_dir=... \
    --test_split_path=$OSWORLD_ROOT/evaluation_examples/test_all.json \
    --task_index=$SLURM_ARRAY_TASK_ID ...

# Aggregate results:
uv run python osworld_score.py --base_output_dir=... \
    --test_split_path=$OSWORLD_ROOT/evaluation_examples/test_all.json
```

## Output contract

Every runner writes `result.json` to its output dir on success. Schema (consumed by pmanager):

```json
{
  "schema_version": 1,
  "task": "...",
  "scores": { "<name>/<metric>": <float>, ... },
  "params": { ... },
  "inputs": { ... },
  "n_samples": <int>,
  "elapsed_s": <int>,
  "completed_at": <unix_ts>,
  "pmanager_run_id": "...",
  "pmanager_parent_run_id": "...",
  "pmanager_parent_step": "..."
}
```

## Development

```bash
uvx ruff check .       # lint
uvx ruff format .      # format
```

`pyproject.toml` carries the strict rule set (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, ruff, pylint, tidy-imports, use-pathlib, return, comprehensions, pep8-naming).
