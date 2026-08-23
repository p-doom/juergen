# cua-micro-evals

A fast, closed-form evaluation ladder for desktop-agent checkpoints: every
task is one (or a few) sampled model turns with an automatic pass/fail
verifier, so checkpoint progress is visible without eyeballing long
freeroll GIFs. Ported from `yll/cua-micro-evals` (the `juergen` repo) onto
this branch, with one addition: the suite now also speaks this branch's own
`ordered_events_v3` (`cua_ordered_typing_v1`) action format, not just yll's
original `computer_use_rel_step_v1` / `qwen3vl_native_cua_v1`. See
`MICRO_EVAL_PORT_NOTES.md` at the repo root for the full history of what was
ported, what was added, and why.

## Layout

| File | Role |
| --- | --- |
| `cua_micro_eval.py` | The runner: loads the suite, boots a fresh VM per attempt, calls the model, parses+dispatches the action, scores against the task's verifier. |
| `cua_micro_tasks.json` | The 18-task suite: `native_launch` / `native_app` / `chrome_control` / `multi_turn` categories. |
| `cua_micro_fixture.py` | Standard-library Tk fixture copied into the guest for editor/terminal/calculator/files/settings tasks; writes exact widget bboxes + semantic state to guest JSON. |
| `cua_micro_action_parser.py` | Strict parsers for `computer_use_rel_step_v1` and `qwen3vl_native_cua_v1` (yll's original two formats). Kept separate from `action_parser.py` because both files independently evolved a class named `OrderedPrimitive`/`OrderedAction` into incompatible shapes — see the module docstring. |
| `action_parser.py` | This branch's own action-format library (`ordered_events_v2/v3`, aggregate, computer_use tool calls) — used here for the added `cua_ordered_typing_v1` support. |
| `osworld_vm_client.py` | The in-VM Flask-agent client. `dispatch_ordered_action()` (ported from yll) executes parsed primitives from ANY of the three supported formats; `dispatch_ordered()`/`dispatch_action()`/`dispatch_computer_use()` are this branch's own, unused by this suite but left in place. |
| `osworld_system_prompts.py` / `osworld_runtime.py` | Shared prompt table + sglang-call/VM-boot helpers (this branch's own; additive edits only, see port notes). |
| `sampling.py` | Qwen-recommended sampling-tuple module (temperature/top_p/top_k/penalties by Instruct vs Thinking regime). |
| `patches/` | Native-runner sampling patch for off-the-shelf baselines. |

## Setup

```bash
uv sync  # creates .venv with sglang, PIL, requests, etc.
```

Point `OSWORLD_ROOT` at an OSWorld checkout if you use anything that imports
`desktop_env`/`mm_agents` (this suite's own tasks don't need it — no
`evaluation_examples/`, no `SetupController` — but keep it set for parity
with other eval tooling on this branch).

## Running

```bash
# Exercise setup and every verifier with known-correct synthetic actions; no GPU, no model.
uv run --project . python cua_micro_eval.py --validate_setups_only \
    --output_dir=/tmp/cua_micro_validate

# Checkpoint trained on this branch's ordered_events_v3 / cua_ordered_typing_v1:
# --vms_per_sglang (default 4) runs that many VMs concurrently against the
# one shared sglang instance instead of one VM at a time -- a single VM's
# step time is dominated by sglang prefill (batch size 1), badly underusing
# the GPU. Pass --vms_per_sglang=1 for the historical one-VM-at-a-time
# behaviour.
uv run --project . python cua_micro_eval.py \
    --model_path=/path/to/checkpoint --attempts=4 \
    --system_prompt_id=cua_ordered_typing_v1 \
    --action_format=cua_ordered_typing_v1 \
    --model_resolution=1280x720 --vms_per_sglang=4 \
    --output_dir=/tmp/cua_micro_ordered

# yll's original computer_use_rel_step_v1 checkpoint:
uv run --project . python cua_micro_eval.py \
    --model_path=/path/to/checkpoint --attempts=4 \
    --system_prompt_id=cua_rel_step_v1_thinking \
    --output_dir=/tmp/cua_micro_relstep

# Native Qwen3-VL-Instruct baseline (off-the-shelf, no fine-tuning):
uv run --project . python cua_micro_eval.py \
    --model_path=Qwen/Qwen3-VL-8B-Instruct --attempts=4 \
    --system_prompt_id=qwen3vl_native_cua_v1 \
    --action_format=qwen3vl_native_cua_v1 \
    --sampling_mode=instruct --output_dir=/tmp/cua_micro_qwen
```

`--action_format` defaults to whatever `--system_prompt_id` implies (see
`_PROMPT_FORMATS` in `cua_micro_eval.py`) — pass it explicitly only to assert
the pairing. All three formats normalize into the same internal
`OrderedAction`/`OrderedPrimitive` shape before scoring and dispatch, so
`movement_metrics`, `serialize_action`, `action_matches_expected`, etc. are
format-agnostic; only parsing (`parse_computer_use_rel_step_action` /
`parse_qwen3vl_computer_use_action` / `parse_ordered_action_tolerant`) and
resolution-scaling (`denormalize_action` / `denormalize_native_ordered_action`)
differ per format.

Task success is exact end-to-end success. `pass_at_4` means at least one of
the first four attempts succeeded; `all_4_success` separately detects tasks
an off-the-shelf model solved four times out of four. Multi-turn partial
credit is the verified prefix fraction, with scheduled-turn parse,
expected-action, completion, and verifier rates reported alongside
task-level metrics. Each turn persists its prompt, response, before/after
frames, overlay, parsed action, and semantic verifier state. `result.json`
is rewritten after every attempt for live monitoring.

## Development

```bash
uv run --project . python -m unittest test_cua_micro_eval.py test_sampling.py
uvx ruff check .
uvx ruff format .
```
