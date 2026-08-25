# micro_evals — the CUA micro-eval suite

Per-interaction-type canary tasks (click, key, type, scroll, drag, multi-step)
driven against a real VM. 18 tasks in `cua_micro_tasks.json`; larger suites live
in the labctl recipe tree.

## Status: a staged port

This is the suite as it ran on `feat/cua-micro-evals`, moved under `evals/` and
wired to this repo's grammars and prompt registry. It still carries its own VM
lifecycle, its own sglang launch and its own turn loop — it runs today, and it is
**not yet an `evals/` task family**. Every duplicated block carries a `PORT:`
comment naming the counterpart it should become; `__init__.py` has the full map.

## Running it

From the **repo root**, so `evals`, `grammars` and `prompts` import as packages:

```
uv run --project evals/micro_evals python -m evals.micro_evals.cua_micro_eval \
    --model_path <hf id or checkpoint> \
    --output_dir  <dest> \
    --suite       evals/micro_evals/cua_micro_tasks.json \
    --system_prompt_id ordered_events_v3_no_goal \
    --vms_per_sglang 8
```

`--project` is this directory and not the workspace: the suite imports only PIL,
requests and wandb, but it *launches* `python -m sglang.launch_server` as a
subprocess, so the torch/sglang resolve lives in this directory's own lock rather
than in the root's 124-package one.

## System prompts

`--system_prompt_id` resolves from two sources, in order:

1. **`prompts/`** — this repo's registry, where a prompt is a named edit of a
   grammar's own spec (`ordered_events_v3_goal`, `..._no_goal`, and the thinking
   variants). Prefer these: the digest is the one stage 04 writes into the
   dataset manifest, so a checkpoint and the arm scoring it are comparable.
2. **`osworld_system_prompts.SYSTEM_PROMPTS`** — the sealed historical prompts,
   kept because earlier checkpoints were trained under them and no codec can
   re-derive them.

A registry prompt is bound to its grammar: `ordered_events_v3` resolves to the
`cua_ordered_typing_v1` parser, because they are the same grammar — same
primitives, same `; ` separator, same `NO_OP`. A prompt over any other grammar is
refused rather than parsed with the wrong reader.

## The xcursor patch

`osworld_vm_client.patch_xcursor_leak` rewrites `pyxcursor.py` inside the guest
at VM start. The OSWorld agent assigns `XOpenDisplay`'s result to `self.display`
where the shared-connection cache expects the class attribute, so it leaks one X
connection per `/screenshot`; Xorg refuses clients past 256, and a multiturn task
dies around turn ~55.

`evals/vm.py` and the pinned `desktop` package carry no equivalent, and the other
suites never hit it — cuagym caps at 25 steps and sign-of-life at 12, roughly 150
connections. Anything here that runs dozens of turns does hit it. Best effort by
design: a VM that refuses the patch still runs, with the old ceiling.

## Tests

```
uv run --project evals/micro_evals pytest evals/micro_evals/tests/
```

Three `SuiteContractTests` failures are **pre-existing** — they fail identically
on `feat/cua-micro-evals` and are assertions about suite content, not about this
port. Four oracle tests skip unless `slurm/` is present beside the checkout (it
is untracked); point `$JUERGEN_SLURM_ROOT` at it to run them.
