# OSWorld parity — the `deltatype-v2` pipeline

Production code for the pipeline that produces the `s900` checkpoint: a
synthetic compact-delta curriculum (Phase A), supervised transfer onto real
OSWorld teacher trajectories in the `deltatype_raw_v2` grammar (Phase B), and
the teacher-forced evaluator that scores both splits.

This is the pipeline only — one canonical copy of each stage, no experiment
arms, no comparison harnesses, no one-shot custody scaffolding. What was
deliberately left out, and where it is still recoverable from, is listed under
[Not in this branch](#not-in-this-branch).

## Lineage

```
relative_factorial_reltool_pre_r256_s750_capacity_v1_run_019fb4beda3472b289ae60fc612c1cea
  │   (pinned labctl checkpoint artifact — see "Entry point" below)
  │
  ├─ Phase A — compact-raw transfer  (A→B)
  │    fresh LoRA r256/alpha256 on the merged warm start, 750 steps, lr 1e-4,
  │    wsd, max_length 4096, target format `deltatype_raw_pre`
  │    synthetic_multistep/build_episodes.py
  │    synthetic_multistep/build_curriculum_stage2.py
  │    synthetic_multistep/curriculum_train_export.sh
  │    → synthetic_multistep_curriculum_A_to_B_raw_pre_r256_s750_recovered_v1_run_019fb56fb2f471118f1a9ed683def8b0
  │
  └─ Phase B — OSWorld teacher-trajectory SFT
       fresh LoRA r256/alpha256, lr 1e-4, wsd (stable 0.7, end 0.0),
       warmup 30, batch 1, grad_accum 8, 900 steps, max_length 16384,
       no in-loop validation
       phaseb_deltatype_raw_v2/build.py
       phaseb_deltatype_raw_v2/tokenize_authorize.{py,sh}
       phaseb_deltatype_raw_v2/train_production_r256.sh
       → phaseb_raw_deltatype_v2_A_to_B_r256_s900_..._run_019fb7c24ffc7c62970c97d0b5e9af0b
```

### Entry point

Phase A warm-starts from a **pinned labctl checkpoint artifact**, not from
`Qwen/Qwen3-VL-8B-Instruct`. That artifact is the endpoint of a 750-step
tool-call warm-up whose code is *not* in this branch: the tool-call scaffold is a
resolved experiment (it beat the B→B control on first-attempt reach at lr 1e-4,
0.9969 vs 0.9438, Δ +0.0531, p≈7.6e-5, but the sign flipped at lr 5e-5 —
0.7906 vs 0.9375, p≈9.4e-8), and the deployment format is compact, so it is
science rather than production. **Consequence: this branch rebuilds `s900` from
that checkpoint, not from the base model.** The checkpoint lives at

```
…/labctl/checkpoints/franz.srambical/relative_factorial_reltool_pre_r256_s750_capacity_v1_run_019fb4beda3472b289ae60fc612c1cea
```

and its build/train code is recoverable only from that run's
`.lab/provenance/<repo>/{tracked,untracked}.patch` (plus run
`019fb43ff2927ee093a50f2f8577c7db` for its corpus). If a from-base rebuild ever
matters again, restore those files from those patches rather than guessing.

### Checkpoints

| What | Path |
|---|---|
| Phase A endpoint = Phase B warm start | `…/labctl/checkpoints/franz.srambical/synthetic_multistep_curriculum_A_to_B_raw_pre_r256_s750_recovered_v1_run_019fb56fb2f471118f1a9ed683def8b0` |
| **Phase B `s900` LoRA (Orbax, the literal adapter state)** | `…/phaseb_raw_deltatype_v2_A_to_B_r256_s900_conditional_exact_continuation_v2_run_019fb7c24ffc7c62970c97d0b5e9af0b/orbax/000900` |
| **Phase B `s900` merged HF export** (what the evals scored) | `…/phaseb_raw_deltatype_v2_A_to_B_r256_s900_continuation_hf_v4_run_019fba52e90778e0b8ae170058c814e7/hf` |

`…` = `/fast/project/HFMI_SynergyUnit/p-doom_shared`. The merged HF export is a
standalone checkpoint; initialising a **new** LoRA on top of it is a stacked
adapter, not a continuation of `s900`, and must be labelled as such.

## Layout

One self-contained mount point that mirrors the `experiments/` sibling layout the
code executed under, because several modules resolve imports with
`__file__`-relative `sys.path` inserts (`_EXPERIMENTS / "phaseb_relative"`,
`_EXPERIMENTS / "phaseb_deltatype_raw_v2"`) and their bytes are frozen by hash —
relocating them into a package would require editing hash-frozen files.

```
osworld_parity/
  phaseb_oracle_eval.py            teacher-forced evaluator (oracle/gold-prefix history)
  phaseb_canonical_eval.py         cross-format semantic canonicalizer
  phaseb_deltatype_raw_v2/         the deltatype-v2 codec + Phase-B build/tokenize/train/eval
    action_v2.py                   grammar: parse / format / ordered_plan / dispatch   [FROZEN]
    prompt.py                      the pinned system prompt (model-facing)             [FROZEN]
    converter.py                   byte-preserving assistant action-span replacement   [FROZEN]
    readiness.py                   fail-closed response canonicality check             [FROZEN]
    build.py                       OSWorld trajectory → deltatype-v2 records+manifest  [FROZEN]
    vendor/action_span_conversion.py   vendored contract module                        [FROZEN]
    verify_sealed_dataset.py       equivalence proof against the sealed artifact
    tokenize_authorize.{py,sh}     chat.jsonl → omegalax grain records, sealed
    production_train_contract.py   preflight/finalize gate for the SFT job
    train_production_r256.sh       the Phase-B SFT invocation (hyperparameter source of truth)
    eval.sh                        vLLM boot + evaluator driver
    aggregate_train_eval.py        4-shard train-split aggregator
    labctl/recipes/                dispatch recipes
    tests/
  phaseb_relative/
    relative_eval.py               wire/transport base for the evaluator
    preflight_vision_budget.py     per-record image/patch budget gate
    server_readiness.py            vLLM server readiness/vision probe (used by eval.sh)
  synthetic_multistep/             Phase A
    build_episodes.py              frozen held-out episode specs + oracle prefixes
    build_curriculum_stage2.py     the compact corpus generator + 4-key overlap gate
    curriculum_train_export.sh     750-step A→B transfer + Orbax→HF export
    labctl/recipes/
  split/                           the hash-pinned OSWorld task splits                [FROZEN]
  tests/
eval/action_parser.py              the production action parser                       [FROZEN]
```

## Reproducing

Dispatch is **hai-login2 only**. Register this checkout as the labctl repo alias
`juergen_phaseb_deltatype_v2` (every recipe declares `repo =` that alias and
asserts it is running from the immutable `$LABCTL_RUN_DIR/source/…` snapshot).

The trainer, [`omegalax`](https://github.com/p-doom/omegalax), is a separate
repo. Recipes name it by alias, not by path — bind your own checkout once:

```bash
labctl register-external --alias omegalax_trainer --path <your-omegalax-checkout>
```

Phase A and Phase B both ran at omegalax
`b3f32c002998a1134c78845847a53ca9cc17fb10`.

```bash
# ---- Phase A ----------------------------------------------------------------
labctl run osworld_parity/synthetic_multistep/labctl/recipes/build_episodes.toml
labctl run osworld_parity/synthetic_multistep/labctl/recipes/build_curriculum_stage2_pinned.toml
labctl run osworld_parity/synthetic_multistep/labctl/recipes/train_curriculum_A_to_B_r256_pinned.toml

# ---- Phase B ----------------------------------------------------------------
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/build_audit_v1.toml
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/tokenize_authorize_production_r256_v1.toml
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/train_production_A_to_B_r256_s900_v1.toml

# ---- Evaluation (teacher-forced, oracle/gold-prefix history) ----------------
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/eval_continuation_A_to_B_r256_s900_v1.toml
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/eval_train_A_to_B_r256_s900_shard{0,1,2,3}_v1.toml
labctl run osworld_parity/phaseb_deltatype_raw_v2/labctl/recipes/aggregate_train_A_to_B_r256_s900_v1.toml
```

There is one evaluator, and it takes `--dataset-kind val|train` plus
`--shard-count/--shard-index`, so the held-out cell and the four train-split
cells are the *same* measurement code, byte-gated identically by all five
recipes. The four shard recipes are mechanically identical except for
`shard_index`, and they are four files on purpose: `labctl run` takes no
per-dispatch arg override, and `labctl run-sweep`'s `[sweep]` fan-out submits a
single SLURM array job as **one** labctl run with **one** output artifact, so all
four shards would write `report.json`/`rows.jsonl`/`eval_manifest.json` into the
same directory and collide. Collapsing them needs `eval.sh` to shard its own
output subdirectory and `aggregate_train_eval.py` to read one directory instead
of four artifacts — a change to working measurement code, deliberately not made
here.

No stage hard-pins a lineage digest. `tokenize_authorize.py` asserts the dataset
manifest's self-seal, the record schema and the action-span suffix invariant, and
records the digests it observes; *which* dataset and warm start are in play is
pinned once, in the recipe's `[inputs]`, and the sealed split digests are
asserted once, by `tests/test_full_source.py`. `production_train_contract.py`
likewise takes its shapes and LoRA geometry from `train_production_r256.sh` — the
single source of truth for the hyperparameters — and pins that script's SHA-256
into the artifact manifest. Both stages therefore re-dispatch unchanged against a
new dataset or warm start.

Every hardcoded scratch path is an environment variable whose default is the
value that executed: `JAX_CACHE_ROOT` (default `…/p-doom_shared/franz/jax_cache`),
`VLLM_CACHE_ROOT` (default `…/p-doom_shared/franz/tmp`), `CUDA_HOME`
(default `/fast/service/apps/software/CUDA/12.6.0`), `NCCL_PRELOAD` (default: the
runtime artifact's own `payload/native/libnccl.so.2`).

### Tests

```bash
/opt/miniforge3/bin/python osworld_parity/phaseb_deltatype_raw_v2/readiness.py
/opt/miniforge3/bin/python -m pytest -q osworld_parity/phaseb_deltatype_raw_v2/tests osworld_parity/tests
```

`/opt/miniforge3/bin/python` is the interpreter `build.py` pins by SHA-256
(3.12.9, conda-forge); it is asserted, so the build fails loudly on any other.
Tests that need cluster-local corpora skip when those are absent:
`PHASEB_SEALED_DATASET`, `PHASEB_AUDIT_OPERAND`, `PHASEB_ROLLOUTS`,
`PHASEB_ONPOLICY_SCRIPTS`.

## The `deltatype-v2` contract

The grammar the `s900` checkpoint emits. One bare action line, mouse values are
**raw pixel deltas** from the current cursor:

```
dx dy scroll [ ; ELEM ELEM …]
ELEM := +NAME | -NAME | type("JSON string") | MOVE(dx,dy)
```

`MOVE` is legal in exactly one shape — a left-button drag:

```
initial_dx initial_dy 0 ; +LMB MOVE(drag_dx,drag_dy) -LMB
```

Elements execute left to right. Special lines: `NO_OP`, `TERMINATE`, `FAIL`.
`readiness.validate_response` fails closed on any non-canonical rendering.

**These bytes are frozen.** `prompt.py` and `action_v2.py` are the model-facing
contract the `s900` weights were fit to; `tests/test_pinned_contract.py` asserts
their SHA-256 against the sealed dataset manifest. Changing them silently
invalidates the checkpoint.

| file | SHA-256 |
|---|---|
| `phaseb_deltatype_raw_v2/action_v2.py` | `1ded3d5a…3075ccb7` |
| `phaseb_deltatype_raw_v2/build.py` | `c3562ebe…c27e7c99` |
| `phaseb_deltatype_raw_v2/converter.py` | `7338b12a…cce5070e` |
| `phaseb_deltatype_raw_v2/prompt.py` | `c6c32ea2…c06c798072` |
| `phaseb_deltatype_raw_v2/readiness.py` | `4672752d…8a341f4408` |
| `phaseb_deltatype_raw_v2/vendor/action_span_conversion.py` | `65397c1d…02c84497` |
| `eval/action_parser.py` | `f916757d…6338caae4a9a` |
| `split/osworld_train.json` | `1a5cb5bf…6399700ae` |
| `split/osworld_eval_heldout.json` | `9bdb3e46…f0c89f8ba7e7c` |

### Equivalence proof

`verify_sealed_dataset.py` re-derives the model-facing bytes from the committed
codec and asserts identity against the sealed artifact
(`phaseb_raw_deltatype_v2_build_audit_v1_run_019fb5a5564e7a71b3ad6e55426af463`):

```
records                          2616
assistant_spans_roundtripped    10721   parse → format byte-identical, all spans
unique_decisions_plan_checked    2616   ordered_plan replayed from screen centre
tasks                             239
system_prompt_sha256           57f7d0b2…  identical in every record
train/chat.jsonl               5f449f3d…
val/chat.jsonl                 a819011d…
```

`tests/test_full_source.py` goes further: it re-runs the whole builder against
the committed codec, the vendored converter and the in-repo parser, and asserts
the rebuilt `train/chat.jsonl` and `val/chat.jsonl` SHA-256 equal those sealed
digests — a byte-identical rebuild from source.

## Results (teacher-forced, oracle/gold-prefix history, greedy T=0)

Not rollouts. `estimand = oracle_history_single_turn_greedy_generation`.

| metric | train (n=2383, 215 tasks) | held-out val (n=233) | Δ |
|---|---|---|---|
| parse rate | 1.000 | 1.000 | 0.000 |
| action-sequence agreement | 0.9979 | 0.9013 | +0.0966 |
| canonical exact-plan agreement | 0.9345 | 0.3562 | **+0.5783** |
| canonical tolerant 50 px | 0.9824 | 0.5622 | +0.4201 |
| canonical tolerant 100 px | 0.9912 | 0.6567 | +0.3345 |
| within 50 px | 0.9782 | 0.5337 | +0.4445 |
| median error (px) | 0.0 | 42.06 | −42.06 |

Train-split 95% task-cluster bootstrap (5000 replicates, seed 20260803, unit
`(app, task_id)`): exact-plan `[0.9226, 0.9460]`, within-50 px `[0.9706, 0.9850]`.
Results live in
`…/labctl/eval_logs/franz.srambical/phaseb_raw_deltatype_v2_eval_continuation_A_to_B_r256_s900_v1_run_019fbcddc0a17213a98735fe1a0d72a7`
(held-out) and `…/phaseb_raw_s900_train_tf_aggregate_v1_run_019fc744dedc750282b10019b0bb67b6`
(train). The near-total train fit against a much weaker held-out score is the
central Phase-B finding.

## Known limitation: `build.py` needs two out-of-repo contract modules

`build.py` loads `build_osworld_format_records.py` and
`convert_abs_to_deltatype.py` by path from
`…/p-doom_shared/franz/onpolicy_distill/scripts` (declared as an `external`
recipe input) and **asserts their SHA-256** — `28b5cbe1…` and `e9424d31…` — as
part of `EXPECTED_CONTRACT_SHA256`. Phase B uses only three symbols from that
surface (`parse_computer_use_tool_calls`, `deltatype_conv.action_to_label`,
`_COORD_ACTIONS`), but the pinned bytes of `build_osworld_format_records.py`
insert personal home directories on `sys.path` and read
`/fast/home/franz.srambical/osworld_parity_split/eval_system_prompt.txt` at
import time, so the builder cannot run for anyone but its author from a clean
checkout.

Vendoring the minimal three-symbol surface therefore **cannot** be done without
editing `build.py`, whose bytes are one of the five `implementation_sha256`
entries the sealed dataset manifest pins and the `s900` behaviour depends on.
That is a re-seal, not a patch:

1. vendor a self-contained converter under `vendor/`;
2. point `build.py` at it and refresh `EXPECTED_CONTRACT_SHA256`;
3. re-run `tests/test_full_source.py` and confirm the rebuilt splits still hash
   to `5f449f3d…` / `a819011d…` (they should: the three symbols are pure);
4. update `implementation_sha256` for `build.py` in the dataset manifest, in
   `tests/test_pinned_contract.py` and in the table above, in one commit that
   says exactly that.

Until then: no committed file names a personal path, and `--onpolicy-scripts`
points at a shared project directory, but a fresh `build_audit_v1` dispatch
still depends on those two out-of-repo files existing at those exact hashes.

### Other rough edges

1. **The evaluator needs `openai` in the runtime venv** (`relative_eval.py`
   imports it at module scope) — even the GPU-free aggregator. It is supplied by
   the labctl `runtime` environment artifact, not by this repo's `uv.lock`.
2. **The five codec modules import each other as flat siblings.** That is how the
   sealed build executed them and their bytes are frozen, so they must be run as
   scripts, or with their directory on `sys.path` (which `tests/conftest.py`
   does). There is no `__init__.py` in that directory by design.
3. **`build_episodes.py` loads `rung2_scene` from an external `--audit-dir`**
   (`…/franz/audit_operand`). Arg-derived, not hardcoded, but still an
   out-of-repo dependency of Phase A.
4. **The train-split shard recipes' NCCL preload** now takes the runtime
   artifact's own `payload/native/libnccl.so.2` instead of a personal scratch
   copy. Those shards ran against runtime **v1** (`artifact_866218c1d704e6f8`);
   if v1 does not ship that library they fail loudly on `test -f` and need
   `NCCL_PRELOAD` set. Unverified — no shard was re-dispatched after the change.
5. **`server_readiness.py` is the only executable in the eval chain that no
   recipe SHA-gates**, even though `eval.sh` invokes it.

## Not in this branch

* **Return-supervision remediation** — the intended next commit. It is validating
  on live training arm C; it lands here once that arm reports.
* **The Phase-A tool-call arm** (`relative_factorial`: the 2×2×2 factorial
  builder, its trainer and export scripts, its recipes) — resolved experiment,
  see [Entry point](#entry-point). Recoverable from `.lab/provenance/` patches of
  runs `019fb43ff2927ee093a50f2f8577c7db` and `019fb4beda3472b289ae60fc612c1cea`.
* **The older evaluator revision.** The held-out cell originally ran from a
  diverged branch (`phaseb_oracle_eval.py` `d4193c2a…`, `phaseb_canonical_eval.py`
  `08bad72b…`, `eval.sh` `51e5428a…`). Only the surviving revision is here, and
  the held-out recipe's hash gates point at it. The deltas were a `forwarddelete`
  keysym alias, the dataset-kind/sharding plumbing and the removal of a
  `PHASEB_EXPORT_STEPS` step map — none touch scoring or inference on the
  step-900 path, so the two cells stay comparable. Those exact older bytes are
  now reproducible **only** from the `.lab/provenance/` patches of run
  `019fbcddc0a17213a98735fe1a0d72a7`, not from this branch.
* **Comparison and ablation machinery** — the A→B vs B→B curriculum comparison,
  the lr-5e-5 rescue arm, `phaseb_normalized_v2`, `compact_scale_ablation`,
  `typing_prose_factorial`, Phase-A closed-loop evaluation
  (`evaluate.py`/`contract.py`/`metrics.py`/`compare.py`/`capacity.py`/
  `uncertainty.py`/`effects.py`) and their recipes.
* **One-shot custody, authorization and recovery scaffolding** —
  `conditional_exact_continuation.*`, `resume_contingency*`, `resume_trigger.py`,
  `authorize_*.py`, `custody_copy.py`, `final_trust_audit.py`,
  `storage_emergency_*`, `bind_eval_provenance.py`,
  `curriculum_export_recovery.sh`, the `RAW_*`/`SCHEDULER_EVIDENCE*`
  chronologies. These encode one job's slurm ids, deadlines,
  iterator/optimizer/RNG state and node allowlists.
* **`proper_vm_capability_ladder/`** — rung scaffolding, labelled non-promotable
  by its own handoff.
* **The sign-of-life v2 suite** — already on
  `origin/franz/sign-of-life-phaseb-compact-baseline-20260803`.
* Generated data and artifacts: rollout trees, tokenized grain records,
  checkpoints, `eval_logs`, images, venvs, caches, `__pycache__`.
