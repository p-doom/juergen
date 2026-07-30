# Synthetic relative 2x2x2 factorial

This experiment completes rung 3 without retraining its four absolute cells.
It adds four relative cells over the exact `audit_operand/r3data_2k` records,
scenes, image paths, and order:

| cell | absolute source twin | grammar | assistant form |
|---|---|---|---|
| `reltool_act` | `abstool_act` | normalized-delta `move_rel` | action only |
| `relraw_act` | `absraw_act` | pixel-delta `deltatype_raw` | action only |
| `reltool_pre` | `abstool_pre` | normalized-delta `move_rel` | preamble |
| `relraw_pre` | `absraw_pre` | pixel-delta `deltatype_raw` | preamble |

`build_relative.py` uses `audit_operand/action_span_conversion.py` for every
assistant conversion and asserts prefix/suffix byte identity. System and user
text intentionally change to the exact relative prompts imported from
`audit_operand/rung2_scene.py`. The fail-loud invariant report covers converter
7/7, 2000/200 counts, geometry leak 0/0, exact record/image/order matching,
assistant outside-action identity, prose identity, digit leak 0, preamble-twin
action identity, gold parse-and-land, and train/eval prompt equality.

## 18-stage labctl pipeline

```text
build_tokenize
  ├─ train_reltool_act ─ eval_reltool_act ─┐
  ├─ train_relraw_act  ─ eval_relraw_act  ─┤
  ├─ train_reltool_pre ─ eval_reltool_pre ─┤
  └─ train_relraw_pre  ─ eval_relraw_pre  ─┤
                                           ├─ effects
abs abstool_act 000750 ─ export ─ eval ────┤
abs absraw_act  000750 ─ export ─ eval ────┤
abs abstool_pre 000750 ─ export ─ eval ────┤
abs absraw_pre  000750 ─ export ─ eval ────┘
```

The four absolute training jobs already produced intact Orbax `000750`
checkpoints but their original inline exports failed because
`export_to_hf.py` called `jax.distributed.initialize()` outside an `srun` job
step. `export_checkpoint.sh` is now the single export implementation for both
relative train/export and the four CPU absolute-recovery stages. It uses the
known-working ordering:

```bash
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py ...
```

The common exporter validates the Orbax marker and LoRA metadata, copies and
checks tokenizer/runtime files, repairs and checks `architectures`, verifies HF
weights, and writes the manifest last with the source checkpoint, Omegalax
commit/diff hash, and `export_ran_inside_srun=true`. Absolute eval stages consume
the corresponding export-stage `/hf` artifact; they no longer depend on
untracked `_hf` directories.

Each relative training stage is one GPU, LoRA r32/alpha32, 750 steps,
max-length 4096, `num_loss_tiles=8`, and `val_steps=15`. Existing GPU recipes
retain their `hai003,hai004,hai007` nodelist pins. Export stages are CPU-only.
All stages use 8 hours, low QoS, requeue, and project-backed outputs/caches.

Evaluation is generic across all four rung-2 grammars, greedy temperature 0,
k=1, seed 0, requires a real chat completion, runs the known-answer self-test,
and aborts on any request error. The effect stage consumes eight reports using
`+1 = relative/tool-call/preamble`, `-1 = absolute/bare/action-only`; an effect
is the positive-product mean minus the negative-product mean.

All experiment-script recipes use repo key `juergen_rft`; Omegalax is an
explicit external runtime input. No job is submitted by this code.

## Preserved pre-registration

- Managing-agent prediction: grammar dominates, but preamble rescues bare-token,
  producing a strong positive grammar-by-preamble interaction.
- Ladder-owner prediction: grammar main effect at least 40 points, preamble at
  most 10, interaction at most 10; `absraw_act` at least 50% in-box and absolute
  tool cells at least 90%.
- The easy one-box scene is a weak preamble test; a small interaction bounds the
  effect here rather than superseding Phase B.
