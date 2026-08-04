# Short-goal golden-trajectory ladder — runbook

Plan of record: `~/.claude/plans/iterative-herding-crescent.md`. Everything below is
copy-pasteable in order. Every step ends in a hard gate; a failed gate stops the ladder
(see §5 for what each failure means).

Arms (they differ ONLY in the move primitive):

| slug | `--arm` | system prompt id | stage_04 alias prefix |
|---|---|---|---|
| `oev4rel` | `ordered_events_v4_rel` | `shortgoal_oev4_rel_v1` | `shortgoal_oev4rel_…` |
| `oev4abs` | `ordered_events_v4_abs` | `shortgoal_oev4_abs_v1` | `shortgoal_oev4abs_…` |

## Preamble — source this in every shell

```bash
export WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-shortgoal
export LABCTL=/fast/project/HFMI_SynergyUnit/yll/slurm/dev/yll/berlin/labctl
export CLUSTER=$LABCTL/cluster.berlin.toml
export OMEGALAX=/fast/project/HFMI_SynergyUnit/yll/omegalax
export DS=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu
export REC=$DS/shortgoal_golden_stage_03_recordings_v1
export WORK=/fast/project/HFMI_SynergyUnit/yll/shortgoal_runs
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
mkdir -p "$REC" "$WORK"/{logs,rung0,rung1,rung2,rung3,exports,armcheck}
```

Facts the commands rely on: 25 templates x 6 seeds = 150 tasks; splits
train=105 / tier_a=21 / tier_b=24; `overfit1` = `term_touch__s00`; `overfit32` = 32 train
task ids; screen 1920x1080, model view 1280x720 q90; `Qwen/Qwen3-VL-4B-Instruct` is already
in `$HF_HOME` (all jobs run `HF_HUB_OFFLINE=1`).

Recipes: `$LABCTL/recipes/training/jobs/shortgoal/*.toml` (7). Deviations from the plan
table, deliberate and load-bearing:

- `batch_size = "4"` on every rung. `process_local_batch_size` requires
  `batch_size % (dp_size*fsdp_size) == 0`, and fsdp 4 is a locked full-FT flag, so 4 is the
  smallest legal global batch = **1 sample per device**. On the 1-record overfit-1 dataset the
  batch is that record four times, so the gradient is identical to batch 1.
- `lr_schedule = "linear"` is omegalax's name for *warmup then constant* — that is the plan's
  "1e-5 constant" for overfit-1. `wsd` rungs carry `lr_stable_fraction=0.7`, `lr_end_factor=0.0`.
- Every rung with a real val set validates on the **sighted** tier-A records, including the
  blank-image control, so the val curves of all three full-150 runs are directly comparable.
  Val loss is curiosity; the gates are in §4.
- **JAX-under-SLURM binds exactly one GPU per process.** `--gres=gpu:4` with `--ntasks=1` does
  NOT give one process 4 local devices — confirmed twice: a 1-task/4-gres job logged
  `local_device_count=1`, and forcing `fsdp_size=4` under it fails
  `ValueError: Mesh shape (1, 4, 1) does not match device_count=1`. A 4-way `fsdp_size` (or any
  mesh dim > 1) requires `--ntasks=N --ntasks-per-node=N` matching that dim, one task per GPU —
  this is why every recipe's `sbatch_extra` carries `--ntasks=4 --ntasks-per-node=4` (bare
  `srun`, no extra GPU-binding flags needed; Slurm binds one GPU per task by default).
- `overfit1` chat builds pass `--replicas=4` (`shortgoal_build.py`): independent of process
  topology, omegalax's grain pipeline computes `data_parallel_size = dp_size * fsdp_size = 4`
  from the device mesh, and `drop_remainder=True` sharding needs `>= data_parallel_size` records
  (`ValueError: Compiled Grain dataset has 1 records, too small to shard across
  data_parallel_size=4`) — so the fix for a 1-episode dataset is record count, not process
  topology. `--replicas=4` writes 4 copies of the same episode (`conversation_id` suffixed
  `__cNN`, `manifest.counts.replicas`), and the `overfit1` recipes keep their
  originally-authored `sbatch_extra` (`--ntasks=4 --ntasks-per-node=4`, matching every other
  rung) — gradient-identical to batch 1 either way, same reasoning as the `batch_size=4` note
  above.
- **Required `train_vlm_sft.py` flags** (`omegalax/scripts/train_vlm_sft.py::_REQUIRED`,
  validated at startup, missing any errors loudly before the run starts):
  `model_id, max_length, num_steps, batch_size, learning_rate, weight_decay, warmup_steps,
  lr_schedule, max_grad_norm, grad_accum_steps, gc_period, seed, tp_size, fsdp_size, dp_size,
  save_dir, jax_cache_dir, save_every, keep_period, keep_latest, log_every, log_memory, resume,
  pad_id, peak_tflops, grain_read_threads, grain_read_buffer_size, grain_workers,
  grain_worker_buffer_size, max_vision_patches_per_sample, max_vision_images_per_sample,
  num_loss_tiles, text_attn_backend, enable_lora, freeze_vision_tower` — plus `lora_rank`/
  `lora_alpha` whenever `enable_lora`/`freeze_vision_tower` full-FT metadata is written (every
  rung: `lora_rank=0 lora_alpha=0`, else `_write_lora_metadata` crashes on `int(None)`), and
  `lr_end_factor`/`lr_stable_fraction` for `wsd`/`cosine` schedules — never
  `lr_schedule=linear` with `warmup_steps=0` (that returns a bare `float`, not a schedule
  callable, and crashes at the first metrics log). If `wandb_project` is set (every recipe),
  set `WANDB_MODE=offline` in the job env — compute nodes have no egress for live W&B logging.

## 0. Record the 150 golden trajectories (KVM node)

### 0.1 Preflight (no VM, no GPU)

```bash
cd "$WT/eval" && uv run pytest -q test_shortgoal_grammar.py test_shortgoal_templates.py \
  test_shortgoal_build.py test_shortgoal_runtime.py test_shortgoal_contract.py
cd "$WT" && uv run --project=eval -- python eval/shortgoal_record.py \
  --output_dir="$WORK/dryrun" --dry_run
jq -e '.n_failed == 0 and .n_tasks == 150' "$WORK/dryrun/dry_run.json"
```

### 0.2 Pin the split manifest (one file every later step reads)

```bash
cd "$WT/eval" && SPLITS="$REC/splits.json" uv run -- python -c 'import json, os, shortgoal_templates as t; json.dump(t.build_split_manifest(), open(os.environ["SPLITS"], "w"), indent=2, sort_keys=True)'
jq -e '.n_tasks == 150 and .counts.train == 105 and .counts.tier_a == 21 and .counts.tier_b == 24' "$REC/splits.json"
```

### 0.3 Record all 150 (one job, one fresh VM per task, ~5 h)

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=shortgoal_record
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/shortgoal_runs/logs/record_%j.out
set -euo pipefail
WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-shortgoal
REC=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/shortgoal_golden_stage_03_recordings_v1
cd "$WT"
exec uv run --project=eval -- python eval/shortgoal_record.py --output_dir="$REC"
EOF
```

No GPU is needed (no model in the loop) but the node must expose `/dev/kvm` — every
`standard` node does. Ports are derived from `SLURM_JOB_ID` (`vm_port = 5000 + (JOB_ID%200)*10`,
`vnc = 5900 + …`), so concurrent recorder jobs do not collide.

Gate: `jq -e '.passed and .n_recorded == 150' "$REC/summary.json"`. Rejected tasks leave
`<task_id>/failure.json` (and no `recording.json`) — read `rejected_reason`, fix the template,
re-run only the stragglers with `--task_ids=a,b,c`.

Shard variant (6 x 25 tasks, ~50 min each) when wall clock matters — separate output roots
because each job writes its own `summary.json`, then fold the task dirs into `$REC`:

```bash
sbatch --array=0-5 --partition=standard --cpus-per-task=8 --mem=48G --time=04:00:00 --qos=low \
  --job-name=shortgoal_record_shard \
  --output="$WORK/logs/record_shard_%A_%a.out" --wrap \
  'cd '"$WT"' && uv run --project=eval -- python eval/shortgoal_record.py \
     --output_dir='"$WORK"'/record_shards/s$SLURM_ARRAY_TASK_ID --seeds=$SLURM_ARRAY_TASK_ID'
# after all 6 finish:
for d in "$WORK"/record_shards/s*/*__s*/; do cp -r "$d" "$REC/"; done
test "$(ls -d "$REC"/*/recording.json 2>/dev/null | wc -l)" = 150
```

### 0.4 Rung 0 gate — golden replay, no model, 2 repeats

```bash
for rep in 1 2; do
  sbatch --partition=standard --cpus-per-task=8 --mem=48G --time=12:00:00 --qos=low \
    --job-name="shortgoal_rung0_rep$rep" --output="$WORK/logs/rung0_rep${rep}_%j.out" --wrap \
    'cd '"$WT"' && uv run --project=eval -- python eval/shortgoal_record.py \
       --replay_from='"$REC"' --output_dir='"$WORK"'/rung0/rep'"$rep"
done
# gate (both repeats):
for rep in 1 2; do jq -e '.passed and .n_passed == 150' "$WORK/rung0/rep$rep/replay_summary.json"; done
```

**Gate: 150/150 in both repeats.** Any flake is template non-determinism, not the model —
`jq '[.tasks[]|select(.passed==false)|{task_id,reason,max_cursor_drift_px}]'` on
`replay_summary.json`, fix determinism, re-record, re-gate. Do not proceed on 149/150.

## 1. Build the chat subsets (both arms + blank-image control)

```bash
cd "$WT"
for pair in oev4rel:ordered_events_v4_rel oev4abs:ordered_events_v4_abs; do
  slug=${pair%%:*}; arm=${pair#*:}
  for subset in overfit1 overfit32 full tiera_val tierb_val; do
    uv run --project=eval -- python eval/shortgoal_build.py \
      --recordings_root="$REC" \
      --output_dir="$DS/shortgoal_${slug}_${subset}_stage_04_chat_v1" \
      --arm="$arm" --subset="$subset" --splits="$REC/splits.json" \
      --model_resolution=1280x720 --jpeg_quality=90
  done
done
uv run --project=eval -- python eval/shortgoal_build.py \
  --recordings_root="$REC" \
  --output_dir="$DS/shortgoal_oev4rel_full_blankimg_stage_04_chat_v1" \
  --arm=ordered_events_v4_rel --subset=full --splits="$REC/splits.json" \
  --model_resolution=1280x720 --jpeg_quality=90 --blank_images
```

Cross-arm identity check (the builder compares `<dir>/oev4rel` vs `<dir>/oev4abs`, so point
symlinks at the two alias dirs and use `--check_only`):

```bash
for subset in overfit1 overfit32 full tiera_val tierb_val; do
  mkdir -p "$WORK/armcheck/$subset"
  for slug in oev4rel oev4abs; do
    ln -sfn "$DS/shortgoal_${slug}_${subset}_stage_04_chat_v1" "$WORK/armcheck/$subset/$slug"
  done
  cd "$WT" && uv run --project=eval -- python eval/shortgoal_build.py \
    --recordings_root="$REC" --output_dir="$WORK/armcheck/$subset" \
    --arm=both --subset="$subset" --splits="$REC/splits.json" --check_only
done
```

Gates: each build prints `<n> records from <n> tasks` — expect tasks
1 / 32 / 105 / 21 / 24; `--check_only` must report the arms identical after masking move
tokens; per-dataset invariants restated outside the builder's own assertions:

```bash
for d in "$DS"/shortgoal_oev4*_stage_04_chat_v1; do
  jq -e '.counts.max_live_images <= 6 and .counts.n_resupervised_turns == 0' "$d/manifest.json" >/dev/null \
    || echo "CHECK $d"
done
```

## 2. Stages 05 + 06 (unmodified, finevision precedent)

`tierb_val` gets a chat only — it is never trained on and never a val set, so it needs no
records. That leaves 9 record datasets.

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=shortgoal_stage0506
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/shortgoal_runs/logs/stage0506_%j.out
set -euo pipefail
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-shortgoal
OMEGALAX=/fast/project/HFMI_SynergyUnit/yll/omegalax
DS=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu
cd "$WT/data_pipeline"
for name in oev4rel_overfit1 oev4rel_overfit32 oev4rel_full oev4rel_full_blankimg \
            oev4rel_tiera_val oev4abs_overfit1 oev4abs_overfit32 oev4abs_full \
            oev4abs_tiera_val; do
  chat="$DS/shortgoal_${name}_stage_04_chat_v1"
  meas="$DS/shortgoal_${name}_stage_05_measure_qwen3vl4b_instruct_v1"
  recs="$DS/shortgoal_${name}_stage_06_records_len8192_v1"
  uv run --locked -- python realigned_pipeline/stage_05_measure_lengths.py \
    --source_path="$chat" --chat_relpath=train/chat.jsonl --output_dir="$meas" \
    --model_id=Qwen/Qwen3-VL-4B-Instruct --processor=Qwen/Qwen3-VL-4B-Instruct \
    --num_workers=16 --omegalax_repo="$OMEGALAX"
  uv run --locked -- python realigned_pipeline/stage_06_training_records.py \
    --source_path="$chat" --chat_relpath=train/chat.jsonl \
    --message_lengths_path="$meas" --output_dir="$recs" \
    --model_id=Qwen/Qwen3-VL-4B-Instruct --processor=Qwen/Qwen3-VL-4B-Instruct \
    --max_length=8192 --overflow_mode=split --records_per_shard=10000 \
    --num_workers=16 --val_fraction=0 --omegalax_repo="$OMEGALAX"
  jq -e '.sessions.truncated_total == 0 and .sessions.split_into_multiple_chunks == 0 and .messages.dropped == 0' \
    "$recs/train/truncation_stats.json"
done
EOF
```

**Gate: zero overflow.** The builder already cut records at eviction points, so any
`split_into_multiple_chunks > 0` means a record does not fit 8192 — fix the builder, not
`--max_length`. Training reads `<recs>/train`.

## 3. Register the artifacts, then launch the 7 trainings

labctl snapshots **committed** source: commit `omegalax` and the `slurm` tree before running.
The juergen worktree is not a labctl repo and is not consumed by these recipes.

```bash
labctl --cluster "$CLUSTER" register-external --alias shortgoal_golden_stage_03_recordings_v1 \
  --path "$REC" --kind dataset
for name in oev4rel_overfit1 oev4rel_overfit32 oev4rel_full oev4rel_full_blankimg \
            oev4rel_tiera_val oev4rel_tierb_val oev4abs_overfit1 oev4abs_overfit32 \
            oev4abs_full oev4abs_tiera_val oev4abs_tierb_val; do
  for suffix in stage_04_chat_v1 stage_05_measure_qwen3vl4b_instruct_v1 stage_06_records_len8192_v1; do
    p="$DS/shortgoal_${name}_${suffix}"
    [ -d "$p" ] && labctl --cluster "$CLUSTER" register-external \
      --alias "shortgoal_${name}_${suffix}" --path "$p" --kind dataset
  done
done
```

`register-external` is idempotent (artifact id = sha256 of the canonical path; the alias
upserts). The path must already be `.../labctl/datasets/<user>/<alias>` — steps 1–2 write
there for exactly that reason.

```bash
cd "$LABCTL"
for f in recipes/training/jobs/shortgoal/*.toml; do labctl --cluster "$CLUSTER" validate "$f"; done
```

Launch in ladder order, gating between rungs (§4):

```bash
R=recipes/training/jobs/shortgoal
# rung 1
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4rel_overfit1_fullft.toml
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4abs_overfit1_fullft.toml
# rung 2
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4rel_overfit32_fullft.toml
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4abs_overfit32_fullft.toml
# rung 3 + control
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4rel_full150_fullft.toml
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4abs_full150_fullft.toml
labctl --cluster "$CLUSTER" run $R/qwen3vl4b_shortgoal_oev4rel_full150_blankimg_fullft.toml
labctl --cluster "$CLUSTER" status
```

Rungs: overfit-1 300 steps @ lr 1e-5 linear; overfit-32 1500 @ 5e-6 wsd; full-150 4000 @ 5e-6
wsd, grad_accum 2. Checkpoints land in
`/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/yll.kryeziu/<recipe.name>_<run.id>/<step>`.
Sanity while running: train loss must fall toward ~0 on the overfit rungs; a flat or NaN
loss is an optimization/plumbing failure before any eval is worth running.

## 4. Evaluate — export, then both modes per rung

### 4.1 Export a step to HF and complete the sidecars

```bash
CKPT=<checkpoints>/qwen3vl4b_shortgoal_oev4rel_overfit1_fullft_<run.id>/300
EXPORT=$WORK/exports/oev4rel_overfit1_300
sbatch --partition=standard --gres=gpu:1 --cpus-per-task=16 --mem=192G --time=01:00:00 \
  --qos=low --job-name=shortgoal_export --output="$WORK/logs/export_%j.out" --wrap \
  'export JAX_PLATFORMS=cpu; cd '"$OMEGALAX"' && uv run --project='"$OMEGALAX"' -- python \
     scripts/export_to_hf.py --model_id=Qwen/Qwen3-VL-4B-Instruct --out_dir='"$EXPORT"' \
     --checkpoint_path='"$CKPT"' --tp_size=1 --fsdp_size=1 --dp_size=1 \
     --max_grad_norm=1.0 --grad_accum_steps=1'
cd "$WT/eval" && EXPORT="$EXPORT" uv run -- python -c 'import os; from pathlib import Path; from hf_complete import complete_export_dir, find_hf_snapshot; print(complete_export_dir(Path(os.environ["EXPORT"]), find_hf_snapshot("Qwen/Qwen3-VL-4B-Instruct", Path(os.environ["HF_HOME"]))))'
```

Repeat per rung/arm (`oev4abs_overfit1_300`, `*_overfit32_1500`, `*_full150_<step>`,
`*_full150_blankimg_<step>`). Harness sanity first — no GPU, no network, no VM:

```bash
cd "$WT" && uv run --project=eval -- python eval/shortgoal_eval.py \
  --self_check --output_dir="$WORK/rung1/self_check"
```

### 4.2 One submit helper for every eval below

```bash
sg_eval() {   # sg_eval <job-name> <walltime> <mem-fraction> <out-dir> <eval flags…>
  local job=$1 time=$2 mf=$3 out=$4; shift 4
  sbatch --partition=standard --gres=gpu:1 --cpus-per-task=8 --mem=96G --time="$time" \
    --qos=low --job-name="$job" --output="$WORK/logs/${job}_%j.out" --wrap \
    "cd $WT && uv run --project=eval -- python eval/shortgoal_eval.py \
       --output_dir=$out --splits=$REC/splits.json --recordings_root=$REC \
       --model_resolution=1280x720 --jpeg_quality=90 --max_steps=12 \
       --mem_fraction_static=$mf --context_length=16384 $*"
}
```

`offline_exact` needs a GPU but no VM (`mf` 0.80); `closed_loop` shares the node with qemu
(`mf` 0.55). `--attempts=1` decodes greedily; `--attempts=4` samples the Qwen Instruct tuple
under seed `sha256("<task_id>:<attempt>")` and records tuple + provenance in `result.json`.

### 4.3 Rung 1 — overfit-1 (both arms, greedy)

```bash
for pair in oev4rel:ordered_events_v4_rel oev4abs:ordered_events_v4_abs; do
  S=${pair%%:*}; A=${pair#*:}; E=$WORK/exports/${S}_overfit1_300
  sg_eval "sg_r1a_$S" 02:00:00 0.80 "$WORK/rung1/${S}_offline" \
    --mode=offline_exact --arm="$A" --model_path="$E" \
    --chat="$DS/shortgoal_${S}_overfit1_stage_04_chat_v1/train/chat.jsonl"
  sg_eval "sg_r1b_$S" 04:00:00 0.55 "$WORK/rung1/${S}_closed" \
    --mode=closed_loop --arm="$A" --model_path="$E" --subset=overfit1 --attempts=1
done
# gates, per arm:
for S in oev4rel oev4abs; do
  jq -e '.scores.exact_line_rate == 1.0 and .scores.parse_valid_rate == 1.0' \
    "$WORK/rung1/${S}_offline/result.json"
  jq -e '.scores.pass_at_1 == 1.0' "$WORK/rung1/${S}_closed/result.json"
done
```

**Gates.** (a) 100% byte-exact lines including the whole-line `TERMINATE` — anything less is a
plumbing bug: abort and fix (§5.2). (b) only meaningful once (a) is green; a failure there is
denorm/env (§5.3). Run (b) only after (a) passes. Arms gate independently.

### 4.4 Rung 2 — overfit-32 (both arms, greedy)

```bash
for pair in oev4rel:ordered_events_v4_rel oev4abs:ordered_events_v4_abs; do
  S=${pair%%:*}; A=${pair#*:}; E=$WORK/exports/${S}_overfit32_1500
  sg_eval "sg_r2a_$S" 04:00:00 0.80 "$WORK/rung2/${S}_offline" \
    --mode=offline_exact --arm="$A" --model_path="$E" \
    --chat="$DS/shortgoal_${S}_overfit32_stage_04_chat_v1/train/chat.jsonl"
  sg_eval "sg_r2b_$S" 08:00:00 0.55 "$WORK/rung2/${S}_closed" \
    --mode=closed_loop --arm="$A" --model_path="$E" --subset=overfit32 --attempts=1
done
# gates, per arm:
for S in oev4rel oev4abs; do
  jq -e '.scores.exact_line_rate >= 0.95 and .scores.parse_valid_rate == 1.0' \
    "$WORK/rung2/${S}_offline/result.json"
  jq -e '.scores.pass_at_1 >= 0.90625' "$WORK/rung2/${S}_closed/result.json"
  jq -e '[.scores.by_category[].pass_at_1] | min > 0' "$WORK/rung2/${S}_closed/result.json"
done
```

**Gates: offline exact ≥0.95, strict parse = 1.0, closed loop ≥29/32 (0.90625), no category
at 0.**

### 4.5 Rung 3 — full-150 (the measurement, one gate)

```bash
for pair in oev4rel:ordered_events_v4_rel oev4abs:ordered_events_v4_abs; do
  S=${pair%%:*}; A=${pair#*:}; E=$WORK/exports/${S}_full150_4000
  sg_eval "sg_r3_train_$S" 1-00:00:00 0.55 "$WORK/rung3/${S}_train" \
    --mode=closed_loop --arm="$A" --model_path="$E" --subset=full --attempts=1
  for tier in tier_a tier_b; do
    sg_eval "sg_r3_${tier}_p1_$S" 08:00:00 0.55 "$WORK/rung3/${S}_${tier}_p1" \
      --mode=closed_loop --arm="$A" --model_path="$E" --split="$tier" --attempts=1
    sg_eval "sg_r3_${tier}_p4_$S" 1-00:00:00 0.55 "$WORK/rung3/${S}_${tier}_p4" \
      --mode=closed_loop --arm="$A" --model_path="$E" --split="$tier" --attempts=4 \
      --sampling_mode=instruct --presence_penalty=0
  done
  for subset in full tiera_val tierb_val; do
    sg_eval "sg_r3_off_${subset}_$S" 06:00:00 0.80 "$WORK/rung3/${S}_off_${subset}" \
      --mode=offline_exact --arm="$A" --model_path="$E" \
      --chat="$DS/shortgoal_${S}_${subset}_stage_04_chat_v1/train/chat.jsonl"
  done
done
# blank-image control (rel arm), evaluated on the SAME sighted tiers:
E=$WORK/exports/oev4rel_full150_blankimg_4000
for tier in tier_a tier_b; do
  sg_eval "sg_r3_blank_${tier}_p1" 08:00:00 0.55 "$WORK/rung3/oev4rel_blank_${tier}_p1" \
    --mode=closed_loop --arm=ordered_events_v4_rel --model_path="$E" --split="$tier" --attempts=1
done
```

Budget: 105 boots ≈ 5 h, 24 tasks x 4 attempts ≈ 5 h. Only gate here:

```bash
for S in oev4rel oev4abs; do jq -e '.scores.pass_at_1 >= 0.9' "$WORK/rung3/${S}_train/result.json"; done
```

Everything else is read, not gated:

```bash
for d in "$WORK"/rung3/*/; do
  jq -r --arg d "$d" '[$d, (.scores.pass_at_1|tostring), (.scores.pass_at_k|tostring),
    (.scores.never_terminate_rate|tostring)] | @tsv' "$d/result.json"
done
jq -r '.scores.by_category | to_entries[] | "\(.key)\t\(.value.pass_at_1)"' \
  "$WORK/rung3/oev4rel_tier_a_p1/result.json"
```

Blank-image control (rel arm, mandatory): it trains on `full_blankimg` but is evaluated on the
**same sighted** tiers as the main rel run, so any surviving grounding skill cannot have come
from pixels. Compare per category:

```bash
join -t $'\t' \
  <(jq -r '.scores.by_category|to_entries[]|"\(.key)\t\(.value.pass_at_1)"' "$WORK/rung3/oev4rel_tier_a_p1/result.json"|sort) \
  <(jq -r '.scores.by_category|to_entries[]|"\(.key)\t\(.value.pass_at_1)"' "$WORK/rung3/oev4rel_blank_tier_a_p1/result.json"|sort)
```

Expected: `fixture`/`browser` (grounding) collapse while `terminal`/`editor` (typing) survive.
If sighted ≈ blanked on the grounding categories, the model is not using vision — flag loudly
and stop treating the rel/abs comparison as meaningful.

## 5. Decision tree — what a failed gate localizes

1. **Rung 0 replay flake (any of 150 fails, either repeat) → template/env determinism.**
   Not the model, not the format. Read `replay_summary.json` `reason` +
   `max_cursor_drift_px`; suspects in order: fixture launch race (raise `--settle_s` /
   `--settle_stable_timeout_s`), a verifier that reads state before the app flushes
   (`--verify_timeout_s`), guest-app drift (gedit vs gnome-text-editor, chromium vs
   google-chrome), seeded geometry landing off-screen. Fix the template, re-record that
   task, re-gate. Never train on a corpus that replays at 149/150.
2. **Rung 1(a) offline exact < 1.0 → DATA PLUMBING.** The model was trained on 300 steps of
   one record; if it cannot reproduce that record's lines byte-for-byte under teacher
   forcing, the eval prompt is not the training prompt. Check, in this order: system-prompt
   bytes (`params.system_prompt_id` in `result.json` vs the arm's registered prompt file vs
   the builder `manifest.json`), chat-template application, `--model_resolution` /
   `--jpeg_quality` matching the builder's 1280x720 q90, image liveness/placeholder
   positions (keep-text K=6 assembly), and record cut points. `--self_check` and
   `test_shortgoal_contract.py` are the byte-identity oracles; `.worst_examples` in
   `result.json` shows golden vs predicted per turn. Abort the ladder until green.
3. **Rung 1(a) = 1.0 but 1(b) fails → ENV / denormalization, not data.** The bytes are right,
   the effect is wrong. Look at `$WORK/rung1/<slug>_closed/<task>/attempt_00/`:
   `conversation.jsonl`, `prompt_*.json`, `step_*.png`, `rollout.gif`. Read the episode row:
   `stop_reason=max_steps` + `never_terminate` → the model never emits `TERMINATE`;
   `spurious_terminate` → it terminates before success; `failed_steps` with
   `tolerant_rescue > 0` → strict-parse regression; `blind_history_steps > 0` → placeholder
   drift. Success-with-wrong-pixels means `denorm_v4` / `move_to` dispatch / grid snapping /
   cursor start; verifier-only failure (`verifier_error`) means the guest check, not the model.
4. **Rung 2 offline < 0.95 while rung 1(a) was 1.0 → OPTIMIZATION, not plumbing.** 32 records
   is a capacity/steps/lr question: check the train-loss curve first (still descending at
   1500 → raise `num_steps`; plateaued high → raise `learning_rate` or, per the plan's first
   knob, unfreeze the vision tower — note that `freeze_vision_tower` and `enable_lora` are
   mutually exclusive so only flip the former). Do not re-audit the pipeline; rung 1 already
   proved the bytes.
5. **Rung 2 offline ≥0.95 but closed loop < 29/32, or a category at 0 → per-category
   grounding/execution.** A single category at 0 with the others green points at that
   category's targets (fixture geometry, page px vs screen px in the browser templates, key
   names in the editor templates), not at the format.
6. **One arm passes and the other fails at the same rung → COORDINATE FRAME.** That is the
   experiment's signal, not a bug to route around: the arms are line-identical modulo the
   move token (step 1 checked this), the same prompt scaffold, the same records, so a
   rel-only failure isolates to the relative move primitive — thousandths-of-screen rendering,
   `denorm_v4` rounding, cursor-position dependence, or mickeys-vs-pixels at dispatch.
   Report per-arm and keep both arms running.
7. **Blank-image control ≈ sighted on grounding categories → the model is not using vision.**
   Then no rel/abs conclusion holds. Look for the answer leaking into text (instruction
   wording, seeded params visible in the GOAL line, targets constant across seeds) before
   touching the model.
8. **`parse_valid_rate < 1.0` anywhere → grammar regression.** Compare
   `.kind_confusion` / `.golden_kind_totals` (offline) or `tolerant_rescue_rate` (closed) and
   re-run `test_shortgoal_grammar.py`; a strict-parse failure that the tolerant twin rescues
   is a formatting drift the model learned from the data, not a parser bug.

## Appendix — cluster migration checklist (written for HoreKA, generalizes)

1. Workspace layout: `$WS/{repos,datasets,runs/logs,huggingface}`; rsync in
   `shortgoal_golden_stage_03_recordings_v1/` (322M, includes splits.json) and, if absent
   on the target, `osworld_vm/` + `qemu/` (34G; the Nix-wrapped qemu may not run off-host —
   fall back to the system `qemu-system-x86_64` via `--qemu_bin`).
2. Clone juergen (branch `yll/shortgoal-golden-ladder`), slurm (recipes as templates),
   omegalax; `uv sync` in juergen/eval, juergen/data_pipeline, omegalax; prefetch
   `Qwen/Qwen3-VL-4B-Instruct` into `$WS/huggingface` (`HF_HOME`).
3. Discover SLURM facts: partition/qos names, gres syntax, and CRITICALLY whether GPU
   nodes expose `/dev/kvm` (closed-loop eval needs GPU+KVM on one node; if not, the eval
   needs a split sglang/VM design).
4. Gates in order, each blocking the next: unit tests (eval ~363 pass; data_pipeline
   legacy identity suites) -> single-task VM replay smoke (drift 0) -> dataset rebuild
   from recordings (NEVER rsync built datasets: chat.jsonl/records embed absolute image
   paths) with `--replicas=4` for overfit1, zero-overflow gate -> 3-step train smoke
   (keep --ntasks=4 geometry + required-flags list; adjust partition, peak_tflops, and
   possibly text_attn_backend on non-Hopper GPUs) -> rung-1 retrain + both eval gates.
5. Only after rung-1 re-passes is the new cluster equivalent; continue at the next rung.
