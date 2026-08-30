# On-policy rejection-sampling — round 0 runbook

Scope: one full round of on-policy collection → acceptance filter → SFT anneal → matched evaluation for the oev3 parity project (plan: `~/.claude/plans/virtual-beaming-waffle.md`, Phase 4). All dispatch goes through labctl; nothing in this runbook is auto-dispatched.

Paths used throughout:

```
WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-oev3-cuagym
LAB=/fast/project/HFMI_SynergyUnit/yll/slurm/dev/yll/berlin/labctl
DS=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu
EV=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/eval_logs/yll.kryeziu
CK=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/yll.kryeziu
OMEGALAX=/fast/project/HFMI_SynergyUnit/yll/omegalax
```

New recipes (validated, not dispatched):

- `$LAB/recipes/datagen/onpolicy_round0_collect.toml` — export + serve + k-sample rollout collection, swept over 24 disjoint task shards.
- `$LAB/recipes/training/jobs/oev3_parity/qwen35_9b_lora_success_onpolicy_anneal.toml` — WSD decay-heavy LoRA anneal on a 0.35/0.65 on-policy/original mix.

## 0. Prerequisites (hard gates, all three before step 1)

1. **Strong checkpoint chosen by matched pass@4**, never pass@1 (pass@1 on 38 tasks is a trigger, not a measurement). Candidates come from the post-vision-fix runs only (lora_success `run_01a0487af31a79b29aad20ee74d062a8`, fullft_drills `run_01a0487b112076d28d2b4dce6f5cea68`); the collect recipe hard-refuses pre-fix run hashes. Record the chosen run + step + artifact id here before proceeding.
2. **Worktree committed.** labctl snapshots committed source; `$WT` currently carries uncommitted eval fixes (FAIL terminal, truncated-think retry, `--retry_on_env_error`) plus the four in-flight components (`eval/cuagym_rollout_runner.py`, `data_pipeline/cuagym_pipeline/onpolicy_curriculum.py`, `data_pipeline/cuagym_pipeline/onpolicy_accept.py`, `data_pipeline/cuagym_pipeline/stage_04o_onpolicy_conversations.py`). Commit them all on the worktree branch first.
3. **Curriculum built** (step 1 below) with the OSWorld-369 contamination blocklist actually populated. As of 2026-08-29, `$DS/cuagym_osworld_blocklist_v1.json` is an EMPTY list (`[]`) — round 0 must not start until the curriculum enforces a real blocklist (franz `stage5_rft` blocklists are the reference).

Contract assumptions to confirm against the in-flight components before dispatch:

- `round0_tasks.jsonl` rows carry `app_family` and `task_id` — the collect recipe derives its output tree `<out>/<app_family>/<task_id>/sample_N/` from exactly these two fields.
- `cuagym_rollout_runner.py` accepts, beyond the agreed `--tasks_file/--task_index/--sample_index/--k/--temperature/--max_steps/--dry_run`, the serving flags the recipe passes (`--base_output_dir`, `--model_path`, `--served_model_name`, `--sglang_port`, `--sglang_api_key`, `--retry_on_env_error`) and reads `SGLANG_URL`/`OSWORLD_*` env, mirroring `eval/osworld_oev3_kvm.py`.
- Each finished sample writes `sample_N/result.json` with `scores.reward` (resume + manifest counting key off this).

## 1. Build the curriculum

CPU job (never on the login node):

```
cd $WT && uv sync --all-packages
mkdir -p $DS/cuagym_onpolicy_round0_stage_00_tasks_v1
srun --qos=low --cpus-per-task=8 --mem=32G --time=01:00:00 bash -c '
  cd '$WT'/data_pipeline
  uv run --no-sync -- python cuagym_pipeline/onpolicy_curriculum.py \
    --trajectories=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/datasets--p-doom--cuagym-qwen35-rollouts/snapshots/ac6a484bd3e9dbe232163e200e80613cc99146b2/p2_9b_think/trajectories.jsonl \
    --blocklist='$DS'/cuagym_osworld_blocklist_v1.json \
    --output_dir='$DS'/cuagym_onpolicy_round0_stage_00_tasks_v1
'
```

(Flag names per `onpolicy_curriculum.py`; adjust if the landed CLI differs.) Emits `round0_tasks.jsonl` (target ~600 tasks) + `curriculum_report.md`. Read the report: tier mix, blocklist hits, per-app_family counts. Sanity: `wc -l round0_tasks.jsonl` and confirm zero OSWorld-369 overlaps.

## 2. Register the tasks file

```
cd $LAB
labctl register-external \
  --alias cuagym_onpolicy_round0_stage_00_tasks_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_00_tasks_v1 \
  --kind dataset
```

The collect recipe's `[inputs.tasks]` already points at this path and appends `/round0_tasks.jsonl`.

## 3. Dispatch collection

Set the policy checkpoint. The recipe ships with a clearly marked placeholder:

```
# PLACEHOLDER: pin the round-0 policy checkpoint here (...)
[inputs.checkpoint]
type     = "artifact"
artifact = "artifact_2417f65f1411cad8"
```

Resolve the real artifact id with the `labctl show` trick: `labctl show run_<training run>` lists every checkpoint output with its `artifact_*` id and step (e.g. the lora_success run's outputs include `artifact_2417f65f1411cad8` = step 010000); `labctl show run_<eval run>` shows in `inputs[].artifact_id` which checkpoint a given eval consumed — use the eval run that produced the winning matched pass@4 to read off the exact artifact. Edit the `artifact =` line to that id.

Preflight (compute node, no VM boot):

```
srun --qos=low --cpus-per-task=4 --mem=16G --time=00:20:00 bash -c '
  cd '$WT'
  .venv/bin/python eval/cuagym_rollout_runner.py \
    --tasks_file '$DS'/cuagym_onpolicy_round0_stage_00_tasks_v1/round0_tasks.jsonl \
    --task_index 0 --sample_index 0 --k 16 --temperature 0.8 --max_steps 80 --dry_run
'
```

Then dispatch the sweep (24 shard jobs, 1 GPU / 24 h each, qos=low, hai001/hai002/hai008 excluded):

```
cd $LAB
labctl run-sweep recipes/datagen/onpolicy_round0_collect.toml
```

Each shard covers task indices `shard, shard+24, ...` for all k=16 samples at temperature 0.8, max_steps 80; shards are disjoint by construction so no cross-job claims are needed. Rollouts land per shard under `$EV/onpolicy_round0_rollouts_stage_r0_<artifact>_<run>/<app_family>/<task_id>/sample_N/`; each shard finalizes a `manifest.json` (marker) with `n_results`, `n_solved_geq_0999`, `n_null_reward`. `OEV3_PROMPT_FAIL` is gated OFF (`"0"` in `[env]`); flip to `"1"` only for a deliberate FAIL-prompt A/B round.

## 4. Monitor + go/no-go gate

Monitor:

```
squeue -u $USER -o "%i %j %T %M"
ls -d $EV/onpolicy_round0_rollouts_stage_r0_* \
  | xargs -I{} sh -c 'echo {} $(find {} -name result.json | wc -l)'
```

Early preflight on rewards (first ~20 results of the first shard): the in-guest `reward.py` dependency issue makes ~5% null rewards expected; if the null fraction is far above that, `scancel` the sweep and fix the guest image before burning GPU-hours.

**Go/no-go gate.** Once ~50 distinct tasks have completed sample sets (≈800 result.json across shards), run the acceptance filter in report-only mode over the partial trees:

```
cd $WT/data_pipeline
uv run --no-sync -- python cuagym_pipeline/onpolicy_accept.py \
  --rollout_roots="$EV/onpolicy_round0_rollouts_stage_r0_*" \
  --output_dir=/var/tmp/onpolicy_r0_gate_check
```

GO if acceptance rate on the **persist-verified tier** is >= 8% (SOLVED means reward >= 0.999 — see risks). Otherwise STOP: `scancel` all remaining sweep jobs (the collect recipe does not set `--requeue`, a plain scancel suffices) and invoke the fallback:

> **Prefix-restart suffix-resampling (fallback — NOT YET BUILT).** Instead of resampling whole episodes from the initial state, take failed rollouts, truncate each at the last verified-good prefix step, restore the VM to that point by deterministically replaying the prefix, and resample only the suffix at fresh temperature. This concentrates the sampling budget on the hard tail of each task instead of re-paying the easy prefix, lifting effective acceptance when full-episode success is rare. It requires deterministic prefix replay plus a mid-episode reward probe in `cuagym_rollout_runner.py`; scope it only if this gate fails.

## 5. Acceptance filter + dataset build (stage_03 → stage_01 → stage_04 → stage_05 → stage_06)

All CPU SLURM jobs from the committed worktree; register every artifact with a `stage_NN` alias immediately after it lands.

Accept (stage_03, filter):

```
srun --qos=low --cpus-per-task=16 --mem=64G --time=02:00:00 bash -c '
  cd '$WT'/data_pipeline
  uv run --no-sync -- python cuagym_pipeline/onpolicy_accept.py \
    --rollout_roots="'$EV'/onpolicy_round0_rollouts_stage_r0_*" \
    --output_dir='$DS'/cuagym_onpolicy_round0_stage_03_accepted_v1
'
labctl register-external --alias cuagym_onpolicy_round0_stage_03_accepted_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_03_accepted_v1 --kind dataset
```

Emits `accepted.jsonl` + report. Read the report before continuing (acceptance by tier, per-app_family counts, null-reward tally).

Image store (stage_01) over the accepted episodes' screenshots (one job per tar; layout must match what `onpolicy_accept.py` emits — it mirrors the HF corpus `screenshots-*.tar` convention consumed by `stage_01_image_store.py`):

```
NTARS=$(ls $DS/cuagym_onpolicy_round0_stage_03_accepted_v1/screenshots-*.tar | wc -l)
for i in $(seq 0 $((NTARS-1))); do
  srun --qos=low --cpus-per-task=16 --mem=64G --time=02:00:00 bash -c '
    cd '$WT'/data_pipeline
    uv run --no-sync -- python cuagym_pipeline/stage_01_image_store.py \
      --tar_dir='$DS'/cuagym_onpolicy_round0_stage_03_accepted_v1 \
      --tar_index='$i' \
      --output_root='$DS'/cuagym_onpolicy_round0_stage_01_image_store_v1
  ' &
done; wait
labctl register-external --alias cuagym_onpolicy_round0_stage_01_image_store_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_01_image_store_v1 --kind dataset
```

Conversations (stage_04o — episodes are already oev3-native, no translation):

```
srun --qos=low --cpus-per-task=16 --mem=64G --time=02:00:00 bash -c '
  cd '$WT'/data_pipeline
  uv run --no-sync -- python cuagym_pipeline/stage_04o_onpolicy_conversations.py \
    --accepted='$DS'/cuagym_onpolicy_round0_stage_03_accepted_v1/accepted.jsonl \
    --image_store='$DS'/cuagym_onpolicy_round0_stage_01_image_store_v1 \
    --output_dir='$DS'/cuagym_onpolicy_round0_stage_04_conversations_v1
'
labctl register-external --alias cuagym_onpolicy_round0_stage_04_conversations_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_04_conversations_v1 --kind dataset
```

Measure (stage_05) and records (stage_06), existing realigned-pipeline stages:

```
srun --qos=low --cpus-per-task=64 --mem=128G --time=04:00:00 bash -c '
  cd '$WT'/data_pipeline
  uv run --no-sync -- python realigned_pipeline/stage_05_measure_lengths.py \
    --source_path='$DS'/cuagym_onpolicy_round0_stage_04_conversations_v1 \
    --output_dir='$DS'/cuagym_onpolicy_round0_stage_05_message_lengths_v1 \
    --omegalax_repo='$OMEGALAX' \
    --model_id=Qwen/Qwen3.5-9B --processor=Qwen/Qwen3.5-9B --num_workers=64
'
labctl register-external --alias cuagym_onpolicy_round0_stage_05_message_lengths_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_05_message_lengths_v1 --kind dataset

srun --qos=low --cpus-per-task=64 --mem=200G --time=04:00:00 bash -c '
  cd '$WT'/data_pipeline
  uv run --no-sync -- python realigned_pipeline/stage_06_training_records.py \
    --source_path='$DS'/cuagym_onpolicy_round0_stage_04_conversations_v1 \
    --message_lengths_path='$DS'/cuagym_onpolicy_round0_stage_05_message_lengths_v1 \
    --output_dir='$DS'/cuagym_onpolicy_round0_stage_06_records_seqlen_24576_v1 \
    --omegalax_repo='$OMEGALAX' \
    --model_id=Qwen/Qwen3.5-9B --processor=Qwen/Qwen3.5-9B \
    --max_length=24576 --records_per_shard=8_000 --num_workers=64 \
    --overflow_mode=split --val_fraction=0.1
'
labctl register-external --alias cuagym_onpolicy_round0_stage_06_records_seqlen_24576_v1 \
  --path $DS/cuagym_onpolicy_round0_stage_06_records_seqlen_24576_v1 --kind dataset
```

Never run a bare `uv sync --project X` anywhere in this flow (it prunes the shared workspace venv); the one sanctioned sync is `uv sync --all-packages` at the worktree root.

## 6. Anneal

Recipe: `$LAB/recipes/training/jobs/oev3_parity/qwen35_9b_lora_success_onpolicy_anneal.toml`. Three edits before dispatch:

1. **GRAFT_SRC** (bash preamble, ships as `/REPLACE_ME_GRAFT_SRC/...` and the recipe exits 1 if left unset). Graft adopts the LATEST numeric step under GRAFT_SRC, so stage a directory containing only the chosen strong checkpoint:

```
GS=/fast/project/HFMI_SynergyUnit/yll/checkpoints/onpolicy_r0_graft_src
SRC=$CK/<stream_alias_of_the_strong_run>
mkdir -p $GS
cp $SRC/config.json $SRC/lora_metadata.json $GS/
cp -r $SRC/<step> $GS/<step>
```

Point GRAFT_SRC at `$GS`.

2. **num_steps** = graft step + 7500. The trainer resumes at the adopted step number, so a literal 7500 with a 10000-step graft would exit immediately; the shipped default `17500` assumes a 10000-step graft — adjust when the graft step changes. With `lr_stable_fraction = 0.2` the stable window ends well before the graft step, so the entire 7500-step anneal runs in WSD decay from roughly half-peak LR down to `lr_end_factor = 0.0`. LR is `1e-4` (LoRA convention; never the full-FT 1e-5).

3. **[inputs.onpolicy]** already points at `cuagym_onpolicy_round0_stage_06_records_seqlen_24576_v1`; bump the version suffix if step 5 produced a different alias. Mix is `--data_mix` with on-policy weight 0.35 / original success stage_06 weight 0.65.

Dispatch:

```
cd $LAB
labctl run recipes/training/jobs/oev3_parity/qwen35_9b_lora_success_onpolicy_anneal.toml
```

Note the recipe carries `--requeue`; killing it later needs `scancel --full`, often twice.

## 7. Evaluate

Resolve anneal checkpoint artifact ids via `labctl show run_<anneal run>` (checkpoints land at graft+2500/5000/7500, e.g. 12500/15000/17500 for a 10000-step graft).

1. **Probe**: `eval/cursor_probe.py` on each anneal checkpoint (existing probe sbatch flow); use `frac_compensating` / `frac_ignoring`, never aim cosine.
2. **Screen**: strat40 single-sample screen per anneal checkpoint — clone `recipes/eval/osworld_strat40_oev3_lora_success.toml` with the arm renamed (claims are keyed run-scoped, so new runs do not collide). The screen is a trigger only.
3. **Matched pass@4, before/after on the SAME steps**: clone `recipes/eval/osworld_strat38_pass4_oev3_lora_success_fixed_step_007500.toml`, pin `[inputs.checkpoint] artifact = ...` per run. Compare the anneal run at steps {15000, 17500} against the continued baseline lora_success run at the SAME optimizer steps {15000, 17500} (reuse the baseline's pass@4 results where they already exist). Single-checkpoint deltas are noise (±0.10–0.20 swings between adjacent checkpoints); require the verdict to replicate across both matched steps. **Report task ids, never counts** — two screens reading "3/38" can share only 1 task; the interesting outcome is the solved-task-id set difference, and whether the anneal adds ids outside the baseline's union.

## 8. Iterate-or-stop rules

- **Iterate (round 1)** if BOTH: (a) the anneal arm's pass@4 solved-task-id set contains >= 2 task ids the matched-step baseline does not solve, replicated across both matched steps; and (b) round-0 acceptance on the persist-verified tier was >= 8%. Round 1 = repeat from step 1 with the anneal checkpoint as the policy, curriculum re-weighted toward tasks that were near the acceptance boundary, all aliases bumped `round0`→`round1` (`stage_r0`→`stage_r1`) with fresh `_vN`.
- **Stop and escalate** if: no new task ids across both matched steps for two consecutive rounds, or acceptance stays < 8% and the prefix-restart fallback is judged not worth building — escalation path per the plan is GRPO/slime discussion, not more of the same loop.
- Do not blend rounds: each round's dataset is its own stage_06 artifact; the anneal always mixes exactly one on-policy round against the original stage_06 success corpus.

## 9. Budget (round 0)

| Item | Math | GPU-hours |
|---|---|---|
| Collection | 600 tasks x k=16 = 9,600 episodes x ~12 min = 1,920 episode-hours; 4 workers/GPU => 480 GPU-h of rollout compute; dispatched as 24 x 1-GPU x 24 h sweep jobs (per shard: 400 episodes / 4 workers x 12 min = 20 h + export/boot margin) | ~480 (576 walltime ceiling) |
| Anneal | 7,500 steps at the measured lora_success cadence (2,500 steps / ~4.1 h on 4xH100) = ~12.3 h wall x 4 GPUs (the "~1 GPU-day" planning figure is ~2x optimistic) | ~49 |
| Probe | cursor_probe x 3 checkpoints | ~2 |
| Screen | strat40 x 3 anneal checkpoints x <=6 h | ~18 |
| Matched pass@4 | 12 h x 1 GPU x (2 anneal steps + 2 baseline steps; baseline sides reusable if already measured) | 24–48 |
| **Total** | | **~570–650** (~24–27 GPU-days) |

## Known risks

- in-guest reward.py deps (~5% null rewards — preflight sample)
- success-convention reconciliation (>0 vs >=0.999 — we standardize on >=0.999 for SOLVED)
- mock_web deferred
- cursor_before not captured (cursor drills can't be built from these rollouts yet)

Additional operational notes: hai001/hai002/hai008 stay excluded everywhere (hai002 has a dead GPU SLURM thinks is healthy); the collect recipe refuses pre-vision-fix checkpoints by run-hash prefix; requeued labctl jobs whose source snapshot was GC'd die in ~1 s — recover via the GRAFT_SRC pattern rather than requeue.
