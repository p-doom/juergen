#!/bin/bash
#SBATCH --job-name=onpol_collect_osworld
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/collect_osworld_%j.out
set -euo pipefail
# ================================ TRACK 2 (GATED) ================================
# Off-the-shelf Qwen3-VL-8B teacher rollouts on REAL OSWorld TRAIN tasks (the no-leak
# train split), NATIVE absolute computer_use -> teacher trajectories for the parity
# experiment. Reuses the SAME off-shelf model, computer_use_v1 prompt, coord_grid 1000,
# and (critically) writes the SAME per-rollout dir structure the converters already consume:
#   <run_dir>/result.json  (instruction, system_prompt, screen_size, slug, stop_reason)
#   <run_dir>/trajectory.jsonl  (step_num, info.{parsed, cursor_before, intended_target, parse_error})
#   <run_dir>/steps/step_NNN.png
# => convert_abs_to_{absolute,moverel,diffabs}.py + fmt_sft_8b.sbatch work UNCHANGED on the output.
#
# *** BLOCKED on subagent a4781827 (VM provider restore + OSWorld train split + harness). ***
# collect_absolute_rollouts.py TODAY takes a free-form --instructions_file and boots a single
# generic qcow2 (no per-task OSWorld setup/evaluator). OSWorld tasks need per-task VM
# setup/config + the OSWorld env. Fill these from a4781827 before firing:
#   OSWORLD_TASK_SHARD = a4781827's train-task shard (task configs/ids for this shard)
#   VM_PROVIDER        = restored VM provider / per-task qcow2 or provider endpoint
#   OUT_DIR            = rollout output root (e.g. .../rollouts/teacher_8b_osworld_train_v1/<shard>)
#   SAMPLES            = samples per task (parallelize: many shards x VMs x GPUs via `gpufree`)
# AND wire collect_absolute_rollouts.py (or a4781827's OSWorld runner) to load the OSWorld task
# config + run OSWorld setup per task, keeping --system_prompt_id computer_use_v1 / --coord_grid 1000.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12; export PATH=$CUDA_HOME/bin:$PATH
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
MODEL_DIR=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
: "${OSWORLD_TASK_SHARD:?set to a4781827 train-task shard}"
: "${VM_PROVIDER:?set to a4781827 restored VM provider / qcow2}"
: "${OUT_DIR:?set rollout output root}"
: "${SAMPLES:=4}"
: "${RUN_PREFIX:=osworld_train}"
echo "OSWorld teacher rollout-gen: shard=$OSWORLD_TASK_SHARD provider=$VM_PROVIDER out=$OUT_DIR samples=$SAMPLES node=$(hostname)"
# NOTE: --instructions_file / --qcow2 args below are placeholders reusing the current harness;
# replace with a4781827's OSWorld-task-aware invocation once the harness interface is known.
uv run --project "$EVAL" python "$SCR/collect_absolute_rollouts.py" \
  --model_path "$MODEL_DIR" \
  --output_dir "$OUT_DIR" \
  --instructions_file "$OSWORLD_TASK_SHARD" \
  --qcow2 "$VM_PROVIDER" \
  --run_prefix "$RUN_PREFIX" \
  --samples_per_instruction "$SAMPLES" \
  --temperature 0.7 \
  --system_prompt_id computer_use_v1 \
  --coord_grid 1000 \
  --max_steps 30 \
  --max_tokens 512 \
  --n_history_frames 6 \
  --mem_fraction_static 0.85
echo "=== OSWORLD SHARD $RUN_PREFIX DONE ==="
