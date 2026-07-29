#!/bin/bash
#SBATCH --job-name=onpol_sweep
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=4:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/sweep_%j.out
set -euo pipefail
# Parameterized via env (sbatch --export): MODEL_DIR, THINKING(0/1), SYSTEM_PROMPT_ID,
# COORD_TYPE(relative|absolute), TASKS_FILE, OUT_DIR, RUN_PREFIX, MAX_TOKENS, MAX_STEPS, MEMFRAC.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
: "${SYSTEM_PROMPT_ID:=canonical}"; : "${COORD_TYPE:=relative}"; : "${THINKING:=0}"
: "${MAX_TOKENS:=1024}"; : "${MAX_STEPS:=20}"; : "${MEMFRAC:=0.85}"; : "${SGLANG_TP:=1}"; : "${EVALUATE:=0}"
THINK_FLAG=""; [ "$THINKING" = "1" ] && THINK_FLAG="--thinking"
EVAL_FLAG=""; [ "$EVALUATE" = "1" ] && EVAL_FLAG="--evaluate"
echo "arm: model=$MODEL_DIR thinking=$THINKING prompt=$SYSTEM_PROMPT_ID coord=$COORD_TYPE prefix=$RUN_PREFIX node=$(hostname)"
uv run --project "$EVAL" python "$SCR/collect_osworld_rollouts.py" \
  --model_path "$MODEL_DIR" \
  --output_dir "$OUT_DIR" \
  --tasks_file "$TASKS_FILE" \
  --run_prefix "$RUN_PREFIX" \
  --system_prompt_id "$SYSTEM_PROMPT_ID" \
  --coord_type "$COORD_TYPE" \
  --coord_grid 1000 --max_steps "$MAX_STEPS" --max_tokens "$MAX_TOKENS" \
  --sglang_tp "$SGLANG_TP" \
  --temperature 0.0 --mem_fraction_static "$MEMFRAC" $THINK_FLAG $EVAL_FLAG
echo "=== SWEEP ARM $RUN_PREFIX DONE ==="
