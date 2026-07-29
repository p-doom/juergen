#!/bin/bash
#SBATCH --job-name=onpol_collect_shard
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/collect_shard_%j.out
set -euo pipefail
# Parameterized via env (sbatch --export): SHARD_FILE, RUN_PREFIX, OUT_DIR, SAMPLES.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
MODEL_DIR=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
: "${SAMPLES:=4}"
echo "shard=$RUN_PREFIX file=$SHARD_FILE out=$OUT_DIR samples=$SAMPLES node=$(hostname)"
uv run --project "$EVAL" python "$SCR/collect_absolute_rollouts.py" \
  --model_path "$MODEL_DIR" \
  --output_dir "$OUT_DIR" \
  --instructions_file "$SHARD_FILE" \
  --run_prefix "$RUN_PREFIX" \
  --samples_per_instruction "$SAMPLES" \
  --temperature 0.7 \
  --system_prompt_id computer_use_v1 \
  --coord_grid 1000 \
  --max_steps 15 \
  --max_tokens 512 \
  --n_history_frames 6 \
  --mem_fraction_static 0.85
echo "=== SHARD $RUN_PREFIX DONE ==="
