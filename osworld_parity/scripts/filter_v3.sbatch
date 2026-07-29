#!/bin/bash
#SBATCH --job-name=onpol_filter
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/filter_%j.out
set -euo pipefail

ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH

EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
# Judge = off-the-shelf 8B-Instruct (fast + fits one H100). Upgrade to
# Qwen3-VL-32B-Instruct for a stronger, more decorrelated judge if keep-rate/quality
# needs it (mem_fraction_static ~0.9, may need a 2nd GPU).
JUDGE_DIR=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)

uv run --project "$EVAL" python "$SCR/quality_filter.py" \
  --rollouts_dir "$ROOT/onpolicy_distill/rollouts/teacher_8b_v3" \
  --out_dir "$ROOT/onpolicy_distill/filtered/teacher_8b_v3" \
  --judge_model "$JUDGE_DIR" \
  --min_score 5.0 \
  --max_judge_frames 4 \
  --mem_fraction_static 0.85

echo "=== FILTER DONE ==="
