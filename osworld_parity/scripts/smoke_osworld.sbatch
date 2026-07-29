#!/bin/bash
#SBATCH --job-name=onpol_smoke_osw
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/smoke_osw_%j.out
set -euo pipefail

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
THINK=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/*/ | head -1)
INSTR=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
echo "node=$(hostname) thinking=$THINK instruct=$INSTR"

# PRIMARY: 8B-THINKING under the canonical OSWorld prompt (reasons natively; no preamble)
uv run --project "$EVAL" python "$SCR/collect_osworld_rollouts.py" \
  --model_path "$THINK" \
  --output_dir "$ROOT/onpolicy_distill/smoke/osworld_thinking_canonical" \
  --tasks_file "$SCR/tasks_smoke_osworld.txt" \
  --system_prompt_id canonical \
  --coord_grid 1000 --max_steps 20 --max_tokens 1536 --temperature 0.0 \
  --mem_fraction_static 0.85 || echo "THINKING config errored (continuing to instruct)"

# COMPARISON: 8B-INSTRUCT under the SAME canonical prompt (emits 'Action:' one-liner, no <think>)
uv run --project "$EVAL" python "$SCR/collect_osworld_rollouts.py" \
  --model_path "$INSTR" \
  --output_dir "$ROOT/onpolicy_distill/smoke/osworld_instruct_canonical" \
  --tasks_file "$SCR/tasks_smoke_osworld.txt" \
  --system_prompt_id canonical \
  --coord_grid 1000 --max_steps 20 --max_tokens 1024 --temperature 0.0 \
  --mem_fraction_static 0.85 || echo "INSTRUCT config errored"

echo "=== OSWORLD SMOKE DONE ==="
