#!/bin/bash
#SBATCH --job-name=onpol_smoke
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/smoke_%j.out
set -euo pipefail

ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
export PYTHONUNBUFFERED=1
# SGLang 0.5.10 JIT-compiles a CUDA kernel at startup and needs nvcc; the default
# /usr/local/cuda symlink lacks nvcc on these nodes, so pin CUDA_HOME (matches the
# osworld eval recipes) and put its nvcc on PATH.
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH

EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
MODEL=$ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/

# resolve the snapshot dir (single snapshot expected)
MODEL_DIR=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
echo "teacher model dir: $MODEL_DIR"
echo "node: $(hostname)  /dev/kvm: $(ls -l /dev/kvm 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# coord_grid=1000 (canonical Qwen3-VL 1000x1000 normalized grid; ÷1000 per-axis to VM px).
uv run --project "$EVAL" python "$SCR/collect_absolute_rollouts.py" \
  --model_path "$MODEL_DIR" \
  --output_dir "$ROOT/onpolicy_distill/smoke/teacher_8b_grid1000" \
  --instructions_file "$SCR/instructions_smoke.txt" \
  --system_prompt_id computer_use_v1 \
  --coord_grid 1000 \
  --max_steps 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --mem_fraction_static 0.85

echo "=== SMOKE DONE ==="
