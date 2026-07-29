#!/bin/bash
#SBATCH --job-name=offshelf_nr_typing
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/eval_offshelf_nr_typing_%j.out
set -euo pipefail
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
EVAL=/fast/home/franz.srambical/juergen/eval
CKPT=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
OUTT=$ROOT/onpolicy_distill/eval/offshelf_8b_instruct_nativerel_typing
mkdir -p "$OUTT"
cd "$EVAL"
echo "=== TYPING (native-rel freeroll) OFF-THE-SHELF on $CKPT ==="
uv run --project="$EVAL" python "$EVAL/freeroll.py" \
  --model_path="$CKPT" \
  --output_dir="$OUTT" \
  --max_steps=60 \
  --desktop_setup=terminal \
  --sglang_port=30000 \
  --sglang_api_key=osworld \
  --mem_fraction_static=0.70 \
  --system_prompt_id=native_rel_v1 \
  --action_format=computer_use_relative \
  --max_tokens=256 \
  --instruction='In the already opened terminal, write "Hello, World! This is a typing test. I am a trained by omegalax."
In the already opened terminal, write a short text about the transformer
In the already opened terminal, list the files in the home directory
In the already opened terminal, open vim and write a script that prints "Hello, World!" ten times. Then run the script'
echo "=== EVAL DONE ==="
