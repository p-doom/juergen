#!/bin/bash
#SBATCH --job-name=onpol_eval_think_v2
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=20:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/eval_%j.out
set -euo pipefail
# RELAUNCH of job 132530 (thinking-distilled anneal eval anneal_8b_osworld_thinking_v1_hf),
# which died at sglang startup with rc=137 (host-RAM OOM: 64G too small for the 8B load).
# Fix: host --mem 64G->192G and GPU mem_fraction_static 0.70->0.55. Grammar UNCHANGED
# (native_rel_think + computer_use_relative + rel_coord_grid=1000 -- the thinking model's
# matched format). The HF checkpoint already exists, so no re-export.
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
CKPT=${CKPT:-$ROOT/onpolicy_distill/checkpoints/anneal_8b_osworld_thinking_v1_hf}
OUTG=$ROOT/onpolicy_distill/eval/anneal_osworld_thinking_grounding
OUTT=$ROOT/onpolicy_distill/eval/anneal_osworld_thinking_typing
mkdir -p "$OUTG" "$OUTT"
cd "$EVAL"

echo "=== GROUNDING (native-rel-think) on $CKPT ==="
uv run --project="$EVAL" python osworld_grounding_runner.py \
  --no_register_artifacts --no_frames \
  --bboxes_jsonl="$ROOT/osworld_grounding_eval_v0/bboxes.jsonl" \
  --model_path="$CKPT" \
  --output_dir="$OUTG" \
  --run_alias=grounding \
  --max_steps=60 \
  --sglang_port=30000 \
  --sglang_api_key=osworld \
  --mem_fraction_static=0.55 \
  --system_prompt_id=native_rel_think \
  --action_format=computer_use_relative \
  --rel_coord_grid=1000 \
  --max_tokens=512
cp "$OUTG/_grounding_run_grounding/summary.json" "$OUTG/result.json" 2>/dev/null || true

echo "=== TYPING (native-rel-think freeroll) on $CKPT ==="
uv run --project="$EVAL" python "$EVAL/freeroll.py" \
  --model_path="$CKPT" \
  --output_dir="$OUTT" \
  --max_steps=60 \
  --desktop_setup=terminal \
  --sglang_port=30000 \
  --sglang_api_key=osworld \
  --mem_fraction_static=0.55 \
  --system_prompt_id=native_rel_think \
  --action_format=computer_use_relative \
  --rel_coord_grid=1000 \
  --max_tokens=512 \
  --instruction='In the already opened terminal, write "Hello, World! This is a typing test. I am a trained by omegalax."
In the already opened terminal, write a short text about the transformer
In the already opened terminal, list the files in the home directory
In the already opened terminal, open vim and write a script that prints "Hello, World!" ten times. Then run the script'

echo "=== EVAL THINKING v2 DONE ==="
