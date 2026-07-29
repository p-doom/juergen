#!/bin/bash
#SBATCH --job-name=onpol_eval_v4_diffabs
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
# FORMAT-MATCHED eval of the custom-grammar (diffabs/psai) distillation anneal.
# vs eval_v3.sbatch the ONLY changes: system_prompt_id psai_v1 (NOT native_rel_v1),
# action_format delta (the "dx dy scroll ; +KEY -KEY" parser, NOT computer_use_relative),
# NO rel_coord_grid (diffabs deltas are RAW PIXELS), and 192G host RAM (the thinking
# eval OOM-killed sglang at 64G). Matched to osworld_grounding/typing_..._8b_lora_v1
# (psai_v1) -- the diffabs BC's own matched-prompt eval.
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
CKPT=${CKPT:-$ROOT/onpolicy_distill/checkpoints/anneal_8b_diffabs_onpolicy_v4_hf}
OUTG=$ROOT/onpolicy_distill/eval/anneal_v4_diffabs_grounding
OUTT=$ROOT/onpolicy_distill/eval/anneal_v4_diffabs_typing
mkdir -p "$OUTG" "$OUTT"
cd "$EVAL"

echo "=== GROUNDING (diffabs/psai) on $CKPT ==="
uv run --project="$EVAL" python osworld_grounding_runner.py \
  --no_register_artifacts --no_frames \
  --bboxes_jsonl="$ROOT/osworld_grounding_eval_v0/bboxes.jsonl" \
  --model_path="$CKPT" \
  --output_dir="$OUTG" \
  --run_alias=grounding \
  --max_steps=60 \
  --sglang_port=30000 \
  --sglang_api_key=osworld \
  --mem_fraction_static=0.60 \
  --system_prompt_id=psai_v1 \
  --action_format=delta \
  --max_tokens=256
cp "$OUTG/_grounding_run_grounding/summary.json" "$OUTG/result.json" 2>/dev/null || true

echo "=== TYPING (diffabs/psai freeroll) on $CKPT ==="
uv run --project="$EVAL" python "$EVAL/freeroll.py" \
  --model_path="$CKPT" \
  --output_dir="$OUTT" \
  --max_steps=60 \
  --desktop_setup=terminal \
  --sglang_port=30000 \
  --sglang_api_key=osworld \
  --mem_fraction_static=0.60 \
  --system_prompt_id=psai_v1 \
  --action_format=delta \
  --max_tokens=256 \
  --instruction='In the already opened terminal, write "Hello, World! This is a typing test. I am a trained by omegalax."
In the already opened terminal, write a short text about the transformer
In the already opened terminal, list the files in the home directory
In the already opened terminal, open vim and write a script that prints "Hello, World!" ten times. Then run the script'

echo "=== EVAL DIFFABS v4 DONE ==="
