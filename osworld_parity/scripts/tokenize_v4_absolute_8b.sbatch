#!/bin/bash
#SBATCH --job-name=onpol_tok_v4_abs
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/tok_v4_abs_%j.out
set -euo pipefail
# Tokenize the ABSOLUTE positive-control synthetic-format dataset for Qwen3-VL-8B,
# same recipe as the move_rel/diffabs synthetic datasets (Qwen3-VL-2B tokenizer, 16384, truncate).
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
CONV=$ROOT/onpolicy_distill/converted/teacher_8b_v4_absolute
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
for split in train val; do
  echo "=== tokenizing absolute $split @ $(date) ==="
  uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
    --data_path "$CONV/_normalized/$split/chat.jsonl" \
    --out_dir "$CONV/$split" \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --tokenizer Qwen/Qwen3-VL-2B-Instruct \
    --processor Qwen/Qwen3-VL-2B-Instruct \
    --max_length 16384 --overflow_mode truncate \
    --records_per_shard 8000 --num_workers 8 --overwrite
done
echo "=== TOKENIZE v4 ABSOLUTE 8B DONE @ $(date) ==="
ls -la "$CONV/train/metadata.json" "$CONV/val/metadata.json"
