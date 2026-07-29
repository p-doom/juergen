#!/bin/bash
#SBATCH --job-name=onpol_conv
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/conv_%j.out
set -euo pipefail

ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
export PYTHONUNBUFFERED=1

SCR=$ROOT/onpolicy_distill/scripts
OMX=/fast/home/franz.srambical/omegalax
CONV=$ROOT/onpolicy_distill/converted/teacher_8b_v3

# 1) diff-of-absolute conversion (CPU, pure python) using the quality-filter keep list
python3 "$SCR/convert_abs_to_relative.py" \
  --rollouts_dir "$ROOT/onpolicy_distill/rollouts/teacher_8b_v3" \
  --out_dir "$CONV" \
  --keep_slugs "$ROOT/onpolicy_distill/filtered/teacher_8b_v3/keep_slugs.txt" \
  --min_valid_actions 2

# 2) tokenize -> array_record chunk-index (identical recipe to videocua_nativerel_v1)
for split in train val; do
  if [ -f "$CONV/_normalized/$split/chat.jsonl" ]; then
    echo "=== tokenizing $split ==="
    uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
      --data_path "$CONV/_normalized/$split/chat.jsonl" \
      --out_dir "$CONV/$split" \
      --model_id Qwen/Qwen3-VL-2B-Instruct \
      --tokenizer Qwen/Qwen3-VL-2B-Instruct \
      --processor Qwen/Qwen3-VL-2B-Instruct \
      --max_length 16384 --overflow_mode truncate \
      --records_per_shard 8000 --num_workers 8 --overwrite
  fi
done
echo "=== CONVERT+TOKENIZE DONE ==="
