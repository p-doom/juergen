#!/bin/bash
#SBATCH --job-name=onpol_tok_v4_8b
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/tok_v4_8b_%j.out
set -euo pipefail
# Tokenize BOTH v4 distillation-format sweep grammars (move_rel + custom diffabs)
# for Qwen3-VL-8B, using the SAME recipe as convert_tokenize_v3 / videocua_nativerel_v2
# (Qwen/Qwen3-VL-2B-Instruct tokenizer+processor, max_length 16384, truncate).
# The _normalized/{train,val}/chat.jsonl were already written by the converters.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
export PYTHONUNBUFFERED=1

for CONV in "$ROOT/onpolicy_distill/converted/teacher_8b_v4_moverel" \
            "$ROOT/onpolicy_distill/converted/teacher_8b_v4_diffabs"; do
  for split in train val; do
    if [ -f "$CONV/_normalized/$split/chat.jsonl" ]; then
      echo "=== tokenizing $CONV [$split] @ $(date) ==="
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
done
echo "=== TOKENIZE v4 8B DONE @ $(date) ==="
ls -la "$ROOT/onpolicy_distill/converted/teacher_8b_v4_moverel/train/metadata.json" \
       "$ROOT/onpolicy_distill/converted/teacher_8b_v4_diffabs/train/metadata.json"
