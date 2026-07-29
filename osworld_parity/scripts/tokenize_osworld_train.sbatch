#!/bin/bash
#SBATCH --job-name=tok_osw_train
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/tok_osw_train_%j.out
set -euo pipefail
# Tokenize the 3 Track-2 OSWorld-teacher format datasets (abs/move_rel/diffabs) for Qwen3-VL-8B.
# Same recipe as the move_rel BC suite (Qwen3-VL-2B tokenizer/processor, max_length 16384, truncate).
# Records are per-step eval-scaffold examples (system + history_n=4 window + current frame -> target);
# images are the eval-processed (smart_resize) PNGs referenced by path.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
for fmt in absolute moverel diffabs; do
  CONV=$ROOT/onpolicy_distill/converted/osworld_train_${fmt}
  for split in train val; do
    echo "=== tokenizing $fmt $split @ $(date) ==="
    uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
      --data_path "$CONV/_normalized/$split/chat.jsonl" \
      --out_dir "$CONV/$split" \
      --model_id Qwen/Qwen3-VL-2B-Instruct \
      --tokenizer Qwen/Qwen3-VL-2B-Instruct \
      --processor Qwen/Qwen3-VL-2B-Instruct \
      --max_length 16384 --overflow_mode truncate \
      --records_per_shard 8000 --num_workers 8 --overwrite
  done
done
echo "=== TOKENIZE OSWORLD TRAIN (3 formats) DONE @ $(date) ==="
ls -la $ROOT/onpolicy_distill/converted/osworld_train_{absolute,moverel,diffabs}/train/metadata.json
