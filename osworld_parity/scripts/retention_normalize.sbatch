#!/bin/bash
#SBATCH --job-name=onpol_retnorm
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/retnorm_%j.out
set -euo pipefail
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
SCR=$ROOT/onpolicy_distill/scripts
OMX=/fast/home/franz.srambical/omegalax
OUT=$ROOT/datasets/franz.srambical/videocua_nativerel_norm_v1

# 1) rescale raw-px deltas -> 0-999 / COORD_SCALE=1000 (per-recording source resolution)
python3 "$SCR/retention_normalize.py" --out "$OUT"

# 2) tokenize -> array_records (same recipe as the on-policy convert)
for split in train val; do
  if [ -f "$OUT/_normalized/$split/chat.jsonl" ]; then
    echo "=== tokenizing retention $split ==="
    uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
      --data_path "$OUT/_normalized/$split/chat.jsonl" --out_dir "$OUT/$split" \
      --model_id Qwen/Qwen3-VL-2B-Instruct --tokenizer Qwen/Qwen3-VL-2B-Instruct \
      --processor Qwen/Qwen3-VL-2B-Instruct \
      --max_length 16384 --overflow_mode truncate --records_per_shard 8000 --num_workers 8 --overwrite
  fi
done
echo "=== RETENTION NORMALIZE+TOKENIZE DONE -> $OUT ==="
