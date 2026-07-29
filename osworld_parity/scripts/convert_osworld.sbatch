#!/bin/bash
#SBATCH --job-name=onpol_conv_osw
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/conv_osw_%j.out
set -euo pipefail
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
SCR=$ROOT/onpolicy_distill/scripts
OMX=/fast/home/franz.srambical/omegalax
CONV=$ROOT/onpolicy_distill/converted/osworld_thinking_v1

# combine BOTH thinking scale rounds; keep ONLY deterministic task-successes;
# retain reasoning prose; 0-999 normalized deltas end-to-end.
python3 "$SCR/convert_abs_to_relative.py" \
  --rollouts_dir "$ROOT/onpolicy_distill/sweep_scale_v1/scale_thinking,$ROOT/onpolicy_distill/sweep_scale_v2/scale_thinking" \
  --out_dir "$CONV" \
  --min_task_success 1.0 --keep_prose --coord_space normalized --min_valid_actions 2

for split in train val; do
  if [ -f "$CONV/_normalized/$split/chat.jsonl" ]; then
    echo "=== tokenizing $split ==="
    uv run --project "$OMX" -- python "$OMX/scripts/build_sft_records_from_chat.py" \
      --data_path "$CONV/_normalized/$split/chat.jsonl" --out_dir "$CONV/$split" \
      --model_id Qwen/Qwen3-VL-2B-Instruct --tokenizer Qwen/Qwen3-VL-2B-Instruct \
      --processor Qwen/Qwen3-VL-2B-Instruct \
      --max_length 16384 --overflow_mode truncate --records_per_shard 8000 --num_workers 8 --overwrite
  fi
done
echo "=== CONVERT+TOKENIZE (osworld_thinking_v1) DONE ==="
cat "$CONV/convert_manifest.json" 2>/dev/null || true
