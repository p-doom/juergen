#!/bin/bash
#SBATCH --job-name=onpol_tok_v4_q35
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/tok_v4_q35_%j.out
set -euo pipefail
# Qwen3.5-9B RE-TOKENIZED copies of BOTH v4 grammars, mirroring how the move_rel BC
# suites were retok'd (qwen35_retok/tokenize_nativerel_qwen35.sbatch): Qwen/Qwen3.5-9B
# tokenizer+processor, max_length 16256, same qwen35 venv + worktree. Reads the
# processor-agnostic _normalized/{train,val}/chat.jsonl; writes *_qwen35_len16256 dirs.
OMX=/fast/home/franz.srambical/.tmp_suite/omegalax-qwen35
VENV=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/omegalax-qwen35-venv
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/uv_cache
export TMPDIR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp
RETOK=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/qwen35_retok

for name in teacher_8b_v4_moverel teacher_8b_v4_diffabs; do
  SRC=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/converted/$name
  OUT=$RETOK/${name}_qwen35_len16256
  mkdir -p "$OUT"
  for split in train val; do
    if [ -f "$SRC/_normalized/$split/chat.jsonl" ]; then
      echo "=== [$name] tokenizing $split @ $(date) -> $OUT/$split ==="
      rm -rf "$OUT/$split"
      "$VENV/bin/python" "$OMX/scripts/build_sft_records_from_chat.py" \
        --data_path "$SRC/_normalized/$split/chat.jsonl" \
        --out_dir "$OUT/$split" \
        --model_id Qwen/Qwen3.5-9B \
        --tokenizer Qwen/Qwen3.5-9B \
        --processor Qwen/Qwen3.5-9B \
        --max_length 16256 --overflow_mode truncate \
        --records_per_shard 8000 --num_workers 8 --overwrite
    fi
  done
done
echo "=== TOKENIZE v4 QWEN3.5 DONE @ $(date) ==="
ls -la "$RETOK/teacher_8b_v4_moverel_qwen35_len16256/train/metadata.json" \
       "$RETOK/teacher_8b_v4_diffabs_qwen35_len16256/train/metadata.json"
