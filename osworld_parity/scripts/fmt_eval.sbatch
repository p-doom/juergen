#!/bin/bash
#SBATCH --job-name=fmt_eval_8b
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/fmt_eval_%j.out
set -euo pipefail
# TRACK-1 format-validity + loss readout for one format SFT (FMT=absolute|moverel|diffabs).
# (1) export the SFT LoRA orbax -> merged HF (CPU), (2) offline first-action FORMAT-VALIDITY
# probe via sglang on held-out synthetic val prompts, (3) echo the SFT val-loss from its log.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
EVAL=/fast/home/franz.srambical/juergen/eval
SCR=$ROOT/onpolicy_distill/scripts
FMT="${FMT:?set FMT=absolute|moverel|diffabs}"
SAVE=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${FMT}
OUT=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${FMT}_hf
VAL=$ROOT/onpolicy_distill/converted/teacher_8b_v4_${FMT}/_normalized/val/chat.jsonl
STOCK=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0; export PATH=$CUDA_HOME/bin:$PATH

# (1) export orbax -> HF (CPU)
CKPT=$(ls -d "$SAVE"/[0-9]*/ 2>/dev/null | sort | tail -1); CKPT=${CKPT%/}
[ -n "$CKPT" ] || { echo "no SFT checkpoint in $SAVE"; exit 1; }
echo "exporting $CKPT -> $OUT"; mkdir -p "$OUT"
( cd "$OMX"; JAX_PLATFORMS=cpu srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
    --model_id=Qwen/Qwen3-VL-8B-Instruct --checkpoint_path="$CKPT" --out_dir="$OUT" \
    --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8 )
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$STOCK/$f" ] && [ ! -f "$OUT/$f" ] && cp "$STOCK/$f" "$OUT/" || true
done
python3 - "$STOCK/config.json" "$OUT/config.json" <<'PY'
import json,sys
b=json.load(open(sys.argv[1])); o=json.load(open(sys.argv[2]))
for k in ("architectures","transformers_version","vision_end_token_id"):
    if k in b and k not in o: o[k]=b[k]
json.dump(o,open(sys.argv[2],"w"),indent=2)
PY

# (2) offline format-validity probe (sglang, GPU)
PORT=$(( 31500 + (SLURM_JOB_ID % 100) * 10 ))
uv run --project "$EVAL" python "$SCR/fmt_validity_check.py" \
  --model_path "$OUT" --val_jsonl "$VAL" --format "$FMT" --n 8 \
  --out_json "$ROOT/onpolicy_distill/eval/fmt_validity_${FMT}.json" \
  --sglang_port "$PORT" --mem_fraction_static 0.80 --max_tokens 256

# (3) SFT val-loss readout
echo "=== $FMT SFT val-loss (from training log) ==="
grep -iE "val[_ ]?loss|eval.*loss" $ROOT/onpolicy_distill/logs/fmt_sft_*.out 2>/dev/null | grep -i "$FMT" | tail -4 || true
echo "=== FMT EVAL ($FMT) DONE ==="
