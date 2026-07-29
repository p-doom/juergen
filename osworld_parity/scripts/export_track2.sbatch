#!/bin/bash
#SBATCH --job-name=export_track2
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/export_track2_%j.out
set -euo pipefail
# Export a Track-2 parity SFT (orbax LoRA) -> merged HF for a4781827's full path-A eval.
# TAG = osw_absolute|osw_moverel|osw_diffabs. CPU (JAX_PLATFORMS=cpu). LoRA auto-merged.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
TAG="${TAG:?set TAG=osw_absolute|osw_moverel|osw_diffabs}"
SAVE=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${TAG}
STOCK=$(ls -d $ROOT/../huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
export JAX_PLATFORMS=cpu HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
# STEP (optional): export that specific step subdir (e.g. 000500 for the early abs read) into a
# step-tagged HF dir; otherwise export the highest step into the canonical _hf dir.
if [ -n "${STEP:-}" ]; then
  CKPT=$SAVE/$STEP
  OUT=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${TAG}_step${STEP}_hf
else
  CKPT=$(ls -d "$SAVE"/[0-9]*/ 2>/dev/null | sort | tail -1); CKPT=${CKPT%/}
  OUT=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${TAG}_hf
fi
[ -n "$CKPT" ] && [ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
echo "exporting $CKPT -> $OUT"; mkdir -p "$OUT"; cd "$OMX"
srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
  --model_id=Qwen/Qwen3-VL-8B-Instruct --checkpoint_path="$CKPT" --out_dir="$OUT" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
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
test -f "$OUT/model.safetensors" || test -f "$OUT/model.safetensors.index.json"
echo "=== EXPORT TRACK2 ($TAG) DONE -> $OUT ==="
