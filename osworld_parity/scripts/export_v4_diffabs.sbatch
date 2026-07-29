#!/bin/bash
#SBATCH --job-name=onpol_export_v4_diffabs
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/export_%j.out
set -euo pipefail
# orbax->merged-HF export of the CUSTOM-GRAMMAR (diffabs) distillation anneal output
# (anneal_8b_diffabs_onpolicy_v4, highest step). Mirror of export_v3.sbatch; only the
# SAVE/OUT and the tokenizer-source STUDENT (the diffabs BC base) differ.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
# tokenizer/processor + config-key source = the diffabs BC merged-HF base the anneal warm-started from
STUDENT=$ROOT/onpolicy_distill/checkpoints/diffabs_sol_8b_bc_base_hf/009000
SAVE=$ROOT/onpolicy_distill/checkpoints/anneal_8b_diffabs_onpolicy_v4
OUT=$ROOT/onpolicy_distill/checkpoints/anneal_8b_diffabs_onpolicy_v4_hf

export JAX_PLATFORMS=cpu
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp

CKPT=$(ls -d "$SAVE"/[0-9]*/ 2>/dev/null | sort | tail -1); CKPT=${CKPT%/}
echo "exporting orbax checkpoint: $CKPT"
mkdir -p "$OUT"
cd "$OMX"
srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
  --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_path="$CKPT" \
  --out_dir="$OUT" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --max_grad_norm=1.0 \
  --grad_accum_steps=2

for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$STUDENT/$f" ] && [ ! -f "$OUT/$f" ] && cp "$STUDENT/$f" "$OUT/" || true
done
python3 - "$STUDENT/config.json" "$OUT/config.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1])); out = json.load(open(sys.argv[2]))
for k in ("architectures", "transformers_version", "vision_end_token_id"):
    if k in base and k not in out: out[k] = base[k]
json.dump(out, open(sys.argv[2], "w"), indent=2)
PY
test -f "$OUT/model.safetensors" || test -f "$OUT/model.safetensors.index.json"
echo "=== EXPORT DIFFABS v4 DONE -> $OUT ==="
ls -la "$OUT"
