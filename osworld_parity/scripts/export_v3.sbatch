#!/bin/bash
#SBATCH --job-name=onpol_export
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/export_%j.out
set -euo pipefail

ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
STUDENT=$ROOT/../labctl/checkpoints/franz.srambical/bc_export_hf_8b_nativerel_artifact_6a26a4e2452eb6c5/001125
SAVE=$ROOT/onpolicy_distill/checkpoints/anneal_8b_nativerel_onpolicy_v3
OUT=$ROOT/onpolicy_distill/checkpoints/anneal_8b_nativerel_onpolicy_v3_hf

# export is a CPU weight-conversion (orbax->HF, LoRA merge). Mirror the proven
# export_recipe.json: JAX_PLATFORMS=cpu + srun (so jax.distributed.initialize gets
# a Slurm coordinator_address — the reason the no-srun run failed).
export JAX_PLATFORMS=cpu
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp

# highest-numbered orbax step dir written by the anneal (e.g. 000600)
CKPT=$(ls -d "$SAVE"/[0-9]*/ 2>/dev/null | sort | tail -1)
CKPT=${CKPT%/}
echo "exporting orbax checkpoint: $CKPT"
mkdir -p "$OUT"

cd "$OMX"
# --model_id resolves the ARCHITECTURE ONLY (omegalax registry needs a supported HF id,
# not a local path). load_pretrained(model_id) builds a skeleton whose weights are then
# FULLY OVERWRITTEN by the 38GB orbax checkpoint (base+LoRA) via nnx.update — and because
# the anneal warm-started from the merged native-rel student, orbax's base == that student.
# So the exported weights are the anneal's (student BC + on-policy LoRA), correct.
# --max_grad_norm / --grad_accum_steps MUST match anneal_v3.sbatch (optimizer pytree shape);
# LoRA auto-detected via lora_metadata.json.
srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
  --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_path="$CKPT" \
  --out_dir="$OUT" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --max_grad_norm=1.0 \
  --grad_accum_steps=2

# copy tokenizer/processor files the eval sglang server needs (export_to_hf writes
# weights+config only). Source = the student export (same Qwen3-VL-8B processor/tokenizer).
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$STUDENT/$f" ] && [ ! -f "$OUT/$f" ] && cp "$STUDENT/$f" "$OUT/" || true
done
# patch config.json keys export_to_hf omits (architectures etc.), from the student config
python3 - "$STUDENT/config.json" "$OUT/config.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))
out = json.load(open(sys.argv[2]))
for k in ("architectures", "transformers_version", "vision_end_token_id"):
    if k in base and k not in out:
        out[k] = base[k]
json.dump(out, open(sys.argv[2], "w"), indent=2)
print("patched config keys:", [k for k in ("architectures","transformers_version","vision_end_token_id") if k in out])
PY
echo "=== EXPORT DONE -> $OUT ==="
ls -la "$OUT"
