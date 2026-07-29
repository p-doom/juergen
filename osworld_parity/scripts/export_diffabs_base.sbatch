#!/bin/bash
#SBATCH --job-name=onpol_export_diffabs_base
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/export_diffabs_base_%j.out
set -euo pipefail
# One-shot orbax -> merged-HF export of the custom-grammar (diffabs/psai) BC warm-start
# base for the custom-grammar distillation anneal. Source = the pure-LoRA videocua
# sol_diffabs 8B BC (job 130720), step 009000 -- the direct diffabs analog of v3's
# videocua-nativerel LoRA warm-start. Uses the SAME invocation as the shared export
# recipe bc_export_hf_per_checkpoint_8b_lora_v1 (JAX_PLATFORMS=cpu; LoRA auto-injected
# from lora_metadata.json; optimizer-shape flags max_grad_norm=1.0 grad_accum_steps=2
# only fix the pytree SIGN, not numeric values).
export JAX_PLATFORMS=cpu
OMX=/fast/home/franz.srambical/omegalax
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp

MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
CKPT=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical/bc_qwen3vl8b_lora_videocua_sol_diffabs_v1_run_019f90e52b197f91a64d8f3c123e4032/009000
OUT=$ROOT/onpolicy_distill/checkpoints/diffabs_sol_8b_bc_base_hf/009000
mkdir -p "$OUT"
cd "$OMX"

srun uv run --project="$OMX" -- python scripts/export_to_hf.py \
    --model_id="$MODEL_ID" \
    --out_dir="$OUT" \
    --checkpoint_path="$CKPT" \
    --tp_size=1 --fsdp_size=1 --dp_size=1 \
    --max_grad_norm=1.0 --grad_accum_steps=2

# Copy tokenizer/processor sidecars + patch config from the base snapshot (same as the export recipe).
SNAP_BASE="${HF_HOME}/hub/models--${MODEL_ID//\//--}/snapshots"
SNAP=$(ls -1d "$SNAP_BASE"/* | tail -1)
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
    if [ -f "${SNAP}/${f}" ] && [ ! -f "${OUT}/${f}" ]; then cp "${SNAP}/${f}" "${OUT}/${f}"; fi
done
python3 - "${SNAP}/config.json" "${OUT}/config.json" <<'PY'
import json, sys
snap = json.load(open(sys.argv[1])); out = json.load(open(sys.argv[2]))
for k in ("architectures", "transformers_version", "vision_end_token_id"):
    if k in snap and k not in out: out[k] = snap[k]
json.dump(out, open(sys.argv[2], "w"), indent=2)
PY
test -f "$OUT/model.safetensors" || test -f "$OUT/model.safetensors.index.json"
echo "=== EXPORT DIFFABS BASE DONE @ $(date) -> $OUT ==="
