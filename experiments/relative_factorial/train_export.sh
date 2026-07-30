#!/bin/bash
# Train one relative factorial cell and export its step-750 checkpoint to HF.
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
for key in omegalax_repo dataset arm output model_id; do
    if [[ -z "${A[$key]:-}" ]]; then
        echo "FATAL missing --$key" >&2
        exit 2
    fi
done
case "${A[arm]}" in
    reltool_act|relraw_act|reltool_pre|relraw_pre) ;;
    *) echo "FATAL invalid arm: ${A[arm]}" >&2; exit 2 ;;
esac

OMX="${A[omegalax_repo]}"
DATA="${A[dataset]}/${A[arm]}"
OUT="${A[output]}"
ORBAX="$OUT/orbax"
for split in train val; do
    metadata="$DATA/$split/metadata.json"
    [[ -f "$metadata" ]] || { echo "FATAL missing tokenized split: $metadata" >&2; exit 2; }
    python3 - "$metadata" "$split" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
expected = 2000 if sys.argv[2] == "train" else 200
if meta.get("num_records") != expected:
    raise SystemExit(f"FATAL {sys.argv[1]} num_records={meta.get('num_records')} expected={expected}")
if meta.get("max_length") != 4096:
    raise SystemExit(f"FATAL {sys.argv[1]} max_length={meta.get('max_length')} expected=4096")
PY
done

job_tag="${SLURM_JOB_ID:-local}_${A[arm]}"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/relative_factorial_${job_tag}"
mkdir -p "$JAX_CACHE" "$ORBAX"

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
    --jax_cache_dir="$JAX_CACHE" \
    --model_id="${A[model_id]}" \
    --processor="${A[model_id]}" \
    --data_path="$DATA/train" \
    --val_data_path="$DATA/val" \
    --save_dir="$ORBAX" \
    --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
    --tp_size=1 --fsdp_size=1 --dp_size=1 \
    --batch_size=1 --grad_accum_steps=8 \
    --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 \
    --lr_end_factor=0.0 --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
    --max_length=4096 --num_steps=750 --num_loss_tiles=8 \
    --keep_latest=3 --keep_period=250 --save_every=250 \
    --val_every=250 --val_steps=15 \
    --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
    --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
    --max_vision_images_per_sample=2 --max_vision_patches_per_sample=16000 \
    --grain_read_threads=4 --grain_read_buffer_size=4 \
    --grain_workers=2 --grain_worker_buffer_size=2 \
    --wandb_entity=pdoom --wandb_project=omegalax \
    --wandb_group="relative_factorial_${A[arm]}" \
    --wandb_name="relative_factorial_${job_tag}" \
    --wandb_tags=berlin,rung3,relative_factorial,lora,r32

CKPT="$ORBAX/000750"
[[ -d "$CKPT" ]] || { echo "FATAL step-750 checkpoint missing: $CKPT" >&2; exit 3; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/export_checkpoint.sh" \
    --omegalax_repo="$OMX" --checkpoint_path="$CKPT" --output="$OUT" \
    --arm="${A[arm]}" --model_id="${A[model_id]}" \
    --manifest_name=train_export_manifest.json

echo "relative factorial train+export complete: ${A[arm]} -> $OUT/hf"
