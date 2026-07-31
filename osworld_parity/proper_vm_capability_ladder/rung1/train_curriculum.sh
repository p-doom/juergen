#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
for key in omegalax_repo dataset arm output model_id exporter; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}" in
    native_absolute_control|compact_raw_phaseb) ;;
    *) echo "FATAL invalid curriculum arm: ${A[arm]}" >&2; exit 2 ;;
esac

OMX="${A[omegalax_repo]}"
DATA="${A[dataset]}"
ARM="${A[arm]}"
OUT="${A[output]}"
ORBAX="$OUT/orbax"
[[ -f "$DATA/build_tokenize_manifest.json" ]] || {
    echo "FATAL validation-gated tokenized curriculum marker missing" >&2; exit 2;
}
python3 - "$DATA" "$ARM" <<'PY'
import json, sys
from pathlib import Path
root, arm = Path(sys.argv[1]), sys.argv[2]
ready = json.loads((root / "build_tokenize_manifest.json").read_text())
if ready.get("status") != "complete" or ready.get("all_prelaunch_gates_passed") is not True:
    raise SystemExit(f"FATAL curriculum dataset is not launch-ready: {ready}")
for split, expected in (("train", 1664), ("val", 208)):
    meta = json.loads((root / arm / split / "metadata.json").read_text())
    if meta.get("num_records") != expected or meta.get("max_length") != 16384:
        raise SystemExit(f"FATAL tokenized curriculum mismatch {arm}/{split}: {meta}")
PY

job_tag="${SLURM_JOB_ID:-local}_${ARM}"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/rung1_curriculum_${job_tag}"
mkdir -p "$JAX_CACHE" "$ORBAX"
cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
    --jax_cache_dir="$JAX_CACHE" \
    --model_id="${A[model_id]}" --processor="${A[model_id]}" \
    --data_path="$DATA/$ARM/train" --val_data_path="$DATA/$ARM/val" \
    --save_dir="$ORBAX" \
    --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
    --tp_size=1 --fsdp_size=1 --dp_size=1 \
    --batch_size=1 --grad_accum_steps=8 \
    --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 \
    --lr_end_factor=0.0 --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
    --max_length=16384 --num_steps=750 --num_loss_tiles=8 \
    --keep_latest=3 --keep_period=250 --save_every=250 \
    --val_every=250 --val_steps=26 \
    --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
    --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
    --max_vision_images_per_sample=10 --max_vision_patches_per_sample=100000 \
    --grain_read_threads=4 --grain_read_buffer_size=4 \
    --grain_workers=2 --grain_worker_buffer_size=2 \
    --wandb_entity=pdoom --wandb_project=omegalax \
    --wandb_group=rung1_synthetic_capability_curriculum \
    --wandb_name="rung1_curriculum_${job_tag}" \
    --wandb_tags=berlin,rung1,synthetic,curriculum,lora,r32

CKPT="$ORBAX/000750"
[[ -d "$CKPT" ]] || { echo "FATAL step-750 checkpoint missing: $CKPT" >&2; exit 3; }
bash "${A[exporter]}" \
    --omegalax_repo="$OMX" --checkpoint_path="$CKPT" --output="$OUT" \
    --arm="$ARM" --model_id="${A[model_id]}" \
    --manifest_name=train_export_manifest.json
python3 - "$OUT/train_export_manifest.json" "$DATA/build_tokenize_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, dataset = map(Path, sys.argv[1:])
value = json.loads(manifest.read_text())
value["artifact_type"] = "rung1_synthetic_curriculum_hf_checkpoint"
value["max_length"] = 16384
value["curriculum_dataset_manifest_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
