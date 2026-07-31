#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in dataset source_model output omegalax_repo experiment_dir; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

DATA="${A[dataset]}"
SOURCE="${A[source_model]}"
SOURCE_HF="$SOURCE/hf"
OUT="${A[output]}"
ORBAX="$OUT/orbax"
PYTHON=/opt/miniforge3/bin/python
[[ -x "$PYTHON" ]] || { echo "FATAL pinned Python missing: $PYTHON" >&2; exit 3; }
[[ ! -e "$OUT/train_manifest.json" ]] || {
  echo "FATAL refusing to overwrite completed normalized-v2 training artifact" >&2; exit 2;
}
mkdir -p "$ORBAX"
"$PYTHON" "${A[experiment_dir]}/train_contract.py" \
  --dataset "$DATA" --source-model "$SOURCE" --output "$OUT" --stage preflight

tag="${SLURM_JOB_ID:-local}_normalized_v2_AtoA_r256"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/phaseb_${tag}"
mkdir -p "$JAX_CACHE"
cd "${A[omegalax_repo]}"
uv run --project="${A[omegalax_repo]}" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAX_CACHE" \
  --model_id="$SOURCE_HF" --processor="$SOURCE_HF" \
  --data_path="$DATA/normalized_v2/train" --save_dir="$ORBAX" \
  --enable_lora=true --lora_rank=256 --lora_alpha=256 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 \
  --lr_end_factor=0.0 --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
  --max_length=16384 --num_steps=900 --num_loss_tiles=16 \
  --keep_latest=3 --keep_period=300 --save_every=300 \
  --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=4 \
  --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group=phaseb_normalized_v2_production_control \
  --wandb_name="phaseb_${tag}" \
  --wandb_tags=berlin,phaseb,normalized_v2,natural_prose,A_to_A,lora,r256,best_pipeline_control

"$PYTHON" "${A[experiment_dir]}/train_contract.py" \
  --dataset "$DATA" --source-model "$SOURCE" --output "$OUT" --stage finalize
echo "Phase-B normalized-v2 A-to-A production-control training complete"
