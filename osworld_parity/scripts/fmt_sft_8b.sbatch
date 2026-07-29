#!/bin/bash
#SBATCH --job-name=fmt_sft_8b
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=8:00:00
#SBATCH --exclude=hai001
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/fmt_sft_%j.out
set -euo pipefail
# TRACK-1 format-teaching SFT: fresh-from-stock Qwen3-VL-8B LoRA on ONE action-format dataset
# (absolute / move_rel / diffabs) built from the SYNTHETIC sh* teacher rollouts. Validates the
# convert+SFT pipeline + gives a "does SFT teach this format at all" signal on toy tasks (NOT an
# OSWorld-parity measurement -- that is track 2). Single-GPU (tp=fsdp=dp=1 => no NCCL/LD_PRELOAD).
# LoRA lr 1e-4, freeze_vision_tower=false (enable_lora freezes vision regardless; the current
# omegalax checkout REJECTS enable_lora+freeze_vision_tower=true). Parameterized:
#   FMT  = absolute|moverel|diffabs   (tag for save/wandb)
#   DATA = tokenized dataset dir (contains train/ and val/)
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
FMT="${FMT:?set FMT=absolute|moverel|diffabs}"
DATA="${DATA:?set DATA=<tokenized dataset dir>}"
SAVE=$ROOT/onpolicy_distill/checkpoints/fmt_sft_8b_${SAVE_TAG:-$FMT}

export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
JAXC=$ROOT/jax_cache/fmt_sft_${FMT}_${SLURM_JOB_ID}
mkdir -p "$JAXC" "$SAVE"
[ -f "$DATA/train/metadata.json" ] || { echo "missing tokenized train: $DATA/train"; exit 1; }

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAXC" \
  --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --processor=Qwen/Qwen3-VL-8B-Instruct \
  --data_path="$DATA/train" \
  --val_data_path="$DATA/val" \
  --save_dir="$SAVE" \
  --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 --lr_end_factor=0.0 --warmup_steps=30 \
  --weight_decay=0.01 --max_grad_norm=1.0 --max_length=16384 \
  --num_steps=${STEPS:-300} --num_loss_tiles=16 \
  --keep_latest=1 --keep_period=${SAVEEVERY:-150} --save_every=${SAVEEVERY:-150} --val_every=${SAVEEVERY:-150} --val_steps=15 \
  --log_every=10 --log_memory=true --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=2 --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group=fmt_sft_8b_${FMT}_${SLURM_JOB_ID} \
  --wandb_name=fmt_sft_8b_${FMT}_${SLURM_JOB_ID} \
  --wandb_tags berlin lora format_teaching pipeline_validation synthetic_sh $FMT single_gpu

echo "=== FMT SFT ($FMT) DONE -> $SAVE ==="
