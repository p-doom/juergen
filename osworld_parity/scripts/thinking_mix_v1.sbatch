#!/bin/bash
#SBATCH --job-name=bc_qwen3vl8b_lora_videocua_thinking_crowdcast_mix_v1
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=hai001
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/thinking_mix_%j.out
set -euo pipefail
# THINKING mix: mirror of bc_qwen3vl8b_lora_videocua_crowdcast_nativerel_mix_v1 with the
# videocua half swapped for the CoT-augmented move_rel dataset (videocua_nativerel_v2_thinking_v1).
# 50/50 iid record mix. videocua turns carry <think> + native_rel_v2_think system prompt;
# crowd-cast (crowdcast_nativerel_v1) has NO thoughts -> plain move_rel + native_rel_v2 system
# prompt. Per-record system prompts (think vs no-think) are coherent -- the same split the
# native_rel v3 distillation used. Fresh from stock Qwen3-VL-8B; single-GPU (tp=fsdp=dp=1 => no
# NCCL/LD_PRELOAD); lr 1e-4 (franz's standing LoRA lr); wsd stable 0.7 warmup 1125 num_steps 10000.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
VC=$ROOT/datasets/franz.srambical/videocua_moverel_thinking_v1
CC=$ROOT/datasets/franz.srambical/crowdcast_moverel
SAVE=$ROOT/onpolicy_distill/checkpoints/thinking_mix_8b_v1

export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
JAXC=$ROOT/jax_cache/thinking_mix_${SLURM_JOB_ID}
mkdir -p "$JAXC" "$SAVE"

# Wait for both tokenized datasets (train+val metadata.json) before launching.
deadline=$(( $(date +%s) + 28800 ))
until [ -f "$VC/train/metadata.json" ] && [ -f "$VC/val/metadata.json" ] \
   && [ -f "$CC/train/metadata.json" ] && [ -f "$CC/val/metadata.json" ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "TIMEOUT waiting for tokenization"; exit 1; }
  echo "[$(date +%T)] waiting for thinking-videocua + crowdcast tokenization ..."
  sleep 60
done

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAXC" \
  --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --processor=Qwen/Qwen3-VL-8B-Instruct \
  --data_mix='[{"path": "'"$VC"'/train", "weight": 0.5}, {"path": "'"$CC"'/train", "weight": 0.5}]' \
  --val_data_path="$VC/val" \
  --save_dir="$SAVE" \
  --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 --lr_end_factor=0.0 --warmup_steps=1125 \
  --weight_decay=0.01 --max_grad_norm=1.0 --max_length=16384 \
  --num_steps=10000 --num_loss_tiles=16 \
  --keep_latest=1 --keep_period=2500 --save_every=2500 --val_every=2500 --val_steps=112 \
  --log_every=10 --log_memory=true --resume=if_present --gc_period=10000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=2 --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group=thinking_mix_8b_v1_${SLURM_JOB_ID} \
  --wandb_name=thinking_mix_8b_v1_${SLURM_JOB_ID} \
  --wandb_tags berlin lora len_16k nativerel_v2 move_rel thinking videocua crowdcast mix_50_50 fresh_stock lr1e4 single_gpu

echo "=== THINKING MIX v1 DONE -> $SAVE ==="
