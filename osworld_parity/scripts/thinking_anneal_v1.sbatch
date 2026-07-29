#!/bin/bash
#SBATCH --job-name=bc_qwen3vl8b_lora_videocua_thinking_anneal_v1
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
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/thinking_anneal_%j.out
set -euo pipefail
# THINKING anneal: mirror of bc_qwen3vl8b_lora_videocua_nativerel_anneal_v1 with the videocua
# data swapped for the CoT-augmented move_rel dataset. Warm-starts from the move_rel crowd-cast
# BASE run's (job 133240) first HF export (step 1000) -- the same warm-start the move_rel videocua
# anneal uses -- and anneals on videocua_nativerel_v2_thinking_v1 ALONE. Single-GPU (no NCCL);
# lr 1e-4; wsd pure-decay (stable 0.0, warmup 50); num_steps 3000.
# GATED: submitted afterok on thinking_anneal_wait_base (which resolves STUDENT). Not force-launched.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
VC=$ROOT/datasets/franz.srambical/videocua_moverel_thinking_v1
STUDENT_FILE=$ROOT/onpolicy_distill/checkpoints/thinking_anneal_base_path.txt
STUDENT="${STUDENT:-$(cat "$STUDENT_FILE" 2>/dev/null || true)}"
[ -n "${STUDENT:-}" ] && [ -f "$STUDENT/model.safetensors" ] || { echo "warm-start base not resolved ($STUDENT)"; exit 1; }
SAVE=$ROOT/onpolicy_distill/checkpoints/thinking_anneal_8b_v1

export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
JAXC=$ROOT/jax_cache/thinking_anneal_${SLURM_JOB_ID}
mkdir -p "$JAXC" "$SAVE"
echo "warm-start STUDENT = $STUDENT"

# Wait for thinking-videocua tokenization (train+val) before launching.
deadline=$(( $(date +%s) + 28800 ))
until [ -f "$VC/train/metadata.json" ] && [ -f "$VC/val/metadata.json" ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "TIMEOUT waiting for thinking-videocua tokenization"; exit 1; }
  echo "[$(date +%T)] waiting for thinking-videocua tokenization ..."; sleep 60
done

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAXC" \
  --model_id="$STUDENT" \
  --processor=Qwen/Qwen3-VL-8B-Instruct \
  --data_path="$VC/train" \
  --val_data_path="$VC/val" \
  --save_dir="$SAVE" \
  --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.0 --lr_end_factor=0.0 --warmup_steps=50 \
  --weight_decay=0.01 --max_grad_norm=1.0 --max_length=16384 \
  --num_steps=3000 --num_loss_tiles=16 \
  --keep_latest=1 --keep_period=1000 --save_every=1000 --val_every=1000 --val_steps=112 \
  --log_every=10 --log_memory=true --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=2 --grain_workers=2 --grain_worker_buffer_size=2 \
  --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group=thinking_anneal_8b_v1_${SLURM_JOB_ID} \
  --wandb_name=thinking_anneal_8b_v1_${SLURM_JOB_ID} \
  --wandb_tags berlin lora len_16k nativerel_v2 move_rel thinking videocua anneal warmstart_crowdcast_base lr1e4 single_gpu

echo "=== THINKING ANNEAL v1 DONE -> $SAVE ==="
