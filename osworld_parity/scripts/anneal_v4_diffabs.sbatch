#!/bin/bash
#SBATCH --job-name=onpol_anneal_8b_diffabs_v4
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --exclude=hai006,hai001
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/anneal_%j.out
set -euo pipefail

# CUSTOM-GRAMMAR (diffabs/psai) arm of the distillation FORMAT SWEEP. Identical
# training geometry to anneal_v3.sbatch (the native_rel v3 arm); the ONLY changes
# are the target action grammar (custom diffabs vs native_rel), the warm-start base
# (diffabs BC vs nativerel BC), and the retain set (videocua_diffabs_v1 vs
# videocua_nativerel_v1). Same teacher rollouts (teacher_8b_v3), same on-policy data,
# same LoRA hyperparameters -> a clean differential on which grammar lets the
# distilled grounding stick.
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
OMX=/fast/home/franz.srambical/omegalax
CONV=$ROOT/onpolicy_distill/converted/teacher_8b_v4_diffabs
RETAIN=$ROOT/datasets/franz.srambical/videocua_diffabs_v1
# Student warm-start = the merged-HF export of the pure-LoRA videocua sol_diffabs 8B
# BC (job 130720, step 9000) -- the direct custom-grammar analog of v3's videocua
# nativerel LoRA warm-start. Produced by export_diffabs_base.sbatch.
STUDENT=${STUDENT:-$ROOT/onpolicy_distill/checkpoints/diffabs_sol_8b_bc_base_hf/009000}
SAVE=$ROOT/onpolicy_distill/checkpoints/anneal_8b_diffabs_onpolicy_v4

export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export LD_PRELOAD=/fast/home/franz.srambical/.tmp_suite/mihir_ablation/nccl_2.28.3/libnccl.so.2
export UV_CACHE_DIR=$ROOT/uv_cache
export TMPDIR=$ROOT/tmp
JAXC=$ROOT/jax_cache/onpol_anneal_${SLURM_JOB_ID}
mkdir -p "$JAXC" "$SAVE"

cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAXC" \
  --model_id="$STUDENT" \
  --processor=Qwen/Qwen3-VL-8B-Instruct \
  --data_mix='[{"path": "'"$CONV"'/train", "weight": 0.5}, {"path": "'"$RETAIN"'/train", "weight": 0.5}]' \
  --val_data_path="$RETAIN/val" \
  --save_dir="$SAVE" \
  --enable_lora=true \
  --lora_rank=32 \
  --lora_alpha=32 \
  --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=4 --dp_size=1 \
  --learning_rate=3e-6 \
  --lr_schedule=wsd \
  --lr_stable_fraction=0.0 \
  --lr_end_factor=0.0 \
  --warmup_steps=50 \
  --num_steps=600 \
  --save_every=300 \
  --val_every=300 \
  --val_steps=20 \
  --keep_period=300 \
  --keep_latest=1 \
  --log_memory=true \
  --num_loss_tiles=16 \
  --resume=if_present \
  --batch_size=4 \
  --grad_accum_steps=2 \
  --max_grad_norm=1.0 \
  --max_length=16384 \
  --max_vision_images_per_sample=29 \
  --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 \
  --grain_read_buffer_size=2 \
  --grain_workers=2 \
  --grain_worker_buffer_size=2 \
  --weight_decay=0.01 \
  --text_attn_backend=mosaic_gpu \
  --peak_tflops=h100_sxm \
  --pad_id=0 \
  --seed=0 \
  --log_every=10 \
  --gc_period=10000 \
  --wandb_entity=pdoom \
  --wandb_project=omegalax \
  --wandb_group=onpol_anneal_8b_diffabs_v4_${SLURM_JOB_ID} \
  --wandb_name=onpol_anneal_8b_diffabs_v4_${SLURM_JOB_ID} \
  --wandb_tags berlin lora diffabs custom_grammar onpolicy_distill anneal format_sweep

echo "=== ANNEAL DIFFABS v4 DONE -> $SAVE ==="
