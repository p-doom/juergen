#!/bin/bash
#SBATCH --job-name=osw_fmt_pp
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --nodelist=hai002
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/slurm_logs/%x_%A_%a.out
# Proper Qwen3-VL-Instruct sampling with anti-repetition: temperature 0.7, top_p 0.8,
# top_k 20, presence_penalty 1.5 (Qwen's recommended anti-repeat). Tests whether proper
# sampling + presence 1.5 lifts the deltatype_norm held-out number above the 2.0% greedy.
# 4-shard array pinned to hai002 (4 free GPUs). Same 110-task held-out + backend.
set -uo pipefail
: "${ACTION_FORMAT:?}"
: "${MODEL_PATH:?}"
: "${RUN_NAME:?}"
N_HISTORY="${N_HISTORY:-4}"
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
BASE=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/$RUN_NAME
export OSWORLD_VM_LOGDIR=$BASE/qemu_logs_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OSWORLD_VM_LOGDIR"
cd /fast/home/franz.srambical/juergen/eval

/fast/home/franz.srambical/juergen/.venv/bin/python \
  /fast/home/franz.srambical/osworld_parity_split/format_eval_shard.py \
  --base_output_dir="$BASE" \
  --test_split_path=/fast/home/franz.srambical/osworld_parity_split/osworld_eval_heldout.json \
  --num_shards=4 --shard_index="${SLURM_ARRAY_TASK_ID}" \
  --action_format="$ACTION_FORMAT" \
  --model_path="$MODEL_PATH" --served_model_name=policy \
  --max_steps=15 --temperature=0.7 --max_tokens=1024 --n_history_frames="$N_HISTORY" \
  --top_p=0.8 --top_k=20 --presence_penalty=1.5 \
  --mem_fraction_static=0.85 --chunked_prefill_size=2048
