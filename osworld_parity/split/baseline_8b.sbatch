#!/bin/bash
#SBATCH --job-name=osw_parity_8b
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/fast/home/franz.srambical/osworld_parity_split/baseline_8b_instruct/slurm_%A_%a.log

# Off-the-shelf Qwen3-VL-8B-Instruct closed-loop OSWorld task-success baseline
# on the 110-task HELD-OUT eval split (native computer_use, apptainer provider).
# 6 shards amortize the sglang load; each shard ~18-19 tasks. Resumable.

set -euo pipefail
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# flashinfer JIT (sglang) needs a working nvcc; the 12.6.0 tree on /fast has one
# (torch is 2.9.1+cu128 — nvcc 12.6 JIT-compiles fine against it).
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
BASE=/fast/home/franz.srambical/osworld_parity_split/baseline_8b_instruct
export OSWORLD_VM_LOGDIR=$BASE/qemu_logs_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OSWORLD_VM_LOGDIR"
cd /fast/home/franz.srambical/juergen/eval

/fast/home/franz.srambical/juergen/.venv/bin/python \
  /fast/home/franz.srambical/osworld_parity_split/baseline_eval_shard.py \
  --test_split_path=/fast/home/franz.srambical/osworld_parity_split/osworld_eval_heldout.json \
  --base_output_dir="$BASE" \
  --num_shards=6 --shard_index="${SLURM_ARRAY_TASK_ID}" \
  --model_path=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/ \
  --served_model_name=Qwen3-VL-8B-Instruct \
  --max_steps=15 --temperature=0.0 --top_p=0.9 --max_tokens=1024 \
  --history_n=4 --coordinate_type=relative \
  --screen_width=1920 --screen_height=1080 --sleep_after_execution=1.0 \
  --provider_name=apptainer --mem_fraction_static=0.85 --chunked_prefill_size=2048
