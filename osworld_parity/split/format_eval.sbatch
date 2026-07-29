#!/bin/bash
#SBATCH --job-name=osw_fmt_eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/slurm_logs/%x_%A_%a.out

# Format-AWARE closed-loop OSWorld eval for a move_rel or diffabs checkpoint,
# on the SAME 110-task held-out split + DesktopEnv/apptainer evaluate backend
# as the absolute control (apples-to-apples), but with that format's own
# prompt+parser+dispatch.
#
# Launch:
#   ACTION_FORMAT=move_rel MODEL_PATH=/path/to/hf RUN_NAME=moverel_ckptNNNN \
#     sbatch --export=ALL,ACTION_FORMAT,MODEL_PATH,RUN_NAME format_eval.sbatch
#   ACTION_FORMAT=diffabs MODEL_PATH=/path/to/hf RUN_NAME=diffabs_ckptNNNN \
#     sbatch --export=ALL,ACTION_FORMAT,MODEL_PATH,RUN_NAME format_eval.sbatch
# Optional: N_HISTORY (default 4 = match abs control; set to the format's training value).

set -uo pipefail  # NOT -e: a transient non-zero (write hiccup, preempt grace) must not nuke the shard
: "${ACTION_FORMAT:?set ACTION_FORMAT=move_rel|diffabs}"
: "${MODEL_PATH:?set MODEL_PATH=/path/to/hf checkpoint}"
: "${RUN_NAME:?set RUN_NAME=<short label>}"
N_HISTORY="${N_HISTORY:-4}"

export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
# Output on /fast/project (8TB free) — /fast/home 95G quota fills on a full run.
BASE=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/$RUN_NAME
export OSWORLD_VM_LOGDIR=$BASE/qemu_logs_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OSWORLD_VM_LOGDIR"
cd /fast/home/franz.srambical/juergen/eval

/fast/home/franz.srambical/juergen/.venv/bin/python \
  /fast/home/franz.srambical/osworld_parity_split/format_eval_shard.py \
  --base_output_dir="$BASE" \
  --test_split_path=/fast/home/franz.srambical/osworld_parity_split/osworld_eval_heldout.json \
  --num_shards=6 --shard_index="${SLURM_ARRAY_TASK_ID}" \
  --action_format="$ACTION_FORMAT" \
  --model_path="$MODEL_PATH" --served_model_name=policy \
  --max_steps=15 --temperature=0.0 --max_tokens=256 --n_history_frames="$N_HISTORY" \
  --mem_fraction_static=0.85 --chunked_prefill_size=2048
