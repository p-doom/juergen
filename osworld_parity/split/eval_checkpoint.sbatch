#!/bin/bash
#SBATCH --job-name=osw_ckpt_eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/slurm_logs/%x_%A_%a.out

# Reusable: eval ANY checkpoint on the SAME 110-task HELD-OUT split + SAME harness
# as the off-shelf baseline, so parity comparison stays apples-to-apples.
#
# Launch:
#   MODEL_PATH=/path/to/hf_export RUN_NAME=absolute_ckpt7875 \
#     sbatch --export=ALL,MODEL_PATH,RUN_NAME eval_checkpoint.sbatch
# Optional: SERVED_NAME (default basename of MODEL_PATH), COORD_TYPE (default relative).

set -uo pipefail  # NOT -e: a transient non-zero (write hiccup, preempt grace) must not nuke the shard
: "${MODEL_PATH:?set MODEL_PATH=/path/to/hf checkpoint dir}"
: "${RUN_NAME:?set RUN_NAME=<short label>}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"
# coordinate_type is HARDCODED to 'relative' == byte-identical to the off-shelf
# baseline (OSWorld 'relative' = the 0-999 normalized grid the abs control was
# distilled from). Do NOT wire this to a4b609f6's "COORD_TYPE=absolute" — that is
# a FORMAT name (absolute coords), NOT the OSWorld harness param; overriding it
# would break eval-config==baseline-config and invalidate the zero-shift control.

export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
# Output on /fast/project (8TB free) — /fast/home is a tiny 95G quota that a full
# eval run (screenshots x tasks x shards x formats) fills, which crashed the last run.
BASE=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/$RUN_NAME
export OSWORLD_VM_LOGDIR=$BASE/qemu_logs_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OSWORLD_VM_LOGDIR"
cd /fast/home/franz.srambical/juergen/eval

/fast/home/franz.srambical/juergen/.venv/bin/python \
  /fast/home/franz.srambical/osworld_parity_split/baseline_eval_shard.py \
  --test_split_path=/fast/home/franz.srambical/osworld_parity_split/osworld_eval_heldout.json \
  --base_output_dir="$BASE" \
  --num_shards=6 --shard_index="${SLURM_ARRAY_TASK_ID}" \
  --model_path="$MODEL_PATH" --served_model_name="$SERVED_NAME" \
  --max_steps=15 --temperature=0.0 --top_p=0.9 --max_tokens=1024 \
  --history_n=4 --coordinate_type=relative \
  --screen_width=1920 --screen_height=1080 --sleep_after_execution=1.0 \
  --provider_name=apptainer --mem_fraction_static=0.85 --chunked_prefill_size=2048
