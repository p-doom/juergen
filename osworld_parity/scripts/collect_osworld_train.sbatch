#!/bin/bash
#SBATCH --job-name=osw_collect_train
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%16
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/rollouts/teacher_8b_osworld_train_v1/slurm_%A_%a.log
set -euo pipefail
# TRACK 2 rollout-gen: off-the-shelf Qwen3-VL-8B teacher trajectories on the 259 no-leak
# OSWorld TRAIN tasks (osworld_train.json). REUSES a4781827's VALIDATED runner
# baseline_eval_shard.py verbatim -- DesktopEnv.reset(task_config) applies each task's own
# setup/snapshot/evaluator automatically (no per-task setup), Qwen3VLAgent (native
# computer_use, coordinate_type=relative => 0-1000 grid), env.evaluate() -> reward.
# Saves {out}/{app}/{task_id}/{result.json,traj.jsonl,steps/step_NNN.png}.
# 16 shards (~16 tasks each) for parallelism; flows as GPUs free. NEVER the eval split.
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
OUT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/rollouts/teacher_8b_osworld_train_v1
export OSWORLD_VM_LOGDIR=$OUT/qemu_logs_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OSWORLD_VM_LOGDIR"
cd /fast/home/franz.srambical/juergen/eval

/fast/home/franz.srambical/juergen/.venv/bin/python \
  /fast/home/franz.srambical/osworld_parity_split/baseline_eval_shard.py \
  --test_split_path=/fast/home/franz.srambical/osworld_parity_split/osworld_train.json \
  --base_output_dir="$OUT" \
  --num_shards=16 --shard_index="${SLURM_ARRAY_TASK_ID}" \
  --model_path=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/ \
  --served_model_name=Qwen3-VL-8B-Instruct \
  --max_steps=15 --temperature=0.0 --top_p=0.9 --max_tokens=1024 \
  --history_n=4 --coordinate_type=relative \
  --screen_width=1920 --screen_height=1080 --sleep_after_execution=1.0 \
  --provider_name=apptainer --mem_fraction_static=0.85 --chunked_prefill_size=2048
echo "=== OSWORLD TRAIN COLLECT shard ${SLURM_ARRAY_TASK_ID} DONE ==="
