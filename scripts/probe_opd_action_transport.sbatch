#!/usr/bin/env bash
#SBATCH --job-name=opd-action-xport
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/eval_logs/opd_action_transport/slurm-%j.log

set -euo pipefail

repo=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-oev3-cuagym
model=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
out=/fast/project/HFMI_SynergyUnit/yll/eval_logs/opd_action_transport/result.json

mkdir -p "$(dirname "$out")"
cd "$repo"
export HF_HOME=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

.venv/bin/python eval/probe_opd_action_transport.py \
  --model-path "$model" \
  --include-click \
  --radius 1 \
  --step 25 \
  --out "$out"
