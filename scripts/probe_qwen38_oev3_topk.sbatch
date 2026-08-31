#!/usr/bin/env bash
#SBATCH --job-name=q38-oev3-topk
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/eval_logs/qwen38_oev3_topk/slurm-%j.log

set -euo pipefail

repo=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-oev3-cuagym
model=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
out=/fast/project/HFMI_SynergyUnit/yll/eval_logs/qwen38_oev3_topk/result.json

mkdir -p "$(dirname "$out")"
cd "$repo"
export HF_HOME=/fast/project/HFMI_SynergyUnit/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

.venv/bin/python eval/probe_oev3_topk.py \
  --model-path "$model" \
  --tp-size 2 \
  --top-k 16 \
  --out "$out"
