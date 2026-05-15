#!/usr/bin/env bash
#SBATCH --job-name=smoke_mm_sglang
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:45:00
#SBATCH --output=/home/franz.srambical/data_pipeline/_smoke_mm_sglang_%j.log
#SBATCH --error=/home/franz.srambical/data_pipeline/_smoke_mm_sglang_%j.log

set -uo pipefail
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export UV_CACHE_DIR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/uv_cache
export CUDA_HOME=/usr/local/cuda-12
export SGLANG_DISABLE_CUDNN_CHECK=1
export PATH="/home/franz.srambical/.local/bin:/usr/local/cuda-12/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

uv run --project=/fast/home/franz.srambical/eval \
  python /fast/home/franz.srambical/data_pipeline/_smoke_mm_sglang.py
