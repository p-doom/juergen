#!/usr/bin/env bash
#SBATCH --job-name=v3_pipeline
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --time=08:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%j.log

# Single-node entrypoint: serve Qwen3.6-27B, then run stages 00->04 over the
# requested clips. One-time env setup first: sbatch slurm/setup_env.sbatch
# Usage: sbatch slurm/run_pipeline.sbatch --run-name v1 --clips all

set -euo pipefail
V3=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline
mkdir -p "$V3/logs"
source "$V3/slurm/_serve_qwen.sh"
serve_qwen
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
"$V3/.venv/bin/python" "$V3/run_pipeline.py" "$@"
