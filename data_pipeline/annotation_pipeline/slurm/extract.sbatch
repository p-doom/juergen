#!/usr/bin/env bash
#SBATCH --job-name=v3_extract
#SBATCH --partition=standard
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%A_%a.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline/logs/%x_%A_%a.log

# Phase 1 of 3 (CPU, 0 GPU): extract 2fps 720p frames + actions for every clip
# into the shared frame cache. Holds no GPUs, so the annotate phase never idles
# on ffmpeg. Run as a wide array for parallelism:
#   sbatch --array=0-15 slurm/extract.sbatch dataset_full_20260615

set -euo pipefail
RUN_NAME=${1:?usage: sbatch --array=0-<N-1> slurm/extract.sbatch <run-name>}
V3=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation-pipeline
N=${SLURM_ARRAY_TASK_COUNT:?must be an array job}
I=${SLURM_ARRAY_TASK_ID:?}
mkdir -p "$V3/logs"

"$V3/.venv/bin/python" "$V3/run_pipeline.py" \
  --run-name "$RUN_NAME" --clips-file "$V3/clips_dataset.json" --clips all \
  --shard "$I/$N" --resume --stages frames
