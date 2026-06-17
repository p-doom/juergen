#!/usr/bin/env bash
#SBATCH --job-name=v3_assemble
#SBATCH --partition=standard
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%A_%a.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%A_%a.log

# Phase 3 of 3 (CPU, 0 GPU): assemble verified trajectories into neutral
# samples (stage 03) and export the run-level canonical SFT artifact (stage 04).
# Holds no GPUs; can overlap the annotate phase (consumes stage-02 outputs as
# they land).
#   sbatch --array=0-7 slurm/assemble.sbatch dataset_full_20260615

set -euo pipefail
RUN_NAME=${1:?usage: sbatch --array=0-<N-1> slurm/assemble.sbatch <run-name>}
V3=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline
DATA_PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
UV=${UV:-/home/yll.kryeziu/.local/bin/uv}
N=${SLURM_ARRAY_TASK_COUNT:?must be an array job}
I=${SLURM_ARRAY_TASK_ID:?}
mkdir -p "$V3/logs"

"$UV" run --project "$DATA_PIPELINE" --locked python -m annotation_pipeline.run_pipeline \
  --run-name "$RUN_NAME" --clips-file "$V3/clips_dataset.json" --clips all \
  --shard "$I/$N" --resume --stages assemble
