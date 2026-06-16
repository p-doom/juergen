#!/usr/bin/env bash
#SBATCH --job-name=v3_annotate
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%A_%a.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%A_%a.log

# Phase 2 of 3 (GPU): stage 02 VLM annotation only. Frames are pre-cached by the
# extract phase, so each node's 8 H100s stay on the VLM instead of ffmpeg.
#   sbatch --array=0-1 slurm/annotate.sbatch dataset_full_20260615   # 2 nodes
# Re-run the same line after 24h timeouts/failures: --resume skips done clips.

set -euo pipefail
RUN_NAME=${1:?usage: sbatch --array=0-<N-1> slurm/annotate.sbatch <run-name>}
V3=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline
DATA_PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
UV=${UV:-/home/yll.kryeziu/.local/bin/uv}
N=${SLURM_ARRAY_TASK_COUNT:?must be an array job}
I=${SLURM_ARRAY_TASK_ID:?}
mkdir -p "$V3/logs"

source "$V3/slurm/_serve_qwen.sh"
serve_qwen
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

export V3_MAX_CONCURRENCY="${V3_MAX_CONCURRENCY:-5}"
"$UV" run --project "$DATA_PIPELINE" --locked python -m annotation_pipeline.run_pipeline \
  --run-name "$RUN_NAME" --clips-file "$V3/clips_dataset.json" --clips all \
  --shard "$I/$N" --resume --stages annotate
