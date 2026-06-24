#!/usr/bin/env bash
#SBATCH --job-name=annotation_iter
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log

# CPU-only (0 GPU) run of the redesigned annotation loop on a compute node, so
# the login node is never hit with ffmpeg. Resumable: clips already done in the
# run dir are skipped. Usage:
#   bash -lc 'cd /fast/.../data_pipeline && sbatch annotation_pipeline/slurm/run_iteration.sbatch clean'
# (submit from a login shell so AZURE_OPENAI_* propagate via --export=ALL).
# Extra args after the run name are forwarded to run_iteration, e.g.
#   sbatch annotation_pipeline/slurm/run_iteration.sbatch kimi_2pass --clips safari_bf75aa71_s1

set -eo pipefail
RUN_NAME="${1:-clean}"
shift || true
EXTRA_ARGS=("$@")
DATA_PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
export PYTHONPATH="$DATA_PIPELINE"
# Absolute ffmpeg so we don't depend on the module system inside the job.
export JUERGEN_ANNOTATION_FFMPEG_BIN="${JUERGEN_ANNOTATION_FFMPEG_BIN:-/fast/service/apps/software/FFmpeg/7.0.2-GCCcore-13.3.0/bin/ffmpeg}"
# A compute node isn't a shared login node, so ffmpeg can use more threads.
export JUERGEN_ANNOTATION_FFMPEG_THREADS="${JUERGEN_ANNOTATION_FFMPEG_THREADS:-8}"
# v2 describe pass narrates ~150 frames. max_completion_tokens is RESERVED against
# the 262K context, so keep it modest (observed output maxed ~21K) or the input
# frames overflow the window. Frame height is 720 (config); a wide ~150-frame clip
# can still overflow at 720p and is skipped (accepted for now).
export LABELER_MAX_TOKENS="${LABELER_MAX_TOKENS:-32000}"

# Azure creds: normally inherited via sbatch --export=ALL; fall back to bashrc.
if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" || -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  source "$HOME/.bashrc" 2>/dev/null || true
fi

cd "$DATA_PIPELINE"
mkdir -p annotation_pipeline/logs
echo "host=$(hostname)  run=$RUN_NAME  ffmpeg=$JUERGEN_ANNOTATION_FFMPEG_BIN"
"$JUERGEN_ANNOTATION_FFMPEG_BIN" -version 2>/dev/null | head -1 || echo "WARN: ffmpeg not runnable"
python3 -c "from array_record.python.array_record_module import ArrayRecordWriter; print('array_record OK')" || echo "WARN: array_record missing"

# Internet/Azure reachability pre-check (compute nodes may have no egress).
code=$(curl -s -o /dev/null -m 20 -w "%{http_code}" \
  "${AZURE_OPENAI_ENDPOINT%/}/models" -H "api-key: ${AZURE_OPENAI_API_KEY:-none}" 2>/dev/null || echo "000")
echo "azure /models reachability: HTTP $code  (200 => stage 02 will work; 000 => no egress, extraction still runs & caches to grain)"

# Reasoning effort: left at the model default (Kimi-K2.6 reasons in-band; the
# describe pass is reasoning-heavy). Set LABELER_REASONING_EFFORT or pass
# --reasoning-effort to override.
# Judge is disabled (--no-judge): we are iterating on the describe→extract prompts.
python3 -m annotation_pipeline.run_iteration --run-name "$RUN_NAME" --no-judge "${EXTRA_ARGS[@]}"
echo "DONE run=$RUN_NAME"
