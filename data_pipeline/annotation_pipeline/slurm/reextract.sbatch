#!/usr/bin/env bash
#SBATCH --job-name=reextract
#SBATCH --partition=standard
#SBATCH --account=hfmi_synergyunit
#SBATCH --qos=low
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline/logs/%x_%j.log

# Re-run ONLY the extract pass over an existing dataset_run, in place, to
# validate an extract-prompt change cheaply (describe stays cached).
#   REEX_RUN_DIR=annotation_pipeline/dataset_runs/qc30 sbatch annotation_pipeline/slurm/reextract.sbatch
set -eo pipefail
DATA_PIPELINE=/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
export PYTHONPATH="$DATA_PIPELINE"
export LABELER_MAX_TOKENS="${LABELER_MAX_TOKENS:-32000}"
RUN_DIR="${REEX_RUN_DIR:-annotation_pipeline/dataset_runs/qc30}"
CONC="${REEX_CONC:-8}"

if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" || -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  source "$HOME/.bashrc" 2>/dev/null || true
fi
cd "$DATA_PIPELINE"
echo "host=$(hostname)  run_dir=$RUN_DIR  conc=$CONC"
code=$(curl -s -o /dev/null -m 20 -w "%{http_code}" \
  "${AZURE_OPENAI_ENDPOINT%/}/models" -H "api-key: ${AZURE_OPENAI_API_KEY:-none}" 2>/dev/null || echo "000")
echo "azure /models reachability: HTTP $code"

python3 -m annotation_pipeline.reextract_run --run-dir "$RUN_DIR" --concurrency "$CONC"
echo "DONE reextract $RUN_DIR"
