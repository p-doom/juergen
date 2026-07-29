#!/bin/bash
#SBATCH --job-name=isonorm_tf
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --nodelist=hai006
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/isonorm_tf_%j.out
set -euo pipefail
# Teacher-forced single-step direction decomposition over the 3 OSW-lineage checkpoints:
# diffabs(raw) vs diffabsnorm(0-999 bare-token) vs moverel(0-999 tool-call).
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz
export HF_HOME=$ROOT/../huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$ROOT/uv_cache TMPDIR=$ROOT/tmp
cd /fast/home/franz.srambical/juergen/eval
/fast/home/franz.srambical/juergen/.venv/bin/python \
  $ROOT/onpolicy_distill/eval/_tf_decomp/tf_decomp_iso.py diffabs,diffabsnorm,moverel
echo "=== ISONORM TFDECOMP DONE @ $(date) ==="
