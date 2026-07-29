#!/bin/bash
#SBATCH --job-name=onpol_diff
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/diff_%j.out
set -euo pipefail
export OSWORLD_ROOT=/fast/home/franz.srambical/OSWorld
SCR=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts
: "${SWEEP_ROOT:?set SWEEP_ROOT}"
python3 "$SCR/differential.py" --sweep_root "$SWEEP_ROOT" --show_prose
echo "=== DIFFERENTIAL DONE for $SWEEP_ROOT ==="
