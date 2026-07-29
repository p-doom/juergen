#!/bin/bash
#SBATCH --job-name=onpol_failnotify
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:05:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/failnotify_%j.out
# Fires ONLY when its afternotok target FAILED -> surfaces a dead chain stage instead
# of a silent PENDING-on-Dependency hang. Env: STAGE, WATCH_JID.
set -uo pipefail
: "${STAGE:=unknown}"; : "${WATCH_JID:=?}"
LOG=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/CHAIN_FAILED_${STAGE}.txt
echo "CHAIN STAGE FAILED: stage=$STAGE watched_job=$WATCH_JID at $(date)" | tee "$LOG"
sacct -j "$WATCH_JID" --format=JobID,JobName%24,State,ExitCode,Elapsed 2>/dev/null | tee -a "$LOG" || true
