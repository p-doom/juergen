#!/bin/bash
#SBATCH --job-name=thinking_anneal_wait_base
#SBATCH --account=hfmi_synergyunit
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=hai001
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/logs/thinking_anneal_wait_%j.out
set -uo pipefail
# GATE for the thinking anneal: block (exit 0) until the move_rel crowd-cast BASE run
# (job 133240, bc_qwen3vl8b_lora_crowdcast_nativerel_base_v1) writes its FIRST HF export.
# The base's export cadence is every 1000 steps and the mix's (133241) is every 2500, so a
# step-001000 nativerel_v2 export uniquely identifies the crowd-cast BASE (no DB access,
# no base/mix ambiguity). Writes the resolved warm-start dir to STUDENT_FILE; the anneal
# job (submitted afterok on THIS job) reads it. Times out after ~2 days -> anneal never runs.
CKROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/franz.srambical
STUDENT_FILE=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/checkpoints/thinking_anneal_base_path.txt
deadline=$(( $(date +%s) + 84600 ))
echo "[$(date +%T)] waiting for crowd-cast base (133240) first export: $CKROOT/bc_export_hf_8b_nativerel_v2_*/001000/model.safetensors"
while [ "$(date +%s)" -lt "$deadline" ]; do
  hit=$(ls -d "$CKROOT"/bc_export_hf_8b_nativerel_v2_*/001000 2>/dev/null | while read d; do
          [ -f "$d/model.safetensors" ] && echo "$d"; done | head -1)
  if [ -n "${hit:-}" ]; then
    echo "$hit" > "$STUDENT_FILE"
    echo "[$(date +%T)] base export found -> $hit (written to $STUDENT_FILE)"
    exit 0
  fi
  sleep 120
done
echo "[$(date +%T)] TIMEOUT: crowd-cast base step-1000 export never appeared"; exit 1
