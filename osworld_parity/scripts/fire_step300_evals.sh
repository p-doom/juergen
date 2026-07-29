#!/bin/bash
# Session-independent poll-and-fire for the step-300 pipeline-validation reads.
# ABS FIRST (the gate) so its eval grabs the first freed GPU; move_rel/diffabs backfill.
# Exports are CPU-only (no GPU competition). Run via nohup on the login node.
set -uo pipefail
R=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill
SCR=$R/scripts
SPLIT=/fast/home/franz.srambical/osworld_parity_split
CK=$R/checkpoints
STEP=000300
LOG=$R/logs/fire_step300.log
echo "[$(date +%T)] poll-and-fire started" | tee -a "$LOG"

wait_ckpt() { until [ -f "$CK/fmt_sft_8b_osw_$1/$STEP/_CHECKPOINT_METADATA" ]; do sleep 45; done; }

# ---- ABS (gate) FIRST ----
wait_ckpt absolute
EXP_A=$(sbatch --parsable --export=ALL,TAG=osw_absolute,STEP=$STEP "$SCR/export_track2.sbatch")
EV_A=$(sbatch --parsable --dependency=afterok:$EXP_A \
  --export=ALL,MODEL_PATH=$CK/fmt_sft_8b_osw_absolute_step${STEP}_hf,RUN_NAME=abs_step300 \
  "$SPLIT/eval_checkpoint.sbatch")
echo "[$(date +%T)] ABS FIRED: export=$EXP_A eval=$EV_A run=abs_step300" | tee -a "$LOG"

# ---- move_rel (backfill) ----
wait_ckpt moverel
EXP_M=$(sbatch --parsable --export=ALL,TAG=osw_moverel,STEP=$STEP "$SCR/export_track2.sbatch")
EV_M=$(sbatch --parsable --dependency=afterok:$EXP_M \
  --export=ALL,ACTION_FORMAT=move_rel,MODEL_PATH=$CK/fmt_sft_8b_osw_moverel_step${STEP}_hf,RUN_NAME=moverel_step300 \
  "$SPLIT/format_eval.sbatch")
echo "[$(date +%T)] MOVEREL FIRED: export=$EXP_M eval=$EV_M run=moverel_step300" | tee -a "$LOG"

# ---- diffabs (backfill) ----
wait_ckpt diffabs
EXP_D=$(sbatch --parsable --export=ALL,TAG=osw_diffabs,STEP=$STEP "$SCR/export_track2.sbatch")
EV_D=$(sbatch --parsable --dependency=afterok:$EXP_D \
  --export=ALL,ACTION_FORMAT=diffabs,MODEL_PATH=$CK/fmt_sft_8b_osw_diffabs_step${STEP}_hf,RUN_NAME=diffabs_step300 \
  "$SPLIT/format_eval.sbatch")
echo "[$(date +%T)] DIFFABS FIRED: export=$EXP_D eval=$EV_D run=diffabs_step300" | tee -a "$LOG"
echo "[$(date +%T)] all step-300 chains fired" | tee -a "$LOG"
