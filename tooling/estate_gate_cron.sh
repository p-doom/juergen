#!/usr/bin/env bash
# Run the estate gate nightly and queue one successor after a successful run.
# SLURM scrontab is disabled on this cluster, so the chain uses delayed sbatch.

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
: "${ESTATE_GATE_LOG_DIR:=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/estate_gate_log}"
: "${ESTATE_GATE_AT:=03:30}"
JOB_NAME=estate_gate

scripts_match_head() {
  git -C "$REPO" ls-files --error-unmatch \
    tooling/estate_gate.sh tooling/estate_gate_cron.sh >/dev/null 2>&1 &&
    git -C "$REPO" diff --quiet HEAD -- \
      tooling/estate_gate.sh tooling/estate_gate_cron.sh
}

if ! scripts_match_head; then
  echo "refusing to run: estate gate scripts do not match HEAD" >&2
  exit 2
fi

umask 007
mkdir -p "$ESTATE_GATE_LOG_DIR" || exit 2
chmod g+rx "$ESTATE_GATE_LOG_DIR" || exit 2
verdicts="$ESTATE_GATE_LOG_DIR/verdicts.log"
run_id="${SLURM_JOB_ID:-manual-$$}"
log="$ESTATE_GATE_LOG_DIR/gate-$(date +%Y%m%dT%H%M%S)-$run_id.log"

if ESTATE_GATE_STRICT=0 bash "$HERE/estate_gate.sh" > "$log" 2>&1; then
  rc=0
else
  rc=$?
fi
chmod g+r "$log"

verdict="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep '^ESTATE GATE:' | tail -1)"
: "${verdict:=NO VERDICT -- the gate did not reach one}"
printf '%s  rc=%d  %s  (%s)\n' \
  "$(date -Is)" "$rc" "$verdict" "$(basename "$log")" >> "$verdicts"
if [ "$rc" -ne 0 ]; then
  printf '%s  ALERT  rc=%d  %s\n' "$(date -Is)" "$rc" "$log" >> "$verdicts"
  chmod g+r "$verdicts"
  exit "$rc"
fi

if ! scripts_match_head; then
  printf '%s  ALERT  gate scripts no longer match HEAD; successor not queued\n' \
    "$(date -Is)" >> "$verdicts"
  chmod g+r "$verdicts"
  exit 2
fi

exec 9>"$ESTATE_GATE_LOG_DIR/schedule.lock"
if ! flock -x 9; then
  printf '%s  ALERT  schedule lock failed; successor not queued\n' \
    "$(date -Is)" >> "$verdicts"
  chmod g+r "$verdicts"
  exit 2
fi
if ! pending="$(squeue -h --me -n "$JOB_NAME" -t PENDING -o %i)"; then
  printf '%s  ALERT  squeue failed; successor not queued\n' "$(date -Is)" >> "$verdicts"
  chmod g+r "$verdicts"
  exit 2
fi
if [ -z "$pending" ]; then
  if ! sbatch --job-name="$JOB_NAME" --partition=standard --cpus-per-task=2 \
    --mem=8G --time=02:00:00 \
    --begin="$(date -d "tomorrow $ESTATE_GATE_AT" +%Y-%m-%dT%H:%M:%S)" \
    --chdir="$REPO" --output="$ESTATE_GATE_LOG_DIR/slurm-%j.out" "$SELF"; then
    printf '%s  ALERT  sbatch failed; successor not queued\n' "$(date -Is)" >> "$verdicts"
    chmod g+r "$verdicts"
    exit 2
  fi
fi
chmod g+r "$verdicts"
exit 0
