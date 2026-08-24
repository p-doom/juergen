#!/usr/bin/env bash
# Run the estate gate nightly and queue one successor after a successful run.
# SLURM scrontab is disabled on this cluster, so the chain uses delayed sbatch.
#
# Usage: estate_gate_cron.sh ABSOLUTE_REPO EXPECTED_HEAD REMOTE_REF
# REMOTE_REF is the exact refs/remotes/* ref that published EXPECTED_HEAD.

set -uo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: estate_gate_cron.sh ABSOLUTE_REPO EXPECTED_HEAD REMOTE_REF" >&2
  exit 2
fi

REPO="$1"
EXPECTED_HEAD="$2"
REMOTE_REF="$3"
: "${ESTATE_GATE_LOG_DIR:=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/estate_gate_log}"
: "${ESTATE_GATE_AT:=03:30}"
JOB_NAME=estate_gate
GATE="$REPO/tooling/estate_gate.sh"
CRON="$REPO/tooling/estate_gate_cron.sh"
RUNNING_SCRIPT="${BASH_SOURCE[0]}"
CONTRACT_ERROR=

contract_error() {
  CONTRACT_ERROR="$1"
  return 1
}

check_checkout() {
  local canonical head remote_head status top

  case "$REPO" in
    /*) ;;
    *) contract_error "repository root is not absolute"; return ;;
  esac
  canonical="$(cd -- "$REPO" 2>/dev/null && pwd -P)" || {
    contract_error "repository root is not readable"
    return
  }
  if [ "$canonical" != "$REPO" ]; then
    contract_error "repository root is not canonical"
    return
  fi
  if [[ ! "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
    contract_error "expected HEAD is not a full lowercase commit ID"
    return
  fi
  case "$REMOTE_REF" in
    refs/remotes/*) ;;
    *) contract_error "remote ref must be under refs/remotes"; return ;;
  esac
  if ! git check-ref-format "$REMOTE_REF" >/dev/null 2>&1; then
    contract_error "remote ref is malformed"
    return
  fi
  top="$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)" || {
    contract_error "repository root is not a git checkout"
    return
  }
  if [ "$top" != "$REPO" ]; then
    contract_error "repository root is not the checkout top level"
    return
  fi
  head="$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" || {
    contract_error "checkout HEAD is unreadable"
    return
  }
  if [ "$head" != "$EXPECTED_HEAD" ]; then
    contract_error "checkout HEAD moved"
    return
  fi
  remote_head="$(git -C "$REPO" rev-parse --verify "${REMOTE_REF}^{commit}" 2>/dev/null)" || {
    contract_error "remote ref is unreadable"
    return
  }
  if [ "$remote_head" != "$EXPECTED_HEAD" ]; then
    contract_error "remote ref does not publish expected HEAD"
    return
  fi
  status="$(git -C "$REPO" status --porcelain=v1 --untracked-files=all --ignore-submodules=none 2>/dev/null)" || {
    contract_error "checkout status is unreadable"
    return
  }
  if [ -n "$status" ]; then
    contract_error "checkout is dirty"
    return
  fi
  if ! git -C "$REPO" ls-files --error-unmatch -- \
      tooling/estate_gate.sh tooling/estate_gate_cron.sh >/dev/null 2>&1; then
    contract_error "estate gate scripts are not tracked"
    return
  fi
  if [ ! -f "$GATE" ] || [ -L "$GATE" ] || [ ! -f "$CRON" ] || [ -L "$CRON" ]; then
    contract_error "estate gate scripts are not regular files"
    return
  fi
  if ! cmp -s -- "$RUNNING_SCRIPT" "$CRON"; then
    contract_error "invoked cron script does not match the remote checkout"
    return
  fi
  return 0
}

if ! check_checkout; then
  printf 'refusing to run: %s\n' "$CONTRACT_ERROR" >&2
  exit 2
fi

umask 007
mkdir -p "$ESTATE_GATE_LOG_DIR" || exit 2
chmod g+rx "$ESTATE_GATE_LOG_DIR" || exit 2
verdicts="$ESTATE_GATE_LOG_DIR/verdicts.log"
run_id="${SLURM_JOB_ID:-manual-$$}"
log="$ESTATE_GATE_LOG_DIR/gate-$(date +%Y%m%dT%H%M%S)-$run_id.log"

if (cd -- "$REPO" && ESTATE_GATE_STRICT=0 bash "$GATE") > "$log" 2>&1; then
  rc=0
else
  rc=$?
fi
chmod g+r "$log"

mapfile -t verdict_lines < <(
  sed 's/\x1b\[[0-9;]*m//g' "$log" | grep '^ESTATE GATE:' || true
)
verdict="${verdict_lines[0]:-NO VERDICT -- the gate did not reach one}"
printf '%s  rc=%d  %s  (%s)\n' \
  "$(date -Is)" "$rc" "$verdict" "$(basename "$log")" >> "$verdicts"
if [ "${#verdict_lines[@]}" -ne 1 ]; then
  printf '%s  ALERT  gate produced %d authoritative verdicts; successor not queued\n' \
    "$(date -Is)" "${#verdict_lines[@]}" >> "$verdicts"
  echo "gate did not produce exactly one authoritative verdict" >&2
  chmod g+r "$verdicts"
  exit 2
fi
case "$rc:$verdict" in
  "0:ESTATE GATE: GREEN"*) gate_rc=0 ;;
  "1:ESTATE GATE: RED"*) gate_rc=1 ;;
  *)
    printf '%s  ALERT  rc=%d disagrees with authoritative verdict; successor not queued\n' \
      "$(date -Is)" "$rc" >> "$verdicts"
    echo "gate exit status disagrees with authoritative verdict" >&2
    chmod g+r "$verdicts"
    exit 2
    ;;
esac
if [ "$gate_rc" -eq 1 ]; then
  printf '%s  ALERT  rc=%d  %s\n' "$(date -Is)" "$rc" "$log" >> "$verdicts"
fi

exec 9>"$ESTATE_GATE_LOG_DIR/schedule.lock"
if ! flock -x 9; then
  printf '%s  ALERT  schedule lock failed; successor not queued\n' \
    "$(date -Is)" >> "$verdicts"
  chmod g+r "$verdicts"
  exit 2
fi
if ! check_checkout; then
  printf '%s  ALERT  %s; successor not queued\n' \
    "$(date -Is)" "$CONTRACT_ERROR" >> "$verdicts"
  printf 'refusing to rearm: %s\n' "$CONTRACT_ERROR" >&2
  chmod g+r "$verdicts"
  exit 2
fi

PENDING_IDS=()
read_pending() {
  local id output
  if ! output="$(squeue -h --me -n "$JOB_NAME" -t PENDING -o %i)"; then
    contract_error "squeue failed"
    return
  fi
  PENDING_IDS=()
  if [ -z "$output" ]; then
    return 0
  fi
  while IFS= read -r id; do
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
      contract_error "squeue returned a non-numeric job ID"
      return
    fi
    PENDING_IDS+=("$id")
  done <<< "$output"
}

if ! read_pending; then
  printf '%s  ALERT  %s; successor not queued\n' \
    "$(date -Is)" "$CONTRACT_ERROR" >> "$verdicts"
  printf '%s\n' "$CONTRACT_ERROR" >&2
  chmod g+r "$verdicts"
  exit 2
fi
if [ "${#PENDING_IDS[@]}" -gt 1 ]; then
  printf '%s  ALERT  multiple pending successors; successor not queued\n' \
    "$(date -Is)" >> "$verdicts"
  echo "multiple pending successors" >&2
  chmod g+r "$verdicts"
  exit 2
fi

if [ "${#PENDING_IDS[@]}" -eq 0 ]; then
  if ! begin="$(date -d "tomorrow $ESTATE_GATE_AT" +%Y-%m-%dT%H:%M:%S)"; then
    printf '%s  ALERT  invalid ESTATE_GATE_AT; successor not queued\n' \
      "$(date -Is)" >> "$verdicts"
    chmod g+r "$verdicts"
    exit 2
  fi
  if ! submitted="$(sbatch --parsable --job-name="$JOB_NAME" --partition=standard \
      --cpus-per-task=2 --mem=8G --time=02:00:00 --begin="$begin" \
      --chdir="$REPO" --output="$ESTATE_GATE_LOG_DIR/slurm-%j.out" \
      "$CRON" "$REPO" "$EXPECTED_HEAD" "$REMOTE_REF")"; then
    printf '%s  ALERT  sbatch failed; successor not queued\n' "$(date -Is)" >> "$verdicts"
    chmod g+r "$verdicts"
    exit 2
  fi
  if [[ ! "$submitted" =~ ^[0-9]+$ ]]; then
    printf '%s  ALERT  sbatch returned a non-numeric job ID\n' \
      "$(date -Is)" >> "$verdicts"
    echo "sbatch returned a non-numeric job ID" >&2
    chmod g+r "$verdicts"
    exit 2
  fi
  if ! read_pending; then
    printf '%s  ALERT  %s after sbatch\n' "$(date -Is)" "$CONTRACT_ERROR" >> "$verdicts"
    printf '%s\n' "$CONTRACT_ERROR" >&2
    chmod g+r "$verdicts"
    exit 2
  fi
  if [ "${#PENDING_IDS[@]}" -ne 1 ] || [ "${PENDING_IDS[0]}" != "$submitted" ]; then
    printf '%s  ALERT  submitted successor is not the only pending successor\n' \
      "$(date -Is)" >> "$verdicts"
    echo "submitted successor is not the only pending successor" >&2
    chmod g+r "$verdicts"
    exit 2
  fi
  printf '%s  QUEUED  job=%s  head=%s  ref=%s\n' \
    "$(date -Is)" "$submitted" "$EXPECTED_HEAD" "$REMOTE_REF" >> "$verdicts"
fi
chmod g+r "$verdicts"
exit "$gate_rc"
