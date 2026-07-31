#!/bin/bash
# CPU episode build wrapper. labctl expands templates in [args], never command.
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in runtime_python audit_dir out; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
[[ "${A[runtime_python]}" = /* && -x "${A[runtime_python]}" ]] || {
  echo "FATAL runtime_python must be an absolute executable: ${A[runtime_python]}" >&2
  exit 3
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${A[runtime_python]}" "$SCRIPT_DIR/build_episodes.py" \
  --audit-dir "${A[audit_dir]}" --out "${A[out]}"

