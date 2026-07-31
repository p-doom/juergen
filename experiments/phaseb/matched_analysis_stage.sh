#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in absolute_eval relative_eval absolute_dataset relative_dataset output experiment_dir; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

/opt/miniforge3/bin/python "${A[experiment_dir]}/matched_analysis.py" \
  --absolute-eval "${A[absolute_eval]}" \
  --relative-eval "${A[relative_eval]}" \
  --absolute-val "${A[absolute_dataset]}/prose_keep/_normalized/val/chat.jsonl" \
  --relative-val "${A[relative_dataset]}/prose_keep/_normalized/val/chat.jsonl" \
  --invariant-report "${A[relative_dataset]}/invariant_report.json" \
  --out "${A[output]}" --n-boot 20000
