#!/bin/bash
set -euo pipefail
declare -A A
for arg in "$@"; do case "$arg" in --*=*) k="${arg%%=*}"; k="${k#--}"; A[$k]="${arg#*=}";;
  *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac; done
for k in source_root raw_twin collected_root audit_dir onpolicy_scripts source_model omegalax output experiment_dir; do
  [[ -n "${A[$k]:-}" ]] || { echo "FATAL missing --$k" >&2; exit 2; }
done
OUT="${A[output]}"; PY=/opt/miniforge3/bin/python
[[ ! -e "$OUT/tokenization_manifest.json" ]] || { echo "FATAL completed artifact exists" >&2; exit 2; }
"$PY" "${A[experiment_dir]}/build.py" \
  --source-root "${A[source_root]}" --raw-twin "${A[raw_twin]}" \
  --collected-root "${A[collected_root]}" --audit-dir "${A[audit_dir]}" \
  --onpolicy-scripts "${A[onpolicy_scripts]}" --output "$OUT"
for split in train val; do
  uv run --project="${A[omegalax]}" -- \
    python "${A[omegalax]}/scripts/build_sft_records_from_chat.py" \
    --data_path "$OUT/_normalized/$split/chat.jsonl" \
    --out_dir "$OUT/normalized_v2/$split" \
    --model_id "${A[source_model]}/hf" --tokenizer "${A[source_model]}/hf" \
    --processor "${A[source_model]}/hf" --max_length 16384 \
    --overflow_mode truncate --records_per_shard 8000 --num_workers 8 --overwrite
done
"$PY" "${A[experiment_dir]}/../phaseb_relative/preflight_vision_budget.py" \
  --dataset "$OUT" --arm normalized_v2 --max-images 29 --max-patches 64000 \
  --out "$OUT/vision_budget_preflight.json"
"$PY" "${A[experiment_dir]}/finalize.py" --output "$OUT" \
  --source-model "${A[source_model]}" --omegalax "${A[omegalax]}"
echo "full-call normalized-v2 build, audit, and tokenization complete"
