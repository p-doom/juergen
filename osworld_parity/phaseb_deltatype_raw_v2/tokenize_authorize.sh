#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in dataset source_model output omegalax_repo experiment_dir; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

OUT="${A[output]}"
PYTHON="/opt/miniforge3/bin/python"
[[ -x "$PYTHON" ]] || { echo "FATAL pinned Python missing: $PYTHON" >&2; exit 3; }
[[ ! -e "$OUT/tokenization_manifest.json" ]] || {
  echo "FATAL refusing to overwrite completed tokenization artifact" >&2; exit 2;
}
"$PYTHON" "${A[experiment_dir]}/tokenize_authorize.py" \
  --dataset "${A[dataset]}" --source-model "${A[source_model]}" \
  --output "$OUT" --stage preflight

for split in train val; do
  uv run --project="${A[omegalax_repo]}" -- \
    python "${A[omegalax_repo]}/scripts/build_sft_records_from_chat.py" \
    --data_path "${A[dataset]}/$split/chat.jsonl" \
    --out_dir "$OUT/raw_v2/$split" \
    --model_id "${A[source_model]}/hf" \
    --tokenizer "${A[source_model]}/hf" \
    --processor "${A[source_model]}/hf" \
    --max_length 16384 --overflow_mode truncate \
    --records_per_shard 8000 --num_workers 8 --overwrite
done

"$PYTHON" "${A[experiment_dir]}/../phaseb_relative/preflight_vision_budget.py" \
  --dataset "$OUT" --arm raw_v2 --max-images 29 --max-patches 64000 \
  --out "$OUT/vision_budget_preflight.json"
"$PYTHON" "${A[experiment_dir]}/tokenize_authorize.py" \
  --dataset "${A[dataset]}" --source-model "${A[source_model]}" \
  --output "$OUT" --stage finalize
echo "raw-v2 tokenization sealed"
