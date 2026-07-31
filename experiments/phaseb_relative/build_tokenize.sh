#!/bin/bash
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in source out audit_dir onpolicy_scripts collected_root omegalax_repo; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
OUT="${A[out]}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
if [[ ! -f "$OUT/build_manifest.json" ]]; then
  python3 "$EXP/build_relative.py" \
    --source-root "${A[source]}" --out-root "$OUT" \
    --audit-dir "${A[audit_dir]}" --onpolicy-scripts "${A[onpolicy_scripts]}" \
    --collected-root "${A[collected_root]}"
fi
python3 - "$OUT/invariant_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get("status") != "pass": raise SystemExit(f"FATAL invariant report: {r}")
for key in ("assistant_outside_action_identity","task_split_order_identity","user_image_identity"):
    x=r[key]
    if x["passing"] != x["total"]: raise SystemExit(f"FATAL {key}: {x}")
g=r["common_pixel_landing"]
if g["within_2px"] != g["total_coordinate_turns"] or g["max_linf_error_px"] > 2:
    raise SystemExit(f"FATAL common-pixel audit: {g}")
if r["new_numeric_tokens_outside_action"]["leaking"] != 0 or r["fallback_turns"] != 0:
    raise SystemExit("FATAL leakage or fallback conversion")
PY

for arm in prose_keep; do
  for split in train val; do
    src="$OUT/$arm/_normalized/$split/chat.jsonl"
    uv run --project="${A[omegalax_repo]}" -- \
      python "${A[omegalax_repo]}/scripts/build_sft_records_from_chat.py" \
      --data_path "$src" --out_dir "$OUT/$arm/$split" \
      --model_id Qwen/Qwen3-VL-2B-Instruct \
      --tokenizer Qwen/Qwen3-VL-2B-Instruct \
      --processor Qwen/Qwen3-VL-2B-Instruct \
      --max_length 16384 --overflow_mode truncate \
      --records_per_shard 8000 --num_workers 8 --overwrite
  done
done

OMX_SHA="$(git -C "${A[omegalax_repo]}" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "${A[omegalax_repo]}" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$OUT" "$OMX_SHA" "$OMX_DIFF_SHA" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
for arm in ("prose_keep",):
  for split,expected in (("train",2383),("val",233)):
    p=out/arm/split/"metadata.json"; m=json.loads(p.read_text())
    if m.get("num_records") != expected or m.get("max_length") != 16384:
      raise SystemExit(f"FATAL tokenization invariant {p}: {m}")
(out/"build_tokenize_manifest.json").write_text(json.dumps({
 "artifact_type":"phaseb_relative_tokenized","schema_version":1,"status":"complete",
 "arms":["prose_keep"],"train_records_per_arm":2383,
 "val_records_per_arm":233,"max_length":16384,"omegalax_commit":sys.argv[2],
 "omegalax_tracked_diff_sha256":sys.argv[3]},indent=2,sort_keys=True)+"\n")
PY
