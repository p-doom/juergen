#!/bin/bash
set -euo pipefail
# Common Orbax -> HF export. export_to_hf.py initializes JAX distributed, so it
# MUST run inside an srun step even when JAX_PLATFORMS=cpu and world size is one.
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) k="${arg%%=*}"; k="${k#--}"; A[$k]="${arg#*=}";;
    *) echo "FATAL unexpected arg: $arg" >&2; exit 2;; esac
done
for k in omegalax_repo checkpoint_path output arm model_id; do
  [[ -n "${A[$k]:-}" ]] || { echo "FATAL missing --$k" >&2; exit 2; }
done
MANIFEST="${A[manifest_name]:-export_manifest.json}"
[[ "$MANIFEST" != */* ]] || { echo "FATAL manifest_name must be a basename" >&2; exit 2; }
LORA_RANK="${A[lora_rank]:-32}"
LORA_ALPHA="${A[lora_alpha]:-$LORA_RANK}"
[[ "$LORA_RANK" =~ ^[1-9][0-9]*$ && "$LORA_ALPHA" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL lora_rank/lora_alpha must be positive integers" >&2; exit 2;
}
OMX="${A[omegalax_repo]}"; CKPT="${A[checkpoint_path]}"; OUT="${A[output]}"; HF="$OUT/hf"
[[ -f "$OMX/scripts/export_to_hf.py" ]] || { echo "FATAL bad Omegalax: $OMX" >&2; exit 2; }
[[ -d "$CKPT" && -f "$CKPT/_CHECKPOINT_METADATA" ]] || {
  echo "FATAL intact Orbax checkpoint missing: $CKPT" >&2; exit 2;
}
STEP="$(basename "$CKPT")"; [[ "$STEP" =~ ^[0-9]{6}$ ]] || exit 2
[[ -f "$(dirname "$CKPT")/lora_metadata.json" ]] || {
  echo "FATAL lora_metadata.json missing beside checkpoint" >&2; exit 2;
}
mkdir -p "$HF"; cd "$OMX"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id="${A[model_id]}" \
  --checkpoint_path="$CKPT" --out_dir="$HF" --tp_size=1 --fsdp_size=1 \
  --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8

SLUG="${A[model_id]//\//--}"
BASE="${HF_HOME}/hub/models--${SLUG}/snapshots"
STOCK="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
[[ -f "$STOCK/config.json" ]] || { echo "FATAL stock snapshot missing: $BASE" >&2; exit 3; }
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$STOCK/$f" || -f "$HF/$f" ]] || cp "$STOCK/$f" "$HF/"
done
SHA="$(git -C "$OMX" rev-parse HEAD)"
DIFF="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$STOCK/config.json" "$HF/config.json" "$OUT/$MANIFEST" \
  "${A[arm]}" "${A[model_id]}" "$CKPT" "$STEP" "$SHA" "$DIFF" \
  "$LORA_RANK" "$LORA_ALPHA" <<'PY'
import json,sys
from pathlib import Path
stock,out,manifest=map(Path,sys.argv[1:4])
arm,model,ckpt,step,sha,diff=sys.argv[4:10]
rank=int(sys.argv[10]); alpha=int(sys.argv[11])
meta=json.loads((Path(ckpt).parent/"lora_metadata.json").read_text())
if (int(meta.get("lora_rank", -1)) != rank
        or float(meta.get("lora_alpha", -1)) != float(alpha)):
    raise SystemExit(f"FATAL LoRA metadata mismatch: {meta} vs r={rank} alpha={alpha}")
s=json.loads(stock.read_text()); o=json.loads(out.read_text())
for k in ("architectures","transformers_version","vision_end_token_id"):
    if k in s and k not in o: o[k]=s[k]
if not o.get("architectures"): raise SystemExit("FATAL config lacks architectures")
out.write_text(json.dumps(o,indent=2)+"\n"); hf=out.parent
if not ((hf/"model.safetensors").is_file() or (hf/"model.safetensors.index.json").is_file()):
    raise SystemExit("FATAL weights missing")
for f in ("tokenizer_config.json","chat_template.json","preprocessor_config.json"):
    if not (hf/f).is_file(): raise SystemExit(f"FATAL runtime file missing: {f}")
manifest.write_text(json.dumps({"artifact_type":"relative_factorial_hf_checkpoint",
 "schema_version":1,"status":"complete","arm":arm,"model_id":model,
 "source_checkpoint":ckpt,"step":int(step),"lora_rank":rank,"lora_alpha":alpha,
 "max_length":4096,"hf_subdir":"hf","export_ran_inside_srun":True,
 "omegalax_commit":sha,"omegalax_tracked_diff_sha256":diff},indent=2,sort_keys=True)+"\n")
PY
echo "checkpoint export complete: ${A[arm]} $CKPT -> $HF"
