#!/bin/bash
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in source output omegalax_repo model_id; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

SRC="${A[source]}"
CKPT="$SRC/orbax/000900"
OUT="${A[output]}"
HF="$OUT/hf"
[[ -f "$SRC/train_manifest.json" && -f "$CKPT/_CHECKPOINT_METADATA" ]] || {
  echo "FATAL complete normalized-v2 step-900 input missing" >&2; exit 3;
}
mkdir -p "$HF"
cd "${A[omegalax_repo]}"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="${A[omegalax_repo]}" -- \
  python scripts/export_to_hf.py --model_id="${A[model_id]}" \
  --checkpoint_path="$CKPT" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8

SLUG="${A[model_id]//\//--}"
BASE="${HF_HOME}/hub/models--${SLUG}/snapshots"
STOCK="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json preprocessor_config.json \
  video_preprocessor_config.json; do
  [[ ! -f "$STOCK/$file" || -f "$HF/$file" ]] || cp "$STOCK/$file" "$HF/"
done

python3 - "$STOCK/config.json" "$HF/config.json" "$OUT/export_manifest.json" \
  "$CKPT" "$SRC/train_manifest.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

stock, config, manifest = map(Path, sys.argv[1:4])
checkpoint, train_path = Path(sys.argv[4]), Path(sys.argv[5])
source = json.loads(train_path.read_text())
if (source.get("artifact_type") !=
        "phaseb_normalized_move_rel_v2_A_to_A_production_control_orbax"
        or source.get("status") != "complete" or source.get("step") != 900
        or source.get("lora_rank") != 256 or source.get("lora_alpha") != 256):
    raise SystemExit("FATAL wrong normalized-v2 training endpoint")
stock_cfg, exported_cfg = json.loads(stock.read_text()), json.loads(config.read_text())
for key in ("architectures", "transformers_version", "vision_end_token_id"):
    if key in stock_cfg and key not in exported_cfg:
        exported_cfg[key] = stock_cfg[key]
if not exported_cfg.get("architectures"):
    raise SystemExit("FATAL exported architectures missing")
config.write_text(json.dumps(exported_cfg, indent=2) + "\n")
weights = sorted(config.parent.glob("*.safetensors"))
if not weights:
    raise SystemExit("FATAL exported weights missing")
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
value = {
    "artifact_type": "phaseb_normalized_move_rel_v2_A_to_A_hf_checkpoint",
    "schema_version": 1,
    "status": "complete",
    "arm": "normalized_v2",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "source_checkpoint": str(checkpoint),
    "step": 900,
    "lora_rank": 256,
    "lora_alpha": 256,
    "max_length": 16384,
    "hf_subdir": "hf",
    "train_manifest_sha256": sha(train_path),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "config_sha256": sha(config),
    "weights": [{"name": path.name, "size": path.stat().st_size} for path in weights],
}
manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
