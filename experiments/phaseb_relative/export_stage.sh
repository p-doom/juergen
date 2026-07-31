#!/bin/bash
set -euo pipefail
declare -A A
for arg in "$@"; do case "$arg" in --*=*) k="${arg%%=*}"; k="${k#--}"; A[$k]="${arg#*=}";;
  *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac; done
for k in source output omegalax_repo model_id arm; do [[ -n "${A[$k]:-}" ]] || exit 2; done
SRC="${A[source]}"; CKPT="$SRC/orbax/000900"; OUT="${A[output]}"; HF="$OUT/hf"
[[ -f "$SRC/train_manifest.json" && -f "$CKPT/_CHECKPOINT_METADATA" ]] || exit 3
mkdir -p "$HF"; cd "${A[omegalax_repo]}"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="${A[omegalax_repo]}" -- \
 python scripts/export_to_hf.py --model_id="${A[model_id]}" --checkpoint_path="$CKPT" \
 --out_dir="$HF" --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
SLUG="${A[model_id]//\//--}"; BASE="${HF_HOME}/hub/models--${SLUG}/snapshots"
STOCK="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
 special_tokens_map.json chat_template.json generation_config.json preprocessor_config.json \
 video_preprocessor_config.json; do [[ ! -f "$STOCK/$f" || -f "$HF/$f" ]] || cp "$STOCK/$f" "$HF/"; done
python3 - "$STOCK/config.json" "$HF/config.json" "$OUT/export_manifest.json" \
 "${A[arm]}" "$CKPT" "$SRC/train_manifest.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
stock,config,manifest=map(Path,sys.argv[1:4]); arm=sys.argv[4]; ckpt=Path(sys.argv[5]); train=Path(sys.argv[6])
s=json.loads(stock.read_text()); c=json.loads(config.read_text())
for k in ("architectures","transformers_version","vision_end_token_id"):
 if k in s and k not in c: c[k]=s[k]
if not c.get("architectures"): raise SystemExit("FATAL architectures missing")
config.write_text(json.dumps(c,indent=2)+"\n")
weights=sorted(config.parent.glob("*.safetensors"))
if not weights: raise SystemExit("FATAL weights missing")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest.write_text(json.dumps({"artifact_type":"phaseb_relative_hf_checkpoint",
 "schema_version":1,"status":"complete","arm":arm,"model_id":"Qwen/Qwen3-VL-8B-Instruct",
 "source_checkpoint":str(ckpt),"step":900,"lora_rank":32,"lora_alpha":32,
 "max_length":16384,"hf_subdir":"hf","train_manifest_sha256":sha(train),
 "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
 "config_sha256":sha(config),"weights":[{"name":p.name,"size":p.stat().st_size} for p in weights]},
 indent=2,sort_keys=True)+"\n")
PY
