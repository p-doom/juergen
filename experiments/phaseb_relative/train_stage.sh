#!/bin/bash
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in dataset arm output omegalax_repo model_id; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}" in prose_keep|prose_strip) ;; *) echo "FATAL bad arm" >&2; exit 2;; esac
DATA="${A[dataset]}/${A[arm]}"; OUT="${A[output]}"; ORBAX="$OUT/orbax"
for spec in train:2383 val:233; do
  split="${spec%%:*}"; expected="${spec#*:}"; meta="$DATA/$split/metadata.json"
  python3 - "$meta" "$expected" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if m.get("num_records") != int(sys.argv[2]) or m.get("max_length") != 16384:
 raise SystemExit(f"FATAL tokenized input mismatch {sys.argv[1]}: {m}")
PY
done
mkdir -p "$ORBAX"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$EXP/preflight_vision_budget.py" \
  --dataset "${A[dataset]}" --arm "${A[arm]}" \
  --max-images 29 --max-patches 64000 \
  --out "$OUT/vision_budget_preflight.json"
tag="${SLURM_JOB_ID:-local}_${A[arm]}"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/phaseb_relative_${tag}"
mkdir -p "$JAX_CACHE"
cd "${A[omegalax_repo]}"
uv run --project="${A[omegalax_repo]}" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAX_CACHE" --model_id="${A[model_id]}" --processor="${A[model_id]}" \
  --data_path="$DATA/train" --val_data_path="$DATA/val" --save_dir="$ORBAX" \
  --enable_lora=true --lora_rank=32 --lora_alpha=32 --freeze_vision_tower=false \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --batch_size=1 --grad_accum_steps=8 \
  --learning_rate=1e-4 --lr_schedule=wsd --lr_stable_fraction=0.7 \
  --lr_end_factor=0.0 --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
  --max_length=16384 --num_steps=900 --num_loss_tiles=16 \
  --keep_latest=1 --keep_period=300 --save_every=300 --val_every=300 --val_steps=15 \
  --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=29 --max_vision_patches_per_sample=64000 \
  --grain_read_threads=4 --grain_read_buffer_size=4 --grain_workers=2 \
  --grain_worker_buffer_size=2 --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group="phaseb_relative_${A[arm]}" --wandb_name="phaseb_relative_${tag}" \
  --wandb_tags=berlin,phaseb,relative,moverel,lora,r32
[[ -f "$ORBAX/000900/_CHECKPOINT_METADATA" ]] || { echo "FATAL step900 missing" >&2; exit 3; }
[[ -f "$ORBAX/lora_metadata.json" ]] || { echo "FATAL LoRA metadata missing" >&2; exit 3; }
python3 - "$OUT/train_manifest.json" "${A[arm]}" "$ORBAX/000900" "${A[dataset]}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); ck=Path(sys.argv[3]); ds=Path(sys.argv[4])
def sha(x): return hashlib.sha256(Path(x).read_bytes()).hexdigest()
p.write_text(json.dumps({"artifact_type":"phaseb_relative_orbax","schema_version":1,
 "status":"complete","arm":sys.argv[2],"model_id":"Qwen/Qwen3-VL-8B-Instruct",
 "step":900,"lora_rank":32,"lora_alpha":32,"max_length":16384,"seed":0,
 "checkpoint":str(ck),"checkpoint_metadata_sha256":sha(ck/"_CHECKPOINT_METADATA"),
 "dataset_manifest_sha256":sha(ds/"build_tokenize_manifest.json"),
 "slurm_job_id":os.environ.get("SLURM_JOB_ID")},indent=2,sort_keys=True)+"\n")
PY
