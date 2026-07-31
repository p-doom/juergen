#!/bin/bash
# Immutable per-cell trainer for the matched typing transfer factorial.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in omegalax_repo dataset source_model output format; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[format]}" in coalesced|perkey) ;; *) echo "FATAL bad format" >&2; exit 2;; esac
OMX="${A[omegalax_repo]}"; DATA="${A[dataset]}"; SOURCE="${A[source_model]}"
SOURCE_HF="$SOURCE/hf"; OUT="${A[output]}"; ORBAX="$OUT/orbax"; HF="$OUT/hf"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"; LR="5e-5"
deadline_epoch="$(date -d '2026-07-31T04:40:00+02:00' +%s)"
now_epoch="$(date +%s)"
(( now_epoch + 90*60 <= deadline_epoch )) || {
  echo "FATAL hard 04:40 CEST finish feasibility gate closed" >&2; exit 2;
}
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
case "$SOURCE_ARM" in reltool_pre) LINEAGE=A;; relraw_pre) LINEAGE=B;;
  *) echo "FATAL wrong stage-1 source arm: $SOURCE_ARM" >&2; exit 2;; esac
python3 - "$DATA" "$SOURCE" "${A[format]}" "$SOURCE_ARM" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,source=map(Path,sys.argv[1:3]); fmt,arm=sys.argv[3:5]
m=json.loads((data/'typing_dataset_manifest.json').read_text())
expected={'artifact_type':'synthetic_typing_factorial_tokenized','status':'complete',
 'formats':['coalesced','perkey'],'train_records_per_format':2000,
 'validation_records_per_format':200,'max_length':4096}
bad={k:(m.get(k),v) for k,v in expected.items() if m.get(k)!=v}
if bad: raise SystemExit(f'FATAL wrong typing dataset: {bad}')
report_path=data/m['pairing_report']; report=json.loads(report_path.read_text())
if hashlib.sha256(report_path.read_bytes()).hexdigest()!=m.get('pairing_report_sha256'):
 raise SystemExit('FATAL pairing report hash mismatch')
if report.get('status')!='pass' or report.get('typed_string_exact')!={'passing':4400,'total':4400}:
 raise SystemExit(f'FATAL typing pairing/oracle gate: {report}')
storage=m.get('storage_gate',{})
if not storage.get('pass') or storage.get('estimated_peak_incremental_bytes',10**20)>700_000_000_000:
 raise SystemExit(f'FATAL storage gate: {storage}')
for split,n in (('train',2000),('val',200)):
 meta=json.loads((data/fmt/split/'metadata.json').read_text())
 if meta.get('num_records')!=n or meta.get('max_length')!=4096:
  raise SystemExit(f'FATAL tokenized cell mismatch {fmt}/{split}: {meta}')
s=json.loads((source/'train_export_manifest.json').read_text())
fixed={'artifact_type':'relative_factorial_hf_checkpoint','status':'complete','arm':arm,
 'model_id':'Qwen/Qwen3-VL-8B-Instruct','step':750,'lora_rank':256,
 'lora_alpha':256,'hf_subdir':'hf'}
bad={k:(s.get(k),v) for k,v in fixed.items() if s.get(k)!=v}
if bad: raise SystemExit(f'FATAL wrong source model: {bad}')
for name in ('config.json','model.safetensors','tokenizer_config.json','chat_template.json','preprocessor_config.json'):
 if not (source/'hf'/name).is_file(): raise SystemExit(f'FATAL source HF missing {name}')
PY
[[ ! -e "$OUT/typing_train_export_manifest.json" ]] || {
  echo "FATAL refusing to overwrite completed typing cell" >&2; exit 2;
}
mkdir -p "$ORBAX"
job_tag="${SLURM_JOB_ID:-local}_${LINEAGE}_${A[format]}_r256"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/typing_${job_tag}"
mkdir -p "$JAX_CACHE"
cd "$OMX"
uv run --project="$OMX" -- srun python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAX_CACHE" --model_id="$SOURCE_HF" --processor="$SOURCE_HF" \
  --data_path="$DATA/${A[format]}/train" --val_data_path="$DATA/${A[format]}/val" \
  --save_dir="$ORBAX" --enable_lora=true --lora_rank=256 --lora_alpha=256 \
  --freeze_vision_tower=false --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 --learning_rate="$LR" \
  --lr_schedule=wsd --lr_stable_fraction=0.7 --lr_end_factor=0.0 \
  --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
  --max_length=4096 --num_steps=750 --num_loss_tiles=8 \
  --keep_latest=3 --keep_period=250 --save_every=250 --val_every=250 --val_steps=15 \
  --log_every=10 --log_memory=false --resume=if_present --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=2 --max_vision_patches_per_sample=16000 \
  --grain_read_threads=4 --grain_read_buffer_size=4 --grain_workers=2 \
  --grain_worker_buffer_size=2 --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group="typing_factorial_r256" --wandb_name="typing_${job_tag}" \
  --wandb_tags="berlin,typing,factorial,lora,r256,${A[format]},lineage_${LINEAGE}"
CKPT="$ORBAX/000750"
[[ -f "$CKPT/_CHECKPOINT_METADATA" && -f "$ORBAX/lora_metadata.json" ]] || {
  echo "FATAL complete step750 typing checkpoint missing" >&2; exit 3;
}
mkdir -p "$HF"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id="$BASE_MODEL" \
  --checkpoint_path="$CKPT" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$SOURCE_HF/$file" || -f "$HF/$file" ]] || cp "$SOURCE_HF/$file" "$HF/"
done
OMX_SHA="$(git -C "$OMX" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$DATA" "$SOURCE" "$OUT" "$LINEAGE" "$SOURCE_ARM" "${A[format]}" \
  "$CKPT" "$OMX_SHA" "$OMX_DIFF_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,source,out=map(Path,sys.argv[1:4]); lineage,arm,fmt=sys.argv[4:7]
ckpt=Path(sys.argv[7]); sha,diff=sys.argv[8:10]; hf=out/'hf'
meta=json.loads((ckpt.parent/'lora_metadata.json').read_text())
if int(meta.get('lora_rank',-1))!=256 or float(meta.get('lora_alpha',-1))!=256:
 raise SystemExit(f'FATAL LoRA mismatch: {meta}')
config=json.loads((hf/'config.json').read_text()); source_config=json.loads((source/'hf/config.json').read_text())
for key in ('architectures','transformers_version','vision_end_token_id'):
 if key in source_config and key not in config: config[key]=source_config[key]
(hf/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ('model.safetensors','config.json','tokenizer_config.json','chat_template.json','preprocessor_config.json'):
 if not (hf/name).is_file(): raise SystemExit(f'FATAL HF export missing {name}')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'artifact_type':'synthetic_typing_factorial_hf_checkpoint','schema_version':1,
 'status':'complete','lineage':lineage,'source_stage1_arm':arm,'target_format':fmt,
 'source_model':str(source.resolve()),'source_manifest_sha256':digest(source/'train_export_manifest.json'),
 'dataset':str(data.resolve()),'dataset_manifest_sha256':digest(data/'typing_dataset_manifest.json'),
 'model_id':'Qwen/Qwen3-VL-8B-Instruct','step':750,'fresh_optimizer':True,
 'lora_rank':256,'lora_alpha':256,'learning_rate':5e-5,'max_length':4096,
 'hf_subdir':'hf','source_checkpoint':str(ckpt.resolve()),
 'checkpoint_metadata_sha256':digest(ckpt/'_CHECKPOINT_METADATA'),
 'omegalax_commit':sha,'omegalax_tracked_diff_sha256':diff,
 'weights':[{'name':'model.safetensors','size':(hf/'model.safetensors').stat().st_size}]}
(out/'typing_train_export_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
echo "typing factorial complete: lineage=$LINEAGE format=${A[format]}"
