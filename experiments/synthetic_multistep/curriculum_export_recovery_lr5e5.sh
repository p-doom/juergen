#!/bin/bash
# Seal an authorized low-LR stage-2 checkpoint with an independently audited CPU export.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in omegalax_repo dataset source_model partial_root run_dir run_id job_id output; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
OMX="${A[omegalax_repo]}"; DATA="${A[dataset]}"; SOURCE="${A[source_model]}"
PARTIAL="${A[partial_root]}"; RUN="${A[run_dir]}"; OUT="${A[output]}"
CKPT="$PARTIAL/orbax/000750"; HF="$OUT/hf"; SOURCE_HF="$SOURCE/hf"
LOG="$(find "$RUN/.lab" -maxdepth 1 -type f -name "*_${A[job_id]}.log" -print -quit)"
[[ -f "$LOG" ]] || { echo "FATAL training log missing" >&2; exit 2; }
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
case "$SOURCE_ARM" in reltool_pre) BRANCH=A_to_B;; relraw_pre) BRANCH=B_to_B;;
  *) echo "FATAL wrong source arm" >&2; exit 2;; esac
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$CKPT" "$RUN" "$LOG" \
  "${A[run_id]}" "${A[job_id]}" <<'PY'
import json,math,re,sys
from pathlib import Path
data,source,partial,checkpoint,run,log=map(Path,sys.argv[1:7]); run_id=sys.argv[7]; job_id=int(sys.argv[8])
d=json.loads((data/'curriculum_dataset_manifest.json').read_text())
if d.get('status')!='complete' or d.get('train_records')!=2000 or d.get('validation_records')!=200:
 raise SystemExit('FATAL wrong curriculum dataset')
s=json.loads((source/'train_export_manifest.json').read_text())
if s.get('status')!='complete' or s.get('lora_rank')!=256 or s.get('arm') not in ('reltool_pre','relraw_pre'):
 raise SystemExit('FATAL wrong stage1 source')
if not (checkpoint/'_CHECKPOINT_METADATA').is_file(): raise SystemExit('FATAL step750 checkpoint missing')
meta=json.loads((checkpoint.parent/'lora_metadata.json').read_text())
if int(meta.get('lora_rank',-1))!=256 or float(meta.get('lora_alpha',-1))!=256:
 raise SystemExit('FATAL LoRA metadata mismatch')
if run.name!=run_id: raise SystemExit('FATAL run identity mismatch')
submit=(run/'.lab/submit.sh').read_text()
if f'--learning_rate=5e-5' not in submit or str(partial.resolve()) not in submit:
 raise SystemExit('FATAL low-LR submit provenance mismatch')
matches=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',log.read_text(errors='replace'))]
if [x[0] for x in matches]!=list(range(10,751,10)) or not all(math.isfinite(x) and math.isfinite(y) for _,x,y in matches):
 raise SystemExit('FATAL incomplete/nonfinite training trace')
if 'finished step=750' not in log.read_text(errors='replace'):
 raise SystemExit('FATAL trainer did not finish step750')
PY
[[ ! -e "$OUT/curriculum_train_export_manifest.json" ]] || { echo "FATAL refusing overwrite" >&2; exit 2; }
mkdir -p "$HF"
cd "$OMX"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id=Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_path="$CKPT" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$SOURCE_HF/$file" || -f "$HF/$file" ]] || cp "$SOURCE_HF/$file" "$HF/"
done
OMX_SHA="$(git -C "$OMX" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$OUT" "$BRANCH" "$SOURCE_ARM" "$CKPT" \
  "$OMX_SHA" "$OMX_DIFF_SHA" "$LOG" "${A[run_id]}" "${A[job_id]}" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
data,source,partial,out=map(Path,sys.argv[1:5]); branch,arm=sys.argv[5:7]; ckpt=Path(sys.argv[7])
sha,diff=sys.argv[8:10]; log=Path(sys.argv[10]); run_id=sys.argv[11]; job_id=int(sys.argv[12]); hf=out/'hf'
config=json.loads((hf/'config.json').read_text()); source_config=json.loads((source/'hf/config.json').read_text())
for key in ('architectures','transformers_version','vision_end_token_id'):
 if key in source_config and key not in config: config[key]=source_config[key]
(hf/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ('model.safetensors','config.json','tokenizer_config.json','chat_template.json','preprocessor_config.json'):
 if not (hf/name).is_file(): raise SystemExit(f'FATAL export missing {name}')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
trace=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',log.read_text(errors='replace'))]
manifest={'artifact_type':'synthetic_multistep_curriculum_hf_checkpoint','schema_version':1,'status':'complete',
 'branch':branch,'source_stage1_arm':arm,'source_model':str(source.resolve()),
 'source_manifest_sha256':digest(source/'train_export_manifest.json'),'dataset':str(data.resolve()),
 'dataset_manifest_sha256':digest(data/'curriculum_dataset_manifest.json'),'target_format':'deltatype_raw_pre',
 'model_id':'Qwen/Qwen3-VL-8B-Instruct','step':750,'fresh_optimizer':True,'lora_rank':256,'lora_alpha':256,
 'learning_rate':5e-5,'max_length':4096,'hf_subdir':'hf','source_checkpoint':str(ckpt.resolve()),
 'recovered_from_terminal_run_root':str(partial.resolve()),'recovered_from_terminal_run_id':run_id,
 'recovered_from_terminal_job_id':job_id,'original_inline_manifest_present':(partial/'curriculum_train_export_manifest.json').is_file(),
 'logged_finite_step_checks':len(trace),'logged_step_sequence':[x[0] for x in trace],
 'final_logged_loss':trace[-1][1],'final_logged_grad_norm':trace[-1][2],
 'endpoint_hashes':{'checkpoint_metadata_sha256':digest(ckpt/'_CHECKPOINT_METADATA'),
  'train_state_metadata_sha256':digest(ckpt/'train_state/_METADATA'),
  'input_iterator_sha256':digest(ckpt/'input_iter/process_0-of-1.json'),
  'lora_metadata_sha256':digest(ckpt.parent/'lora_metadata.json'),'terminal_run_log_sha256':digest(log)},
 'omegalax_commit':sha,'omegalax_tracked_diff_sha256':diff}
(out/'curriculum_train_export_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
echo "sealed low-LR curriculum export: $BRANCH"
