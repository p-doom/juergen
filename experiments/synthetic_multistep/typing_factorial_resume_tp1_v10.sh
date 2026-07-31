#!/bin/bash
# Exact TP1 continuation of a frozen typing-factorial step-250 endpoint.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
for key in repo omegalax_repo dataset source_model partial_root parent_run \
  parent_run_id parent_job_id output format expected_checkpoint_sha \
  expected_train_state_sha expected_iterator_sha expected_lora_sha \
  expected_parent_log_sha memory_safe_smoke expected_omegalax_commit \
  expected_omegalax_diff_sha expected_api_sha expected_vlm_sha \
  expected_memory_test_sha expected_entrypoint_sha expected_lora_source_sha \
  expected_lora_test_sha; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[format]}" in coalesced|perkey) ;; *) echo "FATAL bad format" >&2; exit 2;; esac

REPO="${A[repo]}"; OMX="${A[omegalax_repo]}"; DATA="${A[dataset]}"
SOURCE="${A[source_model]}"; SOURCE_HF="$SOURCE/hf"; PARTIAL="${A[partial_root]}"
RUN="${A[parent_run]}"; OUT="${A[output]}"; SOURCE_ORBAX="$PARTIAL/orbax"
SOURCE_CKPT250="$SOURCE_ORBAX/000250"; ORBAX="$OUT/orbax"
CKPT250="$ORBAX/000250"; CKPT750="$ORBAX/000750"; HF="$OUT/hf"
SMOKE="${A[memory_safe_smoke]}/memory_safe_resume_cuda_smoke.json"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"; LR="5e-5"
LOG="$(find "$RUN/.lab" -maxdepth 1 -type f -name "*_${A[parent_job_id]}.log" -print -quit)"
[[ -f "$LOG" ]] || { echo "FATAL parent log missing" >&2; exit 2; }

deadline_epoch="$(date -d '2026-07-31T12:00:00+02:00' +%s)"
now_epoch="$(date +%s)"
(( now_epoch + 120*60 <= deadline_epoch )) || {
  echo "FATAL hard 12:00 CEST two-hour recovery feasibility gate closed" >&2; exit 2;
}
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
case "$SOURCE_ARM" in reltool_pre) LINEAGE=A;; relraw_pre) LINEAGE=B;;
  *) echo "FATAL wrong stage-1 source arm" >&2; exit 2;; esac

mkdir -p "$OUT"
python3 - "$SMOKE" "$OUT/memory_safe_smoke_preflight.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
source,out=map(Path,sys.argv[1:3]); actual=hashlib.sha256(source.read_bytes()).hexdigest()
expected='a5532720767745d3dd0bc0de8613ec250677909c3c526e453d93b76bb1100de0'
r=json.loads(source.read_text()); contract=r.get('exact_restore_attestation',{}).get('optimizer_contract',{})
if (actual!=expected or r.get('status')!='pass' or r.get('backend')!='gpu'
    or r.get('local_device_count')!=1 or not r.get('optimizer_model_bit_exact')
    or not r.get('rng_exact') or not r.get('iterator_exact')
    or contract.get('promoted_leaf_count')!=6 or contract.get('converted_leaf_count')!=0
    or contract.get('promoted_source_bytes')!=50356224
    or contract.get('fresh_zero_state_bytes')!=25178112
    or not contract.get('promoted_arrays_bitwise_untouched')
    or not contract.get('nnx_merge_preserved_promoted_state')):
 raise SystemExit(f'FATAL pinned legacy-dtype CUDA restore smoke mismatch: {actual} {r}')
out.write_text(json.dumps({'status':'pass','sha256':actual,'source':str(source.resolve()),
 'legacy_fp32_promotions_bitwise_preserved':True,'next_full_update_finite':True},
 indent=2,sort_keys=True)+'\n')
PY

python3 - "$DATA" "$SOURCE" "$SOURCE_CKPT250" "$RUN" "$LOG" \
  "${A[parent_run_id]}" "${A[parent_job_id]}" "${A[format]}" \
  "${A[expected_checkpoint_sha]}" "${A[expected_train_state_sha]}" \
  "${A[expected_iterator_sha]}" "${A[expected_lora_sha]}" \
  "${A[expected_parent_log_sha]}" "$OMX" "$REPO" \
  "$OUT/resume_preflight.json" "${A[expected_omegalax_commit]}" \
  "${A[expected_omegalax_diff_sha]}" "${A[expected_api_sha]}" \
  "${A[expected_vlm_sha]}" "${A[expected_memory_test_sha]}" \
  "${A[expected_entrypoint_sha]}" "${A[expected_lora_source_sha]}" \
  "${A[expected_lora_test_sha]}" <<'PY'
import hashlib,json,math,re,shlex,subprocess,sys
from pathlib import Path
data,source,ckpt,run,log=map(Path,sys.argv[1:6]); run_id=sys.argv[6]; job_id=int(sys.argv[7])
fmt=sys.argv[8]; expected=sys.argv[9:14]; omx=Path(sys.argv[14]); repo=Path(sys.argv[15])
out=Path(sys.argv[16]); expected_commit,expected_diff=sys.argv[17:19]
expected_files=sys.argv[19:25]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
paths=[ckpt/'_CHECKPOINT_METADATA',ckpt/'train_state/_METADATA',
 ckpt/'input_iter/process_0-of-1.json',ckpt.parent/'lora_metadata.json',log]
actual=[digest(path) for path in paths]
if actual!=expected: raise SystemExit(f'FATAL step250 endpoint hash mismatch: {list(zip(actual,expected))}')
if run.name!=run_id or ckpt.name!='000250': raise SystemExit('FATAL parent identity mismatch')
d=json.loads((data/'typing_dataset_manifest.json').read_text())
fixed={'artifact_type':'synthetic_typing_factorial_tokenized','status':'complete',
 'train_records_per_format':2000,'validation_records_per_format':200,'max_length':4096}
if any(d.get(k)!=v for k,v in fixed.items()): raise SystemExit('FATAL wrong dataset')
s=json.loads((source/'train_export_manifest.json').read_text())
if s.get('status')!='complete' or s.get('step')!=750 or s.get('lora_rank')!=256:
 raise SystemExit('FATAL wrong source model')
for split,n in (('train',2000),('val',200)):
 meta=json.loads((data/fmt/split/'metadata.json').read_text())
 if meta.get('num_records')!=n or meta.get('max_length')!=4096: raise SystemExit('FATAL wrong data cell')

omx_commit=subprocess.check_output(['git','-C',str(omx),'rev-parse','HEAD'],text=True).strip()
omx_diff=hashlib.sha256(subprocess.check_output(['git','-C',str(omx),'diff','--binary'])).hexdigest()
status=subprocess.check_output(['git','-C',str(omx),'status','--porcelain=v1','--untracked-files=no'],text=True).splitlines()
file_paths=[omx/'omegalax/vlm/api.py',omx/'omegalax/trainers/vlm.py',
 omx/'tests/test_vlm_memory_safe_restore.py',omx/'scripts/train_vlm_sft.py',
 omx/'omegalax/trainers/lora.py',omx/'tests/test_lora.py']
file_actual=[digest(path) for path in file_paths]
if (omx_commit!=expected_commit or omx_diff!=expected_diff or file_actual!=expected_files
    or status!=[' M omegalax/trainers/vlm.py',' M omegalax/vlm/api.py']):
 raise SystemExit(f'FATAL OmegaLAX snapshot mismatch: {omx_commit=} {omx_diff=} {file_actual=} {status=}')

bundle=run/'.lab/provenance/juergen_rft'; patch=bundle/'untracked.patch'
header=('diff --git a/experiments/synthetic_multistep/typing_factorial_train_export.sh '
        'b/experiments/synthetic_multistep/typing_factorial_train_export.sh')
lines=patch.read_text().splitlines(keepends=True)
start=next(i for i,line in enumerate(lines) if line.rstrip('\n')==header)
reconstructed=[]; in_hunk=False
for line in lines[start+1:]:
 if line.startswith('diff --git '): break
 if line.startswith('@@ '): in_hunk=True; continue
 if in_hunk and line.startswith('+') and not line.startswith('+++'): reconstructed.append(line[1:])
parent_text=''.join(reconstructed)
if hashlib.sha256(parent_text.encode()).hexdigest()!='4bff34fa17dfd7b22d215aaa18828cbfac2d9f463f3798c5dd70ca7c12c84aa9':
 raise SystemExit('FATAL parent trainer reconstruction mismatch')
current=(repo/'experiments/synthetic_multistep/typing_factorial_resume_tp1_v10.sh').read_text()
def options(text):
 marker='uv run --project="$OMX" -- srun '
 begin=text.rfind(marker); selected=[]
 if begin<0: raise SystemExit('FATAL training marker missing')
 for line in text[begin:].splitlines():
  selected.append(line)
  if '--wandb_tags=' in line: break
 tokens=shlex.split('\n'.join(selected).replace('\\\n',' ')); tokens=tokens[tokens.index('python')+2:]
 return {t[2:].split('=',1)[0]:t.split('=',1)[1] for t in tokens if t.startswith('--') and '=' in t}
p=options(parent_text); q=options(current)
if p.get('resume')!='if_present' or q.get('resume')!='required': raise SystemExit('FATAL resume policy')
for key in ('val_data_path','val_every','val_steps'):
 if key not in p or key in q: raise SystemExit(f'FATAL validation-only change gate: {key}')
if tuple(p.get(k) for k in ('tp_size','fsdp_size','dp_size'))!=('1','1','1') or tuple(q.get(k) for k in ('tp_size','fsdp_size','dp_size'))!=('1','1','1'):
 raise SystemExit('FATAL topology changed')
for item in (p,q):
 for key in ('resume','val_data_path','val_every','val_steps','wandb_name','wandb_tags'):
  item.pop(key,None)
if p!=q: raise SystemExit(f'FATAL training semantics changed: {p} != {q}')
text=log.read_text(errors='replace')
trace=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',text)]
if [x[0] for x in trace]!=list(range(10,250,10)) or not all(math.isfinite(x) and math.isfinite(y) for _,x,y in trace):
 raise SystemExit('FATAL parent finite trace incomplete')
save=text.find('Saving checkpoint at step 250'); final=text.find('Finished saving checkpoint (finalized tmp dir)',save)
oom=text.find('RESOURCE_EXHAUSTED: Out of memory',final)
if not (0<=save<final<oom): raise SystemExit('FATAL parent save/OOM ordering')
iterator=json.loads((ckpt/'input_iter/process_0-of-1.json').read_text())
wanted={'next_index_in_cycle':0,'next_index_in_datasets':2,'iterators_in_use_indices':[0,1],
 'iterators_in_use_states':[{'next_index':1000},{'next_index':1000}],'exhausted':[0,0]}
if iterator!=wanted: raise SystemExit('FATAL iterator state mismatch')
report={'status':'pass','parent_run_id':run_id,'parent_job_id':job_id,'format':fmt,
 'endpoint_hashes':dict(zip(('checkpoint_metadata_sha256','train_state_metadata_sha256',
  'input_iterator_sha256','lora_metadata_sha256','parent_log_sha256'),actual)),
 'omegalax_snapshot':{'git_commit':omx_commit,'tracked_binary_diff_sha256':omx_diff,
  'file_sha256':dict(zip([str(x.relative_to(omx)) for x in file_paths],file_actual))},
 'parent_finite_steps':[x[0] for x in trace],'input_iterator_state':iterator,
 'parent_step250_finalized_before_validation_oom':True,
 'allowed_change':'disable_in_loop_validation_only','topology_unchanged':{'tp':1,'fsdp':1,'dp':1},
 'training_semantics_match_parent':True,'external_frozen_200_example_evaluation_unchanged':True}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
PY

LORA_TEST_LOG="$OUT/lora_cpu_tests.log"
JAX_PLATFORMS=cpu PYTHONPATH="$OMX" "$OMX/.venv/bin/python" "$OMX/tests/test_lora.py" \
  >"$LORA_TEST_LOG" 2>&1
python3 - "$LORA_TEST_LOG" "$OUT/lora_cpu_tests.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
log,out=map(Path,sys.argv[1:3]); text=log.read_text(errors='replace')
required=('test_grad_isolation_via_wrt_filter','test_base_kernel_bit_exact_after_step','Ran 9 tests','OK')
if any(token not in text for token in required) or 'FAILED (' in text: raise SystemExit('FATAL LoRA CPU gates failed')
out.write_text(json.dumps({'status':'pass','test_count':9,
 'gradient_isolation':True,'base_kernel_bit_exact_after_step':True,
 'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()},indent=2,sort_keys=True)+'\n')
PY

CLONE_TMP="$OUT/.orbax_clone_${SLURM_JOB_ID:-local}.tmp"
[[ ! -e "$ORBAX" && ! -e "$CLONE_TMP" ]] || { echo "FATAL resume destination exists" >&2; exit 2; }
mkdir -p "$CLONE_TMP"
cp -a --reflink=auto --sparse=always "$SOURCE_ORBAX/." "$CLONE_TMP/"
python3 - "$SOURCE_ORBAX" "$CLONE_TMP" "$ORBAX" "$OUT/orbax_clone_manifest.json" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
source,clone,final,manifest=map(Path,sys.argv[1:5])
def entries(root):
 out={}
 for path in sorted(root.rglob('*')):
  rel=path.relative_to(root).as_posix(); mode=path.lstat().st_mode
  if stat.S_ISLNK(mode): out[rel]=('symlink',os.readlink(path))
  elif stat.S_ISDIR(mode): out[rel]=('directory',)
  elif stat.S_ISREG(mode): out[rel]=('file',path.stat().st_size)
  else: raise SystemExit(f'FATAL unsupported path: {path}')
 return out
a=entries(source); b=entries(clone)
if a!=b: raise SystemExit('FATAL clone inventory mismatch')
tree=hashlib.sha256(); count=0; size=0
for rel,info in a.items():
 tree.update(rel.encode()+b'\0'+info[0].encode()+b'\0')
 if info[0]=='symlink': tree.update(info[1].encode()+b'\0')
 if info[0]!='file': continue
 count+=1; size+=info[1]; digest=hashlib.sha256()
 with (source/rel).open('rb') as left,(clone/rel).open('rb') as right:
  while True:
   x=left.read(8*1024*1024); y=right.read(8*1024*1024)
   if x!=y: raise SystemExit(f'FATAL clone differs: {rel}')
   if not x: break
   digest.update(x)
 tree.update(str(info[1]).encode()+b'\0'+digest.digest())
manifest.write_text(json.dumps({'status':'pass','source_root':str(source.resolve()),
 'destination_root':str(final.resolve()),'file_count':count,'logical_bytes':size,
 'tree_sha256':tree.hexdigest(),'byte_compared_every_file':True,
 'parent_source_is_not_training_save_root':True},indent=2,sort_keys=True)+'\n')
PY
mv "$CLONE_TMP" "$ORBAX"
[[ -f "$CKPT250/_CHECKPOINT_METADATA" && ! -e "$CKPT750" ]] || { echo "FATAL bad clone endpoint" >&2; exit 2; }

cd "$OMX"
JAX_PLATFORMS=cpu uv run --project="$OMX" -- \
  python "$REPO/experiments/synthetic_multistep/typing_checkpoint_scalar.py" \
  --checkpoint="$CKPT250" --out="$OUT/resume_scalar_audit.json" \
  --expected-gradient-step=250 --expected-micro-step=2000

job_tag="${SLURM_JOB_ID:-local}_${LINEAGE}_${A[format]}_tp1_v10"
# All four frozen cells have the same model/topology/batch shapes. Reuse the
# pilot's content-addressed JAX executable cache after its finite-step gate;
# cache lookup cannot change training values or arguments.
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/typing_135629_A_coalesced_tp1_v10"
mkdir -p "$JAX_CACHE"
uv run --project="$OMX" -- srun --gpus-per-task=1 python scripts/train_vlm_sft.py \
  --jax_cache_dir="$JAX_CACHE" --model_id="$SOURCE_HF" --processor="$SOURCE_HF" \
  --data_path="$DATA/${A[format]}/train" \
  --save_dir="$ORBAX" --enable_lora=true --lora_rank=256 --lora_alpha=256 \
  --freeze_vision_tower=false --tp_size=1 --fsdp_size=1 --dp_size=1 \
  --batch_size=1 --grad_accum_steps=8 --learning_rate="$LR" \
  --lr_schedule=wsd --lr_stable_fraction=0.7 --lr_end_factor=0.0 \
  --warmup_steps=30 --weight_decay=0.01 --max_grad_norm=1.0 \
  --max_length=4096 --num_steps=750 --num_loss_tiles=8 \
  --keep_latest=3 --keep_period=250 --save_every=250 \
  --log_every=10 --log_memory=false --resume=required --gc_period=3000 \
  --pad_id=0 --seed=0 --text_attn_backend=mosaic_gpu --peak_tflops=h100_sxm \
  --max_vision_images_per_sample=2 --max_vision_patches_per_sample=16000 \
  --grain_read_threads=4 --grain_read_buffer_size=4 --grain_workers=2 \
  --grain_worker_buffer_size=2 --wandb_entity=pdoom --wandb_project=omegalax \
  --wandb_group="typing_factorial_r256" --wandb_name="typing_${job_tag}" \
  --wandb_tags="berlin,typing,factorial,lora,r256,resume250,tp1_v10,${A[format]},lineage_${LINEAGE}"

python3 - "$ORBAX/restore_memory_release.json" "$ORBAX/restore_exact_state.json" \
  "$OUT/production_restore_gate.json" <<'PY'
import json,sys
from pathlib import Path
release,exact,out=map(Path,sys.argv[1:4]); a=json.loads(release.read_text()); b=json.loads(exact.read_text())
c=b.get('optimizer_contract',{}); groups=c.get('groups',{})
wanted={'leaf_count':504,'numel':698351616,'restored_source_bytes':2793406464,
 'fresh_zero_state_bytes':1396703232}
if (a.get('status')!='release_pass' or not a.get('initialized_optimizer_collected')
    or not a.get('initialized_model_collected') or a.get('live_initialized_array_count_after_gc')!=0
    or not a.get('gpu_memory_drop_verified') or b.get('status')!='restore_pass'
    or b.get('resume_step')!=250 or not b.get('written_before_first_optimizer_update')
    or b.get('optimizer_counters')!={'optimizer_micro_step':2000,'global_gradient_step':250,
      'gradient_accumulation_remainder':0,'adam_count_0':250,'adam_count_2':250}
    or b.get('rng_key_data')!=[928981903,3453687069]
    or b.get('target_topology')!={'tp':1,'fsdp':1,'dp':1}
    or c.get('promoted_leaf_count')!=1512 or c.get('promoted_source_bytes')!=8380219392
    or c.get('fresh_zero_state_bytes')!=4190109696 or c.get('converted_leaf_count')!=0
    or not c.get('all_shapes_and_shardings_exact') or not c.get('all_other_dtypes_exact')
    or not c.get('promoted_arrays_bitwise_untouched') or not c.get('nnx_merge_preserved_promoted_state')
    or groups!={'acc_grads':wanted,'mu':wanted,'nu':wanted}):
 raise SystemExit(f'FATAL production restore gate mismatch: {a} {b}')
out.write_text(json.dumps({'status':'pass','release_attestation':a,'exact_restore_attestation':b},
 indent=2,sort_keys=True)+'\n')
PY

[[ -f "$CKPT750/_CHECKPOINT_METADATA" && -f "$ORBAX/lora_metadata.json" ]] || {
  echo "FATAL complete step750 checkpoint missing" >&2; exit 3;
}
mkdir -p "$HF"
JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run --project="$OMX" -- \
  python scripts/export_to_hf.py --model_id="$BASE_MODEL" \
  --checkpoint_path="$CKPT750" --out_dir="$HF" \
  --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
  special_tokens_map.json chat_template.json generation_config.json \
  preprocessor_config.json video_preprocessor_config.json; do
  [[ ! -f "$SOURCE_HF/$file" || -f "$HF/$file" ]] || cp "$SOURCE_HF/$file" "$HF/"
done

OMX_SHA="$(git -C "$OMX" rev-parse HEAD)"
OMX_DIFF_SHA="$(git -C "$OMX" diff --binary | sha256sum | awk '{print $1}')"
RECOVERY_LOG="$(find "$(dirname "$LABCTL_CONTEXT")" -maxdepth 1 -type f -name "*.log" -print -quit)"
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$OUT" "$LINEAGE" "$SOURCE_ARM" \
  "${A[format]}" "$CKPT250" "$CKPT750" "$OMX_SHA" "$OMX_DIFF_SHA" \
  "$LOG" "$RECOVERY_LOG" "${A[parent_run_id]}" "${A[parent_job_id]}" \
  "$SOURCE_CKPT250" <<'PY'
import hashlib,json,math,re,sys
from pathlib import Path
data,source,partial,out=map(Path,sys.argv[1:5]); lineage,arm,fmt=sys.argv[5:8]
ck250,ck750=map(Path,sys.argv[8:10]); sha,diff=sys.argv[10:12]
parent_log,recovery_log=map(Path,sys.argv[12:14]); parent_run_id=sys.argv[14]; parent_job_id=int(sys.argv[15])
source_ck250=Path(sys.argv[16]); hf=out/'hf'
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
config=json.loads((hf/'config.json').read_text()); source_config=json.loads((source/'hf/config.json').read_text())
for key in ('architectures','transformers_version','vision_end_token_id'):
 if key in source_config and key not in config: config[key]=source_config[key]
(hf/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ('model.safetensors','config.json','tokenizer_config.json','chat_template.json','preprocessor_config.json'):
 if not (hf/name).is_file(): raise SystemExit(f'FATAL export missing {name}')
text=recovery_log.read_text(errors='replace')
if 'restored checkpoint at step 250' not in text or 'finished step=750' not in text:
 raise SystemExit('FATAL resume/completion evidence missing')
if 'RESOURCE_EXHAUSTED' in text or 'validation loss=' in text or 'built train step (jit) and eval step (jit)' in text:
 raise SystemExit('FATAL validation or OOM in continuation')
trace=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',text)]
if [x[0] for x in trace]!=list(range(260,751,10)) or not all(math.isfinite(x) and math.isfinite(y) for _,x,y in trace):
 raise SystemExit(f'FATAL incomplete/nonfinite continuation trace: {trace}')
pre=json.loads((out/'resume_preflight.json').read_text()); clone=json.loads((out/'orbax_clone_manifest.json').read_text())
restore=json.loads((out/'production_restore_gate.json').read_text()); lora=json.loads((out/'lora_cpu_tests.json').read_text())
source_hashes={'checkpoint_metadata_sha256':digest(source_ck250/'_CHECKPOINT_METADATA'),
 'train_state_metadata_sha256':digest(source_ck250/'train_state/_METADATA'),
 'input_iterator_sha256':digest(source_ck250/'input_iter/process_0-of-1.json'),
 'lora_metadata_sha256':digest(source_ck250.parent/'lora_metadata.json')}
if source_hashes!={k:pre['endpoint_hashes'][k] for k in source_hashes}:
 raise SystemExit('FATAL sealed parent changed')
manifest={'artifact_type':'synthetic_typing_factorial_hf_checkpoint','schema_version':2,
 'status':'complete','lineage':lineage,'source_stage1_arm':arm,'target_format':fmt,
 'source_model':str(source.resolve()),'source_manifest_sha256':digest(source/'train_export_manifest.json'),
 'dataset':str(data.resolve()),'dataset_manifest_sha256':digest(data/'typing_dataset_manifest.json'),
 'model_id':'Qwen/Qwen3-VL-8B-Instruct','step':750,'fresh_optimizer':False,
 'exact_resume_from_step':250,'lora_rank':256,'lora_alpha':256,'learning_rate':5e-5,
 'max_length':4096,'hf_subdir':'hf','source_checkpoint':str(ck750.resolve()),
 'sealed_parent_checkpoint250':str(source_ck250.resolve()),
 'recovered_from_terminal_run_root':str(partial.resolve()),
 'recovered_from_terminal_run_id':parent_run_id,'recovered_from_terminal_job_id':parent_job_id,
 'recovery_reason':'in_loop_validation_gpu_oom_after_finalized_step250',
 'recovery_change':'in_loop_validation_disabled_only','training_topology':{'tp_size':1,'fsdp_size':1,
  'dp_size':1,'global_batch_size':1,'gradient_accumulation_steps':8,'unchanged_from_parent':True},
 'external_frozen_200_example_evaluation_unchanged':True,'resume_preflight':pre,
 'resume_orbax_clone':clone,'production_restore_gate':restore,'lora_base_frozen_gate':lora,
 'first_finite_post_resume_update':{'step':260,'loss':trace[0][1],'grad_norm':trace[0][2]},
 'logged_finite_recovery_steps':[x[0] for x in trace],
 'final_logged_loss':trace[-1][1],'final_logged_grad_norm':trace[-1][2],
 'sealed_parent_orbax_unchanged':True,
 'endpoint_hashes':{'checkpoint250_metadata_sha256':digest(ck250/'_CHECKPOINT_METADATA'),
  'checkpoint750_metadata_sha256':digest(ck750/'_CHECKPOINT_METADATA'),
  'train_state750_metadata_sha256':digest(ck750/'train_state/_METADATA'),
  'input_iterator750_sha256':digest(ck750/'input_iter/process_0-of-1.json'),
  'lora_metadata_sha256':digest(ck750.parent/'lora_metadata.json'),
  'parent_run_log_sha256':digest(parent_log),'recovery_run_log_sha256':digest(recovery_log)},
 'omegalax_commit':sha,'omegalax_tracked_diff_sha256':diff,
 'weights':[{'name':'model.safetensors','size':(hf/'model.safetensors').stat().st_size}]}
(out/'typing_train_export_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
echo "typing factorial TP1 v10 exact-resume complete: lineage=$LINEAGE format=${A[format]}"
