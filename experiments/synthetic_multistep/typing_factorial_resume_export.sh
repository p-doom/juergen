#!/bin/bash
# Authorized exact-resume recovery after the step-250 validation-only OOM.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in repo omegalax_repo dataset source_model partial_root parent_run parent_run_id \
  parent_job_id output format expected_checkpoint_sha expected_train_state_sha \
  expected_iterator_sha expected_lora_sha expected_parent_log_sha offline_nnx_gate \
  expected_omegalax_commit expected_omegalax_diff_sha expected_lora_source_sha \
  expected_lora_test_sha lora_cuda_smoke; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[format]}" in coalesced|perkey) ;; *) echo "FATAL bad format" >&2; exit 2;; esac
REPO="${A[repo]}"; OMX="${A[omegalax_repo]}"; DATA="${A[dataset]}"
SOURCE="${A[source_model]}"; SOURCE_HF="$SOURCE/hf"; PARTIAL="${A[partial_root]}"
RUN="${A[parent_run]}"; OUT="${A[output]}"; SOURCE_ORBAX="$PARTIAL/orbax"
OFFLINE_NNX_GATE="${A[offline_nnx_gate]}"
LORA_CUDA_SMOKE="${A[lora_cuda_smoke]}"
SOURCE_CKPT250="$SOURCE_ORBAX/000250"; ORBAX="$OUT/orbax"
CKPT250="$ORBAX/000250"; CKPT750="$ORBAX/000750"; HF="$OUT/hf"
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"; LR="5e-5"
TP_SIZE="${A[tp_size]:-1}"
case "$TP_SIZE" in 1|2) ;; *) echo "FATAL tp_size must be 1 or 2" >&2; exit 2;; esac
LOG="$(find "$RUN/.lab" -maxdepth 1 -type f -name "*_${A[parent_job_id]}.log" -print -quit)"
[[ -f "$LOG" ]] || { echo "FATAL parent training log missing" >&2; exit 2; }

# A conservative 120-minute projection covers the sealed checkpoint clone and
# byte comparison, resume compilation, 500 updates, two async checkpoints, and
# CPU export. The corrected recovery
# deadline is 09:00 CEST (the user returns at 10:00); never start an unsafe
# rescue.
deadline_epoch="$(date -d '2026-07-31T09:00:00+02:00' +%s)"
now_epoch="$(date +%s)"
(( now_epoch + 120*60 <= deadline_epoch )) || {
  echo "FATAL hard 09:00 CEST two-hour recovery feasibility gate closed" >&2; exit 2;
}
SOURCE_ARM="$(jq -r '.arm' "$SOURCE/train_export_manifest.json")"
case "$SOURCE_ARM" in reltool_pre) LINEAGE=A;; relraw_pre) LINEAGE=B;;
  *) echo "FATAL wrong stage-1 source arm" >&2; exit 2;; esac

mkdir -p "$OUT"
python3 - "$OFFLINE_NNX_GATE/offline_nnx_restore_gate.json" \
  "$OUT/offline_nnx_restore_gate_preflight.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
source,out=map(Path,sys.argv[1:3])
expected_sha='1bb8554ea58aade68622cacfcc33edd2e6f70faf79e0aa4d8e46371b65807f22'
actual=hashlib.sha256(source.read_bytes()).hexdigest()
report=json.loads(source.read_text())
expected_counters={'global_gradient_step':250,'optimizer_micro_step':2000,
 'gradient_accumulation_remainder':0}
if (actual!=expected_sha or report.get('status')!='pass'
    or report.get('optimizer_state_python_type')!='flax.nnx.statelib.State'
    or report.get('leaf_count')!=2772
    or not report.get('all_source_target_leaf_records_equal')
    or not report.get('nnx_update_preserves_exact_contract')
    or report.get('restored_counters')!=expected_counters):
 raise SystemExit(f'FATAL pinned offline real-NNX restore gate mismatch: {actual} {report}')
out.write_text(json.dumps({'status':'pass','source':str(source.resolve()),
 'sha256':actual,'restored_counters':expected_counters},indent=2,sort_keys=True)+'\n')
PY
python3 - "$LORA_CUDA_SMOKE/lora_cuda_smoke.json" \
  "$OUT/lora_cuda_smoke_preflight.json" "${A[expected_omegalax_diff_sha]}" \
  "${A[expected_lora_source_sha]}" "${A[expected_lora_test_sha]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
source,out=map(Path,sys.argv[1:3]); expected_diff,expected_lora,expected_test=sys.argv[3:6]
expected_sha='5c529367c33e0f942a3378830506af6e12c878640aec555c28798dbc37094a0f'
actual=hashlib.sha256(source.read_bytes()).hexdigest(); report=json.loads(source.read_text())
if (actual!=expected_sha or report.get('status')!='pass' or report.get('backend')!='gpu'
    or report.get('process_count')!=1 or report.get('local_device_count')!=2
    or report.get('global_device_count')!=2 or report.get('mesh_shape')!=[2,1,1]
    or not report.get('forward_exact_to_unsharded')
    or not report.get('optimizer_loss_finite')
    or not report.get('base_kernel_bit_exact_after_step')
    or not report.get('lora_adapter_updated')
    or report.get('stablehlo_has_custom_call_sharding') is not False
    or report.get('omegalax_diff_sha256')!=expected_diff
    or report.get('lora_source_sha256')!=expected_lora
    or report.get('lora_test_sha256')!=expected_test):
 raise SystemExit(f'FATAL pinned two-GPU native-dot LoRA smoke mismatch: {actual} {report}')
out.write_text(json.dumps({'status':'pass','source':str(source.resolve()),
 'sha256':actual,'backend':'gpu','local_device_count':2,
 'base_kernel_bit_exact_after_step':True,'lora_adapter_updated':True,
 'stablehlo_has_custom_call_sharding':False},indent=2,sort_keys=True)+'\n')
PY
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$SOURCE_CKPT250" "$RUN" "$LOG" \
  "${A[parent_run_id]}" "${A[parent_job_id]}" "${A[format]}" \
  "${A[expected_checkpoint_sha]}" "${A[expected_train_state_sha]}" \
  "${A[expected_iterator_sha]}" "${A[expected_lora_sha]}" \
  "${A[expected_parent_log_sha]}" "$OMX" "$REPO" "$OUT/resume_preflight.json" \
  "$TP_SIZE" "${A[expected_omegalax_commit]}" \
  "${A[expected_omegalax_diff_sha]}" "${A[expected_lora_source_sha]}" \
  "${A[expected_lora_test_sha]}" <<'PY'
import hashlib,json,math,re,shlex,subprocess,sys
from pathlib import Path
data,source,partial,ckpt,run,log=map(Path,sys.argv[1:7])
run_id=sys.argv[7]; job_id=int(sys.argv[8]); fmt=sys.argv[9]
expected=sys.argv[10:15]; omx=Path(sys.argv[15]); repo=Path(sys.argv[16]); out=Path(sys.argv[17])
tp_size=int(sys.argv[18])
expected_omx_commit,expected_omx_diff,expected_lora_source,expected_lora_test=sys.argv[19:23]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
paths=[ckpt/'_CHECKPOINT_METADATA',ckpt/'train_state/_METADATA',
       ckpt/'input_iter/process_0-of-1.json',ckpt.parent/'lora_metadata.json',log]
actual=[digest(path) for path in paths]
if actual!=expected: raise SystemExit(f'FATAL step250 endpoint hash mismatch: {list(zip(actual,expected))}')
if ckpt.name!='000250' or not all(path.is_file() for path in paths[:-1]):
 raise SystemExit('FATAL exact finalized step250 checkpoint missing')
if run.name!=run_id: raise SystemExit('FATAL parent run identity mismatch')
d=json.loads((data/'typing_dataset_manifest.json').read_text())
fixed={'artifact_type':'synthetic_typing_factorial_tokenized','status':'complete',
 'train_records_per_format':2000,'validation_records_per_format':200,'max_length':4096}
bad={k:(d.get(k),v) for k,v in fixed.items() if d.get(k)!=v}
if bad: raise SystemExit(f'FATAL wrong dataset: {bad}')
s=json.loads((source/'train_export_manifest.json').read_text())
if s.get('status')!='complete' or s.get('step')!=750 or s.get('lora_rank')!=256:
 raise SystemExit('FATAL wrong stage1 source')
for split,n in (('train',2000),('val',200)):
 meta=json.loads((data/fmt/split/'metadata.json').read_text())
 if meta.get('num_records')!=n or meta.get('max_length')!=4096:
  raise SystemExit(f'FATAL wrong {fmt}/{split} data')
parent_source=run/'source/juergen_rft/experiments/synthetic_multistep/typing_factorial_train_export.sh'
parent_context=json.loads((run/'.lab/context.json').read_text())
bundle=run/'.lab/provenance/juergen_rft'
patch=bundle/'untracked.patch'
header=('diff --git a/experiments/synthetic_multistep/typing_factorial_train_export.sh '
        'b/experiments/synthetic_multistep/typing_factorial_train_export.sh')
lines=patch.read_text().splitlines(keepends=True)
try: start=next(i for i,line in enumerate(lines) if line.rstrip('\n')==header)
except StopIteration: raise SystemExit('FATAL parent trainer absent from sealed untracked.patch')
reconstructed=[]; in_hunk=False
for line in lines[start+1:]:
 if line.startswith('diff --git '): break
 if line.startswith('@@ '): in_hunk=True; continue
 if in_hunk and line.startswith('+') and not line.startswith('+++'):
  reconstructed.append(line[1:])
 elif in_hunk and line.startswith('\\ No newline at end of file'):
  continue
 elif in_hunk and not line.startswith(('-', ' ')):
  raise SystemExit(f'FATAL unsupported parent patch line: {line!r}')
parent_text=''.join(reconstructed)
parent_sha=hashlib.sha256(parent_text.encode()).hexdigest()
if parent_sha!='4bff34fa17dfd7b22d215aaa18828cbfac2d9f463f3798c5dd70ca7c12c84aa9':
 raise SystemExit(f'FATAL reconstructed parent trainer hash mismatch: {parent_sha}')
if (bundle/'git_head.txt').read_text().strip()!='860bb66d3f5755b81edf88d9f1d5bca4ca2e0fdb':
 raise SystemExit('FATAL parent provenance git head mismatch')
if parent_context.get('source_hash')!='ae6a90779440de25021a0b9e05743bb9ee9889474e7fbb5d23e173aee7c9de8f':
 raise SystemExit('FATAL parent context source hash mismatch')
if parent_source.is_file():
 if digest(parent_source)!=parent_sha:
  raise SystemExit('FATAL parent trainer snapshot hash mismatch')
 parent_source_gate='live snapshot and provenance reconstruction both matched'
else:
 parent_source_gate='snapshot GCed; exact trainer reconstructed from sealed provenance bundle'
current_text=(repo/'experiments/synthetic_multistep/typing_factorial_resume_export.sh').read_text()
def train_options(text):
 marker='uv run --project="$OMX" -- srun '
 begin=text.rfind(marker)
 if begin<0: raise SystemExit('FATAL training command marker missing')
 selected=[]
 for line in text[begin:].splitlines():
  selected.append(line)
  if '--wandb_tags=' in line: break
 tokens=shlex.split('\n'.join(selected).replace('\\\n',' '))
 try: tokens=tokens[tokens.index('python')+2:]
 except ValueError: raise SystemExit('FATAL training python marker missing')
 return {token[2:].split('=',1)[0]:token.split('=',1)[1]
         for token in tokens if token.startswith('--') and '=' in token}
parent_options=train_options(parent_text); resume_options=train_options(current_text)
if parent_options.get('resume')!='if_present' or resume_options.get('resume')!='required':
 raise SystemExit('FATAL resume-policy invariance gate failed')
for key in ('val_data_path','val_every','val_steps'):
 if key not in parent_options or key in resume_options:
  raise SystemExit(f'FATAL validation-only change gate failed for {key}')
parent_topology=tuple(parent_options.get(key) for key in ('tp_size','fsdp_size','dp_size'))
resume_topology=tuple(resume_options.get(key) for key in ('tp_size','fsdp_size','dp_size'))
if parent_topology!=('1','1','1') or resume_topology!=('$TP_SIZE','1','1'):
 raise SystemExit(f'FATAL topology gate failed: parent={parent_topology} resume={resume_topology}')
for options in (parent_options,resume_options):
 for key in ('resume','val_data_path','val_every','val_steps','wandb_name','wandb_tags',
             'tp_size','fsdp_size','dp_size'):
  options.pop(key,None)
if parent_options!=resume_options:
 raise SystemExit(f'FATAL training semantics mismatch: parent={parent_options} resume={resume_options}')
reconstruction_proof={'git_head':(bundle/'git_head.txt').read_text().strip(),
 'context_source_hash':parent_context['source_hash'],
 'tracked_patch_sha256':digest(bundle/'tracked.patch'),
 'untracked_patch_sha256':digest(patch),
 'reconstructed_parent_trainer_sha256':parent_sha,
 'reconstructed_parent_trainer_bytes':len(parent_text.encode()),
 'current_resume_semantics_match_parent_except_validation_resume_policy_and_declared_tp_topology':True}
if digest(omx/'omegalax/trainers/vlm.py')!='3467bdec9ba5e7c7f410a95a68c145d8c1c81411ea348ee5500db077eafad2a4':
 raise SystemExit('FATAL Omega trainer source hash mismatch')
if digest(omx/'scripts/train_vlm_sft.py')!='b80d64d2328abe83bb8e41db804f6331361ff29f6c3b7a508f848c2edc21c670':
 raise SystemExit('FATAL Omega entrypoint source hash mismatch')
omx_commit=subprocess.check_output(
 ['git','-C',str(omx),'rev-parse','HEAD'],text=True).strip()
omx_diff=hashlib.sha256(subprocess.check_output(
 ['git','-C',str(omx),'diff','--binary'])).hexdigest()
omx_tracked_status=subprocess.check_output(
 ['git','-C',str(omx),'status','--porcelain=v1','--untracked-files=no'],text=True).splitlines()
detached=subprocess.run(
 ['git','-C',str(omx),'symbolic-ref','-q','HEAD'],capture_output=True).returncode!=0
lora_source_sha=digest(omx/'omegalax/trainers/lora.py')
lora_test_sha=digest(omx/'tests/test_lora.py')
if (omx_commit!=expected_omx_commit or omx_diff!=expected_omx_diff
    or lora_source_sha!=expected_lora_source or lora_test_sha!=expected_lora_test
    or not detached
    or omx_tracked_status!=[' M omegalax/trainers/lora.py',' M tests/test_lora.py']):
 raise SystemExit(f'FATAL OmegaLAX detached TP2 repair snapshot mismatch: '
  f'{omx_commit=} {omx_diff=} {lora_source_sha=} {lora_test_sha=} '
  f'{detached=} {omx_tracked_status=}')
omegalax_snapshot={'detached_head':True,'git_commit':omx_commit,
 'root':str(omx.resolve()),
 'tracked_binary_diff_sha256':omx_diff,'lora_source_sha256':lora_source_sha,
 'lora_test_sha256':lora_test_sha,'tracked_status':omx_tracked_status,
 'repair':'LoRA explicit-mesh output uses jax.sharding.reshard'}
text=log.read_text(errors='replace')
trace=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',text)]
if [x[0] for x in trace]!=list(range(10,250,10)):
 raise SystemExit(f'FATAL incomplete parent finite trace: {[x[0] for x in trace]}')
if not all(math.isfinite(loss) and math.isfinite(grad) for _,loss,grad in trace):
 raise SystemExit('FATAL nonfinite parent training trace')
save=text.find('Saving checkpoint at step 250')
final=text.find('Finished saving checkpoint (finalized tmp dir)',save)
oom=text.find('RESOURCE_EXHAUSTED: Out of memory',final)
if not (0<=save<final<oom): raise SystemExit('FATAL save-before-validation-OOM ordering not proven')
trainer=(omx/'omegalax/trainers/vlm.py').read_text()
save_code='if checkpoint_manager is not None and save_every and step % save_every == 0:'
eval_code='if eval_step is not None and val_every and step % val_every == 0:'
if trainer.find(save_code)<0 or trainer.find(eval_code)<0 or trainer.find(save_code)>trainer.find(eval_code):
 raise SystemExit('FATAL trainer save-before-validation source invariant changed')
eval_block=trainer[trainer.find(eval_code):trainer.find('if requeue_requested:',trainer.find(eval_code))]
if 'next(val_data_iter)' not in eval_block or 'eval_step(optimizer.model, val_batch)' not in eval_block:
 raise SystemExit('FATAL separate validation iterator/model-only evaluation not proven')
if re.search(r'\brng\b',eval_block): raise SystemExit('FATAL validation unexpectedly touches training RNG')
iterator=json.loads((ckpt/'input_iter/process_0-of-1.json').read_text())
if iterator!={'next_index_in_cycle':0,'next_index_in_datasets':2,
 'iterators_in_use_indices':[0,1],
 'iterators_in_use_states':[{'next_index':1000},{'next_index':1000}],
 'exhausted':[0,0]}:
 raise SystemExit(f'FATAL unexpected step250 iterator state: {iterator}')
report={'status':'pass','parent_run_id':run_id,'parent_job_id':job_id,
 'parent_partial_root':str(partial.resolve()),'format':fmt,
 'endpoint_hashes':dict(zip(('checkpoint_metadata_sha256','train_state_metadata_sha256',
  'input_iterator_sha256','lora_metadata_sha256','parent_log_sha256'),actual)),
 'logged_finite_steps':[x[0] for x in trace],
 'step250_completion_evidence':'finalized Orbax checkpoint; scalar counter audit follows',
 'parent_source_gate':parent_source_gate,
 'parent_source_reconstruction':reconstruction_proof,
 'omegalax_snapshot':omegalax_snapshot,
 'input_iterator_state':iterator,
 'save_precedes_validation':True,'validation_uses_separate_iterator':True,
 'validation_touches_training_rng':False,
 'allowed_change':'disable_in_loop_validation_and_matched_tp1_to_tp2_transition',
 'operational_memory_allocator':{
  'allocator':'cuda_async','memory_fraction':0.95,
  'purpose':'avoid default 75% BFC pool starvation during transient Orbax restore',
  'changes_training_arguments':False},
 'operational_topology':{
  'source':'TP1/FSDP1/DP1','target':f'TP{tp_size}/FSDP1/DP1',
  'global_batch_size':1,'gradient_accumulation_steps':8,'data_parallel_size':1,
  'sample_order_and_update_grouping_unchanged':True,
  'numerical_topology_change_disclosed':tp_size==2},
 'external_frozen_200_example_evaluation_unchanged':True}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
PY

LORA_CPU_TEST_LOG="$OUT/lora_tp2_explicit_mesh_cpu_tests.log"
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
PYTHONPATH="$OMX" "$OMX/.venv/bin/python" "$OMX/tests/test_lora.py" \
  >"$LORA_CPU_TEST_LOG" 2>&1
python3 - "$LORA_CPU_TEST_LOG" "$OUT/lora_tp2_explicit_mesh_cpu_tests.json" \
  "${A[expected_lora_test_sha]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
log,out=map(Path,sys.argv[1:3]); expected_test_sha=sys.argv[3]
text=log.read_text(errors='replace')
required=('test_explicit_tp_output_is_resharded_not_asserted',
 'test_explicit_tp_optimizer_step_updates_only_lora','Ran 11 tests','OK')
if any(token not in text for token in required) or 'FAILED (' in text:
 raise SystemExit('FATAL explicit-mesh LoRA CPU tests did not pass')
result={'status':'pass','backend':'cpu','forced_device_count':2,'test_count':11,
 'test_source_sha256':expected_test_sha,
 'test_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
 'explicit_tp_forward_value_and_layout_test':True,
 'explicit_tp_jitted_optimizer_step_test':True,
 'existing_tp1_fresh_behavior_tests':9}
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY

# The parent step-250 tree is a sealed source endpoint. Resume in the new
# artifact's own Orbax root so step-500/750 saves can never mutate the parent.
# Clone through a private staging directory, byte-compare every file, then
# publish the clone atomically. --reflink=auto preserves copy-on-write where
# supported and falls back to an ordinary sparse-preserving copy.
CLONE_TMP="$OUT/.orbax_clone_${SLURM_JOB_ID:-local}.tmp"
[[ ! -e "$ORBAX" && ! -e "$CLONE_TMP" ]] || {
  echo "FATAL refusing to reuse a resume Orbax destination: $ORBAX" >&2; exit 2;
}
mkdir -p "$CLONE_TMP"
cp -a --reflink=auto --sparse=always "$SOURCE_ORBAX/." "$CLONE_TMP/"
python3 - "$SOURCE_ORBAX" "$CLONE_TMP" "$ORBAX" \
  "$OUT/orbax_clone_manifest.json" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
source,clone,final,manifest=map(Path,sys.argv[1:5])
def entries(root):
 out={}
 for path in sorted(root.rglob('*')):
  rel=path.relative_to(root).as_posix()
  mode=path.lstat().st_mode
  if stat.S_ISLNK(mode): out[rel]=('symlink',os.readlink(path))
  elif stat.S_ISDIR(mode): out[rel]=('directory',)
  elif stat.S_ISREG(mode): out[rel]=('file',path.stat().st_size)
  else: raise SystemExit(f'FATAL unsupported Orbax path type: {path}')
 return out
source_entries=entries(source); clone_entries=entries(clone)
if source_entries!=clone_entries:
 raise SystemExit('FATAL resume Orbax clone inventory mismatch')
tree=hashlib.sha256(); file_count=0; logical_bytes=0
for rel,info in source_entries.items():
 tree.update(rel.encode()+b'\0'+info[0].encode()+b'\0')
 if info[0]=='symlink': tree.update(info[1].encode()+b'\0')
 if info[0]!='file': continue
 file_count+=1; logical_bytes+=info[1]
 digest=hashlib.sha256()
 with (source/rel).open('rb') as left,(clone/rel).open('rb') as right:
  while True:
   a=left.read(8*1024*1024); b=right.read(8*1024*1024)
   if a!=b: raise SystemExit(f'FATAL resume Orbax clone differs: {rel}')
   if not a: break
   digest.update(a)
 tree.update(str(info[1]).encode()+b'\0'+digest.digest())
result={'schema_version':1,'status':'pass','copy_mode':'cp -a --reflink=auto --sparse=always',
 'source_root':str(source.resolve()),'destination_root':str(final.resolve()),
 'file_count':file_count,'logical_bytes':logical_bytes,
 'tree_sha256':tree.hexdigest(),'byte_compared_every_file':True,
 'parent_source_is_not_training_save_root':True}
manifest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
mv "$CLONE_TMP" "$ORBAX"
[[ -f "$CKPT250/_CHECKPOINT_METADATA" && ! -e "$CKPT750" ]] || {
  echo "FATAL cloned resume root is not the exact step-250-only endpoint" >&2; exit 2;
}

# Restore only the three scalar optimizer counters on CPU. This proves the
# checkpoint is the completed global update 250 (8 microsteps/update), not a
# merely named or partially committed directory.
cd "$OMX"
JAX_PLATFORMS=cpu uv run --project="$OMX" -- \
  python "$REPO/experiments/synthetic_multistep/typing_checkpoint_scalar.py" \
  --checkpoint="$CKPT250" --out="$OUT/resume_scalar_audit.json" \
  --expected-gradient-step=250 --expected-micro-step=2000

[[ ! -e "$CKPT750" ]] || { echo "FATAL refusing to overwrite existing step750" >&2; exit 2; }
job_tag="${SLURM_JOB_ID:-local}_${LINEAGE}_${A[format]}_resume_r256"
JAX_CACHE="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/jax_cache/typing_${job_tag}"
mkdir -p "$JAX_CACHE"
TRAIN_ENTRYPOINT="scripts/train_vlm_sft.py"
TP2_AUDIT="$OUT/tp2_sharding_audit.json"
TP2_SOURCE_BITWISE="$OUT/tp2_source_bitwise_reference.json"
if [[ "$TP_SIZE" == 2 ]]; then
  TRAIN_ENTRYPOINT="$REPO/experiments/synthetic_multistep/typing_tp2_train_entrypoint.py"
  [[ ! -e "$TP2_AUDIT" && ! -e "$TP2_SOURCE_BITWISE" ]] || {
    echo "FATAL refusing to reuse TP2 audit/reference" >&2; exit 2;
  }
fi
TYPING_TP2_SHARDING_AUDIT="$TP2_AUDIT" TYPING_TP2_CHECKPOINT="$CKPT250" \
TYPING_TP2_SOURCE_BITWISE_REFERENCE="$TP2_SOURCE_BITWISE" \
OMEGALAX_TRAIN_ENTRYPOINT="$OMX/scripts/train_vlm_sft.py" \
uv run --project="$OMX" -- srun --gpus-per-task="$TP_SIZE" python "$TRAIN_ENTRYPOINT" \
  --jax_cache_dir="$JAX_CACHE" --model_id="$SOURCE_HF" --processor="$SOURCE_HF" \
  --data_path="$DATA/${A[format]}/train" \
  --save_dir="$ORBAX" --enable_lora=true --lora_rank=256 --lora_alpha=256 \
  --freeze_vision_tower=false --tp_size="$TP_SIZE" --fsdp_size=1 --dp_size=1 \
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
  --wandb_tags="berlin,typing,factorial,lora,r256,resume250,${A[format]},lineage_${LINEAGE}"
if [[ "$TP_SIZE" == 2 ]]; then
  jq -e '.status == "restore_pass"
    and .device_preflight.status == "pass"
    and .device_preflight.local_device_count == 2
    and .restored_target_shardings_match == true
    and .checkpoint_dtypes_preserved == true
    and .fresh_optimizer_dtype_canonicalization_applied == false
    and .all_train_state_leaves_bitwise_equal_to_cpu_source_restore == true
    and .bitwise_leaf_count == 2772
    and .restored_counters == {"global_gradient_step":250,"optimizer_micro_step":2000,"gradient_accumulation_remainder":0}
    and .restored_iterator_state_exact == true
    and .global_shapes_match == true
    and .physical_rng_shape_exception_resolved == true
    and .target.mesh_shapes == [[2,1,1]]' \
    "$TP2_AUDIT" >/dev/null || { echo "FATAL TP2 restore audit did not pass" >&2; exit 3; }
fi
[[ -f "$CKPT750/_CHECKPOINT_METADATA" && -f "$ORBAX/lora_metadata.json" ]] || {
  echo "FATAL complete step750 typing checkpoint missing" >&2; exit 3;
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
RECOVERY_LOG="$(find "$(dirname "$LABCTL_CONTEXT")" -maxdepth 1 -type f -name "*.log" -print -quit 2>/dev/null || true)"
# LABCTL_CONTEXT lives in .lab/context.json; the Slurm log is its sibling.
[[ -f "$RECOVERY_LOG" ]] || RECOVERY_LOG="$(find "$(dirname "$LABCTL_CONTEXT")" -maxdepth 1 -type f -name "*.log" -print -quit)"
python3 - "$DATA" "$SOURCE" "$PARTIAL" "$OUT" "$LINEAGE" "$SOURCE_ARM" \
  "${A[format]}" "$CKPT250" "$CKPT750" "$OMX_SHA" "$OMX_DIFF_SHA" "$LOG" \
  "$RECOVERY_LOG" "${A[parent_run_id]}" "${A[parent_job_id]}" \
  "$SOURCE_CKPT250" "$OUT/orbax_clone_manifest.json" "$TP_SIZE" "$TP2_AUDIT" \
  "$OUT/offline_nnx_restore_gate_preflight.json" <<'PY'
import hashlib,json,math,re,sys
from pathlib import Path
data,source,partial,out=map(Path,sys.argv[1:5]); lineage,arm,fmt=sys.argv[5:8]
ck250,ck750=map(Path,sys.argv[8:10]); sha,diff=sys.argv[10:12]
parent_log,recovery_log=map(Path,sys.argv[12:14]); parent_run_id=sys.argv[14]; parent_job_id=int(sys.argv[15])
source_ck250=Path(sys.argv[16]); clone_manifest_path=Path(sys.argv[17]); hf=out/'hf'
tp_size=int(sys.argv[18]); tp2_audit_path=Path(sys.argv[19])
offline_nnx_gate=json.loads(Path(sys.argv[20]).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
config=json.loads((hf/'config.json').read_text()); source_config=json.loads((source/'hf/config.json').read_text())
for key in ('architectures','transformers_version','vision_end_token_id'):
 if key in source_config and key not in config: config[key]=source_config[key]
(hf/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ('model.safetensors','config.json','tokenizer_config.json','chat_template.json','preprocessor_config.json'):
 if not (hf/name).is_file(): raise SystemExit(f'FATAL export missing {name}')
recovery_text=recovery_log.read_text(errors='replace')
if 'restored checkpoint at step 250' not in recovery_text or 'finished step=750' not in recovery_text:
 raise SystemExit('FATAL exact resume/completion evidence missing')
if ('RESOURCE_EXHAUSTED' in recovery_text or 'validation loss=' in recovery_text
    or 'built train step (jit) and eval step (jit)' in recovery_text):
 raise SystemExit('FATAL recovery invoked validation or encountered GPU OOM')
trace=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'step=(\d+) loss=([^ ]+) grad_norm=([^ ]+)',recovery_text)]
if [x[0] for x in trace]!=list(range(260,751,10)) or not all(math.isfinite(x) and math.isfinite(y) for _,x,y in trace):
 raise SystemExit(f'FATAL incomplete/nonfinite recovery trace: {trace}')
scalar=json.loads((out/'resume_scalar_audit.json').read_text())
preflight=json.loads((out/'resume_preflight.json').read_text())
cpu_lora_tests=json.loads((out/'lora_tp2_explicit_mesh_cpu_tests.json').read_text())
cuda_lora_smoke=json.loads((out/'lora_cuda_smoke_preflight.json').read_text())
clone_manifest=json.loads(clone_manifest_path.read_text())
if clone_manifest.get('status')!='pass' or not clone_manifest.get('byte_compared_every_file'):
 raise SystemExit('FATAL untrusted resume Orbax clone manifest')
source_hashes={
 'checkpoint_metadata_sha256':digest(source_ck250/'_CHECKPOINT_METADATA'),
 'train_state_metadata_sha256':digest(source_ck250/'train_state/_METADATA'),
 'input_iterator_sha256':digest(source_ck250/'input_iter/process_0-of-1.json'),
 'lora_metadata_sha256':digest(source_ck250.parent/'lora_metadata.json')}
expected_source={k:preflight['endpoint_hashes'][k] for k in source_hashes}
source_entries=sorted(path.name for path in source_ck250.parent.iterdir())
if source_hashes!=expected_source or source_entries!=['000250','config.json','lora_metadata.json']:
 raise SystemExit(f'FATAL sealed parent Orbax changed during recovery: {source_hashes} {source_entries}')
omx_snapshot=preflight['omegalax_snapshot']; omx_root=Path(omx_snapshot['root'])
if (sha!=omx_snapshot['git_commit']
    or diff!=omx_snapshot['tracked_binary_diff_sha256']
    or digest(omx_root/'omegalax/trainers/lora.py')!=omx_snapshot['lora_source_sha256']
    or digest(omx_root/'tests/test_lora.py')!=omx_snapshot['lora_test_sha256']):
 raise SystemExit('FATAL OmegaLAX TP2 repair snapshot changed during recovery')
tp2_audit=None
if tp_size==2:
 tp2_audit=json.loads(tp2_audit_path.read_text())
 if (tp2_audit.get('status')!='restore_pass'
     or not tp2_audit.get('restored_target_shardings_match')
     or not tp2_audit.get('checkpoint_dtypes_preserved')
     or not tp2_audit.get('all_train_state_leaves_bitwise_equal_to_cpu_source_restore')
     or tp2_audit.get('bitwise_leaf_count')!=2772
     or not tp2_audit.get('restored_iterator_state_exact')
     or tp2_audit.get('target',{}).get('mesh_shapes')!=[[2,1,1]]):
  raise SystemExit('FATAL final TP2 topology audit mismatch')
manifest={'artifact_type':'synthetic_typing_factorial_hf_checkpoint','schema_version':1,
 'status':'complete','lineage':lineage,'source_stage1_arm':arm,'target_format':fmt,
 'source_model':str(source.resolve()),'source_manifest_sha256':digest(source/'train_export_manifest.json'),
 'dataset':str(data.resolve()),'dataset_manifest_sha256':digest(data/'typing_dataset_manifest.json'),
 'model_id':'Qwen/Qwen3-VL-8B-Instruct','step':750,'fresh_optimizer':True,
 'exact_resume_from_step':250,'lora_rank':256,'lora_alpha':256,'learning_rate':5e-5,
 'max_length':4096,'hf_subdir':'hf','source_checkpoint':str(ck750.resolve()),
 'sealed_parent_checkpoint250':str(source_ck250.resolve()),
 'recovered_from_terminal_run_root':str(partial.resolve()),
 'recovered_from_terminal_run_id':parent_run_id,'recovered_from_terminal_job_id':parent_job_id,
 'recovery_reason':'in_loop_validation_gpu_oom_after_finalized_step250',
 'recovery_change':'in_loop_validation_disabled_plus_matched_tp1_first250_to_tp2_last500',
 'training_topology':{'tp_size':tp_size,'fsdp_size':1,'dp_size':1,
  'global_batch_size':1,'gradient_accumulation_steps':8,
  'numerical_topology_change_from_parent':tp_size==2},
 'external_frozen_200_example_evaluation_unchanged':True,
 'offline_real_nnx_restore_gate':offline_nnx_gate,
 'lora_tp2_explicit_mesh_cpu_tests':cpu_lora_tests,
 'lora_tp2_native_dot_cuda_smoke':cuda_lora_smoke,
 'resume_preflight':preflight,'resume_scalar_audit':scalar,
 'resume_orbax_clone':clone_manifest,'sealed_parent_orbax_unchanged':True,
 'tp2_sharding_audit':tp2_audit,
 'logged_finite_recovery_steps':[x[0] for x in trace],
 'final_logged_loss':trace[-1][1],'final_logged_grad_norm':trace[-1][2],
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
echo "typing factorial exact-resume complete: lineage=$LINEAGE format=${A[format]}"
