#!/bin/bash
# Recover a validated Phase-B step900 endpoint into a registered HF artifact.
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
for key in arm source_job_id source_checkpoint_root training_log omegalax_repo model_id output; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}" in
    prose_keep) expected_job=135312; expected_root=pb_prose_keep_r32 ;;
    prose_strip) expected_job=135313; expected_root=pb_prose_strip_r32 ;;
    *) echo "FATAL unsupported arm: ${A[arm]}" >&2; exit 2 ;;
esac
[[ "${A[source_job_id]}" == "$expected_job" ]] || { echo "FATAL source job mismatch" >&2; exit 2; }
SRC="${A[source_checkpoint_root]}"; CKPT="$SRC/000900"; OUT="${A[output]}"; HF="$OUT/hf"
[[ "$(basename "$SRC")" == "$expected_root" ]] || { echo "FATAL source root mismatch" >&2; exit 2; }
state="$(sacct -X -n -P -j "$expected_job" -o State | head -1 | cut -d'|' -f1)"
[[ "$state" == FAILED* ]] || { echo "FATAL expected failed-after-export source job, got $state" >&2; exit 2; }
[[ -f "$CKPT/_CHECKPOINT_METADATA" && -f "$SRC/lora_metadata.json" ]] || {
    echo "FATAL complete step900 endpoint or LoRA metadata missing" >&2; exit 3;
}
grep -Fq "finished step=900" "${A[training_log]}" || { echo "FATAL step900 completion line missing" >&2; exit 3; }
grep -Fq "=== R3 SFT DONE ($expected_root, rank=32, steps=900)" "${A[training_log]}" || {
    echo "FATAL fixed training completion line missing" >&2; exit 3;
}
grep -Fq "ValueError: coordinator_address should be defined." "${A[training_log]}" || {
    echo "FATAL source failure is not the audited export-only failure" >&2; exit 3;
}
if find "$CKPT" -name '*.orbax-checkpoint-tmp' -o -name '*PLACEHOLDER*' | grep -q .; then
    echo "FATAL partial content below step900" >&2; exit 3
fi

mkdir -p "$HF"
cd "${A[omegalax_repo]}"
# srun supplies a coherent one-task Slurm cluster environment.  Calling the
# exporter directly from sbatch leaves SLURM_JOB_ID without SLURM_PROCID and
# makes jax.distributed.initialize() fail coordinator autodetection.
JAX_PLATFORMS=cpu srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    uv run --project="${A[omegalax_repo]}" -- \
    python scripts/export_to_hf.py --model_id="${A[model_id]}" \
    --checkpoint_path="$CKPT" --out_dir="$HF" \
    --tp_size=1 --fsdp_size=1 --dp_size=1 --max_grad_norm=1.0 --grad_accum_steps=8

SLUG="${A[model_id]//\//--}"; BASE="${HF_HOME}/hub/models--${SLUG}/snapshots"
STOCK="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
[[ -f "$STOCK/config.json" ]] || { echo "FATAL stock snapshot missing" >&2; exit 3; }
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
    [[ ! -f "$STOCK/$f" || -f "$HF/$f" ]] || cp "$STOCK/$f" "$HF/"
done
python3 - "$STOCK/config.json" "$HF/config.json" "$OUT/export_manifest.json" \
    "${A[arm]}" "$expected_job" "$CKPT" "$SRC/lora_metadata.json" "${A[training_log]}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
stock,config,manifest=map(Path,sys.argv[1:4])
arm,source_job=sys.argv[4:6]; checkpoint,lora,log=map(Path,sys.argv[6:9])
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()
s=json.loads(stock.read_text()); c=json.loads(config.read_text())
for key in ('architectures','transformers_version','vision_end_token_id'):
    if key in s and key not in c: c[key]=s[key]
if not c.get('architectures'): raise SystemExit('FATAL architectures missing')
config.write_text(json.dumps(c,indent=2)+'\n')
weights=sorted(config.parent.glob('*.safetensors'))
if not weights or any(path.stat().st_size <= 0 for path in weights):
    raise SystemExit('FATAL exported weights missing/empty')
payload={
 'artifact_type':'phaseb_absolute_hf_checkpoint','schema_version':1,'status':'complete',
 'arm':arm,'model_id':'Qwen/Qwen3-VL-8B-Instruct','source_training_job_id':source_job,
 'source_training_state':'FAILED_AFTER_COMPLETE_STEP900_DURING_INLINE_EXPORT',
 'source_checkpoint':str(checkpoint.resolve()),'step':900,'lora_rank':32,'lora_alpha':32,
 'max_length':16384,'hf_subdir':'hf','checkpoint_metadata_sha256':sha(checkpoint/'_CHECKPOINT_METADATA'),
 'lora_metadata_sha256':sha(lora),'training_log_sha256':sha(log),
 'export_slurm_job_id':os.environ.get('SLURM_JOB_ID'),'config_sha256':sha(config),
 'weights':[{'name':p.name,'size':p.stat().st_size,'sha256':sha(p)} for p in weights],
}
manifest.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
echo "Phase-B recovery export complete: $OUT"
