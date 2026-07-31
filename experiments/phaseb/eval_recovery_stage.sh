#!/bin/bash
# Evaluate a registered CPU-recovered Phase-B HF artifact on its own validation prompts.
set -euo pipefail
declare -A A
for arg in "$@"; do case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
esac; done
for key in arm source_job_id source_checkpoint_root model_artifact val_chat training_log \
           training_script audit_dir runtime_dir experiment_dir out; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[arm]}" in
    prose_keep) expected_job=135312; expected_root=pb_prose_keep_r32 ;;
    prose_strip) expected_job=135313; expected_root=pb_prose_strip_r32 ;;
    *) echo "FATAL unsupported arm" >&2; exit 2 ;;
esac
[[ "${A[source_job_id]}" == "$expected_job" \
   && "$(basename "${A[source_checkpoint_root]}")" == "$expected_root" ]] || exit 2
MODEL_ARTIFACT="${A[model_artifact]}"; MODEL="$MODEL_ARTIFACT/hf"
EXPORT="$MODEL_ARTIFACT/export_manifest.json"; CKPT="${A[source_checkpoint_root]}/000900"
[[ "${A[val_chat]}" == "${A[audit_dir]}/phaseb/${A[arm]}/_normalized/val/chat.jsonl" ]] || {
    echo "FATAL cross-arm validation prompts" >&2; exit 2;
}
state="$(sacct -X -n -P -j "$expected_job" -o State | head -1 | cut -d'|' -f1)"
[[ "$state" == FAILED* ]] || { echo "FATAL unexpected source state: $state" >&2; exit 2; }
[[ -f "$CKPT/_CHECKPOINT_METADATA" && -f "$EXPORT" && -f "$MODEL/config.json" ]] || exit 3
grep -Fq "=== R3 SFT DONE ($expected_root, rank=32, steps=900)" "${A[training_log]}" || exit 3
grep -Fq "ValueError: coordinator_address should be defined." "${A[training_log]}" || exit 3
python3 - "$EXPORT" "$MODEL" "${A[arm]}" "$expected_job" "$CKPT" <<'PY'
import json,sys
from pathlib import Path
manifest,model=map(Path,sys.argv[1:3]); arm,job=sys.argv[3:5]; ckpt=Path(sys.argv[5]).resolve()
m=json.loads(manifest.read_text())
if (m.get('artifact_type')!='phaseb_absolute_hf_checkpoint' or m.get('status')!='complete'
    or m.get('arm')!=arm or m.get('source_training_job_id')!=job or m.get('step')!=900
    or Path(m.get('source_checkpoint','')).resolve()!=ckpt or not m.get('export_slurm_job_id')):
 raise SystemExit('FATAL recovery export manifest mismatch')
if not json.loads((model/'config.json').read_text()).get('architectures'):
 raise SystemExit('FATAL architectures missing')
if not list(model.glob('*.safetensors')): raise SystemExit('FATAL weights missing')
PY
export_job="$(jq -r '.export_slurm_job_id' "$EXPORT")"
export_state="$(sacct -X -n -P -j "$export_job" -o State | head -1 | cut -d'|' -f1)"
[[ "$export_state" == COMPLETED* ]] || { echo "FATAL export job not complete: $export_state" >&2; exit 3; }

OUT="${A[out]}"; mkdir -p "$OUT"; rm -f "$OUT/eval_manifest.json" "$OUT/report.json"
PY="${A[runtime_dir]}/.venv/bin/python"
"$PY" "${A[audit_dir]}/phaseb_eval.py" --val_chat "${A[val_chat]}" \
    --out "$OUT/selftest" --selftest --tag "phaseb/${A[arm]}/recovery-selftest"
cache="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/phaseb_recovery_vllm_${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="$cache" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
port=$((19000 + (SLURM_JOB_ID % 1000))); url="http://127.0.0.1:${port}/v1"
cd "${A[runtime_dir]}"
uv run --no-sync vllm serve "$MODEL" --host 127.0.0.1 --port "$port" \
    --served-model-name policy --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager \
    >"$OUT/vllm.log" 2>&1 &
pid=$!; trap 'kill -9 "$pid" 2>/dev/null || true' EXIT
"$PY" "${A[experiment_dir]}/../relative_factorial/readiness.py" \
    --base-url "$url" --model policy --grammar absolute_toolcall --timeout-s 900 --pid "$pid"
"$PY" "${A[audit_dir]}/phaseb_eval.py" --val_chat "${A[val_chat]}" \
    --base_url "$url" --model policy --out "$OUT" --max_tokens 256 --concurrency 12 \
    --selftest --tag "phaseb/${A[arm]}/recovery"
"$PY" "${A[experiment_dir]}/finalize_recovery.py" --arm "${A[arm]}" \
    --source-job-id "$expected_job" --source-checkpoint-root "${A[source_checkpoint_root]}" \
    --model-artifact "$MODEL_ARTIFACT" --val-chat "${A[val_chat]}" \
    --training-log "${A[training_log]}" --training-script "${A[training_script]}" \
    --evaluator "${A[audit_dir]}/phaseb_eval.py" --out "$OUT"
echo "Phase-B recovered ${A[arm]} own-val evaluation complete: $OUT"
