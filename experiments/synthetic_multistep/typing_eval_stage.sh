#!/bin/bash
# Immutable generation-execution plus teacher-forced evaluation for one typing cell.
set -euo pipefail
declare -A A
for arg in "$@"; do
  case "$arg" in --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}";;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2;; esac
done
for key in model dataset out runtime parser experiment_dir lineage format; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[lineage]}" in A|B) ;; *) echo "FATAL bad lineage" >&2; exit 2;; esac
case "${A[format]}" in coalesced|perkey) ;; *) echo "FATAL bad format" >&2; exit 2;; esac
MODEL_ROOT="${A[model]}"; MODEL="$MODEL_ROOT/hf"; DATA="${A[dataset]}"; OUT="${A[out]}"
PY="${A[runtime]}/.venv/bin/python"
mkdir -p "$OUT"
if [[ -n "${A[expected_eval_source_sha]:-}" ]]; then
  actual_eval_source_sha="$(sha256sum "${A[experiment_dir]}/typing_evaluate.py" | awk '{print $1}')"
  [[ "$actual_eval_source_sha" == "${A[expected_eval_source_sha]}" ]] || {
    echo "FATAL typing evaluator source snapshot mismatch" >&2; exit 2;
  }
fi

"$PY" - "${A[experiment_dir]}" "$MODEL_ROOT" "$DATA" "${A[lineage]}" "${A[format]}" "${A[parser]}" <<'PY'
import pathlib,sys
sys.path.insert(0,sys.argv[1])
from typing_evaluate import validate
validate(pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]),sys.argv[4],sys.argv[5],pathlib.Path(sys.argv[6]))
print('typing model/dataset/parser preflight passed')
PY

CACHE_ROOT="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/typing_eval_${SLURM_JOB_ID:-local}_${A[lineage]}_${A[format]}"
export VLLM_CACHE_ROOT="$CACHE_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
PORT=$((24000 + (${SLURM_JOB_ID:-1} % 1000)))
URL="http://127.0.0.1:${PORT}/v1"
cd "${A[runtime]}"
setsid --wait uv run --no-sync vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name policy --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --enforce-eager >"$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
shutdown_vllm() {
  # `uv run vllm serve` owns a multiprocessing EngineCore. Killing only the
  # uv wrapper leaves that GPU-resident child alive and starves the subsequent
  # teacher-forced model load. The dedicated session makes the whole server
  # tree one process group that can be drained before scoring begins.
  kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- "-$VLLM_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  if kill -0 -- "-$VLLM_PID" 2>/dev/null; then
    echo "FATAL vLLM process group survived shutdown" >&2
    return 1
  fi
}
trap shutdown_vllm EXIT
for _ in $(seq 1 180); do
  if curl -fsS "$URL/models" >"$OUT/served_models.json" 2>/dev/null; then break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -100 "$OUT/vllm.log"; exit 4; }
  sleep 5
done
curl -fsS "$URL/models" >"$OUT/served_models.json"
"$PY" "${A[experiment_dir]}/typing_evaluate.py" \
  --base-url "$URL" --model policy --model-root "$MODEL_ROOT" --dataset "$DATA" \
  --parser-dir "${A[parser]}" --lineage "${A[lineage]}" --format "${A[format]}" \
  --out "$OUT" --concurrency 24
shutdown_vllm
trap - EXIT

"$PY" "${A[experiment_dir]}/typing_teacher_forced.py" \
  --model-root "$MODEL_ROOT" --dataset "$DATA" --parser-dir "${A[parser]}" \
  --lineage "${A[lineage]}" --format "${A[format]}" --out "$OUT"

python3 - "$OUT" "${A[lineage]}" "${A[format]}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1]); lineage,fmt=sys.argv[2:4]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
generation=json.loads((out/'typing_generation_manifest.json').read_text())
teacher=json.loads((out/'typing_teacher_forced_report.json').read_text())
if generation.get('status')!='complete' or teacher.get('status')!='complete':
 raise SystemExit('FATAL incomplete typing evaluation components')
for value in (generation,teacher):
 if value.get('lineage')!=lineage or value.get('target_format')!=fmt:
  raise SystemExit('FATAL typing evaluation cell mismatch')
g_hash=generation['model_manifest']['evaluation_input_hashes']['model_manifest_sha256']
t_hash=teacher['model_manifest']['evaluation_input_hashes']['model_manifest_sha256']
if g_hash!=t_hash: raise SystemExit('FATAL model changed between generation and teacher scoring')
manifest={'schema_version':1,'artifact_type':'synthetic_typing_factorial_cell_eval',
 'status':'complete','lineage':lineage,'target_format':fmt,'n_examples':200,
 'model_manifest_sha256':g_hash,
 'generation_manifest_sha256':digest(out/'typing_generation_manifest.json'),
 'generation_rows_sha256':digest(out/'typing_generation_rows.jsonl'),
 'generation_report_sha256':digest(out/'typing_generation_report.json'),
 'teacher_forced_rows_sha256':digest(out/'typing_teacher_forced_rows.jsonl'),
 'teacher_forced_report_sha256':digest(out/'typing_teacher_forced_report.json')}
path=out/'typing_eval_manifest.json'; tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
echo "typing evaluation complete: ${A[lineage]}/${A[format]}"
