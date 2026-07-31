#!/bin/bash
# One-checkpoint, one-semantic Phase-A evaluation. Invoked only through labctl.
set -euo pipefail

declare -A A
for arg in "$@"; do
  case "$arg" in
    --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
    *) echo "FATAL unexpected argument: $arg" >&2; exit 2 ;;
  esac
done
PREAMBLE="${A[preamble]:-false}"
case "$PREAMBLE" in true|false) ;; *) echo "FATAL preamble must be true/false" >&2; exit 2 ;; esac
for key in model episodes out runtime_dir experiment_dir audit_dir semantic checkpoint_alias comparison_label; do
  [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done

MODEL="${A[model]}/hf"
EPISODES="${A[episodes]}"
OUT="${A[out]}"
PY="${A[runtime_dir]}/.venv/bin/python"
[[ -f "$MODEL/config.json" && -f "$EPISODES/build_manifest.json" ]] || exit 3
mkdir -p "$OUT"

# Fail before acquiring/serving weights if the CPU-checkable artifact contract drifted.
"$PY" - "${A[experiment_dir]}" "$EPISODES" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from evaluate import validate_episode_artifact
validate_episode_artifact(pathlib.Path(sys.argv[2]))
print("episode artifact and 100% oracle guard validated")
PY

CACHE_ROOT="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/synthetic_multistep_${SLURM_JOB_ID:-local}_${A[semantic]}"
export VLLM_CACHE_ROOT="$CACHE_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

PORT=$((22000 + (${SLURM_JOB_ID:-1} % 1000)))
URL="http://127.0.0.1:${PORT}/v1"
cd "${A[runtime_dir]}"
uv run --no-sync vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name policy --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --enforce-eager >"$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill -9 "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "$URL/models" >"$OUT/served_models.json" 2>/dev/null; then break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -100 "$OUT/vllm.log"; exit 4; }
  sleep 5
done
curl -fsS "$URL/models" >"$OUT/served_models.json"

EXTRA_ARGS=()
if [[ "$PREAMBLE" == true ]]; then
  EXTRA_ARGS+=(--preamble)
fi
"$PY" "${A[experiment_dir]}/evaluate.py" \
  --base-url "$URL" --model policy --model-dir "$MODEL" \
  --checkpoint-alias "${A[checkpoint_alias]}" --episodes "$EPISODES" --out "$OUT" \
  --audit-dir "${A[audit_dir]}" --semantic "${A[semantic]}" \
  --comparison-label "${A[comparison_label]}" --concurrency 24 "${EXTRA_ARGS[@]}"

test -s "$OUT/eval_manifest.json"
