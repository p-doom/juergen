#!/bin/bash
# Serve one exported cell, prove chat-completion readiness, then run greedy k=1 eval.
set -euo pipefail

declare -A A
for arg in "$@"; do
    case "$arg" in
        --*=*) key="${arg%%=*}"; key="${key#--}"; A[$key]="${arg#*=}" ;;
        *) echo "FATAL unexpected positional argument: $arg" >&2; exit 2 ;;
    esac
done
for key in experiment_dir prime_rl_repo model_path out grammar preamble audit_dir; do
    [[ -n "${A[$key]:-}" ]] || { echo "FATAL missing --$key" >&2; exit 2; }
done
case "${A[grammar]}" in
    move_rel|deltatype_raw|absolute_toolcall|absolute_raw) ;;
    *) echo "FATAL unsupported grammar: ${A[grammar]}" >&2; exit 2 ;;
esac
case "${A[preamble]}" in true|false) ;; *) echo "FATAL preamble must be true/false" >&2; exit 2 ;; esac

MODEL_PATH="${A[model_path]}"
OUT="${A[out]}"
mkdir -p "$OUT"
python3 - "$MODEL_PATH" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
config_path = root / "config.json"
if not config_path.is_file():
    raise SystemExit(f"FATAL export config missing: {config_path}")
config = json.loads(config_path.read_text())
if not config.get("architectures"):
    raise SystemExit("FATAL export config lacks architectures; vLLM would resolve to pooling")
if not ((root / "model.safetensors").is_file() or (root / "model.safetensors.index.json").is_file()):
    raise SystemExit(f"FATAL export weights missing: {root}")
PY

cache="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp/relative_factorial_vllm_${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$cache"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export TRITON_CACHE_DIR="$cache/triton"
export CUDA_HOME=/fast/service/apps/software/CUDA/12.6.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

PY="${A[prime_rl_repo]}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "FATAL prime-rl Python missing: $PY" >&2; exit 2; }
port=$((18000 + (${SLURM_JOB_ID:-1} % 1000)))
base_url="http://127.0.0.1:${port}/v1"
cd "${A[prime_rl_repo]}"
uv run --no-sync vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 --port "$port" --served-model-name policy \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager \
    >"$OUT/vllm.log" 2>&1 &
server_pid=$!
trap 'kill -9 "$server_pid" 2>/dev/null || true' EXIT

"$PY" "${A[experiment_dir]}/readiness.py" \
    --base-url "$base_url" --model policy --grammar "${A[grammar]}" \
    --timeout-s 900 --pid "$server_pid"

eval_args=(
    --audit-dir "${A[audit_dir]}"
    --out "$OUT"
    --base-url "$base_url"
    --model policy
    --model-dir "$MODEL_PATH"
    --grammar "${A[grammar]}"
    --concurrency 24
)
if [[ "${A[preamble]}" == true ]]; then
    eval_args+=(--preamble)
fi
"$PY" "${A[experiment_dir]}/evaluate.py" "${eval_args[@]}"
echo "relative factorial eval complete: ${A[grammar]} preamble=${A[preamble]} -> $OUT"
